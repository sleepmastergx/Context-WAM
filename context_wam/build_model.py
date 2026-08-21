"""Build FastWAM-M5-fused_mlp with the per-layer TTT memory attached.

One place that knows how the pieces connect, so the trainer and the eval
cannot drift apart:

    model  = create_fastwam_decoupled(**cfg.model)      # theirs
    memory = PerLayerEpisodeMemory(n_layers=5, ...)     # ours
    tag_action_blocks(model.mot.mixtures['action'])     # stamp layer ids
    patch_mot_action_expert(model.mot, memory)          # read seam

Fast-WAM source resolution (portable): $FASTWAM_SRC if set, else the vendored
copy at <repo>/third_party/fastwam/src (MIT, pinned — see PROVENANCE.md).

Write/read discipline (checks/check_write_once.py gates it):
    memory.advance(latents, proprio)   ONCE per window, before denoising
    the 5 action layers then READ their state on every denoising step
For the sliding-w arm the advance cadence is owned by sliding_chain.py; the
seam and the fp32 rule below are unchanged.
"""
import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[1]
FASTWAM_SRC = os.environ.get(
    "FASTWAM_SRC", str(_REPO / "third_party" / "fastwam" / "src"))
sys.path.insert(0, FASTWAM_SRC)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from per_layer_memory import (PerLayerEpisodeMemory,  # noqa: E402
                              patch_mot_action_expert, tag_action_blocks)


# keys we add to the model config that THEIR factory must never see
OURS = ("memory", "proprio_dim_value")


def build(cfg, device="cuda"):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # hydra passes every key to create_fastwam_decoupled(); our `memory:` block
    # is not one of its kwargs and raised
    #   TypeError: got an unexpected keyword argument 'memory'
    theirs = OmegaConf.create(
        {k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items()
         if k not in OURS})
    model = instantiate(theirs)
    mem_cfg = cfg.model.get("memory", None)
    if mem_cfg is None or not mem_cfg.get("enabled", False):
        return model, None

    action_expert = model.mot.mixtures["action"]
    n_layers = len(action_expert.blocks)
    if int(mem_cfg["n_layers"]) != n_layers:
        raise ValueError(
            f"memory.n_layers={mem_cfg['n_layers']} but the action expert has "
            f"{n_layers} blocks — the per-layer states would silently misalign.")

    # The seam is `mixed_attn_out` (mot.py:121), the attention output BEFORE
    # self_attn.o -- width num_heads*attn_head_dim, not the expert width. Read
    # it off the model rather than re-deriving it from the config, so a config
    # change can never silently desynchronise the readout from the seam.
    seam_dims = [blk.self_attn.o.in_features for blk in action_expert.blocks]

    memory = PerLayerEpisodeMemory(
        n_layers=n_layers,
        hidden_dim=int(mem_cfg["hidden_dim"]),
        seam_dims=seam_dims,
        latent_channels=int(mem_cfg["latent_channels"]),
        proprio_dim=cfg.model.get("proprio_dim", None),
        d_k=int(mem_cfg["d_k"]), d_v=int(mem_cfg["d_v"]),
        d_hidden=int(mem_cfg["d_hidden"]), d_out=int(mem_cfg["d_out"]),
        chunk=int(mem_cfg["chunk"]), gate_init=float(mem_cfg["gate_init"]),
        share_cell=bool(mem_cfg.get("share_cell", False)),
    ).to(device)

    tag_action_blocks(action_expert)
    patch_mot_action_expert(model.mot, memory)
    # fp32 even under bf16 training: the inner TTT update is a gradient taken
    # in the forward pass and quantizes to noise in bf16. This is also why the
    # memory must stay OUTSIDE the DeepSpeed engine (train.py).
    memory.float()
    return model, memory


def trainable_parameters(model, memory):
    """Their recipe: model.dit (= mot, BOTH experts) + proprio_encoder train.

    Mirrors `_apply_dit_only_train_mode` in their trainer — freeze everything,
    then unfreeze the MoT and proprio encoder; VAE and T5 stay frozen. That is
    ~5B trainable, so this MUST be launched under the ZeRO-1 config or AdamW
    state alone (56 GiB) will not fit.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    groups = []
    model.mot.requires_grad_(True)
    groups.append({"params": [p for p in model.mot.parameters()]})
    pro = getattr(model, "proprio_encoder", None)
    if pro is not None:
        pro.requires_grad_(True)
        groups.append({"params": list(pro.parameters())})

    if memory is not None:
        memory.requires_grad_(True)
        # alpha starts at 1e-3 and MUST grow for the memory to do anything;
        # decaying it fights the experiment
        alpha = [memory.alpha]
        rest = [p for n, p in memory.named_parameters() if n != "alpha"]
        groups.append({"params": rest})
        groups.append({"params": alpha, "weight_decay": 0.0})
    return groups


def memory_param_groups(memory):
    """The memory's own optimizer groups (train.py keeps it outside the
    DeepSpeed engine so it stays fp32; see build() note)."""
    alpha = [memory.alpha]
    rest = [p for n, p in memory.named_parameters() if n != "alpha"]
    return [{"params": rest}, {"params": alpha, "weight_decay": 0.0}]

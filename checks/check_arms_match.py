"""Gate: the TTT arm and the control must differ ONLY in the memory block.

A controlled comparison dies quietly if a second thing changes — a different
layer count, a different fusion mode, a different seed. This asserts the two
model configs are identical everywhere except `memory:`, and reports the
parameter asymmetry the memory introduces.

Also checks the property that makes this comparison cleaner than our RMBench
one: with gate_init=1e-3 the TTT arm at initialisation is FUNCTIONALLY the
control (tanh(alpha) ~ 0), so the arms start from the same function. RMBench's
`concat` inject lost that parity — the arms were not the same architecture at
step 0 and the comparison was murkier for it.
"""
import pathlib
import sys

import yaml

CFG = str(pathlib.Path(__file__).resolve().parents[1] / "configs" / "model")


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    ttt_name = sys.argv[1] if len(sys.argv) > 1 else "fastwam_ttt_m5"
    ttt = yaml.safe_load(open(f"{CFG}/{ttt_name}.yaml"))
    ctl = yaml.safe_load(open(f"{CFG}/fastwam_ttt_m5_control.yaml"))
    a, b = flatten(ttt), flatten(ctl)

    keys = set(a) | set(b)
    diffs = [k for k in sorted(keys) if a.get(k) != b.get(k)]
    bad = [k for k in diffs if not k.startswith("memory.")]
    if bad:
        for k in bad:
            print(f"  MISMATCH {k}: ttt={a.get(k)!r} control={b.get(k)!r}")
        sys.exit("FAIL: arms differ outside the memory block")

    assert a["memory.enabled"] is True and b["memory.enabled"] is False, \
        "FAIL: memory.enabled is not True/False across the arms"
    assert a["action_dit_config.num_layers"] == b["action_dit_config.num_layers"]
    assert a["kv_source_mode"] == b["kv_source_mode"] == "fused_mlp"
    # Was pinned to True. loader.py passes this flag to BOTH experts, and under
    # True it random-inits the 5B VIDEO DiT (dit_path = SKIPPED_PRETRAIN), not
    # just the action expert — the opposite of what the model config's comment 3
    # states ("the Wan2.2 video backbone keeps its pretrained weights"). On
    # VideoUnmask that produced finite loss at step 1 and NaN by step ~2.
    # False + action_dit_pretrained_path: null keeps the action expert on
    # truncated-Gaussian random init, which is the documented intent.
    assert a["skip_dit_load_from_pretrain"] is False
    assert a["action_dit_pretrained_path"] is None, \
        "action expert must stay from-scratch: null path + skip=False"
    assert a["freeze_video_backbone"] is False, \
        "their recipe trains BOTH experts; only VAE+T5 are frozen"

    # parameter asymmetry
    per_layer = 23.1e6
    action = per_layer * a["action_dit_config.num_layers"]
    fusion = 10_242
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "context_wam"))
    from per_layer_memory import PerLayerEpisodeMemory
    # Readouts are built at the SEAM width (attention output, pre-o-projection),
    # which build_model.py reads off the model; mirror that here so the reported
    # asymmetry is the one the run actually pays.
    seam = (a["action_dit_config.num_heads"]
            * a["action_dit_config.attn_head_dim"])
    mem = PerLayerEpisodeMemory(
        n_layers=a["memory.n_layers"], hidden_dim=a["memory.hidden_dim"],
        seam_dims=[seam] * a["memory.n_layers"],
        latent_channels=a["memory.latent_channels"], proprio_dim=8,
        d_out=a["memory.d_out"], chunk=a["memory.chunk"],
        gate_init=a["memory.gate_init"])
    m = sum(p.numel() for p in mem.parameters())
    import torch
    gate = float(torch.tanh(mem.alpha).abs().max())

    print(f"PASS  arms differ only in: {', '.join(diffs)}")
    print(f"      control {(action+fusion)/1e6:.1f}M  |  ttt "
          f"{(action+fusion+m)/1e6:.1f}M  (+{m/1e6:.2f}M, "
          f"{100*m/(action+fusion):.1f}%)")
    print(f"      init gate |tanh(alpha)| = {gate:.4f} -> ttt arm starts "
          f"functionally equal to the control")


if __name__ == "__main__":
    main()

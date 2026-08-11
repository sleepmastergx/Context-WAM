import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from fastwam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    DiTBlock,
    RMSNorm,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    RANDOM_INIT_STD = 0.02
    ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        use_gradient_checkpointing: bool = False,
        action_text_cross_attn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.action_text_cross_attn = action_text_cross_attn

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        if action_text_cross_attn:
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.text_embedding = None
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

        if not action_text_cross_attn:
            for block in self.blocks:
                del block.cross_attn
                del block.norm3

    def initialize_random_weights(self, std: float = RANDOM_INIT_STD) -> None:
        """Initialize a non-pretrained ActionDiT with a truncated Gaussian."""
        if std <= 0:
            raise ValueError(f"`std` must be > 0, got {std}")

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=std,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        # DiT modulation tensors are raw Parameters rather than nn.Linear
        # weights, so initialize them explicitly with the same distribution.
        for name, parameter in self.named_parameters():
            if name.endswith("modulation"):
                nn.init.trunc_normal_(
                    parameter,
                    mean=0.0,
                    std=std,
                )

    @classmethod
    def _from_random_init(
        cls,
        action_dit_config: dict[str, Any],
        device: str,
        torch_dtype: torch.dtype,
    ) -> "ActionDiT":
        model = cls(**action_dit_config)
        model.initialize_random_weights()
        return model.to(device=device, dtype=torch_dtype)

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        layer_init_mapping: list[int] | None = None,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert with truncated Gaussian (std=%.3f) "
                "and expecting checkpoint override.",
                cls.RANDOM_INIT_STD,
            )
            return cls._from_random_init(action_dit_config, device, torch_dtype)

        if not action_dit_pretrained_path:
            if layer_init_mapping is not None:
                logger.warning(
                    "layer_init_mapping=%s was provided but action_dit_pretrained_path is not set. "
                    "The mapping will be ignored and weights will be randomly initialized.",
                    layer_init_mapping,
                )
            logger.info(
                "No `action_dit_pretrained_path` provided; initializing ActionDiT "
                "with truncated Gaussian (std=%.3f).",
                cls.RANDOM_INIT_STD,
            )
            return cls._from_random_init(action_dit_config, device, torch_dtype)

        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if layer_init_mapping is not None and key == "num_layers":
                continue
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        if layer_init_mapping is not None:
            checkpoint_num_layers = int(meta["num_layers"])
            model_num_layers = int(action_cfg["num_layers"])
            if len(layer_init_mapping) != model_num_layers:
                raise ValueError(
                    f"layer_init_mapping length {len(layer_init_mapping)} != "
                    f"model num_layers {model_num_layers}"
                )
            for idx in layer_init_mapping:
                if idx < 0 or idx >= checkpoint_num_layers:
                    raise ValueError(
                        f"layer_init_mapping index {idx} out of range "
                        f"[0, {checkpoint_num_layers})"
                    )

            remapped_backbone: dict[str, torch.Tensor] = {}
            for key, value in backbone_state_dict.items():
                if not key.startswith("blocks."):
                    remapped_backbone[key] = value
            for action_idx, source_idx in enumerate(layer_init_mapping):
                source_prefix = f"blocks.{source_idx}."
                target_prefix = f"blocks.{action_idx}."
                for key, value in backbone_state_dict.items():
                    if key.startswith(source_prefix):
                        suffix = key[len(source_prefix):]
                        remapped_backbone[target_prefix + suffix] = value

            backbone_state_dict = remapped_backbone

        if not action_expert.action_text_cross_attn:
            text_keys_to_filter = [
                key for key in backbone_state_dict
                if key.startswith("text_embedding.")
                or ".cross_attn." in key
                or ".norm3." in key
            ]
            for key in text_keys_to_filter:
                del backbone_state_dict[key]
            if text_keys_to_filter:
                logger.info(
                    "Filtered %d text cross-attn keys from backbone "
                    "(action_text_cross_attn=False): %s%s",
                    len(text_keys_to_filter),
                    text_keys_to_filter[:5],
                    "..." if len(text_keys_to_filter) > 5 else "",
                )

        provided_keys = set(backbone_state_dict.keys())
        missing_keys = sorted(expected_backbone_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_backbone_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Action backbone key mismatch in preprocessed payload. "
                f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
            )

        merged_state = dict(action_state)
        for key in expected_backbone_keys:
            value = backbone_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        if layer_init_mapping is not None:
            # Post-load verification: spot-check one param per block against the checkpoint.
            probe_key_suffix = "self_attn.q.weight"
            verification_ok = True
            for action_idx, source_idx in enumerate(layer_init_mapping):
                model_key = f"blocks.{action_idx}.{probe_key_suffix}"
                ckpt_key = f"blocks.{source_idx}.{probe_key_suffix}"
                if model_key not in merged_state or ckpt_key not in payload["backbone_state_dict"]:
                    continue
                model_param = action_expert.state_dict()[model_key].cpu()
                ckpt_param = payload["backbone_state_dict"][ckpt_key].to(dtype=model_param.dtype)
                if not torch.equal(model_param, ckpt_param):
                    logger.error(
                        "INIT VERIFICATION FAILED: action block %d (%s) does not match "
                        "backbone block %d. max_diff=%.6e",
                        action_idx, model_key, source_idx,
                        (model_param - ckpt_param).abs().max().item(),
                    )
                    verification_ok = False
            if verification_ok:
                logger.info(
                    "INIT VERIFICATION PASSED: all %d action blocks match their source backbone blocks.",
                    len(layer_init_mapping),
                )
            else:
                logger.error("INIT VERIFICATION FAILED: see errors above.")

            logger.info(
                "Loaded ActionDiT backbone (selective) from %s: %d-layer checkpoint -> %d action blocks. "
                "Mapping: %s (random_kept_prefixes=%s).",
                action_dit_pretrained_path,
                int(meta["num_layers"]),
                int(action_cfg["num_layers"]),
                [f"action_{i} <- backbone_{s}" for i, s in enumerate(layer_init_mapping)],
                list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
            )
        else:
            logger.info(
                "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
                action_dit_pretrained_path,
                len(expected_backbone_keys),
                list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
            )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")

        batch_size = action_tokens.shape[0]

        if self.text_embedding is None:
            if timestep.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
                )
            if timestep.shape[0] == 1 and batch_size > 1:
                if self.training:
                    raise ValueError("During training, action timestep length must match batch_size.")
                timestep = timestep.expand(batch_size)

            seq_len = action_tokens.shape[1]
            if seq_len > self.freqs.shape[0]:
                raise ValueError(
                    f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
                )

            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
            tokens = self.action_encoder(action_tokens)
            freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

            return {
                "tokens": tokens,
                "freqs": freqs,
                "t": t,
                "t_mod": t_mod,
                "context": None,
                "context_mask": None,
                "meta": {
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                },
            }

        if context is None:
            raise ValueError("`context` is required when `action_text_cross_attn=True`.")
        if context.ndim != 3:
            raise ValueError(
                f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}"
            )

        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        context_emb = self.text_embedding(context)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x,
                    context,
                    t_mod,
                    freqs,
                    context_mask=context_mask,
                )
            else:
                x = block(x, context, t_mod, freqs, context_mask=context_mask)

        return self.post_dit(x, pre_state)

# Author: Rui Heng Yang

from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


def _infer_new_fused_kv_projection_mode(mot_state: dict[str, torch.Tensor]) -> Optional[str]:
    """Infer the new_fused_kv projection mode from its unique state keys."""
    detected_modes = []
    if "k_video_pos_projection" in mot_state:
        detected_modes.append("simple+PE")
    elif "simple_kv_fusing_layer" in mot_state:
        detected_modes.append("simple")
    if any(
        key in mot_state
        for key in (
            "per_head_kv_fusing_layer",
            "k_head_channel_projection",
            "k_head_channel_bias",
        )
    ):
        detected_modes.append("per_head_channel")
    if any(
        key in mot_state
        for key in (
            "kv_fusing_layer",
            "k_channel_projection",
            "v_channel_projection",
            "k_channel_bias",
            "v_channel_bias",
        )
    ):
        detected_modes.append("full")
    has_head_fused_kv_channel_projection = any(
        key in mot_state
        for key in (
            "head_fused_kv_k_channel_projection",
            "head_fused_kv_v_channel_projection",
        )
    )
    has_head_fused_kv_layer_mixing = "head_fused_kv_layer_mixing" in mot_state
    has_head_fused_kv_sin2d = any(
        key.startswith("head_fused_kv_sin2d_pe_mlps.") for key in mot_state
    )
    if has_head_fused_kv_sin2d:
        detected_modes.append("HeadFusedKV+Sin2DPE")
    elif has_head_fused_kv_channel_projection or has_head_fused_kv_layer_mixing:
        if has_head_fused_kv_channel_projection:
            detected_modes.append("HeadFusedKV")
        else:
            detected_modes.append("simple_head_fused")
    if any(key.startswith("mlp_mixer_fused_kv_blocks.") for key in mot_state):
        detected_modes.append("MLPMixerFusedKV")
    if len(detected_modes) > 1:
        raise RuntimeError(
            "Checkpoint contains parameters from multiple new_fused_kv projection "
            f"modes: {detected_modes}. Refusing to infer an ambiguous architecture."
        )
    return detected_modes[0] if detected_modes else None


def _new_fused_kv_projection_signature_matches(
    metadata_mode: str,
    inferred_mode: str,
) -> bool:
    if metadata_mode == inferred_mode:
        return True
    shared_signature_modes = {"simple_head_fused", "simple_head_softmax"}
    return {metadata_mode, inferred_mode} <= shared_signature_modes


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        freeze_video_backbone: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot
        self.freeze_video_backbone = freeze_video_backbone

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        freeze_video_backbone: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        decoupled: bool = False,
        kv_source_mapping: list[int] | None = None,
        kv_source_mode: str = "final_only",
        kv_fusion: "nn.Module | None" = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )


        video_expert = components.dit
        layer_init_mapping = kv_source_mapping
        if decoupled and kv_source_mode in {"fused_kv", "new_fused_kv"}:
            from .mot_decoupled import compute_kv_source_mapping

            action_config_for_init = action_dit_config or {}
            action_num_layers = int(action_config_for_init.get("num_layers", 5))
            if kv_source_mapping is not None and len(kv_source_mapping) == action_num_layers:
                layer_init_mapping = kv_source_mapping
            else:
                layer_init_mapping = compute_kv_source_mapping(
                    mode="uniform_end",
                    video_num_layers=len(video_expert.blocks),
                    action_num_layers=action_num_layers,
                )

        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
            layer_init_mapping=layer_init_mapping if decoupled else None,
        )

        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")

        if decoupled:
            # Decoupled MoT: asymmetric layer counts (e.g., video=30, action=5).
            # Skip the equal-layer-count assertion since experts have different depths.
            from .mot_decoupled import MoTDecoupled

            mot = MoTDecoupled(
                mixtures={"video": video_expert, "action": action_expert},
                video_num_layers=len(video_expert.blocks),
                action_num_layers=len(action_expert.blocks),
                num_heads=int(video_expert.num_heads),
                attn_head_dim=int(video_expert.attn_head_dim),
                mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
                kv_source_mapping=kv_source_mapping,
                kv_source_mode=kv_source_mode,
                kv_fusion=kv_fusion,
            )
        else:
            # Standard MoT: both experts must have the same number of layers.
            if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
                raise ValueError("ActionDiT `num_layers` must match video expert.")

            mot = MoT(
                mixtures={"video": video_expert, "action": action_expert},
                mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
            )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            freeze_video_backbone=freeze_video_backbone,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample.get("video")
        video_latents = sample.get("video_latents")
        if (video is None) == (video_latents is None):
            raise ValueError(
                "FastWAM training requires exactly one of `sample['video']` or "
                "`sample['video_latents']`."
            )
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video is not None:
            if video.ndim != 5:
                raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
            if video.shape[1] != 3:
                raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")
            batch_size, _, num_frames, height, width = video.shape
        else:
            if video_latents.ndim != 5:
                raise ValueError(
                    "`sample['video_latents']` must be 5D [B, C, T, H, W], "
                    f"got shape {tuple(video_latents.shape)}"
                )
            source_video_shape = sample.get("source_video_shape")
            if source_video_shape is None:
                raise ValueError(
                    "`sample['source_video_shape']` is required with cached video latents."
                )
            source_video_shape = torch.as_tensor(source_video_shape)
            if source_video_shape.ndim == 1:
                source_video_shape = source_video_shape.unsqueeze(0)
            if source_video_shape.ndim != 2 or source_video_shape.shape[1] != 4:
                raise ValueError(
                    "`sample['source_video_shape']` must be [B,4] containing [C,T,H,W], "
                    f"got {tuple(source_video_shape.shape)}"
                )
            batch_size = int(video_latents.shape[0])
            if source_video_shape.shape[0] != batch_size:
                raise ValueError(
                    "Cached latent/source shape batch mismatch: "
                    f"{batch_size} vs {source_video_shape.shape[0]}"
                )
            if not torch.equal(source_video_shape, source_video_shape[:1].expand_as(source_video_shape)):
                raise ValueError("All cached samples in a batch must have the same source video shape.")
            channels, num_frames, height, width = (
                int(value) for value in source_video_shape[0].tolist()
            )
            if channels != 3:
                raise ValueError(f"Cached source video channel count must be 3, got {channels}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        if video is not None:
            input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            input_latents = self._encode_video_latents(input_video, tiled=tiled)
        else:
            temporal_factor = int(self.vae.temporal_downsample_factor)
            spatial_factor = int(self.vae.upsampling_factor)
            expected_shape = (
                batch_size,
                int(self.vae.model.z_dim),
                (num_frames - 1) // temporal_factor + 1,
                height // spatial_factor,
                width // spatial_factor,
            )
            if tuple(video_latents.shape) != expected_shape:
                raise ValueError(
                    "Cached video latent shape mismatch: "
                    f"got {tuple(video_latents.shape)} vs expected {expected_shape}"
                )
            input_latents = video_latents.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                } if action_pre["context"] is not None else None,
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                } if action_pre["context"] is not None else None,
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                } if action_pre["context"] is not None else None,
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            } if action_pre["context"] is not None else None,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_action_batch(
        self,
        input_image: torch.Tensor,
        action_horizon: int,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        seeds: Optional[list[Optional[int]]] = None,
        rand_device: str = "cpu",
    ) -> dict:
        """Batched paradigm-C action inference (native port of the fleet's baseline helper).

        Added by the team (Author: Rui Heng Yang) so serving layers that prefer a
        model-native ``infer_action_batch`` route the released baseline checkpoints
        through THIS repository instead of a dispatcher-side re-implementation.
        ``seeds`` draws one independent noise row per slot (precedence over scalar
        ``seed``); returns ``{"action": [B, T, D] float32 CPU, "timing_ms": {...}}``,
        matching ``FastWAMDecoupled.infer_action_batch``'s contract.

        Restricted to the plain paradigm-C class: paradigm A/B subclasses denoise
        video jointly and MUST NOT inherit this single-prefill KV-cache route.
        """
        if type(self) is not FastWAM:
            raise TypeError(
                f"{type(self).__name__} does not support the baseline batched KV-cache "
                "path; paradigm A/B variants denoise video and cannot reuse the "
                "single-prefill route."
            )
        from .fastwam_baseline_batching import baseline_infer_action_batch

        return baseline_infer_action_batch(
            self,
            input_image=input_image,
            action_horizon=action_horizon,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            seeds=seeds,
            rand_device=rand_device,
        )

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        import time as _time

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        def _sync_ms(t0: float) -> float:
            torch.cuda.synchronize()
            return (_time.perf_counter() - t0) * 1000.0

        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        vae_ms = _sync_ms(t0)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        text_enc_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        proprio_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        video_prefill_ms = _sync_ms(t0)

        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        denoise_ms = _sync_ms(t0)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "timing_ms": {
                "text_enc_ms": text_enc_ms,
                "proprio_ms": proprio_ms,
                "vae_ms": vae_ms,
                "video_prefill_ms": video_prefill_ms,
                "denoise_ms": denoise_ms,
                "num_denoise_steps": num_inference_steps,
            },
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        """Save model weights and optional optimizer state to a PyTorch checkpoint.

        For decoupled MoT variants the KV routing metadata (``kv_source_mode``
        and ``kv_source_mapping``) is persisted alongside the weights so
        ``load_checkpoint`` can reject a checkpoint whose routing disagrees
        with the current model. Same-length mappings produce identically
        shaped fused_kv tensors, so shape checks alone cannot catch a
        semantically wrong routing (mot_decoupled.py:284).
        """
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        kv_source_mode = getattr(self.mot, "kv_source_mode", None)
        if kv_source_mode is not None:
            payload["kv_source_mode"] = kv_source_mode
        kv_source_mapping = getattr(self.mot, "kv_source_mapping", None)
        if kv_source_mapping is not None:
            payload["kv_source_mapping"] = list(kv_source_mapping)
        # Persist the MoT class name as the single authoritative identity
        # discriminator. Sibling MoT classes (e.g. the base MoTDecoupled vs the
        # fixed-RoPE MoTDecoupledActionAlignedVideoRoPE subclass) share identical
        # state_dict keys/shapes, so a cross-class load is shape-compatible and
        # would otherwise pass silently despite differing KV-cache RoPE semantics.
        # A legible class-name string lets load_checkpoint reject that case.
        payload["mot_class"] = type(self.mot).__name__
        # Persist the new_fused_kv RoPE mode alongside mot_class. The two rope
        # modes ("aligned_3d" vs "original_3d") live on the SAME MoT class
        # (MoTDecoupledActionAlignedVideoRoPE) and produce byte-identical
        # state_dicts, so mot_class alone cannot discriminate them: an
        # original_3d checkpoint would otherwise load and eval silently as the
        # default aligned_3d, applying the WRONG RoPE to fused video K. A
        # legible mode string lets load_checkpoint reject that mismatch. Models
        # without the attribute (base MoTDecoupled, non-decoupled) write None.
        payload["new_fused_kv_rope_mode"] = getattr(
            self.mot, "new_fused_kv_rope_mode", None
        )
        payload["aligned_3d_action_spatial_anchor_layout"] = getattr(
            self.mot, "aligned_3d_action_spatial_anchor_layout", None
        )
        if kv_source_mode == "new_fused_kv":
            payload["new_fused_kv_projection_mode"] = getattr(
                self.mot, "new_fused_kv_projection_mode", "full"
            )
            payload["new_fused_kv_simple_head_softmax_fuse_mode"] = getattr(
                self.mot, "new_fused_kv_simple_head_softmax_fuse_mode", "all"
            )
            payload["new_fused_kv_head_fused_kv_fuse_mode"] = getattr(
                self.mot, "new_fused_kv_head_fused_kv_fuse_mode", "all"
            )
            # EEF-relative RoPE geometry identity. A CONJUNCTION, not just the
            # calibration digest: a dataset re-exported at a different raw
            # resolution changes every anchor while the digest stays identical
            # (plan Section 20.1).
            eef_identity = getattr(self.mot, "eef_geometry_identity", None)
            if eef_identity is not None:
                payload["eef_geometry_identity"] = dict(eef_identity)
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        """Load a FastWAM checkpoint.

        Fused-MLP checkpoints must include ``kv_fusion.*`` weights whenever the
        current model has ``mot.kv_fusion`` enabled. This prevents accidental
        eval/resume with randomly initialized fusion layers.

        Fused-KV checkpoints must include the five top-level fused_kv mixing
        parameters (``kv_fusing_layer``, ``k_channel_projection``,
        ``v_channel_projection``, ``k_channel_bias``, ``v_channel_bias``)
        whenever the current model is in ``kv_source_mode="fused_kv"``. Those
        parameters live directly on the MoT (not under a ``kv_fusion`` module,
        which stays ``None`` for fused_kv), so the ``kv_fusion``-based guard
        above cannot catch them. Mirroring the fused_mlp guard, a fused_kv
        checkpoint missing any of the five is rejected rather than silently
        loaded at random init (B1-class failure).
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        # Detect fused_kv from the authoritative mode attribute on the MoT
        # (mot_decoupled.py:250). In fused_kv mode the five mixing tensors are
        # real nn.Parameters and hence appear as top-level state_dict keys; in
        # every other mode they are None non-persistent buffers (absent from the
        # state_dict), so this guard only arms for genuine fused_kv models.
        current_kv_source_mode = getattr(self.mot, "kv_source_mode", None)
        current_kv_source_mapping = getattr(self.mot, "kv_source_mapping", None)
        if current_kv_source_mode is not None:
            # Routing-metadata guard: fused_kv/selected-KV tensor shapes depend
            # only on the mapping LENGTH, so a same-length but semantically
            # different routing would load silently if we relied on shapes.
            # Checkpoints written before this metadata existed lack the keys:
            # warn (cannot verify) instead of raising, for backward compat.
            if "kv_source_mode" in payload:
                ckpt_mode = payload["kv_source_mode"]
                if ckpt_mode != current_kv_source_mode:
                    raise RuntimeError(
                        "Checkpoint kv_source_mode "
                        f"{ckpt_mode!r} does not match the current model's "
                        f"kv_source_mode {current_kv_source_mode!r}. Refusing to "
                        f"load a checkpoint with mismatched KV routing. Checkpoint: {path}"
                    )
            else:
                logger.warning(
                    "Checkpoint has no `kv_source_mode` metadata (legacy format); "
                    "cannot verify KV routing against the current model "
                    "(kv_source_mode=%r). Checkpoint: %s",
                    current_kv_source_mode, path,
                )
            if current_kv_source_mapping is not None:
                if "kv_source_mapping" in payload:
                    ckpt_mapping = list(payload["kv_source_mapping"])
                    if ckpt_mapping != list(current_kv_source_mapping):
                        raise RuntimeError(
                            "Checkpoint kv_source_mapping "
                            f"{ckpt_mapping} does not match the current model's "
                            f"kv_source_mapping {list(current_kv_source_mapping)}. "
                            "Same-length mappings load shape-compatibly but route KV from "
                            f"the WRONG video layers. Checkpoint: {path}"
                        )
                elif "kv_source_mode" in payload:
                    # Partial metadata: mode present but mapping absent. New
                    # save_checkpoint always writes both together, so this is a
                    # tampered/hand-built payload; warn (routing unverifiable)
                    # rather than raise. The mode-absent case is already covered
                    # by the legacy warning above.
                    logger.warning(
                        "Checkpoint has `kv_source_mode` but no `kv_source_mapping` "
                        "metadata (partial/hand-built payload); cannot verify KV "
                        "routing mapping against the current model (%s). Checkpoint: %s",
                        list(current_kv_source_mapping), path,
                    )
        is_fused_kv = current_kv_source_mode in {"fused_kv", "new_fused_kv"}
        current_new_fused_kv_projection_mode = getattr(
            self.mot, "new_fused_kv_projection_mode", "full"
        )
        if (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode == "simple"
        ):
            fused_kv_param_names = ["simple_kv_fusing_layer"]
        elif (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode
            in {"simple+PE", "simple+PE-postnorm"}
        ):
            fused_kv_param_names = ["simple_kv_fusing_layer", "k_video_pos_projection"]
        elif (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode == "per_head_channel"
        ):
            fused_kv_param_names = [
                "per_head_kv_fusing_layer",
                "k_head_channel_projection",
                "k_head_channel_bias",
            ]
        elif (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode
            in {"simple_head_fused", "simple_head_softmax"}
        ):
            fused_kv_param_names = ["head_fused_kv_layer_mixing"]
        elif (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode == "HeadFusedKV"
        ):
            fused_kv_param_names = [
                "head_fused_kv_k_channel_projection",
                "head_fused_kv_v_channel_projection",
                "head_fused_kv_layer_mixing",
            ]
        elif (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode == "HeadFusedKV+Sin2DPE"
        ):
            fused_kv_param_names = [
                "head_fused_kv_k_channel_projection",
                "head_fused_kv_v_channel_projection",
                "head_fused_kv_layer_mixing",
            ] + [
                key for key in self.mot.state_dict()
                if key.startswith("head_fused_kv_sin2d_pe_mlps.")
            ]
        elif (
            current_kv_source_mode == "new_fused_kv"
            and current_new_fused_kv_projection_mode == "MLPMixerFusedKV"
        ):
            fused_kv_param_names = [
                key for key in self.mot.state_dict()
                if key.startswith("mlp_mixer_fused_kv_blocks.")
            ]
        else:
            fused_kv_param_names = [
                "kv_fusing_layer",
                "k_channel_projection",
                "v_channel_projection",
                "k_channel_bias",
                "v_channel_bias",
            ]
        k_fused_norm = getattr(self.mot, "k_fused_norm", None)
        if k_fused_norm is not None:
            fused_kv_param_names.extend(
                f"k_fused_norm.{i}.weight" for i in range(len(k_fused_norm))
            )
        if "mot" in payload:
            mot_state = payload["mot"]
            # MoT identity guard (mirrors the kv_source_mode guard above). Sibling
            # MoT classes share identical state_dict keys/shapes, so a cross-class
            # load is shape-compatible and load_state_dict would NOT catch it. The
            # semantics differ at inference: the fixed-RoPE MoT caches RAW K and
            # re-applies RoPE per action step, while the base MoT caches already
            # RoPE'd K -- loading one class's weights into the other silently
            # produces WRONG inference. Backward-compat matrix: field present +
            # mismatch -> raise; present + match -> silent load; absent (legacy
            # checkpoint written before this metadata) -> warn (cannot verify) and
            # load, preserving prior behavior.
            current_mot_class = type(self.mot).__name__
            if "mot_class" in payload:
                ckpt_mot_class = payload["mot_class"]
                if ckpt_mot_class != current_mot_class:
                    raise RuntimeError(
                        "Checkpoint mot_class "
                        f"{ckpt_mot_class!r} does not match the current model's "
                        f"mot_class {current_mot_class!r}. These MoT classes share "
                        "identical state_dict shapes but have DIFFERENT KV-cache "
                        "RoPE semantics (fixed-RoPE MoT caches raw K and re-applies "
                        "RoPE per step; base MoT caches RoPE'd K), so a cross-class "
                        "load loads shape-compatibly yet produces WRONG inference. "
                        f"Refusing to load. Checkpoint: {path}"
                    )
            else:
                logger.warning(
                    "Checkpoint has no `mot_class` metadata (legacy format); "
                    "cannot verify the MoT class against the current model "
                    "(mot_class=%r). Checkpoint: %s",
                    current_mot_class, path,
                )
            # new_fused_kv RoPE-mode identity guard (mirrors the mot_class guard
            # above). The two rope modes ("aligned_3d" vs "original_3d") share the
            # SAME MoT class and produce byte-identical state_dicts, so mot_class
            # cannot discriminate them and load_state_dict sees no shape symptom.
            # They differ only at inference: original_3d keeps the video 3D RoPE on
            # fused video K and the action 1D RoPE on action Q/K, while aligned_3d
            # re-positions video K into the action-attention 3D basis -- loading one
            # mode's weights under the other silently produces WRONG inference.
            # Backward-compat matrix mirrors mot_class: field present + mismatch ->
            # raise ONLY when the current model actually runs new_fused_kv (for
            # every other kv_source_mode the attribute is INERT -- the subclass
            # sets it on all fixed-RoPE instances, so e.g. a fused_kv checkpoint
            # with a CLI-overridden rope-mode value must not be refused); present +
            # match -> silent load; None==None (base MoTDecoupled / non-decoupled
            # models, which never carry the attribute) -> silent pass.
            # Absent key AND the current model is genuinely new_fused_kv (a legacy
            # pre-metadata new_fused_kv checkpoint) -> warn (cannot verify) and load.
            # Cross-kv_source_mode and cross-class loads never reach this guard:
            # the kv_source_mode and mot_class guards above raise first.
            current_rope_mode = getattr(self.mot, "new_fused_kv_rope_mode", None)
            if "new_fused_kv_rope_mode" in payload:
                ckpt_rope_mode = payload["new_fused_kv_rope_mode"]
                if (
                    ckpt_rope_mode != current_rope_mode
                    and current_kv_source_mode == "new_fused_kv"
                ):
                    raise RuntimeError(
                        "Checkpoint new_fused_kv_rope_mode "
                        f"{ckpt_rope_mode!r} does not match the current model's "
                        f"new_fused_kv_rope_mode {current_rope_mode!r}. Both modes "
                        "share the same MoT class and identical state_dict shapes "
                        "but apply DIFFERENT RoPE to the fused video K / action "
                        "Q/K, so a cross-mode load loads shape-compatibly yet "
                        f"produces WRONG inference. Refusing to load. Checkpoint: {path}"
                    )
            elif current_kv_source_mode == "new_fused_kv":
                # Legacy new_fused_kv checkpoint written before this metadata
                # existed: cannot verify which rope mode it was trained with. Warn
                # (not raise) for backward compat. Base/non-new_fused_kv models pass
                # silently: their None==None comparison never reaches this branch.
                logger.warning(
                    "Checkpoint has no `new_fused_kv_rope_mode` metadata (legacy "
                    "format); cannot verify the new_fused_kv RoPE mode against the "
                    "current model (new_fused_kv_rope_mode=%r). Checkpoint: %s",
                    current_rope_mode, path,
                )
            # EEF-relative geometry identity. Checked as a conjunction: any one
            # field changing invalidates the checkpoint, because every one of
            # them moves the anchors (plan Section 20.1). This is the only
            # geometry check that exists -- anchors are computed at load time,
            # so there is no sidecar or dataset fingerprint behind it.
            current_eef_identity = getattr(self.mot, "eef_geometry_identity", None)
            if current_eef_identity is not None or "eef_geometry_identity" in payload:
                ckpt_eef_identity = payload.get("eef_geometry_identity")
                if ckpt_eef_identity is None:
                    raise RuntimeError(
                        "Current model uses EEF-relative camera RoPE "
                        f"(new_fused_kv_rope_mode={current_rope_mode!r}) but the "
                        "checkpoint carries no `eef_geometry_identity`. It cannot "
                        "be verified to have been trained against this geometry. "
                        f"Refusing to load. Checkpoint: {path}"
                    )
                if current_eef_identity is None:
                    raise RuntimeError(
                        "Checkpoint carries `eef_geometry_identity` but the current "
                        "model has none; it was not configured for EEF-relative "
                        f"camera RoPE. Refusing to load. Checkpoint: {path}"
                    )
                differing = {
                    field: (ckpt_eef_identity.get(field), current_eef_identity.get(field))
                    for field in current_eef_identity
                    if ckpt_eef_identity.get(field) != current_eef_identity.get(field)
                }
                if differing:
                    detail = "; ".join(
                        f"{field}: checkpoint={ckpt!r} current={cur!r}"
                        for field, (ckpt, cur) in sorted(differing.items())
                    )
                    raise RuntimeError(
                        "Checkpoint EEF geometry identity does not match the "
                        f"current model ({detail}). Every field here moves the "
                        "anchors, so the loaded weights were trained against a "
                        "different spatial origin and would produce WRONG "
                        f"inference. Refusing to load. Checkpoint: {path}"
                    )

            current_anchor_layout = getattr(
                self.mot, "aligned_3d_action_spatial_anchor_layout", None
            )
            if (
                current_kv_source_mode == "new_fused_kv"
                and current_rope_mode in {
                    "aligned_3d",
                    "aligned_3dp",
                    "aligned_3d_overlap",
                }
            ):
                if "aligned_3d_action_spatial_anchor_layout" in payload:
                    checkpoint_anchor_layout = payload[
                        "aligned_3d_action_spatial_anchor_layout"
                    ]
                    if checkpoint_anchor_layout != current_anchor_layout:
                        raise RuntimeError(
                            "Checkpoint aligned_3d_action_spatial_anchor_layout "
                            f"{checkpoint_anchor_layout!r} does not match the "
                            "current model's "
                            "aligned_3d_action_spatial_anchor_layout "
                            f"{current_anchor_layout!r}. The layouts share "
                            "identical state_dict shapes but assign different "
                            "camera anchors to action-attention heads. Refusing "
                            f"to load. Checkpoint: {path}"
                        )
                else:
                    logger.warning(
                        "Checkpoint has no "
                        "`aligned_3d_action_spatial_anchor_layout` metadata "
                        "(legacy format); cannot verify the %s action "
                        "anchor layout against the current model "
                        "(aligned_3d_action_spatial_anchor_layout=%r). "
                        "Checkpoint: %s",
                        current_rope_mode,
                        current_anchor_layout,
                        path,
                    )
            if current_kv_source_mode == "new_fused_kv":
                inferred_projection_mode = _infer_new_fused_kv_projection_mode(mot_state)
                checkpoint_projection_mode = payload.get(
                    "new_fused_kv_projection_mode", inferred_projection_mode
                )
                if checkpoint_projection_mode is None:
                    # Legacy / unstamped checkpoints: trust the model config the
                    # caller already built (CLI/Hydra). Safer than hard-failing
                    # eval on otherwise loadable weights.
                    logger.warning(
                        "Checkpoint has no `new_fused_kv_projection_mode` metadata "
                        "and its projection mode cannot be inferred from the MoT "
                        "state; trusting the current model's "
                        "new_fused_kv_projection_mode=%r. Checkpoint: %s",
                        current_new_fused_kv_projection_mode,
                        path,
                    )
                    checkpoint_projection_mode = current_new_fused_kv_projection_mode
                if (
                    "new_fused_kv_projection_mode" in payload
                    and inferred_projection_mode is not None
                    and not _new_fused_kv_projection_signature_matches(
                        checkpoint_projection_mode,
                        inferred_projection_mode,
                    )
                ):
                    raise RuntimeError(
                        "Checkpoint new_fused_kv projection metadata disagrees with its "
                        f"parameter signature: metadata={checkpoint_projection_mode!r}, "
                        f"inferred={inferred_projection_mode!r}. Checkpoint: {path}"
                    )
                if checkpoint_projection_mode != current_new_fused_kv_projection_mode:
                    raise RuntimeError(
                        "Checkpoint new_fused_kv_projection_mode "
                        f"{checkpoint_projection_mode!r} does not match the current "
                        "model's new_fused_kv_projection_mode "
                        f"{current_new_fused_kv_projection_mode!r}. Refusing to load "
                        f"a checkpoint with a mismatched projection architecture. Checkpoint: {path}"
                    )
                if "new_fused_kv_projection_mode" not in payload:
                    logger.warning(
                        "Checkpoint has no `new_fused_kv_projection_mode` metadata "
                        "(legacy format); inferred %r from its MoT parameter signature. "
                        "Checkpoint: %s",
                        inferred_projection_mode,
                        path,
                    )
                if current_new_fused_kv_projection_mode == "simple_head_softmax":
                    current_softmax_fuse_mode = getattr(
                        self.mot,
                        "new_fused_kv_simple_head_softmax_fuse_mode",
                        "all",
                    )
                    if "new_fused_kv_simple_head_softmax_fuse_mode" in payload:
                        checkpoint_softmax_fuse_mode = payload[
                            "new_fused_kv_simple_head_softmax_fuse_mode"
                        ]
                        if checkpoint_softmax_fuse_mode == "uni_end":
                            checkpoint_softmax_fuse_mode = "uniform_end"
                        if checkpoint_softmax_fuse_mode != current_softmax_fuse_mode:
                            raise RuntimeError(
                                "Checkpoint new_fused_kv_simple_head_softmax_fuse_mode "
                                f"{checkpoint_softmax_fuse_mode!r} does not match "
                                "the current model's "
                                "new_fused_kv_simple_head_softmax_fuse_mode "
                                f"{current_softmax_fuse_mode!r}. The modes share "
                                "the same state_dict shapes but route video layers "
                                "to action DiT layers differently. Refusing to load. "
                                f"Checkpoint: {path}"
                            )
                    elif current_softmax_fuse_mode != "all":
                        raise RuntimeError(
                            "Checkpoint has no "
                            "`new_fused_kv_simple_head_softmax_fuse_mode` metadata "
                            "(legacy format, treated as 'all') but the current "
                            "model requests "
                            f"{current_softmax_fuse_mode!r}. Refusing to load a "
                            "shape-compatible checkpoint with unverifiable "
                            f"simple_head_softmax routing. Checkpoint: {path}"
                        )
                if current_new_fused_kv_projection_mode in {
                    "HeadFusedKV",
                    "HeadFusedKV+Sin2DPE",
                }:
                    current_head_fused_fuse_mode = getattr(
                        self.mot,
                        "new_fused_kv_head_fused_kv_fuse_mode",
                        "all",
                    )
                    if "new_fused_kv_head_fused_kv_fuse_mode" in payload:
                        checkpoint_head_fused_fuse_mode = payload[
                            "new_fused_kv_head_fused_kv_fuse_mode"
                        ]
                        if checkpoint_head_fused_fuse_mode != current_head_fused_fuse_mode:
                            raise RuntimeError(
                                "Checkpoint new_fused_kv_head_fused_kv_fuse_mode "
                                f"{checkpoint_head_fused_fuse_mode!r} does not "
                                "match the current model's "
                                "new_fused_kv_head_fused_kv_fuse_mode "
                                f"{current_head_fused_fuse_mode!r}. The modes "
                                "share the same state_dict shapes but route "
                                "video layers to action DiT layers differently. "
                                f"Refusing to load. Checkpoint: {path}"
                            )
                    elif current_head_fused_fuse_mode != "all":
                        raise RuntimeError(
                            "Checkpoint has no "
                            "`new_fused_kv_head_fused_kv_fuse_mode` metadata "
                            "(legacy format, treated as 'all') but the current "
                            "model requests "
                            f"{current_head_fused_fuse_mode!r}. Refusing to load "
                            "a shape-compatible checkpoint with unverifiable "
                            f"HeadFusedKV routing. Checkpoint: {path}"
                        )
            if is_fused_kv:
                # Pre-load guard: the fused_kv mixing params must be present
                # in the incoming state before we attempt a (strict=False) load,
                # otherwise strict=False would silently leave them at random init.
                missing_fused_kv_keys = [
                    key for key in fused_kv_param_names
                    if key not in mot_state
                ]
                if missing_fused_kv_keys:
                    raise RuntimeError(
                        "Checkpoint is missing fused_kv mixing parameters for a "
                        "fused_kv model. Refusing to leave the KV-mixing tensors "
                        f"at random init. Missing: {', '.join(missing_fused_kv_keys)}. "
                        f"Checkpoint: {path}"
                    )
            if getattr(self.mot, "kv_fusion", None) is not None:
                expected_fusion_keys = [
                    key for key in self.mot.state_dict().keys()
                    if key.startswith("kv_fusion.")
                ]
                missing_fusion_keys = [
                    key for key in expected_fusion_keys
                    if key not in mot_state
                ]
                if missing_fusion_keys:
                    preview = ", ".join(missing_fusion_keys[:5])
                    if len(missing_fusion_keys) > 5:
                        preview += f", ... ({len(missing_fusion_keys)} total)"
                    raise RuntimeError(
                        "Checkpoint is missing kv_fusion.* weights for a fused_mlp model. "
                        f"Refusing to leave fusion layers at random init. Missing: {preview}. "
                        f"Checkpoint: {path}"
                    )

            incompatible = self.mot.load_state_dict(mot_state, strict=False)
            if getattr(self.mot, "kv_fusion", None) is not None:
                missing_fusion_keys = [
                    key for key in incompatible.missing_keys
                    if key.startswith("kv_fusion.")
                ]
                if missing_fusion_keys:
                    raise RuntimeError(
                        "Checkpoint load reported missing kv_fusion.* weights for a fused_mlp "
                        f"model: {missing_fusion_keys}. Checkpoint: {path}"
                    )
            if is_fused_kv:
                # Post-load defense: even after the pre-load key check, confirm
                # load_state_dict did not report any fused_kv mixing tensors as
                # missing (e.g. a partial/renamed state). Any missing
                # entry means the tensor stayed at random init.
                missing_fused_kv_after = [
                    key for key in incompatible.missing_keys
                    if key in fused_kv_param_names
                ]
                if missing_fused_kv_after:
                    raise RuntimeError(
                        "Checkpoint load reported missing fused_kv mixing "
                        f"parameters for a fused_kv model: {missing_fused_kv_after}. "
                        f"Checkpoint: {path}"
                    )
            if incompatible.missing_keys:
                # General missing-keys guard: strict=False tolerates ANY absent
                # key, so a full `mot` checkpoint missing non-fused weights
                # (e.g. a truncated or foreign-variant state dict) would leave
                # those tensors silently at initialization (B1-class failure).
                # The fused-specific guards above fire first with more
                # actionable messages; this catches everything else. Unexpected
                # EXTRA keys stay tolerated (forward compat with newer saves).
                preview = ", ".join(incompatible.missing_keys[:5])
                if len(incompatible.missing_keys) > 5:
                    preview += f", ... ({len(incompatible.missing_keys)} total)"
                raise RuntimeError(
                    "Checkpoint `mot` state is missing weights present in the "
                    f"current model; those tensors would silently stay at "
                    f"initialization. Missing: {preview}. Checkpoint: {path}"
                )
        elif "dit" in payload:
            if getattr(self.mot, "kv_fusion", None) is not None:
                raise RuntimeError(
                    "Cannot load legacy `dit` checkpoint into a fused_mlp model because "
                    "`dit` checkpoints do not contain kv_fusion.* weights. Load a full "
                    f"`mot` checkpoint instead. Checkpoint: {path}"
                )
            # Mirror the fused_mlp guard above for fused_kv models. A legacy `dit`
            # checkpoint only carries the video expert's weights, so it can never
            # contain the five top-level fused_kv mixing params (kv_fusing_layer /
            # k/v_channel_projection / k/v_channel_bias); loading it would leave
            # them at random init (a B1-class silent corruption). is_fused_kv was
            # computed above from the authoritative mot.kv_source_mode attribute.
            if is_fused_kv:
                raise RuntimeError(
                    "Cannot load legacy `dit` checkpoint into a fused_kv model because "
                    "`dit` checkpoints do not contain the fused_kv mixing parameters "
                    f"({', '.join(fused_kv_param_names)}). Load a full `mot` checkpoint "
                    f"instead. Checkpoint: {path}"
                )
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)

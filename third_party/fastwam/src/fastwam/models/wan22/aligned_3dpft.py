"""Additive ``aligned_3dpft`` integration for FasterWAM decoupled models."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from fastwam.geometry.flow_trajectory import build_aligned_3dpft_geometry

from .fasterwam_decoupled import (
    FasterWAMDecoupled,
    MoTDecoupledActionAlignedVideoRoPE,
    create_fasterwam_decoupled,
)


class MoTAligned3DPFT(MoTDecoupledActionAlignedVideoRoPE):
    """Dynamic trajectory RoPE layered onto the existing ``aligned_3dp`` path."""

    supported_new_fused_kv_rope_modes = (
        MoTDecoupledActionAlignedVideoRoPE.supported_new_fused_kv_rope_modes
        | {"aligned_3dpft"}
    )
    _aligned_3d_family_rope_modes = frozenset(
        {*MoTDecoupledActionAlignedVideoRoPE._aligned_3d_family_rope_modes, "aligned_3dpft"}
    )

    def set_flow_trajectory_context(
        self,
        context: Optional[Mapping[str, torch.Tensor]],
        *,
        expected_batch_size: Optional[int] = None,
    ) -> None:
        """Install the context the next action pass denoises against.

        ``expected_batch_size`` is set by callers that pack independent
        requests into one forward pass. It makes the per-slot correspondence
        between context entries and action rows an enforced contract rather
        than an assumption -- see `build_aligned_3dpft_geometry(strict_batch=)`.
        """
        self._aligned_3dpft_context = context
        self._aligned_3dpft_expected_batch_size = expected_batch_size

    def prepare_flow_trajectory(self, noisy_action: torch.Tensor) -> None:
        context = getattr(self, "_aligned_3dpft_context", None)
        if context is None:
            raise ValueError(
                "aligned_3dpft requires an aligned_3dpft_context for every train/infer call"
            )
        expected_batch_size = getattr(
            self, "_aligned_3dpft_expected_batch_size", None
        )
        if expected_batch_size is not None and noisy_action.shape[0] != expected_batch_size:
            raise ValueError(
                f"aligned_3dpft action batch {noisy_action.shape[0]} does not match "
                f"the {expected_batch_size} requests whose context was installed"
            )
        anchors, visible = build_aligned_3dpft_geometry(
            noisy_action, context, strict_batch=expected_batch_size is not None
        )
        self._aligned_3dpft_anchors = anchors
        self._aligned_3dpft_visible = visible

    def clear_flow_trajectory(self) -> None:
        self._aligned_3dpft_context = None
        self._aligned_3dpft_expected_batch_size = None
        self._aligned_3dpft_anchors = None
        self._aligned_3dpft_visible = None

    def _build_new_fused_kv_3d_rope_freqs(
        self,
        video_freqs: torch.Tensor,
        action_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_rope, static_action_rope = super()._build_new_fused_kv_3d_rope_freqs(
            video_freqs, action_seq_len
        )
        if not bool(getattr(self, "_aligned_3dpft_active", False)):
            return video_rope, static_action_rope

        anchors = getattr(self, "_aligned_3dpft_anchors", None)
        if anchors is None:
            raise ValueError("aligned_3dpft has no trajectory anchors; action hook did not run")
        if anchors.ndim != 4 or anchors.shape[1:] != (action_seq_len, 2, 2):
            raise ValueError(
                "aligned_3dpft anchors must be [B, Sa, 2, 2], got "
                f"{tuple(anchors.shape)}"
            )

        t_dim, h_dim, w_dim = self._split_3d_rope_dims()
        tokens_per_frame = self._infer_video_tokens_per_frame(video_freqs)
        height, width = self._infer_video_spatial_grid_size(
            video_freqs, tokens_per_frame
        )
        if width % 2 != 0:
            raise ValueError(
                f"aligned_3dpft requires a two-camera horizontal grid, got width={width}"
            )
        local_width = width // 2
        if height > 1:
            h_unit = video_freqs[width, 0, t_dim:t_dim + h_dim]
        else:
            raise ValueError("aligned_3dpft requires a non-degenerate spatial grid")
        if width > 1:
            w_unit = video_freqs[1, 0, t_dim + h_dim:]
        else:
            raise ValueError("aligned_3dpft requires a non-degenerate spatial grid")

        anchors = anchors.to(device=video_freqs.device, dtype=torch.float32)
        camera_offset = torch.tensor(
            [0.0, float(local_width)], device=anchors.device, dtype=anchors.dtype
        )
        composite_y = anchors[..., 0]
        composite_x = anchors[..., 1] + camera_offset.view(1, 1, 2)
        head_camera = self._allocate_aligned_3d_head_anchor_indices(anchors.device)
        gather_index = head_camera.view(1, 1, -1).expand(
            anchors.shape[0], action_seq_len, -1
        )
        head_y = torch.gather(composite_y, 2, gather_index)
        head_x = torch.gather(composite_x, 2, gather_index)

        h_angles = head_y[..., None] * torch.angle(h_unit).view(1, 1, 1, -1)
        w_angles = head_x[..., None] * torch.angle(w_unit).view(1, 1, 1, -1)
        spatial = torch.cat(
            [
                torch.polar(torch.ones_like(h_angles), h_angles),
                torch.polar(torch.ones_like(w_angles), w_angles),
            ],
            dim=-1,
        )
        temporal = static_action_rope[..., :t_dim].unsqueeze(0).expand(
            anchors.shape[0], -1, -1, -1
        )
        return video_rope, torch.cat([temporal, spatial], dim=-1)

    def _build_aligned_3dp_per_head_action_mask(
        self,
        base_action_mask: torch.Tensor,
        video_freqs: torch.Tensor,
    ) -> torch.Tensor:
        mask = super()._build_aligned_3dp_per_head_action_mask(
            base_action_mask, video_freqs
        )
        if not bool(getattr(self, "_aligned_3dpft_active", False)):
            return mask
        visible = getattr(self, "_aligned_3dpft_visible", None)
        if visible is None:
            raise ValueError("aligned_3dpft has no depth visibility mask")
        head_camera = self._allocate_aligned_3d_head_anchor_indices(visible.device)
        head_visible = visible.index_select(2, head_camera).permute(0, 2, 1)
        video_seq_len = mask.shape[-1] - mask.shape[-2]
        dynamic = mask.unsqueeze(0).expand(visible.shape[0], -1, -1, -1).clone()
        dynamic[..., :video_seq_len] &= head_visible.unsqueeze(-1)
        return dynamic[0] if dynamic.shape[0] == 1 else dynamic

    def _validate_cached_action_mask(
        self,
        *,
        action_mask: torch.Tensor,
        video_seq_len: int,
        action_seq_len: int,
        use_new_fused_kv: bool,
    ) -> None:
        """Admit the per-request mask a batched flow context produces.

        Depth visibility is per request, so `_build_aligned_3dp_per_head_action_mask`
        keeps a leading batch axis whenever more than one request is in flight.
        The inherited aligned_3dp contract only knows the ``[H, Sa, Sk]`` form.
        """
        anchors = getattr(self, "_aligned_3dpft_anchors", None)
        if (
            not bool(getattr(self, "_aligned_3dpft_active", False))
            or anchors is None
            or action_mask.ndim != 4
        ):
            super()._validate_cached_action_mask(
                action_mask=action_mask,
                video_seq_len=video_seq_len,
                action_seq_len=action_seq_len,
                use_new_fused_kv=use_new_fused_kv,
            )
            return
        expected = (
            anchors.shape[0],
            self.num_heads,
            action_seq_len,
            video_seq_len + action_seq_len,
        )
        if action_mask.shape != torch.Size(expected):
            raise ValueError(
                "aligned_3dpft batched action_attention_mask shape must be "
                f"{expected}, got {tuple(action_mask.shape)}"
            )

    def _run_action_layers_with_video_kv_source(self, *args: Any, **kwargs: Any):
        if self.new_fused_kv_rope_mode != "aligned_3dpft":
            return super()._run_action_layers_with_video_kv_source(*args, **kwargs)
        self._aligned_3dpft_active = True
        self.new_fused_kv_rope_mode = "aligned_3dp"
        try:
            return super()._run_action_layers_with_video_kv_source(*args, **kwargs)
        finally:
            self.new_fused_kv_rope_mode = "aligned_3dpft"
            self._aligned_3dpft_active = False


class Aligned3DPFTFasterWAMDecoupled(FasterWAMDecoupled):
    """FasterWAM whose current noisy action state drives action-token RoPE."""

    def _install_aligned_3dpft_hook(self) -> None:
        if getattr(self, "_aligned_3dpft_hook_handle", None) is not None:
            return

        def _capture_noisy_action(_module, args):
            if not args:
                raise RuntimeError("ActionDiT action_encoder hook received no input")
            self.mot.prepare_flow_trajectory(args[0])

        self._aligned_3dpft_hook_handle = self.action_expert.action_encoder.register_forward_pre_hook(
            _capture_noisy_action
        )

    def _set_flow_context(self, context: Optional[Mapping[str, torch.Tensor]]) -> None:
        self.mot.set_flow_trajectory_context(context)

    def training_loss(self, sample: dict, tiled: bool = False):
        context = sample.get("aligned_3dpft_context")
        self._set_flow_context(context)
        try:
            return super().training_loss(sample, tiled=tiled)
        finally:
            self.mot.clear_flow_trajectory()

    @torch.no_grad()
    def infer_action(
        self,
        *args: Any,
        aligned_3dpft_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._set_flow_context(aligned_3dpft_context)
        try:
            return super().infer_action(*args, **kwargs)
        finally:
            self.mot.clear_flow_trajectory()

    @torch.no_grad()
    def infer_action_batch(
        self,
        *args: Any,
        aligned_3dpft_context: Optional[Mapping[str, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serve ``B`` requests, each against its own observed geometry.

        The context must carry one entry per request in slot order: the caller
        packs unrelated episodes into a single batch, so entry ``i`` is the only
        geometry that may drive row ``i``. That correspondence is enforced
        rather than trusted -- the installed batch size is checked against the
        action rows the hook actually sees.
        """
        input_image = kwargs.get("input_image")
        if input_image is None:
            raise ValueError(
                "aligned_3dpft infer_action_batch requires the keyword-only "
                "`input_image` used by the batched contract"
            )
        self.mot.set_flow_trajectory_context(
            aligned_3dpft_context, expected_batch_size=int(input_image.shape[0])
        )
        try:
            return super().infer_action_batch(*args, **kwargs)
        finally:
            self.mot.clear_flow_trajectory()


def create_aligned_3dpft(
    *,
    new_fused_kv_rope_mode: str = "aligned_3dpft",
    eef_calibration_path: str | None = None,
    eef_raw_source_resolution: int = 512,
    **kwargs: Any,
) -> Aligned3DPFTFasterWAMDecoupled:
    """Hydra factory preserving the original model/checkpoint parameter layout."""
    if new_fused_kv_rope_mode != "aligned_3dpft":
        raise ValueError("create_aligned_3dpft only accepts new_fused_kv_rope_mode=aligned_3dpft")
    if kwargs.get("kv_source_mode") != "new_fused_kv":
        raise ValueError("aligned_3dpft requires kv_source_mode=new_fused_kv")
    if kwargs.get("new_fused_kv_projection_mode") != "simple_head_softmax":
        raise ValueError(
            "aligned_3dpft requires new_fused_kv_projection_mode=simple_head_softmax"
        )
    layout = str(kwargs.get("aligned_3d_action_spatial_anchor_layout", "horizontal"))
    if layout not in {"horizontal", "libero"}:
        raise ValueError("aligned_3dpft requires a two-camera horizontal/libero layout")
    if eef_calibration_path is None:
        raise ValueError("aligned_3dpft requires eef_calibration_path")

    # Build the identical parameterized architecture through the established
    # factory. aligned_3dpft differs only in runtime geometry, so upgrading these
    # two Python classes adds no state_dict entries and changes no tensor shape.
    model = create_fasterwam_decoupled(
        new_fused_kv_rope_mode="aligned_3dp",
        **kwargs,
    )
    model.mot.__class__ = MoTAligned3DPFT
    model.mot.new_fused_kv_rope_mode = "aligned_3dpft"
    model.mot._aligned_3dpft_context = None
    model.mot._aligned_3dpft_expected_batch_size = None
    model.mot._aligned_3dpft_anchors = None
    model.mot._aligned_3dpft_visible = None
    model.__class__ = Aligned3DPFTFasterWAMDecoupled
    model.eef_calibration_path = str(eef_calibration_path)
    model.eef_raw_source_resolution = int(eef_raw_source_resolution)
    model._aligned_3dpft_hook_handle = None
    model._install_aligned_3dpft_hook()
    return model

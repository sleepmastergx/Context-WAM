import logging
import json
import inspect
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import timedelta
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .geometry.identity import assert_calibration_parity, eef_rope_mode
from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


def create_accelerator_from_cfg(cfg: DictConfig) -> Accelerator:
    mixed_precision = str(cfg.mixed_precision).strip().lower()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {cfg.mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    distributed_timeout_minutes = int(
        OmegaConf.select(cfg, "distributed_timeout_minutes", default=60)
    )
    return Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=timedelta(minutes=distributed_timeout_minutes))
        ],
    )


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig, accelerator=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        self.fastwam_dir = Path(__file__).resolve().parents[2]
        self.hf_upload_cfg = cfg.get("hf_upload", {})
        if self.hf_upload_cfg is None:
            self.hf_upload_cfg = {}
        self.hf_upload_enabled = bool(self.hf_upload_cfg.get("enabled", False))
        self.hf_upload_dry_run = bool(self.hf_upload_cfg.get("dry_run", False))
        self.hf_upload_fail_on_error = bool(self.hf_upload_cfg.get("fail_on_error", False))
        self.hf_upload_bucket_run_name = self.hf_upload_cfg.get("bucket_run_name", None)
        self.hf_upload_script = self._resolve_hf_upload_script(
            self.hf_upload_cfg.get("script_path", "huggingface/upload_to_hf.sh")
        )
        self._hf_uploaded_steps = set()
        self._last_hf_upload_error = None
        self.freeze_video_backbone = bool(
            OmegaConf.select(cfg, "freeze_video_backbone", default=False)
        )
        
        self.resume = cfg.resume
        # Escape hatch for legacy full-state directories written before the
        # sibling weights checkpoint existed. Defaults to the safe behavior:
        # an unverifiable full-state resume is refused, not warned about.
        self.allow_unverified_full_state_resume = bool(
            OmegaConf.select(cfg, "allow_unverified_full_state_resume", default=False)
        )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)
        self.distributed_timeout_minutes = int(
            OmegaConf.select(cfg, "distributed_timeout_minutes", default=60)
        )

        self.accelerator = accelerator or create_accelerator_from_cfg(cfg)
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f distributed_timeout_min=%d",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
            self.distributed_timeout_minutes,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        if self.hf_upload_enabled and self.accelerator.is_main_process:
            logger.info(
                "HF checkpoint upload enabled: script=%s dry_run=%s fail_on_error=%s bucket_run_name=%s",
                self.hf_upload_script,
                self.hf_upload_dry_run,
                self.hf_upload_fail_on_error,
                self.hf_upload_bucket_run_name,
            )
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")
        # Model and datasets are both instantiated here and no batch has been
        # drawn yet, so this is the last point at which their geometry can be
        # compared before one reaches training_loss().
        self._assert_eef_anchor_calibration_parity()

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(
            self.model,
            freeze_video_backbone=self.freeze_video_backbone,
        )

        # Log total/trainable/frozen param counts for visibility.
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params_count = total_params - trainable_params_count
        logger.info(
            "Total params: %.2fB | Trainable: %.2fB | Frozen: %.2fB",
            total_params / 1e9,
            trainable_params_count / 1e9,
            frozen_params_count / 1e9,
        )
        if getattr(self.model, 'freeze_video_backbone', False):
            video_params = sum(
                p.numel() for p in self.model.mot.mixtures["video"].parameters()
            )
            logger.info("Video backbone FROZEN: %.2fB params", video_params / 1e9)

        # Filter by requires_grad so frozen params (e.g., video expert when
        # freeze_video_backbone=True) are excluded from the optimizer, saving
        # DeepSpeed optimizer state memory.
        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend([p for p in proprio_encoder.parameters() if p.requires_grad])
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        raw_model = self.accelerator.unwrap_model(self.model)
        mot = getattr(raw_model, "mot", None)
        if mot is not None:
            model_meta = {}
            for name in getattr(mot, "expert_order", []):
                expert = mot.mixtures[name]
                model_meta[f"{name}_num_params_B"] = round(
                    sum(p.numel() for p in expert.parameters()) / 1e9, 2
                )
            if hasattr(mot, "kv_source_mapping"):
                model_meta["kv_source_mapping"] = mot.kv_source_mapping
            if hasattr(mot, "video_num_layers"):
                model_meta["video_num_layers"] = mot.video_num_layers
            if hasattr(mot, "action_num_layers"):
                model_meta["action_num_layers"] = mot.action_num_layers
            wandb.config.update({"model_meta": model_meta}, allow_val_change=True)

        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _assert_eef_anchor_calibration_parity(self) -> None:
        """Refuse to train when dataset anchors and model geometry disagree.

        The dataset resolves its anchors from
        ``data.<split>.eef_anchor_calibration_path`` while the model stamps
        checkpoints with the identity derived from ``model.eef_calibration_path``.
        The two are configured independently and nothing else compares them, so a
        run could consume anchors from calibration B while recording calibration
        A -- after which evaluation loads cleanly against a different spatial
        origin. Anchors from the wrong calibration are ordinary finite token
        coordinates, so there is no other symptom.

        Inert for ``aligned_3d`` and every legacy mode: they resolve no anchors,
        and ``eef_rope_mode()`` returns ``None`` for them.

        Raises:
            RuntimeError: If an EEF RoPE mode is active and a dataset either
                builds no anchor index or built it from another calibration.
        """
        # __init__ calls this before accelerator.prepare(), so self.model is
        # still the bare model and needs no unwrapping.
        model = self.model
        mode = eef_rope_mode(model)
        if mode is None:
            return

        # val_dataset feeds training_loss() during evaluate(), so it is checked
        # too; build_datasets() aliases it to the train dataset when `data.val`
        # is absent, and re-checking the same object would only duplicate logs.
        splits: list[tuple[str, object]] = [("data.train", self.train_dataset)]
        if self.val_dataset is not None and self.val_dataset is not self.train_dataset:
            splits.append(("data.val", self.val_dataset))

        for config_key, dataset in splits:
            anchor_index = getattr(dataset, "eef_anchor_index", None)
            if anchor_index is None:
                raise RuntimeError(
                    f"new_fused_kv_rope_mode={mode!r} consumes projected EEF "
                    f"anchors, but the instantiated {config_key} dataset built no "
                    "anchor index. Set "
                    f"{config_key}.eef_anchor_calibration_path to the same "
                    "artifact as model.eef_calibration_path "
                    f"({getattr(model, 'eef_calibration_path', None)!r})."
                )
            calibration = getattr(anchor_index, "calibration", None)
            assert_calibration_parity(
                model,
                digest=anchor_index.calibration_digest,
                source_label=f"{config_key}.eef_anchor_calibration_path",
                source_path=getattr(calibration, "source_path", None),
            )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    @staticmethod
    def _weights_checkpoint_beside_state_dir(state_dir: Path) -> Path | None:
        """Locate the weights checkpoint written alongside a full-state directory.

        ``save_checkpoint`` always writes ``checkpoints/weights/<step_tag>.pt``
        and ``checkpoints/state/<step_tag>/`` together, so the sibling weights
        file is this state directory's semantic-identity record.
        """
        candidate = state_dir.parent.parent / "weights" / f"{state_dir.name}.pt"
        return candidate if candidate.is_file() else None

    def _validate_full_state_resume_identity(self, state_dir: Path) -> None:
        """Run the checkpoint identity guards before restoring a full state dir.

        ``accelerator.load_state()`` restores tensors directly and never sees the
        semantic metadata (``mot_class``, ``new_fused_kv_rope_mode``,
        ``eef_geometry_identity``, KV routing, projection mode). Those all load
        shape-compatibly, so resuming across RoPE modes or EEF geometries would
        otherwise train silently against the wrong RoPE or spatial origin --
        exactly what the weight-checkpoint guards exist to prevent.

        Validating through the real ``load_checkpoint`` keeps one copy of those
        rules. It costs one extra weight load per resume, which is negligible
        against a training run and is what the ``.pt`` resume branch already
        does; ``accelerator.load_state()`` then restores the authoritative
        weights and optimizer state.

        When the sibling weights checkpoint is absent the identity cannot be
        verified at all, so this **fails closed**: an unverifiable resume is
        refused rather than warned about, because the failure it would admit is
        silent. ``allow_unverified_full_state_resume=true`` (default ``false``)
        is the explicit opt-in for a legacy state directory written before the
        sibling weights checkpoint existed; it is logged loudly when used and
        must not be set to work around a genuinely mismatched run directory.

        Raises:
            RuntimeError: If no sibling weights checkpoint exists and the
                override is not enabled.
        """
        reference = self._weights_checkpoint_beside_state_dir(state_dir)
        if reference is None:
            expected = state_dir.parent.parent / "weights" / f"{state_dir.name}.pt"
            if not getattr(self, "allow_unverified_full_state_resume", False):
                raise RuntimeError(
                    f"No sibling weights checkpoint for state directory {state_dir}; "
                    "cannot verify checkpoint identity (MoT class, RoPE mode, EEF "
                    "geometry, KV routing, projection mode) before "
                    f"accelerator.load_state(). Expected {expected}. Those all "
                    "restore shape-compatibly, so an unverified resume can train "
                    "against the wrong RoPE or spatial origin with no symptom. "
                    "Resume from the run directory that holds the matching "
                    "weights checkpoint, or -- only for a legacy state directory "
                    "written before that file existed, and only after confirming "
                    "the configuration matches the run being resumed -- set "
                    "allow_unverified_full_state_resume=true."
                )
            logger.warning(
                "SAFETY OVERRIDE allow_unverified_full_state_resume=true: "
                "resuming full training state from %s with NO sibling weights "
                "checkpoint (expected %s), so this process cannot verify "
                "checkpoint identity (MoT class, RoPE mode, EEF geometry, KV "
                "routing). If the resumed run used a different RoPE mode or EEF "
                "geometry, training continues against the wrong spatial origin.",
                state_dir,
                expected,
            )
            return
        logger.info(
            "Validating checkpoint identity for full-state resume against %s",
            reference,
        )
        self.accelerator.unwrap_model(self.model).load_checkpoint(
            str(reference), optimizer=None
        )

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self._validate_full_state_resume_identity(resume_path)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        self._apply_dit_only_train_mode(model)

    @staticmethod
    def _apply_dit_only_train_mode(model, freeze_video_backbone: bool | None = None):
        if freeze_video_backbone is None:
            freeze_video_backbone = bool(
                getattr(model, "freeze_video_backbone_for_training", False)
            )
        setattr(model, "freeze_video_backbone_for_training", bool(freeze_video_backbone))

        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        # Re-freeze video expert after blanket unfreeze when freeze_video_backbone is enabled.
        # This must come AFTER dit.requires_grad_(True) so only the video mixture is re-frozen.
        if getattr(model, 'freeze_video_backbone', False):
            model.mot.mixtures["video"].requires_grad_(False)
            model.mot.mixtures["video"].eval()
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    def _collect_trainable_parameters(self, model):
        self._apply_dit_only_train_mode(
            model,
            freeze_video_backbone=self.freeze_video_backbone,
        )
        trainable_params = [p for p in model.dit.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable DiT parameters found for optimizer.")
        logger.info(
            "Training parameters: freeze_video_backbone=%s dit_trainable=%.3fB",
            self.freeze_video_backbone,
            sum(p.numel() for p in trainable_params) / 1e9,
        )
        return trainable_params

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)
        eef_anchor_token = sample.get("eef_anchor_token", None)
        aligned_3dpft_context = sample.get("aligned_3dpft_context", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        eval_sample_is_unbatched = video.ndim == 4
        if eval_sample_is_unbatched:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        # The EEF-relative RoPE modes hard-error rather than defaulting an
        # anchor, so this key must survive the rebuild. Only the eef arms set
        # it; every other run leaves it None and is unaffected.
        if eef_anchor_token is not None:
            if not isinstance(eef_anchor_token, torch.Tensor):
                eef_anchor_token = torch.as_tensor(eef_anchor_token)
            if eef_anchor_token.ndim == 2:
                eef_anchor_token = eef_anchor_token.unsqueeze(0)
            if eef_anchor_token.ndim != 3 or eef_anchor_token.shape[1:] != (2, 2):
                raise ValueError(
                    "`sample['eef_anchor_token']` must be [2,2] or [B,2,2] with "
                    f"cameras (main, wrist) and coords (y, x), got "
                    f"{tuple(eef_anchor_token.shape)}"
                )
            if eef_anchor_token.shape[0] != video.shape[0]:
                raise ValueError(
                    f"`eef_anchor_token` batch {eef_anchor_token.shape[0]} does not "
                    f"match video batch {video.shape[0]}"
                )

        if aligned_3dpft_context is not None:
            if not isinstance(aligned_3dpft_context, Mapping):
                raise TypeError(
                    "`sample['aligned_3dpft_context']` must be a mapping, got "
                    f"{type(aligned_3dpft_context)}"
                )
            batched_flow_context = {}
            for key, value in aligned_3dpft_context.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                if eval_sample_is_unbatched:
                    value = value.unsqueeze(0)
                elif value.ndim == 0 or value.shape[0] != video.shape[0]:
                    raise ValueError(
                        f"aligned_3dpft_context[{key!r}] batch shape "
                        f"{tuple(value.shape)} does not match video batch {video.shape[0]}"
                    )
                batched_flow_context[key] = value
            aligned_3dpft_context = batched_flow_context

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
            "eef_anchor_token": eef_anchor_token,
            "aligned_3dpft_context": aligned_3dpft_context,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt
        # Passed only when the run resolves anchors: `evaluate()` is shared with
        # every variant, and the other `infer()` signatures have no such
        # parameter. Kept batched [1,2,2] -- `_infer_action_core` validates it
        # against the action batch, unlike the [0]-indexed kwargs above.
        if sample.get("eef_anchor_token") is not None:
            infer_kwargs["eef_anchor_token"] = sample["eef_anchor_token"]

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def _resolve_hf_upload_script(self, script_path):
        script = Path(str(script_path)).expanduser()
        if not script.is_absolute():
            script = self.fastwam_dir / script
        return script

    def _upload_checkpoint_to_hf(self, step_tag: str, weights_path: str | None) -> bool:
        self._last_hf_upload_error = None
        if not self.hf_upload_enabled:
            return True
        if step_tag in self._hf_uploaded_steps:
            logger.info("[hf-upload] step=%s already uploaded; skipping duplicate request.", step_tag)
            return True

        if weights_path is None or not Path(weights_path).is_file():
            self._last_hf_upload_error = f"weights checkpoint not found: {weights_path}"
            logger.warning("[hf-upload] %s", self._last_hf_upload_error)
            return False
        if not self.hf_upload_script.is_file():
            self._last_hf_upload_error = f"upload script not found: {self.hf_upload_script}"
            logger.warning("[hf-upload] %s", self._last_hf_upload_error)
            return False

        cmd = ["bash", str(self.hf_upload_script)]
        if self.hf_upload_dry_run:
            cmd.append("--dry-run")
        cmd.extend(["run", str(Path(self.output_dir).resolve()), step_tag])

        bucket_run_name = self.hf_upload_bucket_run_name
        if bucket_run_name not in (None, "", "null"):
            cmd.append(str(bucket_run_name))

        logger.info("[hf-upload] uploading step=%s output_dir=%s", step_tag, self.output_dir)
        try:
            subprocess.run(cmd, cwd=str(self.fastwam_dir), check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            self._last_hf_upload_error = str(exc)
            logger.warning("[hf-upload] failed for step=%s: %s", step_tag, exc)
            return False

        self._hf_uploaded_steps.add(step_tag)
        logger.info("[hf-upload] complete for step=%s", step_tag)
        return True

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        if self.hf_upload_enabled:
            upload_ok = True
            if self.accelerator.is_main_process:
                upload_ok = self._upload_checkpoint_to_hf(step_tag=step_tag, weights_path=ckpt_path)
            upload_status = torch.tensor(
                [0 if upload_ok else 1],
                device=self.accelerator.device,
                dtype=torch.int64,
            )
            gathered_status = self.accelerator.gather(upload_status)
            if int(gathered_status.max().item()) != 0 and self.hf_upload_fail_on_error:
                message = self._last_hf_upload_error or "HF checkpoint upload failed"
                raise RuntimeError(message)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        

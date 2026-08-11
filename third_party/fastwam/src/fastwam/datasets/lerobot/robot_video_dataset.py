# Author: Rui Heng Yang

import hashlib
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from ..video_latent_cache import VideoLatentCache, build_video_preprocess_spec
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


def _is_main_process_without_init() -> bool:
    return pytorch_utils._resolve_global_rank() == 0

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        episode_indices=None,
        episode_indices_path=None,
        episode_indices_key=None,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        video_backend: Optional[str] = None,
        video_latent_cache_dir: Optional[str] = None,
        video_latent_cache_vae_identity: Optional[dict[str, str]] = None,
        strict_getitem: bool = False,
        precompute_video_only: bool = False,
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        eef_anchor_calibration_path: Optional[str] = None,
        eef_anchor_raw_resolution: int = 512,
    ):
        if video_latent_cache_dir is not None and not strict_getitem:
            raise ValueError(
                "`strict_getitem=true` is required with `video_latent_cache_dir` "
                "to prevent latent/action sample misalignment."
            )
        if video_latent_cache_dir is not None and video_backend is None:
            raise ValueError(
                "`video_backend` must be explicitly pinned when using a video latent cache."
            )
        if video_latent_cache_dir is not None and video_latent_cache_vae_identity is None:
            raise ValueError(
                "`video_latent_cache_vae_identity` is required with a video latent cache."
            )
        if strict_getitem and skip_padding_as_possible:
            raise ValueError(
                "`skip_padding_as_possible` must be false in strict/cache mode because "
                "padding retries substitute a random global sample index."
            )
        if precompute_video_only and (not strict_getitem or video_latent_cache_dir is not None):
            raise ValueError(
                "`precompute_video_only` requires strict_getitem=true and no latent cache."
            )

        if (num_frames - 1) % action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames-1 must be divisible by action_video_freq_ratio, got "
                f"{num_frames - 1} and {action_video_freq_ratio}"
            )
        if ((num_frames - 1) // action_video_freq_ratio) % 4 != 0:
            raise ValueError(
                "video transitions must be divisible by 4 for tokenization, got "
                f"{(num_frames - 1) // action_video_freq_ratio}"
            )
        video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))

        shape_meta_container = OmegaConf.to_container(shape_meta, resolve=True)
        processor_config: dict[str, Any] | None = None
        if isinstance(processor, DictConfig):
            processor_config = OmegaConf.to_container(processor, resolve=True)

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta_container,
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            episode_indices=episode_indices,
            episode_indices_path=episode_indices_path,
            episode_indices_key=episode_indices_key,
            global_sample_stride=global_sample_stride,
            video_backend=video_backend,
            strict_getitem=strict_getitem,
            image_only=precompute_video_only,
            image_sample_indices=(video_sample_indices if precompute_video_only else None),
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        self.video_sample_indices = video_sample_indices

        self.camera_key = camera_key
        self.video_latent_cache_dir = video_latent_cache_dir
        self.strict_getitem = bool(strict_getitem)
        self.precompute_video_only = bool(precompute_video_only)
        self.lerobot_dataset._set_return_images(video_latent_cache_dir is None)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        resolved_episode_indices_path = None
        episode_indices_path_sha256 = None
        if episode_indices_path is not None:
            resolved_path = Path(episode_indices_path).expanduser().resolve()
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Episode split file not found: {resolved_path}")
            resolved_episode_indices_path = str(resolved_path)
            episode_indices_path_sha256 = hashlib.sha256(resolved_path.read_bytes()).hexdigest()

        preprocess_spec = build_video_preprocess_spec(
            dataset_dirs=dataset_dirs,
            shape_meta=shape_meta_container,
            processor_config=processor_config,
            num_frames=num_frames,
            action_video_freq_ratio=action_video_freq_ratio,
            video_size=video_size,
            concat_multi_camera=concat_multi_camera,
            video_backend=str(video_backend),
            global_sample_stride=global_sample_stride,
            sample_selection={
                "val_set_proportion": float(val_set_proportion),
                "is_training_set": bool(is_training_set),
                "episode_indices": (
                    None if episode_indices is None else [int(value) for value in episode_indices]
                ),
                "episode_indices_path": (
                    resolved_episode_indices_path
                ),
                "episode_indices_path_sha256": episode_indices_path_sha256,
                "episode_indices_key": episode_indices_key,
            },
        )
        self.video_preprocess_spec = preprocess_spec
        self.video_latent_cache = None
        if video_latent_cache_dir is not None:
            expected_vae_identity = OmegaConf.to_container(
                video_latent_cache_vae_identity, resolve=True
            ) if isinstance(video_latent_cache_vae_identity, DictConfig) else dict(
                video_latent_cache_vae_identity
            )
            self.video_latent_cache = VideoLatentCache(
                video_latent_cache_dir,
                expected_sample_count=len(self.lerobot_dataset),
                expected_preprocess_spec=preprocess_spec,
                expected_vae_identity=expected_vae_identity,
            )

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                dist_initialized = (
                    torch.distributed.is_available() and torch.distributed.is_initialized()
                )
                if _is_main_process_without_init() or not dist_initialized:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    if _is_main_process_without_init():
                        save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if dist_initialized:
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if _is_main_process_without_init():
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
        # EEF-relative camera RoPE: resolve every episode's anchors once, here,
        # and hold them in RAM. No sidecar exists (plan Section 15). Disabled
        # unless a calibration path is configured, so every other mode keeps its
        # existing sample keys and behavior byte for byte.
        self.eef_anchor_index = None
        if eef_anchor_calibration_path:
            from fastwam.datasets.eef_anchors import EpisodeAnchorIndex
            from fastwam.geometry import EEFProjector, load_calibration

            calibration = load_calibration(eef_anchor_calibration_path)
            self.eef_anchor_index = EpisodeAnchorIndex.build(
                [str(d) for d in dataset_dirs],
                calibration_path=eef_anchor_calibration_path,
                projector=EEFProjector(calibration, int(eef_anchor_raw_resolution)),
            )

    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        if not self.precompute_video_only:
            image_is_pad = image_is_pad[self.video_sample_indices]
        video = None
        video_latents = None
        if self.video_latent_cache is None:
            video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
            num_cameras = 1
            if video.ndim == 5:
                if not self.precompute_video_only:
                    video = video[:, self.video_sample_indices, :, :, :]
                num_cameras, T_video, C, H, W = video.shape
            else:
                assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
                if not self.precompute_video_only:
                    video = video[self.video_sample_indices, :, :, :]
                T_video, C, H, W = video.shape

            video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
            if self.concat_multi_camera == "robotwin":
                if num_cameras != 3:
                    raise ValueError(
                        f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                    )
                cam_top = transforms_F.resize(
                    video[0],
                    size=[256, 320],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 256, 320]
                cam_left = transforms_F.resize(
                    video[1],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                cam_right = transforms_F.resize(
                    video[2],
                    size=[128, 160],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, 128, 160]
                bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
                video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
            elif num_cameras > 1:
                if self.concat_multi_camera == "horizontal":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
                elif self.concat_multi_camera == "vertical":
                    video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
                else:
                    raise ValueError(
                        f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                        "Expected one of: horizontal, vertical, robotwin."
                    )
            else:
                video = video.squeeze(0)  # [T_video, C, H, W]

            # final resize and normalization
            video = self.resize_transform(video)
            video = self.crop_transform(video)
            video = self.normalize_transform(video)  # [T_video, C, H, W]

            video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]
        else:
            returned_index = int(sample["idx"])
            video_latents = self.video_latent_cache[returned_index]

        if self.precompute_video_only:
            if video is None:
                raise RuntimeError("precompute_video_only unexpectedly resolved cached latents.")
            return {"video": video}

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        sampled_video_frames = len(self.video_sample_indices)
        if sampled_video_frames <= 1:
            raise ValueError(f"At least two sampled video frames are required, got {sampled_video_frames}")
        if action.shape[0] % (sampled_video_frames - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {sampled_video_frames - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if video is not None:
            data["video"] = video
        else:
            data["video_latents"] = video_latents
            data["source_video_shape"] = torch.tensor(
                [3, sampled_video_frames, int(self.video_size[0]), int(self.video_size[1])],
                dtype=torch.int64,
            )
        if self.eef_anchor_index is not None:
            # Key off the RETURNED sample, never the requested idx. Both the
            # padding retry above and __getitem__'s exception handler resample
            # to a random global index, so an idx-keyed lookup would pair these
            # frames with another episode's gripper position -- silently.
            # Indexed, not `.get(..., 0)`: a missing `dataset_index` would
            # silently key every directory of a concatenated MultiLeRobotDataset
            # to directory 0, serving one suite's geometry for another's frames.
            data["eef_anchor_token"] = torch.from_numpy(
                self.eef_anchor_index.lookup(
                    dataset_index=int(sample["dataset_index"]),
                    episode_index=int(sample["episode_index"]),
                    frame_index=int(sample["frame_index"]),
                ).copy()
            )
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        if self.strict_getitem:
            return self._get(idx)
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data

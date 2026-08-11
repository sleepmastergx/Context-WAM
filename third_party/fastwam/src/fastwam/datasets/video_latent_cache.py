"""Memory-mapped storage for precomputed FastWAM video latents.

Author: Rui Heng Yang
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CACHE_SCHEMA_VERSION = 1
LATENTS_FILENAME = "latents.bf16"
COMPLETION_FILENAME = "complete.u8"
MANIFEST_FILENAME = "manifest.json"
IN_PROGRESS_MANIFEST_FILENAME = "manifest.in_progress.json"
PORTABLE_LEGACY_PREPROCESS_IMPLEMENTATIONS = {
    "f58bb6f97b8d03cb6258b2cba8fb03fcd5944823f6432262fce0b9f998c422f3":
        "8bc26276aaef7cd152c385fae6e3a8f3d687e5c8db337dc1ec0e8bf9aa8678f5",
}


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_index_files(root: Path) -> str:
    """Hash files that define global-index-to-training-sample semantics."""

    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".parquet"}
        and path.relative_to(root).parts
        and path.relative_to(root).parts[0] in {"data", "meta"}
    )
    for path in paths:
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _sha256_video_inventory(root: Path) -> str:
    """Hash source-video relative paths and sizes without rereading all media."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.mp4")):
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, byteorder="big"))
        digest.update(relative_path)
        digest.update(path.stat().st_size.to_bytes(8, byteorder="big"))
    return digest.hexdigest()


def _preprocess_implementation_sha256() -> str:
    """Hash source files that determine decoded pixels passed to the VAE."""

    datasets_dir = Path(__file__).resolve().parent
    source_paths = [
        Path(__file__).resolve(),
        datasets_dir / "dataset_utils.py",
        datasets_dir / "lerobot" / "base_lerobot_dataset.py",
        datasets_dir / "lerobot" / "robot_video_dataset.py",
        datasets_dir / "lerobot" / "processors" / "fastwam_processor.py",
        datasets_dir / "lerobot" / "lerobot" / "lerobot_dataset.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.relative_to(datasets_dir).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _pixel_preprocess_implementation_sha256() -> str:
    """Hash pixel-producing sources without this cache-validation module."""

    datasets_dir = Path(__file__).resolve().parent
    source_paths = [
        datasets_dir / "dataset_utils.py",
        datasets_dir / "lerobot" / "base_lerobot_dataset.py",
        datasets_dir / "lerobot" / "robot_video_dataset.py",
        datasets_dir / "lerobot" / "processors" / "fastwam_processor.py",
        datasets_dir / "lerobot" / "lerobot" / "lerobot_dataset.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.relative_to(datasets_dir).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def vae_implementation_sha256() -> str:
    """Hash native Wan VAE and loader sources that determine encoded latents."""

    source_root = Path(__file__).resolve().parents[1]
    source_paths = [
        source_root / "models" / "wan22" / "wan_video_vae.py",
        source_root / "models" / "wan22" / "helpers" / "loader.py",
        source_root / "models" / "wan22" / "helpers" / "io.py",
        source_root / "models" / "wan22" / "helpers" / "state_dict_converters.py",
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _package_version(package_name: str) -> str | None:
    """Return an installed package version, or ``None`` when unavailable."""

    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def build_video_preprocess_spec(
    *,
    dataset_dirs: Sequence[str | os.PathLike[str]],
    shape_meta: Mapping[str, Any],
    processor_config: Mapping[str, Any] | None,
    num_frames: int,
    action_video_freq_ratio: int,
    video_size: Sequence[int],
    concat_multi_camera: str | None,
    video_backend: str,
    global_sample_stride: int,
    sample_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic cache identity for the video preprocessing path.

    Parameters describe every configured operation that can affect the video
    tensor before VAE encoding. Dataset metadata digests catch replacement of a
    dataset root while avoiding a prohibitively expensive hash of every MP4.
    """

    roots: list[dict[str, Any]] = []
    for raw_root in dataset_dirs:
        root = Path(raw_root).expanduser().resolve()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
        roots.append(
            {
                "path": str(root),
                "info_json_sha256": _sha256_file(info_path),
                "dataset_index_sha256": _sha256_index_files(root),
                "video_inventory_sha256": _sha256_video_inventory(root),
            }
        )

    backend_package = "torchcodec" if video_backend == "torchcodec" else "av"
    return {
        "dataset_roots": roots,
        "shape_meta": json.loads(json.dumps(shape_meta, sort_keys=True)),
        "processor_config": (
            None
            if processor_config is None
            else json.loads(json.dumps(processor_config, sort_keys=True))
        ),
        "num_frames": int(num_frames),
        "action_video_freq_ratio": int(action_video_freq_ratio),
        "sampled_video_indices": list(range(0, int(num_frames), int(action_video_freq_ratio))),
        "video_size": [int(value) for value in video_size],
        "concat_multi_camera": concat_multi_camera,
        "video_backend": video_backend,
        "video_backend_package_version": _package_version(backend_package),
        "preprocess_implementation_sha256": _preprocess_implementation_sha256(),
        "torch_version": torch.__version__,
        "torchvision_version": _package_version("torchvision"),
        "global_sample_stride": int(global_sample_stride),
        "sample_selection": json.loads(json.dumps(sample_selection, sort_keys=True)),
    }


def preprocess_spec_sha256(spec: Mapping[str, Any]) -> str:
    """Return a stable digest for a video preprocessing specification."""

    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def preprocess_specs_compatible_for_cache_consumption(
    cached_spec: Mapping[str, Any], expected_spec: Mapping[str, Any]
) -> bool:
    """Return whether two specs identify the same already-encoded video tensors.

    Absolute installation paths and decoder package versions affect cache
    generation provenance, but they do not change the meaning of an existing
    latent cache at consumption time. All dataset-content hashes, preprocessing
    configuration, and implementation hashes remain mandatory.
    """

    cached_implementation = str(cached_spec.get("preprocess_implementation_sha256"))
    expected_implementation = str(expected_spec.get("preprocess_implementation_sha256"))
    if cached_implementation != expected_implementation:
        compatible_pixel_digest = PORTABLE_LEGACY_PREPROCESS_IMPLEMENTATIONS.get(
            cached_implementation
        )
        if compatible_pixel_digest != _pixel_preprocess_implementation_sha256():
            return False

    def normalize(spec: Mapping[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(spec, sort_keys=True))
        for key in (
            "preprocess_implementation_sha256",
            "torch_version",
            "torchvision_version",
            "video_backend_package_version",
        ):
            normalized.pop(key, None)
        roots = normalized.get("dataset_roots")
        if isinstance(roots, list):
            for root in roots:
                if isinstance(root, dict):
                    root.pop("path", None)
        return normalized

    return normalize(cached_spec) == normalize(expected_spec)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Atomically write a JSON object to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temporary_path, path)


def _content_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that determine the meaning of every cached latent."""

    keys = (
        "schema_version",
        "sample_count",
        "dataset_sample_count",
        "latent_shape",
        "dtype",
        "source_video_shape",
        "preprocess_spec_sha256",
        "vae",
        "encoder_batch_size_per_gpu",
    )
    return {key: manifest.get(key) for key in keys}


def create_or_resume_cache_files(
    *,
    cache_dir: Path,
    sample_count: int,
    latent_shape: Sequence[int],
    in_progress_manifest: Mapping[str, Any],
    overwrite: bool,
) -> None:
    """Create or validate resumable mmap files for a distributed cache build."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_elements = int(sample_count) * int(torch.tensor(latent_shape).prod().item())
    expected_latent_bytes = latent_elements * torch.tensor([], dtype=torch.bfloat16).element_size()
    expected_completion_bytes = int(sample_count)
    latent_path = cache_dir / LATENTS_FILENAME
    completion_path = cache_dir / COMPLETION_FILENAME
    progress_path = cache_dir / IN_PROGRESS_MANIFEST_FILENAME
    complete_manifest_path = cache_dir / MANIFEST_FILENAME

    if overwrite:
        for path in (latent_path, completion_path, progress_path, complete_manifest_path):
            if path.exists():
                path.unlink()

    if complete_manifest_path.exists():
        with complete_manifest_path.open("r", encoding="utf-8") as file_handle:
            existing_complete = json.load(file_handle)
        if existing_complete.get("status") != "complete":
            raise ValueError(f"Existing cache manifest is not complete: {complete_manifest_path}")
        if _content_identity(existing_complete) != _content_identity(in_progress_manifest):
            raise ValueError(
                f"Existing complete cache identity does not match this run: "
                f"{complete_manifest_path}. Use overwrite=true only after confirming the "
                "target cache directory."
            )

    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as file_handle:
            existing = json.load(file_handle)
        if existing != dict(in_progress_manifest):
            raise ValueError(
                f"Existing in-progress manifest does not match this run: {progress_path}. "
                "Use overwrite=true only after confirming the target cache directory."
            )
    else:
        atomic_write_json(in_progress_manifest, progress_path)

    if not latent_path.exists():
        with latent_path.open("wb") as file_handle:
            file_handle.truncate(expected_latent_bytes)
    if not completion_path.exists():
        with completion_path.open("wb") as file_handle:
            file_handle.truncate(expected_completion_bytes)

    if latent_path.stat().st_size != expected_latent_bytes:
        raise ValueError(
            f"Latent file size mismatch: {latent_path.stat().st_size} vs {expected_latent_bytes}"
        )
    if completion_path.stat().st_size != expected_completion_bytes:
        raise ValueError(
            "Completion file size mismatch: "
            f"{completion_path.stat().st_size} vs {expected_completion_bytes}"
        )


class VideoLatentCache:
    """Read a complete FastWAM BF16 latent cache by global dataset index.

    The mmap is opened lazily in each DataLoader worker. Returned tensors are
    cloned so collating cannot alias or mutate the shared disk mapping.
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        expected_sample_count: int,
        expected_preprocess_spec: Mapping[str, Any],
        expected_vae_identity: Mapping[str, Any],
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        manifest_path = self.cache_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Complete video latent cache manifest not found: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as file_handle:
            self.manifest: dict[str, Any] = json.load(file_handle)

        if self.manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported video latent cache schema: "
                f"{self.manifest.get('schema_version')} vs {CACHE_SCHEMA_VERSION}"
            )
        if self.manifest.get("status") != "complete":
            raise ValueError(f"Video latent cache is not complete: {manifest_path}")
        if self.manifest.get("dtype") != "bfloat16":
            raise ValueError(f"Expected BF16 cache, got {self.manifest.get('dtype')}")

        self.sample_count = int(self.manifest["sample_count"])
        if self.sample_count != int(expected_sample_count):
            raise ValueError(
                f"Cache sample count mismatch: {self.sample_count} vs {expected_sample_count}"
            )
        expected_digest = preprocess_spec_sha256(expected_preprocess_spec)
        cached_digest = self.manifest.get("preprocess_spec_sha256")
        if cached_digest != expected_digest:
            cached_spec = self.manifest.get("preprocess_spec")
            cached_spec_is_valid = (
                isinstance(cached_spec, Mapping)
                and preprocess_spec_sha256(cached_spec) == cached_digest
            )
            if not cached_spec_is_valid or not preprocess_specs_compatible_for_cache_consumption(
                cached_spec, expected_preprocess_spec
            ):
                raise ValueError(
                    "Video latent cache preprocessing mismatch: "
                    f"cache={cached_digest} current={expected_digest}"
                )
        expected_vae = {
            key: str(expected_vae_identity.get(key))
            for key in ("model_id", "sha256", "implementation_sha256", "encode_dtype")
        }
        if not expected_vae["model_id"] or expected_vae["model_id"] == "None":
            raise ValueError("Expected VAE identity must include `model_id`.")
        if len(expected_vae["sha256"]) != 64:
            raise ValueError("Expected VAE identity must include a SHA-256 digest.")
        if len(expected_vae["implementation_sha256"]) != 64:
            raise ValueError("Expected VAE identity must include an implementation SHA-256.")
        if expected_vae["encode_dtype"] != "bfloat16":
            raise ValueError(
                f"Video latent caches require BF16 VAE encoding, got {expected_vae['encode_dtype']}."
            )
        cached_vae = self.manifest.get("vae", {})
        cached_vae_identity = {key: str(cached_vae.get(key)) for key in expected_vae}
        if cached_vae_identity != expected_vae:
            raise ValueError(
                "Video latent cache VAE identity mismatch: "
                f"cache={cached_vae_identity} current={expected_vae}"
            )

        self.latent_shape = tuple(int(value) for value in self.manifest["latent_shape"])
        if len(self.latent_shape) != 4 or any(value <= 0 for value in self.latent_shape):
            raise ValueError(f"Invalid cached latent shape: {self.latent_shape}")
        self._elements_per_sample = int(torch.tensor(self.latent_shape).prod().item())
        expected_bytes = (
            self.sample_count
            * self._elements_per_sample
            * torch.tensor([], dtype=torch.bfloat16).element_size()
        )
        latent_path = self.cache_dir / LATENTS_FILENAME
        if not latent_path.is_file() or latent_path.stat().st_size != expected_bytes:
            actual = latent_path.stat().st_size if latent_path.exists() else None
            raise ValueError(f"Cache latent file size mismatch: {actual} vs {expected_bytes}")

        completion_path = self.cache_dir / COMPLETION_FILENAME
        if not completion_path.is_file() or completion_path.stat().st_size != self.sample_count:
            raise ValueError("Cache completion bitmap is missing or has the wrong size.")
        completion = torch.from_file(
            str(completion_path), shared=False, size=self.sample_count, dtype=torch.uint8
        )
        incomplete_count = int((completion != 1).sum().item())
        if incomplete_count:
            raise ValueError(f"Cache contains {incomplete_count} incomplete samples.")

        self._latents: torch.Tensor | None = None

    def __len__(self) -> int:
        """Return the number of cached samples."""

        return self.sample_count

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return one cloned BF16 latent tensor."""

        if index < 0 or index >= self.sample_count:
            raise IndexError(f"Cache index {index} out of bounds for {self.sample_count} samples.")
        if self._latents is None:
            total_elements = self.sample_count * self._elements_per_sample
            self._latents = torch.from_file(
                str(self.cache_dir / LATENTS_FILENAME),
                shared=False,
                size=total_elements,
                dtype=torch.bfloat16,
            ).view(self.sample_count, *self.latent_shape)
        return self._latents[index].clone()

    def __getstate__(self) -> dict[str, Any]:
        """Drop the mmap when the dataset is serialized to a worker."""

        state = self.__dict__.copy()
        state["_latents"] = None
        return state

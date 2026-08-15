"""Pre-download the Wan2.2 weights the Fast-WAM loader needs, onto the volume.

The loader resolves weights under $DIFFSYNTH_MODEL_BASE_PATH (falling back to
./checkpoints RELATIVE TO CWD). Doing the download as its own step means the
first real training run does not spend 20+ minutes pulling ~22 GiB, and — more
useful on a pod that gets recycled — the files land on /workspace and survive
a machine switch.

Uses the repo's OWN ModelConfig (loaded straight from the vendored file, so no
fastwam package import and no torch/deepspeed requirement beyond what io.py
itself imports), so the on-disk layout is exactly what the loader then globs
for: $BASE/<model_id>/<origin_file_pattern>.

    DIFFSYNTH_MODEL_BASE_PATH=/workspace/checkpoints/ \
    python scripts/download_wan_weights.py

Source: modelscope, the loader's default — KEEP IT. Only the DiT lives on
HuggingFace too; `DiffSynth-Studio/Wan-Series-Converted-Safetensors` (the
converted VAE and T5) does not exist there, so DIFFSYNTH_DOWNLOAD_SOURCE=
huggingface 404s on two of the four downloads.

Downloads (all four repos are public; no token needed):
    diffusion_pytorch_model*.safetensors    ~10 GiB  video DiT (Wan2.2-TI2V-5B)
    models_t5_umt5-xxl-enc-bf16.safetensors ~11 GiB  T5-XXL, for text contexts
    Wan2.2_VAE.safetensors                  ~0.5 GiB the VAE the cache uses
    google/umt5-xxl/                        ~5 MiB   tokenizer
"""
import argparse
import importlib.util
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
IO_PY = HERE / "third_party/fastwam/src/fastwam/models/wan22/helpers/io.py"

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"


def _load_io_module():
    spec = importlib.util.spec_from_file_location("_fastwam_io", IO_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH",
                                                     "/workspace/checkpoints/"))
    ap.add_argument("--skip-text-encoder", action="store_true",
                    help="skip the 11 GiB T5 (only needed to encode text "
                         "contexts; training runs with load_text_encoder: false)")
    args = ap.parse_args()

    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.base
    ModelConfig = _load_io_module().ModelConfig

    # the same four configs _resolve_configs() builds with redirect_common_files
    configs = [
        ("video DiT", ModelConfig(
            model_id=MODEL_ID,
            origin_file_pattern="diffusion_pytorch_model*.safetensors")),
        ("VAE", ModelConfig(
            model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            origin_file_pattern="Wan2.2_VAE.safetensors")),
        ("tokenizer", ModelConfig(
            model_id=TOKENIZER_MODEL_ID, origin_file_pattern="google/umt5-xxl/")),
    ]
    if not args.skip_text_encoder:
        configs.insert(1, ("T5 text encoder", ModelConfig(
            model_id="DiffSynth-Studio/Wan-Series-Converted-Safetensors",
            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.safetensors")))

    for name, cfg in configs:
        print(f"==> {name}: {cfg.model_id}/{cfg.origin_file_pattern}", flush=True)
        cfg.download_if_necessary()
        print(f"    -> {cfg.path}", flush=True)

    print(f"\nall weights under {args.base}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

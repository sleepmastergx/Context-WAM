#!/bin/bash
# Environment for the DP pipeline (stage-1 DP + stage-2 arms + sim eval).
# Creates a python 3.11 venv and clones the two PINNED public dependencies.
#
#   VENV=...      (default: <repo>/.venv-dp)
#   EXTERNAL=...  (default: <repo>/external)   where the clones go
#
# When it finishes, export what it prints (DP_REPO + PATH) before running dp/.
# NOTE: this env is SEPARATE from the Fast-WAM training env (torch 2.9.1 vs
# 2.7.1; the benchmark's mani-skill fork pins differ) — do not merge them.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)          # repo root
VENV=${VENV:-$HERE/.venv-dp}
EXT=${EXTERNAL:-$HERE/external}
BENCH_PIN=f9015a5      # github.com/RoboMME/robomme_benchmark
DP_PIN=f333cd6         # github.com/RoboMME/DP
mkdir -p "$EXT"

# --- uv (static binary, no root needed) ---
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.11
uv venv --python 3.11 "$VENV"
export PATH="$VENV/bin:$PATH"

# --- pinned public dependencies ---
[ -d "$EXT/robomme_benchmark" ] || \
    git clone https://github.com/RoboMME/robomme_benchmark.git "$EXT/robomme_benchmark"
git -C "$EXT/robomme_benchmark" checkout -q "$BENCH_PIN"
[ -d "$EXT/DP" ] || git clone https://github.com/RoboMME/DP.git "$EXT/DP"
git -C "$EXT/DP" checkout -q "$DP_PIN"

# benchmark brings mani-skill fork 3.0.0b21, sapien 3.0.2, torch 2.9.1,
# gymnasium 0.29.1
uv pip install --python "$VENV/bin/python" -e "$EXT/robomme_benchmark"

# DP deps. The datasets UPPER bound is load-bearing: lerobot 0.3.3 declares
# datasets<=3.6.0, and from 4.0 ds["col"] returns a lazy Column instead of a
# list, so lerobot's torch.stack(hf_dataset["timestamp"]) raises TypeError.
# 4.0+ also writes the parquet feature as _type:List, which 3.6.0 cannot read
# back -- so writer and reader must agree, and convert_h5_to_lerobot.py must
# run in THIS venv, not a system python that resolved a newer datasets.
# lerobot: their environment.yml pins 0.3.3, but 0.3.3 caps
# torchvision<0.23 while the benchmark pins torchvision 0.24.1 (the authors
# used two envs). Resolution (verified working): install lerobot WITHOUT deps
# — our data is image-parquet, none of the capped pins are exercised. A plain
# "pip install lerobot" DOWNGRADES torch to 2.2.x and breaks transformers.
uv pip install --python "$VENV/bin/python" \
    hydra-core omegaconf diffusers transformers einops wandb \
    pyarrow pandas opencv-python-headless "imageio[ffmpeg]" \
    "datasets>=2.19,<=3.6.0" jsonlines draccus deepdiff h5py
uv pip install --python "$VENV/bin/python" -e "$EXT/DP"
uv pip install --python "$VENV/bin/python" --no-deps "lerobot==0.3.3"
# ...but --no-deps also drops PyAV, and lerobot/datasets/lerobot_dataset.py
# imports video_utils -> `import av` at MODULE level, so LeRobotDataset is
# unimportable without it even though our data is image-parquet, not video.
# Installed on its own: it pulls no torch pin, so 2.9.1 survives.
uv pip install --python "$VENV/bin/python" av
# re-assert the benchmark's torch in case any dep nudged it
uv pip install --python "$VENV/bin/python" "torch==2.9.1" "torchvision==0.24.1"

# --- CLIP weights into the HF cache now (compute nodes may lack egress) ---
"$VENV/bin/python" - <<'EOF'
from transformers import CLIPTextModel, CLIPTokenizer
CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP cached")
EOF

# --- import checks (CPU only) ---
"$VENV/bin/python" - <<'EOF'
import robomme, mani_skill, torch, sapien, gymnasium
print("robomme OK; torch", torch.__version__, "| sapien", sapien.__version__,
      "| gymnasium", gymnasium.__version__)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("lerobot 0.3.x API OK")
EOF

# --- sapien URDF encoding check (a 3.0.0b1 bug needed a source patch;
#     verify it is fixed in the pinned 3.0.2 before trusting sim resets) ---
URDF=$("$VENV/bin/python" -c "import sapien, pathlib; print(pathlib.Path(sapien.__file__).parent / 'wrapper' / 'urdf_loader.py')")
if grep -nE 'open\([^)]*"r"\)' "$URDF" | grep -v encoding; then
    echo "WARNING: sapien urdf_loader.py opens files without encoding=;"
    echo "patch THIS venv's copy to open(..., encoding='utf-8')."
else
    echo "sapien urdf_loader encoding OK"
fi

echo
echo "DONE. Before running anything in dp/:"
echo "  export DP_REPO=$EXT/DP"
echo "  export PATH=$VENV/bin:\$PATH"

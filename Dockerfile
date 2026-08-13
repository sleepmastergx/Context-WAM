
# Context-WAM training image — dependencies ONLY.
#
#   ghcr.io/sleepmastergx/context-wam:cu128-torch271
#
# Deliberately NOT in this image: source code, datasets, model weights, HF/pip
# caches, checkpoints, outputs, secrets. All of those live on the RunPod
# Network Volume mounted at /workspace, so a pod start is:
#
#   1. attach the same Network Volume at /workspace
#   2. launch this image
#   3. git clone/pull into /workspace/projects/Context-WAM
#   4. python train.py --arm ttt        # no pip install, no re-download
#
# Reproduces the environment validated on RunPod (RTX 5090 / sm_120, driver
# 580.x, CUDA 12.8): Ubuntu 24.04 + Python 3.12.3 + torch 2.7.1+cu128.
#
# Build/push:
#   docker build -t ghcr.io/sleepmastergx/context-wam:cu128-torch271 .
#   docker push  ghcr.io/sleepmastergx/context-wam:cu128-torch271

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

LABEL org.opencontainers.image.title="Context-WAM"
LABEL org.opencontainers.image.description="Dependency-only training image for Context-WAM (Fast-WAM + TTT episodic memory). Source, data and weights come from the mounted /workspace volume."
LABEL org.opencontainers.image.source="https://github.com/sleepmastergx/Context-WAM"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------- OS packages
# python3.12 is Ubuntu 24.04's default — the version this stack was validated on.
# build-essential + ninja + libaio-dev: DeepSpeed JIT-compiles its ops on first
# use (fused/cpu adam); nvcc comes from the -devel base.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv \
        build-essential ninja-build libaio-dev \
        git git-lfs curl ca-certificates \
        libglib2.0-0 libgl1 \
    && git lfs install --system \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Ubuntu 24.04 marks the system interpreter externally-managed; this image IS
# the environment, so install into it directly (same as the validated pod).
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1
# (PIP_NO_CACHE_DIR is turned back off in the runtime ENV block below — it is a
# build-time setting, so wheels do not bloat the image layer.)

# ------------------------------------------------------- python dependencies
# The install ORDER is load-bearing (see requirements.txt):
#   1. torch first — DeepSpeed's build imports it
#   2. deepspeed --no-build-isolation so its build env sees torch
#   3. everything else (torch already satisfied → nothing gets downgraded)
# --ignore-installed: Ubuntu's python3-pip is dpkg-owned and has no RECORD
# file, so a plain -U fails with "Cannot uninstall pip 24.0". This installs
# the new pip into /usr/local, shadowing /usr/bin/pip3 — the same layout as
# the validated pod.
RUN pip install -U --ignore-installed pip wheel setuptools

RUN pip install "torch==2.7.1" "torchvision==0.22.1" \
        --index-url https://download.pytorch.org/whl/cu128

RUN pip install "deepspeed==0.18.5" --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# third_party/fastwam is vendored IN THE REPO, so it cannot be baked in here.
# context_wam/build_model.py already puts it on sys.path at import time; this
# .pth makes a plain `import fastwam` work too, for the default clone location.
# Clone elsewhere → export FASTWAM_SRC=<clone>/third_party/fastwam/src.
RUN echo "/workspace/projects/Context-WAM/third_party/fastwam/src" \
        > /usr/local/lib/python3.12/dist-packages/context-wam-fastwam.pth

# --------------------------------------------------------- persistent layout
# Every path below is on the Network Volume. Nothing here is baked into the
# image — these are the defaults the code and setup.sh read.
ENV HF_HOME=/workspace/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODELSCOPE_CACHE=/workspace/hf_cache/modelscope \
    DIFFSYNTH_MODEL_BASE_PATH=/workspace/checkpoints/ \
    CACHE_DIR=/workspace/data/movecube_fastwam \
    OUT_ROOT=/workspace/outputs \
    CHECKPOINT_ROOT=/workspace/checkpoints \
    PIP_NO_CACHE_DIR=0 \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    TORCH_HOME=/workspace/.cache/torch \
    TRITON_CACHE_DIR=/workspace/.cache/triton

VOLUME ["/workspace"]
WORKDIR /workspace/projects/Context-WAM

CMD ["/bin/bash"]

# Portable environment for the DP pipeline (stage-1 DP + stage-2 arms + eval).
#
#   source scripts/env-dp.sh
#
# Everything is derived from this file's own location, so it works from a clone
# anywhere -- a laptop, a workstation, a pod. Nothing here assumes /workspace.
# Separate from the Fast-WAM env by design (torch 2.9.1 vs 2.7.1): do not source
# both into one shell.

_ENV_DP_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT="$_ENV_DP_HERE"
export DP_REPO="${DP_REPO:-$REPO_ROOT/external/DP}"
export PATH="$REPO_ROOT/.venv-dp/bin:$PATH"

# Caches: respect anything already exported (a pod points these at a network
# volume), else fall back to the usual per-user locations.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

export DATASET_ROOT="${DATASET_ROOT:-$REPO_ROOT/data/robomme_data_lerobot_videounmask}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/runs}"

# --- Vulkan ICD selection (sim eval renders through it) --------------------
# Sim eval needs a Vulkan driver. Prefer the GPU, but only when its DRM render
# node is actually openable: some containers bind-mount /dev/dri/renderD* from
# the host with an owner outside the container's user namespace, and then even
# root gets EACCES (cap_dac_override does not cross a userns boundary). The
# NVIDIA ICD fails to initialise and sapien reports the unhelpful "failed to
# find a rendering device". Falling back to lavapipe keeps eval correct, just
# slow. Run `python checks/check_vulkan.py` for a one-second verdict.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
    _nvidia_icd=""
    for _c in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
        [ -f "$_c" ] && { _nvidia_icd="$_c"; break; }
    done
    _lavapipe=""
    for _c in /usr/share/vulkan/icd.d/lvp_icd.json \
              /usr/share/vulkan/icd.d/lvp_icd.x86_64.json; do
        [ -f "$_c" ] && { _lavapipe="$_c"; break; }
    done
    _render_ok=0
    for _n in /dev/dri/renderD*; do
        [ -r "$_n" ] && [ -w "$_n" ] && { _render_ok=1; break; }
    done

    if [ -n "$_nvidia_icd" ] && [ "$_render_ok" = "1" ]; then
        export VK_ICD_FILENAMES="$_nvidia_icd"
    elif [ -n "$_lavapipe" ]; then
        export VK_ICD_FILENAMES="$_lavapipe"
        echo "env-dp: GPU render node unavailable -> CPU Vulkan (lavapipe)." >&2
        echo "        Sim eval will be correct but very slow." >&2
    elif [ -n "$_nvidia_icd" ]; then
        export VK_ICD_FILENAMES="$_nvidia_icd"
    fi
    unset _nvidia_icd _lavapipe _render_ok _c _n
fi

cd "$REPO_ROOT" 2>/dev/null || true

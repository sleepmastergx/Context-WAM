#!/bin/bash
# EXAMPLE Slurm launcher — adapt partition/gres/mem to the target cluster.
# One arm per job; submit control and ttt as INDEPENDENT jobs (no --dependency
# chaining). Do not pin nodes unless the cluster requires it.
#SBATCH --job-name=context_wam
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#
# Usage:  sbatch --export=ALL,ARM=ttt,CACHE_DIR=/path/to/movecube_fastwam \
#                scripts/slurm_example.sbatch
set -euo pipefail
: "${ARM:?export ARM=control|ttt}"
: "${CACHE_DIR:?export CACHE_DIR=<downloaded cache>}"
# NEVER override CUDA_VISIBLE_DEVICES with raw indices here — Slurm exports
# the allocated devices itself; raw indices can land on another user's GPUs.
cd "$(dirname "$0")/.."
exec bash "scripts/launch_${ARM}.sh"

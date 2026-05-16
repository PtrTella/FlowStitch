#!/bin/bash
#SBATCH --job-name=FLUX_STITCH
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/stitching.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/stitching.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch

# Variable
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export DIFFUSERS_NO_FLASH_ATTN=1
export PYTHONUNBUFFERED=1

# Python
source .venv_hpc/bin/activate
python hpc_cluster/flux_latent_stitching.py


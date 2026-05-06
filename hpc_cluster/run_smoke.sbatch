#!/bin/bash
#SBATCH --job-name=FluxFirstRun
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/smoke.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/smoke.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch/

# Variabili d'ambiente fondamentali
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export DIFFUSERS_NO_FLASH_ATTN=1
export PYTHONUNBUFFERED=1

# Pyhton
source .venv_hpc/bin/activate
python hpc_cluster/flux_smoke_gen.py
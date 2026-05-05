#!/bin/bash
#SBATCH --job-name=FluxFirstRun
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00            # 1 ora per stare larghi col download
#SBATCH --output=logs/smoke.out
#SBATCH --error=logs/smoke.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch

# Variabili d'ambiente fondamentali
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
source /scratch.hpc/pietro.tellarini2/.venv/bin/activate

python smoke_test_flux.py
#!/bin/bash
#SBATCH --job-name=FLUX_DATA
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/dataset.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/dataset.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch

# Variable
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export PYTHONUNBUFFERED=1

# Python
source .venv_hpc/bin/activate
python hpc_cluster/run_dataset.py
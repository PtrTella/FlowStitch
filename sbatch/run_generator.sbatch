#!/bin/bash
#SBATCH --job-name=FLUX_DATA
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/dataset_%j.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/dataset_%j.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch

# Variable
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export DIFFUSERS_NO_FLASH_ATTN=1
export PYTHONUNBUFFERED=1

# Python
source /scratch.hpc/pietro.tellarini2/.venv/bin/activate
python dataset_generator.py
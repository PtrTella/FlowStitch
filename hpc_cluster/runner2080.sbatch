#!/bin/bash
#SBATCH --partition=rtx2080
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch
#
# %x verrà sostituito dal --job-name che passi al lancio
# %j verrà sostituito dal JOB_ID numerico
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/%x.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/hpc_cluster/logs/%x.err

# Variabili d'ambiente comuni
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

source .venv_hpc/bin/activate

# Eseguiamo lo script passato come primo argomento ($1)
# Se vuoi passare altri argomenti allo script python, puoi usare "$@"
python $1

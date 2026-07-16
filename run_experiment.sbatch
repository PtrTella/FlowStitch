#!/bin/bash
#SBATCH --job-name=FLUX_STITCH
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/stitching_%j.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/stitching_%j.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch

# Variabili d'ambiente per il cluster
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export DIFFUSERS_NO_FLASH_ATTN=1
export PYTHONUNBUFFERED=1
export FORCE_IPV4=1

# Creazione cartella log
mkdir -p logs
mkdir -p outputs/experiment_cluster

# Attivazione Ambiente (assicurati che .venv o .venv_hpc esista sul nodo)
source .venv_hpc/bin/activate

# Lancio dello script
python run_experiment.py

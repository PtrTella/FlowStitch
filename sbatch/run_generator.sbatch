#!/bin/bash
#SBATCH --job-name=FLUX_DATA
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/dataset_%j.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/dataset_%j.err

# Spostati nella cartella di lavoro
cd /scratch.hpc/pietro.tellarini2/FlowStitch

# Crea logs
mkdir -p logs

# Variabile base di Hugging Face
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"

# LA VARIABILE DELLA DISCORDIA (Se fallisce l'attenzione, togli il commento a questa riga)
export DIFFUSERS_NO_FLASH_ATTN=1

# Attiva l'ambiente virtuale

# Lancia il generatore
python flux_dataset_generator.py
#!/bin/bash
#SBATCH --job-name=FLUX_STITCH
#SBATCH --partition=l40
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/stitching_%j.out
#SBATCH --error=/scratch.hpc/pietro.tellarini2/FlowStitch/logs/stitching_%j.err
#SBATCH --chdir=/scratch.hpc/pietro.tellarini2/FlowStitch

# Variabili d'ambiente per il cluster (TUTTO reindirizzato su scratch)
export HF_HOME="/scratch.hpc/pietro.tellarini2/huggingface_cache"
export HF_HUB_CACHE="/scratch.hpc/pietro.tellarini2/huggingface_cache/hub"
export DIFFUSERS_NO_FLASH_ATTN=1
export PYTHONUNBUFFERED=1
export FORCE_IPV4=1

# Caricamento credenziali protette da .env (mai tracciato in git)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Creazione cartella log e output
mkdir -p logs
mkdir -p outputs/cluster_matrix

# Attivazione Ambiente virtuale
source .venv_hpc/bin/activate

# Esecuzione diretta dello script Python passato come argomento
# Default: scripts/run_cluster_matrix.py
SCRIPT_TO_RUN="${1:-scripts/run_cluster_matrix.py}"
shift $(( $# > 0 ? 1 : 0 ))

echo "=== FlowStitch SLURM Job ==="
echo "Script:     $SCRIPT_TO_RUN"
echo "Argomenti:  $*"
echo "Node:       $(hostname)"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================"

# Controllo esistenza file
if [ ! -f "$SCRIPT_TO_RUN" ]; then
    echo "[ERRORE] Script non trovato: $SCRIPT_TO_RUN"
    exit 1
fi

python "$SCRIPT_TO_RUN" "$@"



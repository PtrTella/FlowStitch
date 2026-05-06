# FlowStitch

Progetto di ricerca sulla Cross-Attention e Latent Stitching per modelli FLUX.1.

## Struttura del Progetto

- `hpc_cluster/`: Codice per l'estrazione dati su cluster HPC (Slurm).
- `local_analysis/`: Notebook Jupyter per l'analisi dei tensori e generazione maschere.
- `data/`: Dataset generato (ignorato da Git).
- `docs/`: Documentazione, abstract e bozze per la tesi.

## Setup

### HPC
```bash
cd hpc_cluster
pip install -r requirements_hpc.txt
sbatch run_dataset.sbatch
```

### Locale
```bash
cd local_analysis
pip install -r requirements_local.txt
jupyter notebook
```

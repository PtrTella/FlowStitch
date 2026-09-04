# FlowStitch

FlowStitch is a research framework for unsupervised latent decomposition and generative stitching in Flow Matching models, specifically implemented for the **FLUX.1 (MM-DiT)** architecture.

The framework extracts semantic concepts from latent space trajectories and injects them into target generative scenes by steering the ODE (Ordinary Differential Equation) velocity field during inference, using **Spectral Matting (DiffCut)**, **Topological Data Analysis (TDA $H_0$)**, and **Kinetic Trajectory Shaping (KTS)**.

---

## Quick Setup

Il repository usa una configurazione unificata in `pyproject.toml` per gestire gli ambienti di sviluppo sia su Mac che sul cluster HPC, senza file di configurazione sparsi.

### 🍎 1. Setup su Mac (Sviluppo Locale con `uv`)
Crea il virtualenv e installa le dipendenze locali (supporto nativo Apple Silicon MPS, Jupyter e testing):
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[local]"
```

### ⚡ 2. Setup su Cluster HPC Giano (Calcolo GPU con `pip`)
Sul cluster lavora sempre dentro `/scratch.hpc/pietro.tellarini2/FlowStitch/`.
Crea l'ambiente virtuale e installa con un solo comando esplicito evitando di intasare la quota home:
```bash
python3 -m venv .venv_hpc
source .venv_hpc/bin/activate
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -e ".[cluster]"
```
> **Perché questi flag?**
> * `--no-cache-dir`: impedisce a pip di salvare copie dei pacchetti nella home (`~/.cache/pip`), proteggendo la quota disco studente di Giano da saturazione.
> * `--extra-index-url https://download.pytorch.org/whl/cu118`: scarica direttamente i binari PyTorch precompilati per le GPU NVIDIA (CUDA 11.8).
> * `-e ".[cluster]"`: installa il pacchetto `flowstitch` in modalità editable assieme all'extra `bitsandbytes` per ottimizzazione VRAM.

---

## Gestione Credenziali e File di Configurazione

Per mantenere il repository ordinato e sicuro:

* **`pyproject.toml` (Root):**  
  L'**unica fonte di verità** per tutte le dipendenze di progetto. Include i profili:
  * `local`: Jupyter, ipykernel, pytest.
  * `cluster`: `bitsandbytes` per quantizzazione e VRAM optimization.
* **`.env` (Root - Ignorato da git):**  
  File privato per i segreti (protetto da permessi `600` e inserito in `.gitignore`). Contiene il token Hugging Face per i modelli gated come FLUX.1:
  ```env
  HF_TOKEN=hf_tuo_token_qui
  ```

---

## Come Eseguire gli Esperimenti

### 1. Esecuzione su Cluster SLURM (Giano - GPU L40)
Lancia la pipeline completa (compilazione maschera ibrida in-place + iniezione ODE con KTS):
```bash
sbatch run_experiment.sbatch
```
Controlla lo stato del job e segui i log in diretta:
```bash
squeue -u $USER
tail -f logs/stitching_*.out
```
L'immagine combinata finale verrà salvata in `outputs/experiment_cluster/stitching_dual.png`.

### 2. Esecuzione e Studio su Mac (Offline)
Sul Mac puoi analizzare i dati e visualizzare l'intero percorso teorico senza bisogno di GPU pesanti aprendo il master notebook della ricerca:
```bash
jupyter notebook notebooks/PhD_Research_Theoretical_Journey.ipynb
```

---

## Struttura del Repository

```text
FlowStitch/
├── flowstitch/              # Pacchetto Python modulare
│   ├── core/                # Hooking MM-DiT, configurazioni, serializzazione safetensors
│   ├── extraction/          # Algoritmi maschere (Spectral/DiffCut, TDA H0, Energy Gating)
│   ├── stitching/           # Perturbazione ODE (KTS) e Look-Back EMA
│   ├── pipelines/           # Orchestrazione (dataset_generation, mask_compilation, latent_stitching)
│   └── evaluation/          # Metriche quantitative (DICE, IoU, CLIPScore)
├── data/dataset_v1/         # Semantic Cache (tensori x0, v0, attention_maps in safetensors)
├── notebooks/               # Notebook teorici di ricerca per la tesi
├── docs/                    # Tesi LaTeX e Academic Report
├── run_experiment.py        # Entrypoint Python dell'esperimento
├── run_experiment.sbatch    # Script SLURM per nodo GPU L40
├── pyproject.toml           # Configurazione pacchetto ed extras (local / cluster)
└── .env                     # Token segreti (HF_TOKEN) mai committati in git
```

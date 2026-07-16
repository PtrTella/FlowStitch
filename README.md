# FlowStitch

Unsupervised latent decomposition and stitching in Flow Matching models (FLUX.1).

This project explores zero-shot semantic injection — transplanting objects from one generation into another scene by manipulating the ODE velocity field during inference. It implements Kinetic Trajectory Shaping (KTS) and Graph-based Spectral Matting.

## Architecture

```mermaid
graph TD
    A[Dataset Generation] --> B[Cross-Attention Maps]
    B --> C[Spectral Matting Fiedler]
    C --> D[Target Mask A_target]
    A --> E[Base Noise x_0]
    A --> F[Initial Velocity v_0]
    D --> G[Latent Stitching ODE]
    E --> G
    F --> G
    G --> H[Final Image]
```

## Structure

- `flowstitch/`: Core python package
  - `core/`: Config, hooking logic, serialization
  - `extraction/`: Mask extraction (Attention, Spectral, Energy, Hybrid, TDA, Decoders)
  - `stitching/`: ODE Perturbation, KTS, EMA Smoothing, Semantic Processor
  - `evaluation/`: DICE, IoU, CLIPScore metrics and benchmark runner
  - `pipelines/`: End-to-end execution (dataset generation, mask compilation, latent stitching)
- `notebooks/`: Jupyter notebooks for exploratory analysis (9 notebooks)
- `docs/`: LaTeX thesis and research documents
- `tests/`: Unit tests
- `run_experiment.py`: HPC experiment entrypoint
- `run_experiment.sbatch`: SLURM batch script

## Setup

### 1. Install Package
```bash
# Basic installation
pip install -e .

# HPC installation (with bitsandbytes for quantization)
pip install -e ".[hpc]"

# Dev installation
pip install -e ".[dev]"
```

### 2. HPC Usage
```bash
cd hpc_cluster
sbatch run_dataset.sbatch
```

## References
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- DiffCut: Zero-Shot Object Segmentation (Wu et al., 2024)
- FLUX.1 (Black Forest Labs, 2024)

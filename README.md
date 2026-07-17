# FlowStitch

FlowStitch is a framework for unsupervised latent decomposition and generative stitching in Flow Matching models, specifically implemented for the FLUX.1 architecture. 

The repository provides tools to extract semantic objects from a generated latent space and inject them into a different generative target scene by manipulating the ODE (Ordinary Differential Equation) velocity field during inference, without relying on external segmentation models or fine-tuning.

For detailed theoretical foundations, mathematical formulations, and the research background (including Spectral Matting and Kinetic Trajectory Shaping), see the [Academic Report](docs/Academic_Report_FlowStitch.md).

## Method Overview

The framework operates in three main stages:
1. **Data Capture:** A target prompt is generated once using FLUX.1. A hooking mechanism captures the initial noise $x_0$, the initial velocity field $v_0$, and cross-attention maps at timestep $t=0$.
2. **Mask Compilation:** The captured data is processed offline to compile a spatial mask of the target object. Supported methods include raw attention, Otsu thresholding, Chebyshev energy gating, Spectral Matting (Fiedler vector decomposition), and a hybrid Fiedler-attention approach.
3. **Latent Stitching:** The target noise is blended with new ambient noise at $t=0$, and the ODE trajectory is guided during the second generation using Kinetic Trajectory Shaping (KTS) and attention-level Semantic Grafting.

## Repository Structure

- `flowstitch/`: Core Python package.
  - `core/`: Hooking mechanics, configurations, and serialization utilities.
  - `extraction/`: Mask extraction algorithms (Spectral, Energy, Attention, TDA).
  - `stitching/`: ODE perturbation (KTS) and attention grafting modules.
  - `pipelines/`: End-to-end orchestration scripts (capture, compile, stitch).
  - `evaluation/`: Validation metrics (DICE, IoU, CLIPScore).
- `docs/`: Technical reports, project updates, and research documents.
- `notebooks/`: Jupyter notebooks detailing the exploratory analysis phase.
- `tests/`: Basic unit tests.
- `run_experiment.py`: Main execution script for experiments.
- `run_experiment.sbatch`: SLURM script for cluster deployment.

## Installation

Install the package in editable mode:
```bash
pip install -e .
```

To include development dependencies or cluster-specific libraries:
```bash
pip install -e ".[dev]"
pip install -e ".[hpc]"
```

## Running Experiments

To run the default experiment pipeline (generating data, compiling a hybrid mask, and executing stitching):
```bash
python run_experiment.py
```

On a SLURM cluster, you can submit the job using the provided batch script:
```bash
sbatch run_experiment.sbatch
```

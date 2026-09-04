# Zero-Shot Semantic Object Extraction and Composition in Flow Matching Generative Models

**Author:** Pietro Tellarini  
**Supervisor:** Prof. Andrea Asperti  
**Degree:** MSc Artificial Intelligence (LM-18 / LM-32)  
**Institution:** Alma Mater Studiorum — Università di Bologna  
**Academic Year:** 2025/2026

---

## Abstract

<!-- TODO: Scrivilo per ultimo, 200 parole max -->

---

## Chapter 1 — Introduction

### 1.1 Motivation

Recent advances in text-to-image generative models have produced architectures capable of synthesizing photorealistic images from natural language descriptions. Models such as Stable Diffusion, DALL-E 3, and FLUX.1 achieve remarkable quality by learning to reverse a noise-corruption process, either through score-based diffusion or, more recently, through Flow Matching — a formulation based on continuous Ordinary Differential Equations (ODEs).

Despite their generative power, these models offer limited control over the spatial composition of the output. A user can describe a scene in text, but cannot directly specify where individual objects should appear, how they should interact with the environment, or transplant an object from one generation into another. Existing approaches to compositional control typically require fine-tuning (LoRA, DreamBooth), external segmentation networks (SAM), or training-time conditioning — all of which impose significant computational overhead and restrict zero-shot applicability.

This thesis investigates whether the internal representations of a pre-trained Flow Matching model contain sufficient structural information to decompose a generated scene into its semantic components and recompose them into a new context, without any external supervision, fine-tuning, or auxiliary models.

### 1.2 Research Questions

The work addresses three interconnected questions:

1. **Extraction:** Can we isolate the spatial extent of a semantic object from the model's internal velocity field and attention maps, without ground-truth segmentation labels?
2. **Injection:** Can we transplant the extracted object's latent representation into a different generative trajectory while preserving the ODE solver's stability?
3. **Quality:** Does the resulting composite image maintain visual coherence at the object boundaries and semantic consistency with both the target object and the ambient scene?

### 1.3 Contributions

This thesis makes the following contributions:

- **Interior Energy Collapse:** We identify and analyze a fundamental failure mode in energy-based masking of velocity fields: the kinetic energy $\|v_0\|_2$ collapses toward zero in the interior of homogeneous objects, making scalar thresholding methods produce hollow masks.
- **Spectral Matting on velocity fields:** We propose applying Fiedler vector decomposition to a cosine-affinity graph built from the initial velocity field $v_0$, achieving threshold-free binary segmentation that correctly captures object interiors.
- **Kinetic Trajectory Shaping (KTS):** We design a differential perturbation mechanism for the ODE velocity field with exponential temporal damping, enabling stable object injection without solver collapse.
- **FlowStitch framework:** We implement a complete, modular Python pipeline for zero-shot object extraction and composition on FLUX.1 (MM-DiT architecture), with six extraction methods, attention-level semantic grafting, and automated evaluation.

### 1.4 Thesis Structure

Chapter 2 provides the theoretical background on Flow Matching and the FLUX.1 architecture. Chapter 3 describes the FlowStitch pipeline architecture. Chapter 4 details the mask extraction methods and the research journey that led to them. Chapter 5 presents the Kinetic Trajectory Shaping injection mechanism. Chapter 6 reports the experimental evaluation. Chapter 7 concludes the thesis and outlines future work.

---

## Chapter 2 — Background

### 2.1 From Diffusion Models to Flow Matching

<!-- TODO: Espandi. Copri:
- Score-based diffusion (Song et al. 2020)
- DDPM (Ho et al. 2020) 
- La transizione da noise prediction a velocity prediction
- Perché Flow Matching è più efficiente (traiettorie rettilinee)
-->

Score-Based Diffusion Models learn to reverse a stochastic corruption process by estimating the score function $\nabla_x \log p_t(x)$. The Denoising Diffusion Probabilistic Model (DDPM) formulation discretizes this into a Markov chain of denoising steps, requiring hundreds of function evaluations for a single image.

Flow Matching (Lipman et al., 2023) reformulates generation as learning a deterministic velocity field that transports a simple prior distribution (Gaussian noise) to the data distribution via an ODE:

$$x_t = (1-t) \, x_0 + t \, x_1$$

where $x_0 \sim \mathcal{N}(0, I)$ is the initial noise and $x_1$ is the target image in latent space. The neural network learns to predict the temporal derivative:

$$v_t = \frac{dx_t}{dt} = x_1 - x_0$$

which is constant along ideal (rectified) trajectories. This rectification enables high-quality generation in as few as 1–4 ODE integration steps, a significant advantage over the hundreds of steps required by DDPM.

The initial velocity $v_0$, predicted at the very first timestep ($t = 0$), encodes the full directional intent of the generation. This vector field is the primary object of analysis in this thesis.

### 2.2 Rectified Flows

<!-- TODO: Espandi. Copri:
- Liu et al. 2022/2023
- Reflow procedure
- Perché le traiettorie rettilinee permettono pochi step
- Connessione con Optimal Transport
-->

### 2.3 The FLUX.1 Architecture

FLUX.1 (Black Forest Labs, 2024) is a 12-billion parameter text-to-image model that replaces the traditional U-Net backbone with a Multimodal Diffusion Transformer (MM-DiT). Text tokens (from a T5-XXL encoder) and image tokens (from a latent VAE encoder) are concatenated into a single sequence and processed jointly through transformer blocks with Rotary Positional Embeddings (RoPE).

This joint processing has a critical consequence for our work: the attention maps in FLUX.1 naturally encode spatial text-to-image alignments. Unlike U-Net architectures where cross-attention occurs in dedicated layers, the MM-DiT's joint attention means that every attention head simultaneously processes both modalities, producing richer spatial correspondences.

FLUX.1-schnell, the distilled variant used in this work, achieves high-quality generation in 4 inference steps using a Euler ODE solver.

### 2.4 Related Work

#### 2.4.1 Zero-Shot Segmentation in Generative Models

<!-- TODO: Copri:
- DiffCut (Couairon et al. 2024) — Normalized Cut su diffusion features
- Come FlowStitch si differenzia (velocity field vs. static features, MM-DiT vs. U-Net)
-->

#### 2.4.2 Tuning-Free Image Editing

<!-- TODO: Copri:
- FlowEdit (Kulikov et al. 2024)
- ConsistEdit (Yin et al. 2025)
- Differenza: editing vs. extraction+composition
-->

#### 2.4.3 Spectral Methods in Computer Vision

<!-- TODO: Copri:
- Normalized Cuts (Shi & Malik, 2000)
- Spectral Clustering applicato a vision
- Fiedler vector e graph partitioning
-->

---

## Chapter 3 — The FlowStitch Pipeline

### 3.1 System Overview

FlowStitch operates in three sequential stages:

1. **Data Capture**: A single forward pass of FLUX.1 with a target prompt captures the initial noise $x_0$, the velocity field $v_0$, and cross-attention maps at $t = 0$.
2. **Mask Compilation**: The captured data is processed offline to produce a spatial binary mask $A_{\text{target}}$ identifying the target object in the $64 \times 64$ latent grid.
3. **Latent Stitching**: A second generation with an ambient prompt blends the target noise into the new scene and perturbs the ODE trajectory using KTS.

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Data Capture  │────▶│ Mask Compilation  │────▶│ Latent Stitching │
│ (1 forward    │     │ (offline,         │     │ (2nd generation  │
│  pass, t=0)   │     │  6 methods)       │     │  with KTS + SG)  │
└──────────────┘     └──────────────────┘     └─────────────────┘
```

### 3.2 Data Capture: Hooking the MM-DiT

The data capture stage intercepts FLUX.1's internal state at the first inference step ($t = 0$) using a custom context manager, `FluxDataCapturer`. This mechanism:

1. Registers a post-forward hook on the transformer to capture the predicted velocity $v_0$ and the initial noise $x_0$.
2. Temporarily replaces each attention processor with a wrapper (`AttnProcessorWrapper`) that materializes and stores the attention weight matrices $\text{Softmax}(QK^T / \sqrt{d})$ for the image→text slice of the joint attention.
3. Restores the original processors after capture, leaving the model unmodified.

All captured tensors are serialized to disk using the SafeTensors format for security (avoiding Python pickle deserialization vulnerabilities).

### 3.3 Mask Compilation

<!-- TODO: Breve overview dei 6 metodi (tabella dal report), rimanda a Chapter 4 per i dettagli -->

### 3.4 Latent Stitching

<!-- TODO: Breve overview del blending + KTS + Semantic Grafting, rimanda a Chapter 5 per i dettagli -->

---

## Chapter 4 — Mask Extraction: Methods and Failures

This chapter traces the chronological research path through six extraction methods. Each method was motivated by the failure of its predecessor, making the progression a narrative of incremental understanding of the velocity field's structure.

### 4.1 Attention-Based Extraction (Baseline)

The simplest approach aggregates the cross-attention weights for the tokens corresponding to the target word. A sliding-window tokenizer utility handles sub-word fragmentation (BPE artifacts in T5): for each candidate span $[i, j)$, the decoded text is checked for containment of the target word, and the minimal span is selected.

The resulting soft attention map can be binarized via Otsu's method. These methods provided a reasonable starting point for simple, single-object scenes, but they lack geometric precision and are sensitive to token overflow in multi-word prompts.

### 4.2 Energy Gating and the Interior Energy Collapse

#### 4.2.1 The Chebyshev Gating Approach

The natural next step was to leverage the velocity field magnitude directly. We compute the per-position kinetic energy $E(z) = \|v_0(z)\|_2$ and apply statistical thresholding:

$$\tau = \mu_E + k \cdot \sigma_E$$

where $\mu_E$ and $\sigma_E$ are the mean and standard deviation of the energy distribution across the latent grid, and $k$ controls the selectivity (default $k = 2.0$, motivated by the Chebyshev inequality).

#### 4.2.2 The Interior Energy Collapse

Empirical analysis revealed a fundamental failure: the kinetic energy $\|v_0(z)\|_2$ is spatially concentrated at object boundaries and collapses toward zero in the interior of homogeneous regions. This means energy-based thresholds act as high-pass filters: they correctly identify object contours but systematically exclude the interior, producing hollow, annular masks.

<!-- TODO: Inserisci figura dai notebook (energy heatmap che mostra il collasso) -->

We refer to this failure mode as the **Interior Energy Collapse**. It motivated the shift toward methods that consider velocity *direction* rather than magnitude.

### 4.3 Spectral Matting (Fiedler Vector Decomposition)

To overcome the Interior Energy Collapse, we project the velocity field $v_0$ into a spatial affinity graph. Each of the $N$ latent positions becomes a node. Edge weights are defined by the clamped cosine similarity of the velocity vectors:

$$W_{ij} = \max\left(0, \; \frac{v_{0,i} \cdot v_{0,j}}{\|v_{0,i}\|_2 \, \|v_{0,j}\|_2}\right)$$

We construct the unnormalized graph Laplacian $L = D - W$ (where $D_{ii} = \sum_j W_{ij}$ is the degree matrix) and compute its spectral decomposition via `torch.linalg.eigh`. The eigenvector corresponding to the second smallest eigenvalue — the **Fiedler vector** — partitions the graph along the boundary of maximum directional divergence. A simple zero-crossing binarization ($\phi_2(i) > 0$) yields the spatial mask.

The key insight is that interior pixels share coherent velocity directions despite their low magnitude. While their energy is negligible, their directional agreement is strong — they all "point" toward the same region of the target image. The Fiedler partition captures this directional coherence, correctly grouping interior pixels with boundary pixels.

#### 4.3.1 Decimation for Numerical Stability

On the full $64 \times 64$ grid, the $4096 \times 4096$ Laplacian matrix has a near-zero spectral gap ($\Delta\lambda = \lambda_2 - \lambda_1 \approx 10^{-6}$), causing eigensolvers to stagnate or return numerically unstable results. We address this by applying bilinear decimation to $32 \times 32$ before computing the Laplacian, which increases the spectral gap to approximately $10^{-2}$ and provides a $\sim 64\times$ speedup. The resulting mask is upsampled back to native resolution via nearest-neighbor interpolation.

### 4.4 Hybrid Method: Fiedler + Semantic Compass

The Fiedler vector partitions the graph into exactly two clusters, but the labeling is arbitrary: the object may end up labeled as either 0 or 1. The `hybrid` method resolves this ambiguity by using the Otsu-thresholded attention map as a **semantic compass**. It computes the overlap between each Fiedler partition and the attention core:

```
overlap_0 = sum(spectral_mask * semantic_core)
overlap_1 = sum((1 - spectral_mask) * semantic_core)
if overlap_1 > overlap_0:
    spectral_mask = 1 - spectral_mask
```

This is the method currently deployed in the experimental pipeline.

### 4.5 Topological Data Analysis (Persistent Homology)

<!-- TODO: Espandi dal report. Copri:
- Motivazione: alternativa theory-grounded al Fiedler
- Costruzione della point cloud nel cosine-distance space
- Calcolo H0 con Ripser
- Single-linkage clustering
- Perché non è il default (dipendenze extra, performance simile all'hybrid)
-->

### 4.6 Comparative Discussion

<!-- TODO: 
- Tabella comparativa dei 6 metodi (DICE, IoU dal notebook 09)
- Discussione su quando ciascun metodo funziona e quando fallisce
- Giustificazione della scelta dell'hybrid come default
-->

---

## Chapter 5 — Kinetic Trajectory Shaping

### 5.1 The Injection Problem

<!-- TODO: Spiega perché il naive overwrite del latent (hard masking) causa:
- Discontinuità nel campo di velocità
- Violazione della condizione di Lipschitz
- Collapse del solver ODE
Usa i risultati dal notebook 03
-->

### 5.2 ODE Velocity Perturbation with Temporal Damping

At each ODE integration step, the ambient velocity field is perturbed toward the captured target velocity:

$$v_{\text{stitch}} = v_{\text{ambient}} + \lambda \cdot D(t) \cdot \left[ A_{\text{phys}} \odot (v_{\text{target}} - v_{\text{ambient}}) \right]$$

where:
- $\lambda$ is the blending strength hyperparameter (default 1.0),
- $D(t) = \exp\left(-\gamma \cdot \max(0, \, t_{\text{norm}} - t_{\text{cutoff}})\right)$ is the temporal damping factor ($\gamma = 5.0$, $t_{\text{cutoff}} = 0.8$).

The damping provides a **soft-landing**: the perturbation is at full strength for $t_{\text{norm}} < 0.8$ and decays exponentially in the final 20% of integration, allowing the model to autonomously harmonize lighting and boundary coherence.

### 5.3 Semantic Grafting

In addition to the velocity perturbation, the `dual` stitching mode injects a `SemanticGraftingProcessor` into every single-stream transformer block. This custom attention processor blends the attention output with the pre-attention hidden states, weighted by the mask:

$$h_{\text{out}} = h_{\text{attn}} \cdot (1 - s \cdot A_{\text{target}}) + h_{\text{input}} \cdot (s \cdot A_{\text{target}})$$

where $s$ is the injection strength (default 0.85). While KTS operates on the velocity field between integration steps, Semantic Grafting operates inside the transformer blocks during each forward pass, directly protecting the target's feature representation.

### 5.4 Trajectory EMA Smoothing

An optional Exponential Moving Average filter suppresses high-frequency oscillations in the perturbed velocity:

$$\bar{V}_t = \alpha \, V_t + (1 - \alpha) \, \bar{V}_{t-1}$$

where $\alpha = 0.3$ by default.

---

## Chapter 6 — Experimental Evaluation

### 6.1 Experimental Setup

<!-- TODO: Copri:
- Modello: FLUX.1-schnell (4 step, Euler solver)
- Hardware: HPC Giano, NVIDIA L40
- Dataset: 10 prompt pairs (cubes, spheres, pyramids, lakes, mountains)
- Metriche: DICE, mIoU, CLIPScore
-->

### 6.2 Quantitative Results

#### 6.2.1 Mask Quality

<!-- TODO: Inserisci tabella risultati dal notebook 09:
- Hybrid: DICE 0.8942, IoU 0.8124
- Spectral: DICE 0.8715, IoU 0.7781
- Chebyshev: DICE 0.6120, IoU 0.4482
- Otsu: DICE 0.7431, IoU 0.5942
-->

#### 6.2.2 Generation Quality

<!-- TODO: CLIPScore results dal cluster -->

#### 6.2.3 Ablation Studies

<!-- TODO: Ablation su lambda, gamma, t_cutoff — da fare sul cluster -->

### 6.3 Qualitative Results

<!-- TODO: Inserisci immagini generate (prima/dopo stitching, confronto maschere) -->

### 6.4 Failure Cases and Limitations

<!-- TODO: Copri:
- Oggetti con texture eterogenee che frammentano il Fiedler
- Scene multi-oggetto con semantica sovrapposta
- Sensibilità dello spectral gap alla complessità della scena
-->

---

## Chapter 7 — Conclusions and Future Work

### 7.1 Summary

<!-- TODO: Scrivi per ultimo -->

### 7.2 Limitations

<!-- TODO: Riassumi da §6.4 -->

### 7.3 Future Directions

- Multi-object decomposition via k-way spectral clustering.
- Continuous masks with variance-preserving square-root blending.
- Multi-step data capture beyond $t = 0$.
- Alternative Laplacian formulations (symmetric normalized).
- Adaptive threshold tuning for KTS hyperparameters.

---

## Bibliography

<!-- TODO: Inserisci con BibTeX. Reference chiave:
- Lipman et al. 2023 - Flow Matching for Generative Modeling
- Liu et al. 2022 - Flow Straight and Fast (Rectified Flow)
- Ho et al. 2020 - DDPM
- Song et al. 2020 - Score-Based Generative Modeling
- Couairon et al. 2024 - DiffCut
- Shi & Malik 2000 - Normalized Cuts
- Black Forest Labs 2024 - FLUX.1
-->

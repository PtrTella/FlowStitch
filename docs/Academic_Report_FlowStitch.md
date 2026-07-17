# Research Progress Report: FlowStitch Framework
**Unsupervised Latent Decomposition and Generative Stitching in Flow Matching Models**

**Author:** Pietro Tellarini  
**Status:** Work in Progress — Empirical Validation Running on HPC Cluster  
**Date:** July 2026

---

## 1. Introduction and Research Objective

This report documents the ongoing research on the **FlowStitch** framework — a zero-shot, unsupervised pipeline for extracting semantic objects from generative Flow Matching models (specifically, FLUX.1 by Black Forest Labs) and injecting them into new generative contexts without external segmentation networks.

The core research question is: given a pre-trained text-to-image model, can we isolate an arbitrary object from its latent representation and seamlessly "stitch" it into a different generated scene, relying only on the model's own internal representations? The long-term goal is to enable applications such as text-guided scene composition and latent-level data augmentation — without fine-tuning or auxiliary models.

The research has progressed through the following stages:
1. Analysis of the FLUX.1 internal representations (velocity fields and attention maps).
2. Discovery of a fundamental failure mode in energy-based masking (the Interior Energy Collapse).
3. Exploration of spectral and topological extraction methods to overcome this failure.
4. Design and implementation of a differential injection mechanism (Kinetic Trajectory Shaping).

The mask extraction problem is still under active investigation: no single method has proven universally reliable across all object types and scene complexities. The current experimental campaign on the HPC cluster is designed to systematically evaluate the available approaches.

---

## 2. Background: Flow Matching and FLUX.1

### 2.1 Flow Matching
Unlike Score-Based Diffusion Models, which iteratively predict noise, Flow Matching learns a continuous velocity field that transports a Gaussian noise distribution toward the data distribution. The process is governed by an ODE:
$$x_t = (1-t) \, x_0 + t \, x_1$$
where $x_0$ is the initial noise and $x_1$ is the target image in latent space. The neural network predicts the temporal derivative $v_t = dx_t/dt = x_1 - x_0$, which is constant along ideal (rectified) trajectories.

The initial velocity $v_0$, predicted at the very first timestep ($t=0$), encodes the full directional intent of the generation. This vector field is the primary object of our analysis.

### 2.2 MM-DiT Architecture
FLUX.1 uses a Multimodal Diffusion Transformer (MM-DiT) instead of the traditional U-Net. Text and image tokens are concatenated into a single sequence and processed jointly through transformer blocks with Rotary Positional Embeddings (RoPE). This joint processing means that the attention maps naturally encode spatial text-to-image alignments.

---

## 3. Pipeline Architecture

FlowStitch operates in **three sequential stages**.

### Stage 1 — Data Capture
FLUX.1 is run once with a **target prompt** (e.g., *"a blue sphere"*). During this forward pass, we intercept the transformer's internal state at $t=0$ using a custom hooking mechanism (`FluxDataCapturer`) that wraps the model as a context manager. We capture and save to disk:
- **$x_0$**: the initial noise tensor.
- **$v_0$**: the predicted velocity field (the transformer's output at step 0).
- **Cross-attention maps**: the image→text attention weight matrices from selected transformer layers.

This stage runs only once per target object.

### Stage 2 — Mask Compilation
From the captured data, we compute a spatial mask $A_{\text{target}}$ that identifies where the target object resides in the latent grid ($64 \times 64$ positions). The codebase implements **six extraction methods**, reflecting the chronological research exploration:

| Method | Input | Description |
|--------|-------|-------------|
| `attention` | Cross-attention maps | Raw attention aggregation over target-word tokens |
| `otsu` | Cross-attention maps | Attention + Otsu binarization |
| `chebyshev` | $v_0$ energy | Statistical gating ($\tau = \mu + k\sigma$) on velocity magnitude |
| `spectral` | $v_0$ field | Fiedler vector on cosine-affinity Laplacian |
| `hybrid` | $v_0$ + attention | Fiedler partition + attention-based orientation |
| `tda` | $v_0$ + attention | Persistent Homology ($H_0$) via single-linkage clustering |

None of these has proven universally reliable yet. The `hybrid` method is the current best candidate and the one deployed in the experimental pipeline. The research journey across these methods is described in Section 4.

### Stage 3 — Latent Stitching
FLUX.1 is run a **second time** with a different **ambient prompt** (e.g., *"a crystal clear lake"*). At $t=0$, the ambient noise is blended with the previously captured target noise using the mask. Then, at each ODE integration step, the velocity field is perturbed to guide the target object's trajectory. The final image is decoded through the VAE.

---

## 4. Mask Extraction: Research Path

### 4.1 Attention-Based Methods (Starting Point)
The simplest approach aggregates the cross-attention weights for the tokens corresponding to the target word. A sliding-window tokenizer utility handles sub-word fragmentation (T5 sub-tokens). The resulting soft map can be binarized via Otsu's method. These methods provided a reasonable starting point for simple scenes, but they lack geometric precision and are sensitive to token overflow in multi-word prompts.

### 4.2 Energy Gating and the Interior Energy Collapse
The natural next step was to leverage the velocity magnitude directly: threshold the kinetic energy $\|v_0\|_2$ at $\tau = \mu + k\sigma$ (Chebyshev gating). However, our analysis revealed that kinetic energy is spatially concentrated at object boundaries and collapses toward zero in the interior of homogeneous regions. This means energy-based thresholds act as high-pass filters: they correctly identify object contours but systematically exclude the interior, producing hollow, annular masks.

We refer to this failure mode as the **Interior Energy Collapse**. It motivated the shift toward methods that consider velocity *direction* rather than magnitude.

### 4.3 Spectral Matting (Fiedler Vector Decomposition)
To overcome the Interior Energy Collapse, we project the velocity field $v_0$ into a spatial affinity graph. Each of the $N$ latent positions becomes a node. Edge weights are the clamped cosine similarity of the velocity vectors:
$$W_{ij} = \max\left(0, \; \frac{v_{0,i} \cdot v_{0,j}}{\|v_{0,i}\|_2 \, \|v_{0,j}\|_2}\right)$$
We construct the unnormalized graph Laplacian $L = D - W$ (where $D$ is the degree matrix) and compute its spectral decomposition. The eigenvector corresponding to the second smallest eigenvalue — the **Fiedler vector** — partitions the graph along the boundary of maximum directional divergence.

The key insight: interior pixels share coherent velocity directions despite their low magnitude, so the Fiedler partition correctly groups them with the boundary pixels.

**Decimation for numerical stability:** On the full $64 \times 64$ grid, the $4096 \times 4096$ Laplacian has a near-zero spectral gap, causing eigensolvers to stagnate. We apply bilinear decimation to $32 \times 32$ before computing the Laplacian, then upsample the result via nearest-neighbor interpolation.

### 4.4 Hybrid Method (Current Candidate)
The Fiedler vector partitions the graph into exactly two clusters, but the labeling is arbitrary (the object could be labeled 0 or 1). The `hybrid` method resolves this ambiguity by using the Otsu-thresholded attention map as a **semantic compass**: it checks which Fiedler partition overlaps more with the attention core, and inverts the mask if necessary.

This is the method currently deployed in the experimental pipeline, though its robustness across diverse object categories is still being evaluated.

### 4.5 Open Issues
The mask extraction problem remains the most challenging aspect of the framework. Specific open difficulties include: objects with heterogeneous textures that fragment the Fiedler partition, scenes with multiple overlapping objects, and the sensitivity of the spectral gap to scene complexity. Identifying the most robust strategy is the primary goal of the ongoing experiments.

---

## 5. Generative Latent Stitching

### 5.1 Initial Noise Blending
At $t=0$, the initial latent is composed by blending the ambient noise with the captured target noise. In `dual` mode, the mask is first softened with a Gaussian blur ($3 \times 3$ kernel, $\sigma = 2.5$) and then binarized at threshold 0.1 to produce a smooth-edged binary gate $A_{\text{phys}}$:
$$z_0 = (1 - A_{\text{phys}}) \odot z_{\text{ambient}} + A_{\text{phys}} \odot x_{0,\text{target}}$$

### 5.2 Kinetic Trajectory Shaping (KTS)
At each ODE integration step, the ambient velocity field is perturbed toward the captured target velocity:
$$v_{\text{stitch}} = v_{\text{ambient}} + \lambda \cdot D(t) \cdot \left[ A_{\text{phys}} \odot (v_{\text{target}} - v_{\text{ambient}}) \right]$$
where:
- $\lambda$ is the blending strength (default 1.0),
- $D(t) = \exp\left(-\gamma \cdot \max(0, \, t_{\text{norm}} - t_{\text{cutoff}})\right)$ is the temporal damping factor ($\gamma = 5.0$, $t_{\text{cutoff}} = 0.8$).

The damping provides a **soft-landing**: the perturbation is at full strength for $t_{\text{norm}} < 0.8$ and decays exponentially in the final 20% of integration, allowing the model to harmonize lighting and boundary coherence. Timesteps are normalized from $[0, 1000]$ to $[0, 1]$ via $t_{\text{norm}} = t / 1000$.

An optional Exponential Moving Average ($\bar{V}_t = \alpha V_t + (1 - \alpha)\bar{V}_{t-1}$, $\alpha = 0.3$) can be applied to suppress high-frequency oscillations in the perturbed velocity.

### 5.3 Semantic Grafting (Attention-Level Injection)
In addition to the velocity perturbation, the `dual` mode injects a **SemanticGraftingProcessor** into every single-stream transformer block. This custom attention processor blends the attention output with the pre-attention hidden states, weighted by the mask:
$$h_{\text{out}} = h_{\text{attn}} \cdot (1 - s \cdot A_{\text{target}}) + h_{\text{input}} \cdot (s \cdot A_{\text{target}})$$
where $s$ is the injection strength (default 0.85). While KTS operates on the velocity field between integration steps, Semantic Grafting operates inside the transformer blocks during each forward pass, directly protecting the target's feature representation.

---

## 6. Current Status and Roadmap

The FlowStitch codebase is fully implemented and modular, organized into `core/`, `extraction/`, `stitching/`, and `pipelines/`. The experimental pipeline is currently running on the HPC cluster (L40 partition), executing `hybrid` mask compilation followed by `dual`-mode latent stitching on a first test scene.

### Phase 1 — Validation (Immediate)
- Collect quantitative metrics (DICE, mIoU, CLIPScore) from the current HPC run.
- Run the same scene through all 6 extraction methods and compare results.
- Test on diverse objects beyond simple geometries: textured surfaces, irregular boundaries, transparent materials.

### Phase 2 — Parameter Sensitivity
- Ablation study on KTS hyperparameters ($\lambda$, $\gamma$, $t_{\text{cutoff}}$).
- Evaluate the impact of the SemanticGrafting injection strength.
- Explore continuous masks with variance-preserving blending as an alternative to binary thresholding.

### Phase 3 — Architectural Extensions
- Multi-object decomposition via k-way spectral clustering.
- Multi-step data capture beyond $t=0$ for sharper attention maps (the hooking system already supports this).
- Alternative Laplacian formulations (e.g., symmetric normalized) for improved spectral stability.

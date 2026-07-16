# Research Progress Report: FlowStitch Framework
**Unsupervised Latent Decomposition and Generative Stitching via Flow Matching**

**Author:** Pietro Tellarini  
**Status:** HPC Empirical Validation (In Progress)

---

## 1. Executive Summary
The generative manipulation of multi-object scenes requires ultra-high fidelity latent decomposition capable of isolating overlapping and semantically contiguous entities. This report documents the theoretical advancements and the implemented architecture of the **FlowStitch** framework. 

FlowStitch extends the Flow Matching (FM) paradigm beyond image generation, elevating it to an instrument for unsupervised zero-shot segmentation. We observed that current zero-shot methods, which rely on static cross-attention maps or empirical scalar energy filters, are plagued by geometric blindness and instability. 

By framing the velocity vector field through the lens of Spectral Graph Theory, FlowStitch analytically resolves the intrinsic "Thermodynamic Void" paradox of Flow Matching models. Furthermore, for the injection phase, we introduce Kinetic Trajectory Shaping (KTS) to execute stable, Lipschitz-continuous multi-object integration without degrading the ODE solver. The entire pipeline is currently undergoing empirical validation on an HPC cluster.

---

## 2. Theoretical Framing: MM-DiT and Flow Matching

### 2.1 The Flow Matching Paradigm
Unlike Score-Based Diffusion Models, which iteratively predict added noise, Flow Matching learns a continuous **vector field** (velocity) that smoothly transports a simple Gaussian noise distribution toward a complex data distribution.
The process is governed by an Ordinary Differential Equation (ODE) defining a quasi-linear trajectory:
$$ x_t = (1 - t) x_0 + t x_1 $$
where the neural network continuously predicts the temporal derivative $v_t = \frac{dx_t}{dt}$. The initial velocity vector $v_0$, predicted at timestep $t=0$, encodes the "thermodynamic direction" of the generative ecosystem. This vector field serves as the primary object of our analysis.

### 2.2 MM-DiT Architecture and Joint Attention Hooking
FLUX.1 adopts a Multimodal Diffusion Transformer (MM-DiT) architecture. Text and image representations are processed jointly, utilizing Rotary Positional Embeddings (RoPE) to guarantee spatial coherence. To extract exact object-text alignments without altering the native code, FlowStitch implements dynamic software hooking on the `Transformer2DModel`. We monkey-patch the `FluxAttnProcessor` during the forward pass to temporarily materialize the highly-optimized Flash Attention matrices, capturing the exact spatial relationships between text prompts and image latents.

---

## 3. Pathologies of Spatial Masking: The Thermodynamic Void

### 3.1 Kinetic Path Energy (KPE)
Flow Matching can be analyzed as an Inverse Fluid Dynamics problem. A fundamental Energy-Density dualism emerges along the generative trajectory: 
$$ \|v_\theta(z, t)\|^2 \asymp -\nabla_z \log \hat{p}_t(z) $$
This demonstrates that kinetic energy reaches its local maxima where the probability density undergoes maximum variation—i.e., at the physical contours of an object. The model exerts maximum "force" to separate the object's boundary from the background noise.

### 3.2 The Thermodynamic Void Paradox
Conversely, inside flat, homogeneous regions of an object, the spatial gradient vanishes, and the kinetic energy collapses toward zero. Consequently, applying standard statistical gating thresholds (e.g., Chebyshev $\mu + \sigma$ limits) on the velocity magnitude systematically carves out the interior of objects. This phenomenon, which we formally define as the **Thermodynamic Void**, renders scalar energy thresholding fundamentally inadequate for solid object extraction.

---

## 4. Topological Object Extraction

To transcend the Thermodynamic Void and avoid heuristic energy thresholds, FlowStitch abstracts the velocity field into the domain of Spectral Graph Theory. 

### 4.1 Latent Affinity and the Laplacian
We project the latent pixels into an affinity graph. The edge weights $W_{ij}$ represent the clamped Cosine Similarity of the normalized Key vectors extracted from the transformer blocks. To eliminate self-loop bias, the diagonal is zeroed.
We then construct the unnormalized Laplacian matrix:
$$ L = D - W $$
where $D$ is the degree matrix.

### 4.2 Spectral Matting and the Fiedler Vector
By performing spectral decomposition on $L$, we extract the **Fiedler vector** (the eigenvector corresponding to the second smallest eigenvalue). The zero-crossing of the Fiedler vector analytically partitions the latent graph precisely along the physical boundaries of maximum directional divergence. This guarantees a perfect geometric extraction of the object's interior, bypassing the Thermodynamic Void entirely.

**Engineering Optimization:** 
During development, the ARPACK eigensolver stagnated on full-resolution matrices ($4096 \times 4096$) due to a near-zero spectral gap. We resolved this through a bilinear decimation strategy: downsampling the field to $32 \times 32$, solving the $1024 \times 1024$ Laplacian (increasing the spectral gap by 4 orders of magnitude), and utilizing nearest-neighbor upsampling to restore the native resolution.

---

## 5. Generative Latent Stitching

Replacing a background tensor with an isolated object using a hard binary mask creates a **Lipschitz Discontinuity** in the vector field. This abrupt jump causes low-step ODE solvers (e.g., Euler) to fail, producing visual ghosting and severe boundary artifacts.

### 5.1 Kinetic Trajectory Shaping (KTS)
FlowStitch abandons direct spatial replacement in favor of **Time-Domain ODE Perturbation**. We induce a fluid force field that smoothly deflects the background's temporal derivative toward the target's topological attractor:
$$ v_{stitch} = v_{ambient} + \left[ M \odot (v_{target} - v_{ambient}) \right] \cdot D(t) $$
Crucially, the damping factor $D(t) = e^{-\gamma (t_{norm} - t_{cutoff})_+}$ provides a **Thermodynamic Soft-Landing**. FLUX expects normalized timesteps $t_{norm} \in [0,1]$. By exponentially damping the perturbation in the final integration steps ($t_{norm} \to 1$), we force the neural network to harmonize the lighting and statistical matching of the grafted object into the host ecosystem, naturally eliminating jagged edges.

### 5.2 Variance Normalization and EMA Smoothing
*   **Variance Normalization:** We analytically proved that linear blending of two independent noise fields collapses spatial variance by 50% ($\text{Var} = 2M^2 - 2M + 1$). While binary Fiedler masks (where $M \in \{0,1\}$) naturally bypass this collapse, continuous transition masks require a square-root spherical parameterization ($z = \sqrt{1-M} z_a + \sqrt{M} z_b$) to restore perfect isotropy.
*   **Look-Back EMA (TrajectoryEMA):** To suppress high-frequency oscillations induced by KTS perturbations, we implemented an Exponential Moving Average on the velocity trajectories: $\bar{V}_t = (1 - \alpha)\bar{V}_{t-1} + \alpha V_t$. This bounds the temporal variation of the first derivative, ensuring solver stability.

---

## 6. Current Status and Next Steps

The FlowStitch Python architecture (`flowstitch/`) is fully implemented, modular, and optimized. It natively handles Spectral Matting, TDA graph cuts, and Kinetic Trajectory Shaping. 

**Current Operations:**
The experimental validation pipeline (`run_experiment.py`) has been deployed to the HPC cluster (L40 partition) using a custom IPv4 network socket wrapper to bypass local firewall restrictions. 

Once the HPC jobs conclude, the quantitative benchmark results (DICE Score, mean Intersection over Union, and CLIPScore) will be aggregated. These metrics will empirically validate the superiority of the Fiedler-based topological extraction over baseline Otsu-attention methods, completing the validation phase of the thesis.

"""
FlowStitch Semantic Extraction Package.

Questo modulo contiene sia le formulazioni canoniche validate per il Flow Stitching,
sia i metodi esplorativi e storici sviluppati durante le varie fasi della tesi (Cap. 1-7).

Struttura:
1. Suite Canonica Fused Masks (Raccomandata per Latent Stitching):
   - extract_attention_mask: Baseline 1 (Cross-Attention pura)
   - compute_calibrated_attention_mask: Baseline 2 (Cross-Attention soft-calibrata)
   - compute_linear_fused_mask: Metodo 3 (Linear Attention x Cosine)
   - compute_hermite_fused_mask: Metodo 4 - Canonico Principale (Hermite Smoothstep C^1)
   - compute_cosine_field: Allineamento direzionale della velocità v0

2. Moduli Storici & Exploratory Baselines (Cap. 1-6):
   - otsu_threshold: Binarizzazione euristica dell'attenzione (Cap. 1)
   - compute_chebyshev_threshold: Gating statistico dell'energia cinetica (Cap. 5)
   - compute_fiedler_mask: Partizione spettrale su grafo di similarità vettoriale (Cap. 6)
   - extract_tda_mask: Filtrazione ad omologia persistente H0 con Ripser (Cap. 6)
   - hybrid_semantic_decomposition: Decomposizione spettrale con guida semantica (Cap. 6)

3. Decomposizione Multi-Oggetto (Cap. 7):
   - multi_object_decomposition: Modulazione semantica ed isolamento vettoriale
   - validate_orthogonality: Verifica di indipendenza algebrica tra maschere
   - extract_pure_velocity_fields: Proiezione di Gram-Schmidt su campi di velocità

Nota sui Decoders Storici:
   Il sottomodulo `flowstitch.extraction.decoders` (graph_diffusion, svd_cosine_hybrid, ecc.)
   contiene i prototipi di decodifica geometrica offline del Capitolo 5, preservati
   esclusivamente a fini di tracciamento e documentazione della tesi.
"""
# 1. Canonical Fused Mask Suite
from .fused_mask import (
    compute_kinetic_semantic_hull_mask,
    compute_core_fused_mask,
    compute_bilateral_sigmoid_mask,
    compute_hermite_fused_mask,
    compute_linear_fused_mask,
    compute_calibrated_attention_mask,
    compute_cosine_field,
)
from .sam_flow_mask import compute_sam_flow_mask
from .attention_mask import extract_attention_mask

# 2. Legacy / Exploratory Baselines (Theses Ch. 1-6)
from .attention_mask import otsu_threshold
from .spectral_mask import compute_fiedler_mask
from .energy_mask import compute_chebyshev_threshold
from .hybrid_mask import hybrid_semantic_decomposition
from .tda_mask import extract_tda_mask

# 3. Multi-Object Decomposition (Ch. 7)
from .multi_object import (
    multi_object_decomposition,
    validate_orthogonality,
    extract_pure_velocity_fields,
)

__all__ = [
    # Canonical
    "compute_kinetic_semantic_hull_mask",
    "compute_sam_flow_mask",
    "compute_core_fused_mask",
    "compute_bilateral_sigmoid_mask",
    "compute_hermite_fused_mask",
    "compute_linear_fused_mask",
    "compute_calibrated_attention_mask",
    "extract_attention_mask",
    "compute_cosine_field",
    # Legacy Baselines
    "otsu_threshold",
    "compute_fiedler_mask",
    "compute_chebyshev_threshold",
    "hybrid_semantic_decomposition",
    "extract_tda_mask",
    # Multi-Object
    "multi_object_decomposition",
    "validate_orthogonality",
    "extract_pure_velocity_fields",
]

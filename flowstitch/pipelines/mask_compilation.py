"""
Mask Compilation Pipeline — compila maschere semantiche per una scena.

Supporta la suite canonica (raccomandata) e tutti i metodi di estrazione storici dei Capitoli 1-7:

Suite Canonica (Stitching Validato):
- "hermite": Transizione cubica Hermite C^1 (Default Canonico)
- "fused_linear": Fusione lineare Attention x Cosine con taglio soglia
- "calibrated_attention": Cross-attention con sottrazione del piedistallo di rumore
- "attention": Cross-attention pura normalizzata in [0, 1] (Baseline)

Metodi Storici & Esplorativi:
- "otsu": Attention + binarizzazione Otsu (Cap. 1)
- "chebyshev": Gating statistico μ+kσ sull'energia cinetica (Cap. 5)
- "spectral": Vettore di Fiedler con decimazione bilineare (Cap. 6)
- "hybrid": Fiedler + bussola semantica (Cap. 6)
- "tda": Persistent Homology H₀ con Ripser (Cap. 6)
"""
import os
import json
import torch
import logging
from transformers import T5Tokenizer

from ..core.tokenizer_utils import find_token_indices
from ..extraction.attention_mask import extract_attention_mask, otsu_threshold
from ..extraction.fused_mask import (
    compute_kinetic_semantic_hull_mask,
    compute_core_fused_mask,
    compute_bilateral_sigmoid_mask,
    compute_hermite_fused_mask,
    compute_linear_fused_mask,
    compute_calibrated_attention_mask,
    compute_cosine_field,
)
from ..extraction.sam_flow_mask import compute_sam_flow_mask
from ..extraction.spectral_mask import compute_fiedler_mask
from ..extraction.energy_mask import compute_chebyshev_threshold
from ..extraction.hybrid_mask import hybrid_semantic_decomposition
from ..extraction.tda_mask import extract_tda_mask
from ..core.serialization import load_tensors, save_tensors

logger = logging.getLogger(__name__)

CANONICAL_METHODS = [
    "kinetic_hull",
    "sam_flow",
    "core_fused",
    "core_fused_soft",
    "bilateral_sigmoid",
    "hermite",
    "fused_linear",
    "calibrated_attention",
    "attention",
]
LEGACY_METHODS = ["otsu", "chebyshev", "spectral", "hybrid", "cosine_direction", "tda"]
SUPPORTED_METHODS = CANONICAL_METHODS + LEGACY_METHODS


def compile_mask_for_scene(
    db_path: str,
    word_to_isolate: str,
    method: str = "hermite",
    manual_tokens: list = None,
    chebyshev_k: float = 1.0,
    spectral_resolution: int = 32,
    hermite_edge0: float = 0.10,
    hermite_edge1: float = 0.65,
    linear_tau_bg: float = 0.12,
    calib_pedestal_quantile: float = 0.15,
) -> torch.Tensor:
    """
    Compiles a semantic mask from cross-attention and/or velocity field data.

    Args:
        db_path: path to scene folder containing attention_maps.pt, v0_velocity.pt, metadata.json
        word_to_isolate: target word to extract
        method: one of "hermite" (default), "fused_linear", "calibrated_attention",
                "attention", "otsu", "chebyshev", "spectral", "hybrid", "tda"
        manual_tokens: optional manual token indices (skip tokenizer)
        chebyshev_k: threshold multiplier for Chebyshev method (τ = μ + k·σ)
        spectral_resolution: downsampled resolution for Fiedler computation
        hermite_edge0: lower edge for Hermite smoothstep transition
        hermite_edge1: upper edge for Hermite smoothstep transition
        linear_tau_bg: background threshold for linear fused mask
        calib_pedestal_quantile: quantile for background noise subtraction in calibrated attention

    Returns:
        A_target tensor [1, seq_len, 1]
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Method '{method}' not supported. Use one of: {SUPPORTED_METHODS}")

    category = "CANONICA" if method in CANONICAL_METHODS else "LEGACY/ESPLORATIVA"
    logger.info(f"--- Compilazione Maschera [{method}] ({category}) per '{word_to_isolate}' in {db_path} ---")

    # Load metadata and find token indices
    metadata_path = os.path.join(db_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    target_prompt = metadata["prompt"]

    tokenizer = T5Tokenizer.from_pretrained(
        "google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True
    )

    if manual_tokens is not None:
        token_indices = manual_tokens
    else:
        token_indices = find_token_indices(tokenizer, target_prompt, word_to_isolate)

    if not token_indices:
        raise ValueError(f"CRITICO: Parola '{word_to_isolate}' non trovata nel prompt '{target_prompt}'.")

    # Load attention maps (always needed)
    attn_target = load_tensors(
        os.path.join(db_path, "attention_maps.pt")
    )
    layer_keys = [k for k in attn_target.keys() if k.startswith("layer_")]
    if not layer_keys:
        raise ValueError(f"No layer keys found in attention maps. Available keys: {list(attn_target.keys())}")
    layer_keys.sort(key=lambda k: int(k.split('_')[1]))
    target_key = layer_keys[-1]  # Use deepest layer
    logger.info(f"Using attention map key: '{target_key}'")
    layer_10 = attn_target[target_key]

    # --- Dispatch by Method ---
    if method == "kinetic_hull":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_kinetic_semantic_hull_mask(
            attn_mask=raw_attn,
            v0=v0,
            plateau=True,
        )

    elif method == "sam_flow":
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_sam_flow_mask(
            attn_mask=raw_attn,
        )

    elif method == "core_fused":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_core_fused_mask(
            attn_mask=raw_attn,
            v0=v0,
            plateau=True,
        )

    elif method == "core_fused_soft":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_core_fused_mask(
            attn_mask=raw_attn,
            v0=v0,
            plateau=False,
        )

    elif method == "bilateral_sigmoid":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_bilateral_sigmoid_mask(
            attn_mask=raw_attn,
            v0=v0,
        )

    elif method == "hermite":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_hermite_fused_mask(
            attn_mask=raw_attn,
            v0=v0,
            edge0=hermite_edge0,
            edge1=hermite_edge1,
        )

    elif method == "fused_linear":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_linear_fused_mask(
            attn_mask=raw_attn,
            v0=v0,
            tau_bg=linear_tau_bg,
        )

    elif method == "calibrated_attention":
        A_target = compute_calibrated_attention_mask(
            layer_attn=layer_10,
            token_indices=token_indices,
            pedestal_quantile=calib_pedestal_quantile,
        )

    elif method == "attention":
        A_target = extract_attention_mask(layer_10, token_indices, normalize=True)

    elif method == "otsu":
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = otsu_threshold(raw_attn).to(torch.float32)

    elif method == "chebyshev":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        energy = torch.norm(v0, p=2, dim=-1, keepdim=True)
        A_target = compute_chebyshev_threshold(energy, k=chebyshev_k).to(torch.float32)

    elif method == "spectral":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        A_target = compute_fiedler_mask(v0, target_resolution=spectral_resolution).to(torch.float32)

    elif method == "hybrid":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = hybrid_semantic_decomposition(
            v0, raw_attn, target_resolution=spectral_resolution
        ).to(torch.float32)

    elif method == "cosine_direction":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = compute_cosine_field(v0=v0, attn_mask=raw_attn, normalize_positive=True).to(torch.float32)

    elif method == "tda":
        v0 = load_tensors(os.path.join(db_path, "v0_velocity.pt"))
        raw_attn = extract_attention_mask(layer_10, token_indices, normalize=True)
        A_target = extract_tda_mask(v0, raw_attn).to(torch.float32)

    # Save
    out_file = os.path.join(db_path, "A_target.pt")
    save_tensors(A_target, out_file)
    logger.info(f"[SUCCESS] Maschera [{method}] compilata e salvata in: {out_file} (range: [{A_target.min():.3f}, {A_target.max():.3f}])")
    return A_target

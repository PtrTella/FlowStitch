"""
Mask Compilation Pipeline — compila maschere semantiche per una scena.

Supporta tutti i metodi di estrazione sviluppati nei Capitoli 1-7:
- "attention": Cross-attention basica (Cap. 1)
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
from ..extraction.spectral_mask import compute_fiedler_mask
from ..extraction.energy_mask import compute_chebyshev_threshold
from ..extraction.hybrid_mask import hybrid_semantic_decomposition
from ..extraction.tda_mask import extract_tda_mask
from ..core.serialization import load_tensors, save_tensors

logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ["attention", "otsu", "chebyshev", "spectral", "hybrid", "tda"]


def compile_mask_for_scene(
    db_path: str,
    word_to_isolate: str,
    method: str = "attention",
    manual_tokens: list = None,
    chebyshev_k: float = 1.0,
    spectral_resolution: int = 32,
):
    """
    Compiles a semantic mask from cross-attention and/or velocity field data.

    Args:
        db_path: path to scene folder containing attention_maps.pt, v0_velocity.pt, metadata.json
        word_to_isolate: target word to extract
        method: one of "attention", "otsu", "chebyshev", "spectral", "hybrid", "tda"
        manual_tokens: optional manual token indices (skip tokenizer)
        chebyshev_k: threshold multiplier for Chebyshev method (τ = μ + k·σ)
        spectral_resolution: downsampled resolution for Fiedler computation

    Returns:
        A_target tensor [1, seq_len, 1]
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Method '{method}' not supported. Use one of: {SUPPORTED_METHODS}")

    logger.info(f"--- Compilazione Maschera [{method}] per '{word_to_isolate}' in {db_path} ---")

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
        os.path.join(db_path, "attention_maps.pt"),
        map_location="cpu",
    )
    layer_10 = attn_target["layer_10"]

    # Basic attention mask (used by all methods as base or directly)
    A_target = extract_attention_mask(layer_10, token_indices, normalize=True)

    if method == "attention":
        pass  # A_target already computed

    elif method == "otsu":
        A_target = otsu_threshold(A_target).to(torch.float32)

    elif method == "chebyshev":
        v0 = load_tensors(
            os.path.join(db_path, "v0_velocity.pt"), map_location="cpu"
        )
        energy = torch.norm(v0, p=2, dim=-1, keepdim=True)
        A_target = compute_chebyshev_threshold(energy, k=chebyshev_k).to(torch.float32)

    elif method == "spectral":
        v0 = load_tensors(
            os.path.join(db_path, "v0_velocity.pt"), map_location="cpu"
        )
        A_target = compute_fiedler_mask(v0, target_resolution=spectral_resolution).to(torch.float32)

    elif method == "hybrid":
        v0 = load_tensors(
            os.path.join(db_path, "v0_velocity.pt"), map_location="cpu"
        )
        A_target = hybrid_semantic_decomposition(
            v0, A_target, target_resolution=spectral_resolution
        ).to(torch.float32)

    elif method == "tda":
        v0 = load_tensors(
            os.path.join(db_path, "v0_velocity.pt"), map_location="cpu"
        )
        A_target = extract_tda_mask(v0, A_target).to(torch.float32)

    # Save
    out_file = os.path.join(db_path, "A_target.pt")
    save_tensors(A_target, out_file)
    logger.info(f"[SUCCESS] Maschera [{method}] compilata e salvata in: {out_file}")
    return A_target

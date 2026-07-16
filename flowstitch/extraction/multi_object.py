"""
Multi-Object Semantic Decomposition — DNA Vettoriale (Capitolo 7).

Scompone una scena multi-oggetto isolando selettivamente un singolo oggetto
attraverso un processo a due stadi:

1. Modulazione semantica dell'energia:
   E_target = ||v₀||₂ · A_token

2. Gating non-lineare adattivo (sigmoid):
   α = σ(k · (E_target - (μ_E + σ_E)))

Dove A_token è la mappa di attenzione per il token target,
e la soglia è adattiva (μ + σ dell'energia pre-filtrata).

Validato su scene con cubo + sfera con verifica di ortogonalità
tra maschere di oggetti diversi.
"""
from __future__ import annotations
import torch
import logging

logger = logging.getLogger(__name__)


def multi_object_decomposition(
    v0: torch.Tensor,
    attn_maps: dict[str, torch.Tensor],
    k_steepness: float = 10.0,
) -> dict[str, torch.Tensor]:
    """
    Decompone una scena multi-oggetto in maschere per singolo oggetto.

    Args:
        v0: [1, seq_len, dim] campo di velocità
        attn_maps: dict {object_name: attn_mask [1, seq_len, 1]}
            mappe di attenzione per ciascun oggetto
        k_steepness: ripidità della sigmoid adattiva

    Returns:
        dict {object_name: alpha_mask [1, seq_len, 1]} maschere continue in [0, 1]
    """
    # Energia cinetica globale
    energy = torch.norm(v0, p=2, dim=-1, keepdim=True)  # [1, seq_len, 1]

    masks = {}
    for name, A_token in attn_maps.items():
        # Stadio 1: Modulazione semantica
        E_target = energy * A_token  # [1, seq_len, 1]

        # Stadio 2: Sigmoid adattiva centrata su μ + σ
        mu_E = E_target.mean()
        sigma_E = E_target.std()
        threshold = mu_E + sigma_E

        alpha = torch.sigmoid(k_steepness * (E_target - threshold))
        masks[name] = alpha

        logger.info(
            "Object '%s': μ=%.4f, σ=%.4f, τ=%.4f, active_pixels=%d",
            name, mu_E.item(), sigma_E.item(), threshold.item(),
            (alpha > 0.5).sum().item(),
        )

    return masks


def validate_orthogonality(
    masks: dict[str, torch.Tensor],
    threshold: float = 0.05,
) -> dict[str, float]:
    """
    Valida l'ortogonalità tra maschere di oggetti diversi.

    Calcola l'overlap (prodotto scalare normalizzato) tra ogni coppia.
    Overlap < threshold indica buona separazione.

    Returns:
        dict {pair_name: overlap_value}
    """
    names = list(masks.keys())
    results = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = masks[names[i]].float().flatten()
            b = masks[names[j]].float().flatten()

            overlap = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
            pair_name = f"{names[i]}_vs_{names[j]}"
            results[pair_name] = overlap.item()

            status = "✅ ortogonali" if overlap.item() < threshold else "⚠️ sovrapposizione"
            logger.info(
                "Overlap %s: %.4f (%s)", pair_name, overlap.item(), status,
            )

    return results


def extract_pure_velocity_fields(
    v0: torch.Tensor,
    masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Estrae i campi di velocità puri (DNA Vettoriale) per ciascun oggetto.

    v0_object = v0 ⊙ α_object

    Questo permette di "trapiantare" un oggetto da una scena all'altra
    preservando la sua identità cinetica.

    Args:
        v0: [1, seq_len, dim] campo di velocità
        masks: dict {name: alpha [1, seq_len, 1]}

    Returns:
        dict {name: v0_pure [1, seq_len, dim]}
    """
    return {name: v0 * alpha for name, alpha in masks.items()}

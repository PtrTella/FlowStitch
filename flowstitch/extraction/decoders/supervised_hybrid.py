"""
SupervisedHybridDecoder — Primo decoder ibrido (Capitolo 5, §5.2).

Combina segnale semantico (cross-attention per token) con sensore topologico
(cosine similarity locale del campo v₀). Fusione via sigmoid:
    M = σ(k · (S_sem · S_phys - λ))

Limitazioni scoperte: dipende fortemente dal parametro λ e non risolve
completamente il Vuoto Termodinamico nelle regioni piatte.
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SupervisedHybridDecoder:
    """Cosine similarity topologica + sigmoid gating."""

    def __init__(self, k: float = 12.0, lambda_val: float = 0.45):
        self.k = k
        self.lambda_val = lambda_val

    def _extract_semantic_signal(
        self,
        attn_tensor: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """Isola il potenziale semantico usando gli indici dei token."""
        H, W = spatial_shape
        # attn_tensor: [Heads, H*W, Text_Seq]
        attn_maps_list = [attn_tensor[:, :, idx].mean(dim=0) for idx in token_indices]
        attn_map = torch.clamp(torch.stack(attn_maps_list).sum(dim=0), min=0.0, max=1.0)

        attn_min, attn_max = attn_map.min(), attn_map.max()
        A_target = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
        return A_target.view(H, W)

    def _compute_topological_sensor(self, v_t: torch.Tensor) -> torch.Tensor:
        """Cosine similarity spaziale del flusso vettoriale (4-connesso)."""
        v_norm = F.normalize(v_t, p=2, dim=0)
        shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        cos_sims = [
            (v_norm * torch.roll(v_norm, shifts=(dy, dx), dims=(1, 2))).sum(dim=0)
            for dy, dx in shifts
        ]
        S_phys = torch.stack(cos_sims, dim=0).mean(dim=0)
        return (S_phys + 1.0) / 2.0

    def __call__(
        self,
        attn_tensor: torch.Tensor,
        v_t: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple = (64, 64),
    ) -> tuple:
        """
        Args:
            attn_tensor: [Heads, H*W, Text_Seq]
            v_t: [C, H, W] velocity field
            token_indices: list of token indices for target word
            spatial_shape: (H, W) of the spatial grid

        Returns:
            (M_hybrid, S_sem, S_phys)
        """
        S_sem = self._extract_semantic_signal(attn_tensor, token_indices, spatial_shape)
        S_phys = self._compute_topological_sensor(v_t)
        M_hybrid = torch.sigmoid(self.k * (S_sem * S_phys - self.lambda_val))
        return M_hybrid, S_sem, S_phys

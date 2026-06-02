"""
SvdCosineHybridDecoder — Decoder con SVD per consenso multi-testa (Capitolo 5, §5.3).

Invece di mediare naïvemente le 24 teste di attenzione, usa SVD per estrarre
la componente a massima varianza. Questo purifica il segnale semantico
eliminando le teste rumorose. Il sensore fisico usa cosine similarity globale
(non locale come nel SupervisedHybrid).

Miglioramento rispetto a §5.2: SVD riempie parzialmente il Vuoto Termodinamico
perché il primo autovettore cattura il consenso globale.
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SvdCosineHybridDecoder:
    """SVD sulle teste di attenzione + cosine topology globale."""

    def __init__(self, k: float = 12.0, lambda_val: float = 0.5):
        self.k = k
        self.lambda_val = lambda_val

    def _compute_svd_semantics(
        self,
        attn_tensor: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """Estrae l'autovettore primario (SVD) lungo la profondità delle Heads."""
        H, W = spatial_shape

        # Somma token (se parola frammentata), ma LASCIA LE HEADS SEPARATE
        # Shape: [Heads, H*W]
        attn_maps = torch.stack([attn_tensor[:, :, idx] for idx in token_indices]).sum(dim=0)

        # Matrice X: [4096 Pixel, Num_Heads]
        X = attn_maps.t()
        X_centered = X - X.mean(dim=0, keepdim=True)

        # SVD — il primo autovettore è il "Consenso Globale" delle Heads
        U, S, V = torch.svd(X_centered)
        S_sem = U[:, 0]

        # Fix polarità: la SVD può invertire il segno
        mean_attention = X.mean(dim=1)
        if torch.sum(S_sem * mean_attention) < 0:
            S_sem = -S_sem

        S_sem = (S_sem - S_sem.min()) / (S_sem.max() - S_sem.min() + 1e-8)
        return S_sem.view(H, W)

    def _compute_cosine_topology(self, v_t: torch.Tensor) -> torch.Tensor:
        """Sensore topologico globale via cosine similarity matrix."""
        C, H, W = v_t.shape
        v_flat = v_t.view(C, -1).t()  # [4096, C]

        v_norm = F.normalize(v_flat, p=2, dim=-1)
        # Affinità cosine elevata al cubo (affila i bordi)
        W_mat = torch.clamp(torch.matmul(v_norm, v_norm.t()), min=0.0) ** 3

        # Coerenza per pixel (somma riga)
        S_phys = torch.sum(W_mat, dim=-1)
        S_phys = (S_phys - S_phys.min()) / (S_phys.max() - S_phys.min() + 1e-8)
        return S_phys.view(H, W)

    def __call__(
        self,
        attn_tensor: torch.Tensor,
        v_t: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple = (64, 64),
    ) -> tuple:
        S_sem = self._compute_svd_semantics(attn_tensor, token_indices, spatial_shape)
        S_phys = self._compute_cosine_topology(v_t)

        M_hybrid = torch.sigmoid(self.k * (S_sem * S_phys - self.lambda_val))
        m_min, m_max = M_hybrid.min(), M_hybrid.max()
        M_hybrid = (M_hybrid - m_min) / (m_max - m_min + 1e-8)

        return M_hybrid, S_sem, S_phys

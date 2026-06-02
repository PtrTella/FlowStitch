"""
SupervisedSpectralHybridDecoder — Decoder spettrale con Fiedler (Capitolo 5, §5.4).

Costruisce il Laplaciano normalizzato dal grafo di cosine similarity del campo v₀,
estrae il vettore di Fiedler (secondo autovettore), e usa l'attenzione semantica
solo come "bussola" per risolvere l'ambiguità di segno.

Questo decoder anticipa il breakthrough del Capitolo 6 (Spectral Matting),
ma opera a risoluzione piena (4096×4096) — il che causa problemi numerici
documentati nel Capitolo 6 (ARPACK failure).
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SupervisedSpectralHybridDecoder:
    """Vettore di Fiedler dal Laplaciano normalizzato + bussola semantica."""

    def __init__(self, k: float = 12.0, lambda_val: float = 0.45):
        self.k = k
        self.lambda_val = lambda_val

    def _extract_semantic_signal(
        self,
        attn_tensor: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """Isola il potenziale semantico (usato solo come bussola di polarità)."""
        H, W = spatial_shape
        attn_maps_list = [attn_tensor[:, :, idx].mean(dim=0) for idx in token_indices]
        attn_map = torch.clamp(torch.stack(attn_maps_list).sum(dim=0), min=0.0, max=1.0)

        attn_min, attn_max = attn_map.min(), attn_map.max()
        A_target = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
        return A_target.view(H, W)

    def _compute_spectral_sensor(self, v_t: torch.Tensor) -> torch.Tensor:
        """
        Spectral Matting: Laplaciano Normalizzato → Vettore di Fiedler.

        ATTENZIONE: opera a risoluzione piena (4096×4096). Per matrici grandi,
        usare flowstitch.extraction.spectral_mask.compute_fiedler_mask che
        applica decimazione bilineare a 32×32.
        """
        C, H, W = v_t.shape
        v_t_flat = v_t.view(C, -1).t()  # [4096, C]

        v_norm = F.normalize(v_t_flat, p=2, dim=-1)
        W_mat = torch.clamp(torch.matmul(v_norm, v_norm.t()), min=0.0) ** 3

        # Laplaciano normalizzato L_sym = I - D^{-1/2} W D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(W_mat, dim=-1) + 1e-8))
        L = torch.eye(W_mat.shape[0], device=v_t.device) - torch.matmul(
            torch.matmul(D_inv_sqrt, W_mat), D_inv_sqrt
        )

        # Eigendecomposition → Fiedler vector (indice 1)
        evals, evecs = torch.linalg.eigh(L)
        fiedler_vector = evecs[:, 1]

        return fiedler_vector.view(H, W)

    def __call__(
        self,
        attn_tensor: torch.Tensor,
        v_t: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple = (64, 64),
    ) -> tuple:
        H, W = spatial_shape

        # 1. Bussola semantica
        S_sem = self._extract_semantic_signal(attn_tensor, token_indices, spatial_shape)

        # 2. Forma topologica spettrale
        S_phys_spectral = self._compute_spectral_sensor(v_t)
        max_val = S_phys_spectral.abs().max() + 1e-8
        S_phys_spectral = S_phys_spectral / max_val

        # 3. Polarizzazione: usa la semantica per orientare il Fiedler
        threshold = torch.quantile(S_sem.view(-1), 0.50)
        core_mask = (S_sem > threshold).float()
        fiedler_target_sign = torch.sum(S_phys_spectral * core_mask) / (core_mask.sum() + 1e-8)

        if fiedler_target_sign < 0:
            S_phys_spectral = -S_phys_spectral
            logger.info("Polarizzazione invertita (segno: %.4f)", fiedler_target_sign.item())

        # 4. Maschera finale (solo fisica, la semantica ha già orientato)
        M_hybrid = torch.sigmoid(self.k * S_phys_spectral)
        M_hybrid = (M_hybrid - M_hybrid.min()) / (M_hybrid.max() - M_hybrid.min() + 1e-8)

        return M_hybrid, S_sem, S_phys_spectral

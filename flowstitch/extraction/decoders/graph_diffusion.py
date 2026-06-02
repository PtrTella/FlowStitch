"""
GraphDiffusionHybridDecoder — Random walk di Markov (Capitolo 5, §5.5).

Abbandona completamente le soglie statistiche. Costruisce un grafo di Markov
sulla matrice di cosine similarity del campo v₀, e propaga un seed semantico
(dalla cross-attention) attraverso random walk iterativo.

Il random walk riempie naturalmente il Vuoto Termodinamico: la probabilità
diffonde dal bordo (alta energia) al centro (bassa energia) attraverso i
"tubi" di alta cosine similarity.

Limitazione: il seed iniziale (media delle heads) è rumoroso →
risolto in §5.6 con SVD.
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class GraphDiffusionHybridDecoder:
    """Random walk di Markov sul grafo coseno del campo v₀."""

    def __init__(self, k: float = 12.0, diffusion_steps: int = 5):
        self.k = k
        self.diffusion_steps = diffusion_steps

    def _extract_semantic_seed(
        self,
        attn_tensor: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """Estrae la mappa di attenzione iniziale (il 'liquido' da diffondere)."""
        H, W = spatial_shape
        attn_maps_list = [attn_tensor[:, :, idx].mean(dim=0) for idx in token_indices]
        attn_map = torch.clamp(torch.stack(attn_maps_list).sum(dim=0), min=0.0, max=1.0)

        attn_min, attn_max = attn_map.min(), attn_map.max()
        A_seed = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
        return A_seed.view(H, W)

    def _compute_transition_matrix(self, v_t: torch.Tensor) -> torch.Tensor:
        """
        Matrice di transizione P = D^{-1} W per random walk.
        W_{ij} = max(0, cos(v_i, v_j))^3 (cosine similarity cubica).
        """
        C, H, W = v_t.shape
        v_flat = v_t.view(C, -1).t()  # [4096, C]

        v_norm = F.normalize(v_flat, p=2, dim=-1)
        W_mat = torch.clamp(torch.matmul(v_norm, v_norm.t()), min=0.0) ** 3

        # Normalizzazione per riga → probabilità di transizione
        D_inv = 1.0 / (torch.sum(W_mat, dim=-1) + 1e-8)
        P = D_inv.unsqueeze(1) * W_mat
        return P

    def __call__(
        self,
        attn_tensor: torch.Tensor,
        v_t: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple = (64, 64),
    ) -> tuple:
        H, W = spatial_shape

        # 1. Seed semantico (il liquido)
        S_seed = self._extract_semantic_seed(attn_tensor, token_indices, spatial_shape)

        # 2. Struttura topologica (i tubi del grafo)
        P_matrix = self._compute_transition_matrix(v_t)

        # 3. Diffusione iterativa di Markov
        S_diffused = S_seed.view(-1).clone()
        for _ in range(self.diffusion_steps):
            S_diffused = torch.matmul(P_matrix, S_diffused)

        S_diffused = S_diffused.view(H, W)
        s_min, s_max = S_diffused.min(), S_diffused.max()
        S_diffused = (S_diffused - s_min) / (s_max - s_min + 1e-8)

        # 4. Sigmoid per indurimento (centrata a 0.5)
        M_hybrid = torch.sigmoid(self.k * (S_diffused - 0.5))
        m_min, m_max = M_hybrid.min(), M_hybrid.max()
        M_hybrid = (M_hybrid - m_min) / (m_max - m_min + 1e-8)

        return M_hybrid, S_seed, S_diffused

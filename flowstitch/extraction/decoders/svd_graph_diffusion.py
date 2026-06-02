"""
SvdSeededGraphDiffusionDecoder — Il decoder definitivo (Capitolo 5, §5.6).

Combina il meglio dei decoder precedenti:
- SVD (da §5.3) per il seed semantico purificato — elimina il rumore delle heads
- Graph Diffusion (da §5.5) per la propagazione strutturale — riempie il vuoto

Il seed SVD è "l'inchiostro purissimo" e la matrice di transizione di Markov
sono "i tubi" attraverso cui l'inchiostro diffonde, riempiendo la forma
dell'oggetto nella sua interezza.

Questo decoder risolve simultaneamente:
1. Il Vuoto Termodinamico (la diffusione riempie il centro)
2. Il rumore semantico (SVD purifica il segnale)
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SvdSeededGraphDiffusionDecoder:
    """SVD seed purificato + diffusione di Markov sul grafo coseno."""

    def __init__(self, k: float = 12.0, diffusion_steps: int = 5):
        self.k = k
        self.diffusion_steps = diffusion_steps

    def _extract_semantic_seed(
        self,
        attn_tensor: torch.Tensor,
        token_indices: list,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """
        Seed semantico via SVD sulle Heads.
        L'autovettore primario cattura il consenso globale, eliminando
        le teste rumorose.
        """
        H, W = spatial_shape

        # Sommiamo i token (se frammentati), ma LASCIAMO LE HEADS INTONSE
        # Shape: [Heads, 4096]
        attn_maps = torch.stack([attn_tensor[:, :, idx] for idx in token_indices]).sum(dim=0)

        # Matrice X per la SVD [4096, Num_Heads]
        X = attn_maps.t()
        X_centered = X - X.mean(dim=0, keepdim=True)

        # Decomposizione consensuale
        U, S, V = torch.svd(X_centered)
        S_seed = U[:, 0]

        # Controllo polarità
        mean_attention = X.mean(dim=1)
        if torch.sum(S_seed * mean_attention) < 0:
            S_seed = -S_seed

        S_seed = (S_seed - S_seed.min()) / (S_seed.max() - S_seed.min() + 1e-8)
        return S_seed.view(H, W)

    def _compute_transition_matrix(self, v_t: torch.Tensor) -> torch.Tensor:
        """Matrice di transizione P = D^{-1} W per random walk termodinamico."""
        C, H, W = v_t.shape
        v_flat = v_t.view(C, -1).t()  # [4096, C]

        v_norm = F.normalize(v_flat, p=2, dim=-1)
        W_mat = torch.clamp(torch.matmul(v_norm, v_norm.t()), min=0.0) ** 3

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
        """
        Args:
            attn_tensor: [Heads, H*W, Text_Seq] cross-attention tensor
            v_t: [C, H, W] velocity field
            token_indices: list of token indices for target word
            spatial_shape: (H, W)

        Returns:
            (M_hybrid, S_seed, S_diffused)
        """
        H, W = spatial_shape

        # 1. Inchiostro purissimo (SVD sulle Heads)
        S_seed = self._extract_semantic_seed(attn_tensor, token_indices, spatial_shape)

        # 2. Tubi termodinamici (cosine similarity)
        P_matrix = self._compute_transition_matrix(v_t)

        # 3. Propagazione di Markov
        S_diffused = S_seed.view(-1).clone()
        for _ in range(self.diffusion_steps):
            S_diffused = torch.matmul(P_matrix, S_diffused)

        S_diffused = S_diffused.view(H, W)
        s_min, s_max = S_diffused.min(), S_diffused.max()
        S_diffused = (S_diffused - s_min) / (s_max - s_min + 1e-8)

        # 4. Sigmoid KTS (centrata a 0.5)
        M_hybrid = torch.sigmoid(self.k * (S_diffused - 0.5))
        m_min, m_max = M_hybrid.min(), M_hybrid.max()
        M_hybrid = (M_hybrid - m_min) / (m_max - m_min + 1e-8)

        return M_hybrid, S_seed, S_diffused

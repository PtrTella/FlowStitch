import torch
import torch.nn.functional as F
import math


def compute_sam_flow_mask(
    attn_mask: torch.Tensor,
    quantile: float = 0.70,
    alpha: float = 15.0,
    tau_core: float = 0.50,
    dilation_radius: int = 2,
    w_ring: float = 0.80,
    s_relax: float = 0.00,
) -> torch.Tensor:
    """
    Implementazione del meccanismo di Dynamic Soft Mask di SAM-Flow (arXiv:2606.06228).

    Fasi dell'algoritmo (Sezione III-C del paper):
    1. Min-max normalization della mappa di cross-attention di base M_base.
    2. Calcolo della soglia quantilica: theta = quantile(M_base, quantile).
    3. Risposta sigmoidea rinforzata: S = sigmoid(alpha * (M_base - theta)).
    4. Estrazione del nucleo (Core): C = (S >= tau_core).
    5. Dilatazione morfologica 2D del nucleo con raggio dilation_radius per ottenere D.
    6. Definizione della fascia/anello di transizione (Transition Ring): R = D - C.
    7. Costruzione del supporto continuo: W_soft = C + w_ring * (R * S) + s_relax.

    Args:
        attn_mask: [1, seq_len, 1] o [seq_len] mappa di attenzione
        quantile: Soglia quantile per isolare le attivazioni salienti (default: 0.70)
        alpha: Coefficiente di sharpening sigmoideo (default: 15.0)
        tau_core: Soglia per l'inclusione nel nucleo solido (default: 0.50)
        dilation_radius: Raggio di dilatazione morfologica in pixel/patch (default: 2)
        w_ring: Peso di amplificazione della transizione (default: 0.80)
        s_relax: Fattore di rilassamento per background (default: 0.00 per zero leak)

    Returns:
        mask: [1, seq_len, 1] float32 tensor
    """
    orig_device = attn_mask.device
    orig_dtype = attn_mask.dtype

    attn_flat = attn_mask.view(-1).to(torch.float32)
    seq_len = attn_flat.shape[0]
    side = int(math.isqrt(seq_len))
    assert side * side == seq_len, f"seq_len={seq_len} non è un quadrato perfetto"

    # 1. Normalizzazione min-max [0, 1]
    a_min, a_max = attn_flat.min(), attn_flat.max()
    if (a_max - a_min) > 1e-8:
        attn_norm_1d = (attn_flat - a_min) / (a_max - a_min)
    else:
        attn_norm_1d = attn_flat

    # Reshape a 2D per operazioni spaziali: [1, 1, H, W]
    attn_2d = attn_norm_1d.view(1, 1, side, side)

    # 2. Soglia quantilica theta
    theta = torch.quantile(attn_2d.flatten(), quantile)

    # 3. Risposta sigmoidea rinforzata (Sharpened Sigmoid)
    S = torch.sigmoid(alpha * (attn_2d - theta))

    # 4. Nucleo solido (Core Region)
    C = (S >= tau_core).to(torch.float32)

    # 5. Dilatazione morfologica 2D tramite max_pool2d
    k_size = 2 * dilation_radius + 1
    D = F.max_pool2d(C, kernel_size=k_size, stride=1, padding=dilation_radius)

    # 6. Anello di transizione (Transition Region: D - C)
    R = torch.clamp(D - C, min=0.0, max=1.0)

    # 7. Supporto continuo SAM-Flow
    W_soft = C + w_ring * (R * S) + s_relax
    W_soft = torch.clamp(W_soft, min=0.0, max=1.0)

    # Normalizzazione per saturare il massimo a 1.0 se non nullo
    max_val = W_soft.max()
    if max_val > 1e-6:
        W_soft = W_soft / max_val

    return W_soft.view(1, -1, 1).to(device=orig_device, dtype=orig_dtype)

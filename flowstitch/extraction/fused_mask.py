"""
Fused Mask Extraction Suite — La Quaterna Ufficiale di FlowStitch.

Combina l'attrattore semantico della Cross-Attention (baricentro globale dell'oggetto)
con l'allineamento direzionale della velocità v0 (definizione nitida dei contorni ad alta frequenza).

Fornisce le 4 formulazioni di riferimento per il framework FlowStitch:
1. extract_attention_mask: Baseline 1 (Cross-Attention pura normalizzata [0, 1])
2. compute_calibrated_attention_mask: Baseline 2 (Cross-Attention con sottrazione del piedistallo di rumore)
3. compute_linear_fused_mask: Metodo 3 (Fusione lineare soft-calibrata Attention x Cosine)
4. compute_hermite_fused_mask: Metodo 4 - Canonico Principale (Transizione cubica Hermite C^1: 3u^2 - 2u^3)
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import math
from typing import Sequence

from .attention_mask import extract_attention_mask


def compute_cosine_field(
    v0: torch.Tensor,
    attn_mask: torch.Tensor,
    normalize_positive: bool = True,
) -> torch.Tensor:
    """
    Calcola il campo di similarità coseno tra ciascun vettore di velocità v0(x)
    e il vettore medio di riferimento dell'oggetto ponderato dalla Cross-Attention:

        v_bar = normalize(sum_x (v0(x) * attn(x)))
        cos(x) = v0_norm(x) @ v_bar

    Args:
        v0: Campo di velocità [1, seq_len, dim] o [seq_len, dim]
        attn_mask: Mappa di attenzione guida [1, seq_len, 1] o [seq_len, 1] o [seq_len]
        normalize_positive: Se True, effettua clamp a [0, 1] (solo allineamento parallelo).

    Returns:
        cos_field: [1, seq_len, 1] float32 tensor
    """
    orig_device = v0.device
    orig_dtype = v0.dtype

    # Standardizza dimensioni a [seq_len, dim] e [seq_len, 1]
    v0_flat = v0.view(-1, v0.shape[-1]).to(torch.float32)
    seq_len = v0_flat.shape[0]

    attn_flat = attn_mask.view(seq_len, 1).to(torch.float32)

    # 1. Normalizzazione L2 dei vettori spaziali
    v0_normed = F.normalize(v0_flat, p=2, dim=-1, eps=1e-8)

    # 2. Calcolo del vettore medio del target ponderato dall'attenzione
    weighted_sum = (v0_normed * attn_flat).sum(dim=0, keepdim=True)  # [1, dim]
    mean_target_vector = F.normalize(weighted_sum, p=2, dim=-1, eps=1e-8)  # [1, dim]

    # 3. Proiezione coseno
    cos_raw = torch.matmul(v0_normed, mean_target_vector.transpose(0, 1))  # [seq_len, 1]

    if normalize_positive:
        cos_field = torch.clamp(cos_raw, min=0.0, max=1.0)
    else:
        cos_field = cos_raw

    return cos_field.view(1, seq_len, 1).to(device=orig_device, dtype=orig_dtype)


def compute_calibrated_attention_mask(
    layer_attn: torch.Tensor,
    token_indices: Sequence[int],
    pedestal_quantile: float = 0.15,
) -> torch.Tensor:
    """
    Baseline 2: Cross-Attention con sottrazione del piedistallo di rumore.

    Rimuove la componente continua di background calcolando il quantile inferiore
    e riscalando linearmente l'intervallo attivo in [0, 1].

    Args:
        layer_attn: Matrice di attenzione [1, heads, seq_len, text_seq_len]
        token_indices: Indici dei token associati al soggetto
        pedestal_quantile: Quantile (es. 0.15 = 15%) per stimare la soglia del rumore di fondo

    Returns:
        calib_attn: [1, seq_len, 1] float32 tensor
    """
    # Estrai attenzione pura normalizzata in [0, 1]
    raw_attn = extract_attention_mask(layer_attn, token_indices, normalize=True)  # [1, seq_len, 1]
    attn_flat = raw_attn.view(-1)

    tau_bg = torch.quantile(attn_flat, pedestal_quantile).item()
    calib = torch.clamp((attn_flat - tau_bg) / max(1.0 - tau_bg, 1e-6), min=0.0, max=1.0)

    max_val = calib.max()
    if max_val > 1e-6:
        calib = calib / max_val

    return calib.view_as(raw_attn)


def compute_linear_fused_mask(
    attn_mask: torch.Tensor,
    v0: torch.Tensor,
    tau_bg: float = 0.12,
) -> torch.Tensor:
    """
    Metodo 3: Fusione lineare soft-calibrata Attention x Cosine.

        prod(x) = attn(x) * max(0, cos(x))
        M(x) = clamp((prod(x) - tau_bg) / (1.0 - tau_bg), 0, 1)

    Offre isolamento netto senza perdite nello sfondo, ma con discontinuità
    della derivata prima sul bordo tau_bg (C^0 piecewise linear).

    Args:
        attn_mask: [1, seq_len, 1] o [seq_len] mappa di attenzione
        v0: [1, seq_len, dim] campo di velocità iniziale
        tau_bg: Soglia di taglio lineare del background

    Returns:
        mask: [1, seq_len, 1] float32 tensor
    """
    orig_device = attn_mask.device
    orig_dtype = attn_mask.dtype

    attn_flat = attn_mask.view(-1, 1).to(torch.float32)
    cos_flat = compute_cosine_field(v0, attn_flat, normalize_positive=True).view(-1, 1).to(torch.float32)

    # Normalizza attenzione se non lo è già
    a_min, a_max = attn_flat.min(), attn_flat.max()
    if (a_max - a_min) > 1e-8:
        attn_norm = (attn_flat - a_min) / (a_max - a_min)
    else:
        attn_norm = attn_flat

    prod = attn_norm * cos_flat
    m_calib = torch.clamp((prod - tau_bg) / max(1.0 - tau_bg, 1e-6), min=0.0, max=1.0)

    max_val = m_calib.max()
    if max_val > 1e-6:
        m_calib = m_calib / max_val

    return m_calib.view(1, -1, 1).to(device=orig_device, dtype=orig_dtype)


def compute_hermite_fused_mask(
    attn_mask: torch.Tensor,
    v0: torch.Tensor,
    edge0: float = 0.10,
    edge1: float = 0.65,
) -> torch.Tensor:
    """
    Metodo 4 — Canonico Principale: Hermite Smoothstep C^1.

    Formula polinomiale cubica:
        prod(x) = attn(x) * max(0, cos(x))
        u = clamp((prod(x) - edge0) / (edge1 - edge0), 0.0, 1.0)
        M(x) = 3*u^2 - 2*u^3

    Proprietà Matematiche:
    - Derivata prima continua su tutto il dominio: dM/du = 6u(1 - u)
    - Gradiente nullo sia al confine inferiore u=0 che a saturazione u=1
    - Soddisfa rigorosamente la condizione di continuità locale Lipschitz
      richiesta dal Teorema di Picard-Lindelöf per l'integrazione ODE di Flow Matching.
    - Zero background leakage e densità del nucleo preservata.

    Args:
        attn_mask: [1, seq_len, 1] o [seq_len] mappa di attenzione
        v0: [1, seq_len, dim] campo di velocità iniziale
        edge0: Limite inferiore della rampa (transizione smooth dal background)
        edge1: Limite superiore della rampa (saturazione piena al centro del soggetto)

    Returns:
        mask: [1, seq_len, 1] float32 tensor
    """
    orig_device = attn_mask.device
    orig_dtype = attn_mask.dtype

    attn_flat = attn_mask.view(-1, 1).to(torch.float32)
    cos_flat = compute_cosine_field(v0, attn_flat, normalize_positive=True).view(-1, 1).to(torch.float32)

    a_min, a_max = attn_flat.min(), attn_flat.max()
    if (a_max - a_min) > 1e-8:
        attn_norm = (attn_flat - a_min) / (a_max - a_min)
    else:
        attn_norm = attn_flat

    prod = attn_norm * cos_flat

    # Cubic Hermite smoothstep transition
    u = torch.clamp((prod - edge0) / max(edge1 - edge0, 1e-6), min=0.0, max=1.0)
    m_hermite = u * u * (3.0 - 2.0 * u)

    max_val = m_hermite.max()
    if max_val > 1e-6:
        m_hermite = m_hermite / max_val

    return m_hermite.view(1, -1, 1).to(device=orig_device, dtype=orig_dtype)


def compute_core_fused_mask(
    attn_mask: torch.Tensor,
    v0: torch.Tensor,
    core_tau_low: float = 0.15,
    core_tau_high: float = 0.45,
    edge0: float = 0.10,
    edge1: float = 0.70,
    plateau: bool = True,
) -> torch.Tensor:
    """
    Metodo 5 — Core-Preserved Soft Trimap Fusion (Hermite C^1).

    Risolve analiticamente il problema del centro svuotato nei soggetti speculari o 3D:
    - Nel Nucleo (A >= core_tau_high): w_core = 1, se plateau=True la maschera è saturata
      a 1.0 solido costante (Core Min = 1.0), proteggendo la superficie da riflessi e ombre speculari.
    - Sul Perimetro (A <= core_tau_low): w_core = 0, il Coseno direzionale ritaglia ad
      alta frequenza il confine ed estingue al 100% l'alone nello sfondo.
    - Nella Fascia di Transizione: interpolazione cubica Hermite C^1 priva di salti.

    Args:
        attn_mask: [1, seq_len, 1] o [seq_len] mappa di attenzione
        v0: [1, seq_len, dim] campo di velocità iniziale
        core_tau_low: soglia inferiore per l'attivazione del nucleo solido
        core_tau_high: soglia superiore dove il nucleo è pienamente protetto
        edge0: taglio inferiore di background nella rampa Hermite
        edge1: saturazione superiore nella rampa Hermite
        plateau: se True (default), satura il nucleo solido a 1.0 (impenetrabile)

    Returns:
        mask: [1, seq_len, 1] float32 tensor
    """
    orig_device = attn_mask.device
    orig_dtype = attn_mask.dtype

    attn_flat = attn_mask.view(-1, 1).to(torch.float32)
    cos_flat = compute_cosine_field(v0, attn_flat, normalize_positive=True).view(-1, 1).to(torch.float32)

    a_min, a_max = attn_flat.min(), attn_flat.max()
    if (a_max - a_min) > 1e-8:
        attn_norm = (attn_flat - a_min) / (a_max - a_min)
    else:
        attn_norm = attn_flat

    # 1. Ponderazione differenziabile del nucleo (smoothstep cubic)
    t_core = torch.clamp((attn_norm - core_tau_low) / max(core_tau_high - core_tau_low, 1e-6), 0.0, 1.0)
    w_core = t_core * t_core * (3.0 - 2.0 * t_core)

    # 2. Fusione bilaterale: nel nucleo domina l'Attenzione (o plateau unitario solido), sul perimetro il Coseno taglia l'alone
    core_target = torch.ones_like(attn_norm) if plateau else attn_norm
    m_raw = w_core * core_target + (1.0 - w_core) * (attn_norm * cos_flat)

    # 3. Transizione cubica Hermite C^1 finale
    u = torch.clamp((m_raw - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    m_core_fused = u * u * (3.0 - 2.0 * u)

    max_val = m_core_fused.max()
    if max_val > 1e-6:
        m_core_fused = m_core_fused / max_val

    return m_core_fused.view(1, -1, 1).to(device=orig_device, dtype=orig_dtype)


def compute_bilateral_sigmoid_mask(
    attn_mask: torch.Tensor,
    v0: torch.Tensor,
    tau_gate: float = 0.50,
    sharpness: float = 10.0,
    bg_cut: float = 0.05,
) -> torch.Tensor:
    """
    Metodo 6 — Bilateral Sigmoidal Gate (Alpha-Gated continuo).

    Modula la Cross-Attention con un cancello sigmoideo continuo basato sulla similarità coseno grezza:
        Gate(x) = sigmoid(sharpness * (cos_raw(x) - tau_gate))
        M(x) = clamp((attn(x) * Gate(x) - bg_cut) / (1.0 - bg_cut), 0, 1)

    Offre una transizione morbida e differenziabile, preservando continuità C-infinito.

    Args:
        attn_mask: [1, seq_len, 1] o [seq_len] mappa di attenzione
        v0: [1, seq_len, dim] campo di velocità iniziale
        tau_gate: soglia di allineamento per l'attivazione del cancello sigmoideo
        sharpness: pendenza del gradiente di transizione
        bg_cut: taglio lineare dei residui di fondo

    Returns:
        mask: [1, seq_len, 1] float32 tensor
    """
    orig_device = attn_mask.device
    orig_dtype = attn_mask.dtype

    attn_flat = attn_mask.view(-1, 1).to(torch.float32)
    cos_raw = compute_cosine_field(v0, attn_flat, normalize_positive=False).view(-1, 1).to(torch.float32)

    a_min, a_max = attn_flat.min(), attn_flat.max()
    if (a_max - a_min) > 1e-8:
        attn_norm = (attn_flat - a_min) / (a_max - a_min)
    else:
        attn_norm = attn_flat

    gate = torch.sigmoid(sharpness * (cos_raw - tau_gate))
    m_gated = attn_norm * gate
    m_cut = torch.clamp((m_gated - bg_cut) / max(1.0 - bg_cut, 1e-6), min=0.0, max=1.0)

    max_val = m_cut.max()
    if max_val > 1e-6:
        m_cut = m_cut / max_val

    return m_cut.view(1, -1, 1).to(device=orig_device, dtype=orig_dtype)


def compute_kinetic_semantic_hull_mask(
    attn_mask: torch.Tensor,
    v0: torch.Tensor,
    quantile: float = 0.65,
    closing_radius: int = 2,
    edge0: float = 0.05,
    edge1: float = 0.35,
    plateau: bool = True,
) -> torch.Tensor:
    """
    Metodo 7 — Kinetic-Semantic Coherent Hull (Fusione Intelligente Multimodale).

    Risolve sinergicamente la perdita delle punte geometriche e il collasso di densità
    sulle superfici piatte (facce della piramide e pareti del cubo):
    1. Calibrazione Quantilica Dinamica: isola la componente semantica saliente rimuovendo
       il piedistallo continuo di fondo (simile a SAM-Flow quantile theta).
    2. Chiusura Morfologica Spaziale (Closing 2D): dilatazione e successiva erosione 2D
       sulla mappa di attenzione per colmare le depressioni concave su facce e pareti uniformi.
    3. Campo di Coerenza Direzionale del Vettore di Velocità: proietta il flusso v0 sul
       baricentro cinetico dell'oggetto.
    4. Sinergia Cinetico-Semantica: potenzia la solidità del corpo geometrico combinando
       l'attenzione morfologicamente chiusa con la proiezione direzionale del coseno.
    5. Nucleo Monolitico a Plateau Unitario con Raccordo Hermite C^1:
       - Nucleo (Core): M(x) = 1.000 costante (nessuna diluizione da parte del flusso ambientale).
       - Perimetro: transizione cubica Hermite 3u^2 - 2u^3 a derivata prima nulla sui bordi.
       - Sfondo: M(x) = 0.00000 esatto (zero background leakage).

    Args:
        attn_mask: [1, seq_len, 1] o [seq_len] mappa di attenzione
        v0: [1, seq_len, dim] o [seq_len, dim] campo di velocità iniziale
        quantile: soglia quantilica per isolare la salienza semantica (default: 0.65)
        closing_radius: raggio del filtro di chiusura morfologica 2D (default: 2)
        edge0: limite inferiore del smoothstep Hermite (taglio background, default: 0.05)
        edge1: limite superiore del smoothstep Hermite (inizio plateau 1.0, default: 0.35)
        plateau: se True (default), satura a 1.000 costante tutto il nucleo

    Returns:
        mask: [1, seq_len, 1] float32 tensor
    """
    orig_device = attn_mask.device
    orig_dtype = attn_mask.dtype

    attn_flat = attn_mask.view(-1).to(torch.float32)
    seq_len = attn_flat.shape[0]
    side = int(math.isqrt(seq_len))
    assert side * side == seq_len, f"seq_len={seq_len} non è un quadrato perfetto"

    # 1. Normalizzazione min-max dell'Attenzione [0, 1]
    a_min, a_max = attn_flat.min(), attn_flat.max()
    if (a_max - a_min) > 1e-8:
        attn_norm = (attn_flat - a_min) / (a_max - a_min)
    else:
        attn_norm = attn_flat

    # 2. Calibrazione quantilica per estrarre la salienza attiva
    theta = torch.quantile(attn_norm, quantile)
    a_calib = torch.clamp((attn_norm - theta) / max(1.0 - theta.item(), 1e-6), min=0.0, max=1.0)
    max_v = a_calib.max()
    if max_v > 1e-6:
        a_calib = a_calib / max_v

    # 3. Chiusura morfologica 2D (Closing: Dilate -> Erode)
    k_size = 2 * closing_radius + 1
    a_2d = a_calib.view(1, 1, side, side)
    if closing_radius > 0:
        d_2d = F.max_pool2d(a_2d, kernel_size=k_size, stride=1, padding=closing_radius)
        a_closed_2d = -F.max_pool2d(-d_2d, kernel_size=k_size, stride=1, padding=closing_radius)
    else:
        a_closed_2d = a_2d

    a_closed = a_closed_2d.view(1, -1, 1)

    # 4. Campo di coerenza coseno pesato sul target
    cos_field = compute_cosine_field(v0, a_closed, normalize_positive=True).view(1, -1, 1)

    # 5. Sinergia Cinetico-Semantica
    s_raw = torch.where(
        a_closed > 0.0,
        torch.maximum(a_closed, (a_closed ** 0.5) * cos_field),
        torch.zeros_like(a_closed),
    )

    # 6. Transizione Hermite C^1 con Plateau
    u = torch.clamp((s_raw - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    if plateau:
        m_final = torch.where(u >= 1.0, torch.ones_like(u), u * u * (3.0 - 2.0 * u))
    else:
        m_final = u * u * (3.0 - 2.0 * u)

    max_val = m_final.max()
    if max_val > 1e-6:
        m_final = m_final / max_val

    return m_final.view(1, -1, 1).to(device=orig_device, dtype=orig_dtype)




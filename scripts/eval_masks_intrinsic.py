#!/usr/bin/env python3
"""
FlowStitch Intrinsic & Structural Mask Evaluator (Famiglia A)
=============================================================
Valutazione sistematica offline su Mac di tutte le rappresentazioni intrinseche del flusso
e dei metodi storici/strutturali dei Capitoli 1-6 della Tesi:

1. Immagine Originale FLUX.1 (RGB)
2. Cross-Attention Pura (Layer 10) [Base semantica testuale]
3. Otsu Hard Threshold [Soglia ottimale globale applicata alla Cross-Attention]
4. Velocità Intensità ||v_0|| [Energia Cinetica continua]
5. Chebyshev Energy Gating [Soglia mu + sigma su ||v_0||: Vuoto Termodinamico]
6. Velocità Direzione C(x) [Allineamento cosinusoidale continuo rispetto a v_bar]
7. Vettore di Fiedler [Taglio spettrale su grafo Laplaciano 32x32]
8. Hybrid [Fiedler orientato da bussola di attenzione]
9. TDA Topological Mask [Single-linkage clustering H0 su distanza coseno]
10. Profilo 1D Sezione Orizzontale al Centro [Confronto diretto M(x)]
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowstitch.core.serialization import load_tensors
from flowstitch.core.tokenizer_utils import find_token_indices
from flowstitch.extraction.attention_mask import extract_attention_mask, otsu_threshold
from flowstitch.extraction.energy_mask import compute_chebyshev_threshold
from flowstitch.extraction.spectral_mask import compute_fiedler_mask
from flowstitch.extraction.hybrid_mask import hybrid_semantic_decomposition
from flowstitch.extraction.fused_mask import compute_cosine_field
from flowstitch.extraction.tda_mask import extract_tda_mask
from transformers import T5Tokenizer

SUBJECTS = [
    ("a_blue_sphere", "sphere"),
    ("a_red_cube", "cube"),
    ("a_yellow_pyramid", "pyramid"),
]


def evaluate_intrinsic_scene(subject_dir: Path, word: str, output_dir: Path, tokenizer: T5Tokenizer):
    if not subject_dir.exists():
        print(f"[ERRORE] Cartella non trovata: {subject_dir}")
        return None

    # 1. Carica metadati e immagine originale
    with open(subject_dir / "metadata.json", "r") as f:
        meta = json.load(f)
    prompt = meta["prompt"]

    orig_img_path = subject_dir / "final_image.png"
    orig_img = Image.open(orig_img_path).convert("RGB") if orig_img_path.exists() else None

    token_indices = find_token_indices(tokenizer, prompt, word)

    # 2. Carica attention_maps e v0
    attn_dict = load_tensors(str(subject_dir / "attention_maps.pt"))
    layer_keys = sorted([k for k in attn_dict.keys() if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
    layer_10 = attn_dict[layer_keys[-1]]

    v0 = load_tensors(str(subject_dir / "v0_velocity.pt"))
    if isinstance(v0, dict):
        v0 = list(v0.values())[0]

    # --- 3. Calcolo di TUTTE le 8 rappresentazioni matematiche ---
    representations = {}

    # A. Cross-Attention Pura [0, 1]
    attn_raw = extract_attention_mask(layer_10, token_indices, normalize=True)
    representations["1. Cross-Attention"] = (attn_raw.view(64, 64).float().cpu(), "magma")

    # B. Otsu Binarization su Attention
    otsu_m = otsu_threshold(attn_raw).to(torch.float32)
    representations["2. Otsu (Thresh Attn)"] = (otsu_m.view(64, 64).float().cpu(), "gray")

    # C. Velocità: Intensità ||v_0|| (Energia Cinetica Continua)
    v0_f32 = v0.view(4096, -1).to(torch.float32)
    energy = torch.norm(v0_f32, p=2, dim=-1, keepdim=True)
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    representations["3. Velocità: Intensità ||v0||"] = (energy_norm.view(64, 64).float().cpu(), "plasma")

    # D. Chebyshev Energy Gating (Soglia mu + sigma: Vuoto Termodinamico)
    cheb_m = compute_chebyshev_threshold(energy, k=1.0).to(torch.float32)
    representations["4. Chebyshev (Vuoto Termo.)"] = (cheb_m.view(64, 64).float().cpu(), "plasma")

    # E. Velocità: Direzione C(x) (Cosine Similarity Continua con v_bar)
    cos_field = compute_cosine_field(v0=v0, attn_mask=attn_raw, normalize_positive=False)
    # Mostriamo la similarità pura in [-1, 1] mappata per visualizzazione o normalizzata
    cos_norm = (cos_field - cos_field.min()) / (cos_field.max() - cos_field.min() + 1e-8)
    representations["5. Velocità: Direzione C(x)"] = (cos_norm.view(64, 64).float().cpu(), "coolwarm")

    # F. Vettore di Fiedler (Grafo Laplaciano 32x32)
    fiedler_m = compute_fiedler_mask(v0, target_resolution=32).to(torch.float32)
    representations["6. Spectral Fiedler"] = (fiedler_m.view(64, 64).float().cpu(), "gray")

    # G. Hybrid (Fiedler + Bussola Attenzione)
    hybrid_m = hybrid_semantic_decomposition(v0, attn_raw, target_resolution=32).to(torch.float32)
    representations["7. Hybrid (Fiedler+Attn)"] = (hybrid_m.view(64, 64).float().cpu(), "gray")

    # H. TDA Topological Mask (Homology H0 Clustering)
    try:
        tda_m = extract_tda_mask(v0, attn_raw).to(torch.float32)
        representations["8. TDA (Omologia H0)"] = (tda_m.view(64, 64).float().cpu(), "gray")
    except Exception as e:
        print(f"[WARN] TDA fallito: {e}")
        tda_m = torch.zeros(64, 64)
        representations["8. TDA (Omologia H0)"] = (tda_m, "gray")

    # Regioni per metriche oggettive
    m_base = representations["1. Cross-Attention"][0]
    core_region = (m_base > 0.50)
    bg_region = (m_base < 0.10)

    # Calcolo metriche quantitative oggettive
    metrics = {}
    for name, (m, _) in representations.items():
        core_mean = m[core_region].mean().item() if core_region.any() else 0.0
        core_min = m[core_region].min().item() if core_region.any() else 0.0
        bg_leak = m[bg_region].mean().item() if bg_region.any() else 0.0
        zero_pct = (m == 0.0).float().mean().item() * 100.0

        dx = torch.abs(m[:, 1:] - m[:, :-1])
        dy = torch.abs(m[1:, :] - m[:-1, :])
        max_grad = max(dx.max().item(), dy.max().item())

        metrics[name] = {
            "core_mean": core_mean,
            "core_min": core_min,
            "bg_leak": bg_leak,
            "zero_pct": zero_pct,
            "max_grad": max_grad,
        }

    # Centro geometrico per la sezione orizzontale 1D
    ys, xs = torch.where(core_region)
    y_center = int(ys.float().mean().item()) if len(ys) > 0 else 32

    # --- 4. Plotting a 10 Pannelli (2 Righe x 5 Colonne) ---
    fig = plt.figure(figsize=(26, 11), dpi=150)
    plt.suptitle(
        f"Famiglia A: Mappe Intrinseche di Flusso e Struttura — {subject_dir.name} ('{word}')",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Pannello 1 (R1, C1): Immagine Originale
    ax_img = plt.subplot(2, 5, 1)
    if orig_img is not None:
        ax_img.imshow(orig_img)
    ax_img.set_title("Originale FLUX.1", fontsize=12, fontweight="bold")
    ax_img.axhline(y_center * 16, color="cyan", linestyle="--", alpha=0.8, label=f"Sezione Y={y_center}")
    ax_img.legend(loc="upper right", fontsize=8)
    ax_img.axis("off")

    rep_keys = list(representations.keys())

    # Pannelli 2-5 (R1, C2..C5): Cross-Attention, Otsu, Intensità ||v0||, Chebyshev
    for i in range(4):
        k = rep_keys[i]
        m, cmap = representations[k]
        ax = plt.subplot(2, 5, i + 2)
        im = ax.imshow(m.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(k, fontsize=11, fontweight="bold")
        ax.axhline(y_center, color="cyan", linestyle="--", alpha=0.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Pannelli 6-9 (R2, C1..C4): Direzione C(x), Fiedler, Hybrid, TDA
    for i in range(4, 8):
        k = rep_keys[i]
        m, cmap = representations[k]
        ax = plt.subplot(2, 5, i + 2)
        im = ax.imshow(m.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(k, fontsize=11, fontweight="bold")
        ax.axhline(y_center, color="cyan", linestyle="--", alpha=0.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Pannello 10 (R2, C5): Profilo 1D della Sezione Orizzontale al Centro
    ax_slice = plt.subplot(2, 5, 10)
    x_axis = np.arange(64)
    colors = ["#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for k, color in zip(rep_keys, colors):
        m = representations[k][0]
        slice_vals = m[y_center, :].numpy()
        linewidth = 2.0 if "Attn" in k or "Direzione" in k else 1.2
        linestyle = "-" if "Direzione" in k or "Cross-Attention" in k else "--"
        ax_slice.plot(x_axis, slice_vals, label=k.split(". ")[-1], color=color, linewidth=linewidth, linestyle=linestyle)

    ax_slice.set_title(f"Profilo 1D Sezione Y={y_center} (Centro)", fontsize=11, fontweight="bold")
    ax_slice.set_xlabel("Coordinata X (Patch 64x64)")
    ax_slice.set_ylabel("Valore M(x)")
    ax_slice.set_ylim(-0.05, 1.05)
    ax_slice.grid(True, linestyle=":", alpha=0.6)
    ax_slice.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_fig = output_dir / f"eval_intrinsic_{subject_dir.name}.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Salvata figura a 10 pannelli in: {out_fig}")
    return metrics


def main():
    data_root = PROJECT_ROOT / "data/dataset_v1"
    output_dir = PROJECT_ROOT / "outputs/eval_masks_intrinsic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FLOWSTITCH INTRINSIC & STRUCTURAL SUITE EVALUATION (10-PANEL COMPARATOR)")
    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")
    print("=" * 80)

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True)

    all_metrics = {}
    for subject_name, word in SUBJECTS:
        print(f"\n>>>> Analisi Intrinseca Completa: {subject_name} (token: '{word}') <<<<")
        m = evaluate_intrinsic_scene(data_root / subject_name, word, output_dir, tokenizer)
        if m:
            all_metrics[subject_name] = m
            print(f"\n--- Metriche Intrinseche: {subject_name} ---")
            print(f"{"Rappresentazione":<32} | {"Core Mean":<10} | {"Core Min":<10} | {"BG Leak":<10} | {"Zero %":<8} | {"Max Grad":<8}")
            print("-" * 90)
            for name, vals in m.items():
                print(f"{name:<32} | {vals["core_mean"]:<10.3f} | {vals["core_min"]:<10.3f} | {vals["bg_leak"]:<10.5f} | {vals["zero_pct"]:<7.1f}% | {vals["max_grad"]:<8.3f}")

    json_path = output_dir / "metrics_intrinsic.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[COMPLETATO] Tutte le metriche salvate in: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

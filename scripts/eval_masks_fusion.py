#!/usr/bin/env python3
"""
FlowStitch Fusion Mask Evaluator (Famiglia B: Attention, Cosine, Trimap)
========================================================================
Valutazione offline su Mac delle formulazioni di fusione multi-modale:
1. pure_attention       : Cross-Attention pura normalizzata [0, 1] (Layer 10)
2. calibrated_attention : Cross-Attention con sottrazione piedistallo di rumore
3. fused_linear         : Fusione lineare soft-calibrata Attention x Cosine
4. hermite              : Transizione cubica Hermite C^1 (Prodotto A x C)
5. bilateral_sigmoid    : Cancello sigmoideo bilaterale continuo
6. core_fused_soft      : Soft Trimap Bilaterale SENZA plateau (attenua riflessi, min ~0.60)
7. core_fused (plateau) : Soft Trimap Bilaterale CON plateau unitario (nucleo solido 1.000)

Genera griglie 3x3 ad alta risoluzione con profilo 1D della sezione mediana.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowstitch.pipelines.mask_compilation import compile_mask_for_scene

SUBJECTS = [
    ("a_blue_sphere", "sphere"),
    ("a_red_cube", "cube"),
    ("a_yellow_pyramid", "pyramid"),
]

FUSION_METHODS = [
    ("1. Pure Attention", "attention"),
    ("2. Calibrated Attention", "calibrated_attention"),
    ("3. Linear Fused (A*C)", "fused_linear"),
    ("4. Hermite Product (A*C)", "hermite"),
    ("5. Bilateral Sigmoid", "bilateral_sigmoid"),
    ("6. Core-Fused Soft", "core_fused_soft"),
    ("7. Core-Fused Plateau", "core_fused"),
]


def evaluate_fusion(data_root: Path, subject_name: str, word: str, output_dir: Path):
    subject_dir = data_root / subject_name
    if not subject_dir.exists():
        print(f"[ERRORE] Cartella non trovata: {subject_dir}")
        return None

    orig_img_path = subject_dir / "final_image.png"
    orig_img = Image.open(orig_img_path).convert("RGB") if orig_img_path.exists() else None

    masks = {}
    for disp_name, method_key in FUSION_METHODS:
        m = compile_mask_for_scene(
            db_path=str(subject_dir),
            word_to_isolate=word,
            method=method_key,
        )
        masks[disp_name] = m.view(64, 64).float().cpu()

    # Riferimento geometrico oggettivo dalla Pure Attention
    m_base = masks["1. Pure Attention"]
    core_region = (m_base > 0.50)
    bg_region = (m_base < 0.10)

    metrics = {}
    for name, m in masks.items():
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

    ys, xs = torch.where(core_region)
    y_center = int(ys.float().mean().item()) if len(ys) > 0 else 32

    # --- Plotting Comparativo a 9 Pannelli (3 Righe x 3 Colonne) ---
    fig = plt.figure(figsize=(18, 17), dpi=150)
    plt.suptitle(
        f"Famiglia B: Maschere Attention & Fusione Multi-Modale — {subject_name} ('{word}')",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Pannello 1 (R1, C1): Originale FLUX.1
    ax_img = plt.subplot(3, 3, 1)
    if orig_img is not None:
        ax_img.imshow(orig_img)
    ax_img.set_title("Originale FLUX.1", fontsize=12, fontweight="bold")
    ax_img.axhline(y_center * 16, color="cyan", linestyle="--", alpha=0.8, label=f"Sezione Y={y_center}")
    ax_img.legend(loc="upper right", fontsize=8)
    ax_img.axis("off")

    # Pannelli 2-8 (R1C2..C3, R2C1..C3, R3C1..C2): Le 7 Maschere
    for idx, (disp_name, _) in enumerate(FUSION_METHODS):
        sub_idx = idx + 2
        ax = plt.subplot(3, 3, sub_idx)
        im = ax.imshow(masks[disp_name].numpy(), cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_title(disp_name, fontsize=11, fontweight="bold")
        ax.axhline(y_center, color="cyan", linestyle="--", alpha=0.5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Pannello 9 (R3, C3): Profilo 1D della Sezione Orizzontale al Centro
    ax_slice = plt.subplot(3, 3, 9)
    x_axis = np.arange(64)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf", "#8c564b"]
    for (disp_name, _), color in zip(FUSION_METHODS, colors):
        slice_vals = masks[disp_name][y_center, :].numpy()
        linewidth = 2.5 if "Plateau" in disp_name else (2.0 if "Soft" in disp_name else 1.2)
        linestyle = "-" if "Core-Fused" in disp_name else "--"
        ax_slice.plot(x_axis, slice_vals, label=disp_name.split(". ")[-1], color=color, linewidth=linewidth, linestyle=linestyle)

    ax_slice.set_title(f"Profilo 1D Sezione Y={y_center} (Centro)", fontsize=11, fontweight="bold")
    ax_slice.set_xlabel("Coordinata X (Patch 64x64)")
    ax_slice.set_ylabel("Intensità Maschera M(x)")
    ax_slice.set_ylim(-0.05, 1.05)
    ax_slice.grid(True, linestyle=":", alpha=0.6)
    ax_slice.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_fig = output_dir / f"eval_fusion_{subject_name}.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Salvata figura 3x3 in: {out_fig}")
    return metrics


def main():
    data_root = PROJECT_ROOT / "data/dataset_v1"
    output_dir = PROJECT_ROOT / "outputs/eval_masks_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FLOWSTITCH FUSION MASK EVALUATION (3x3 SUITE CON ABLATION PLATEAU)")
    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")
    print("=" * 80)

    all_metrics = {}
    for subject_name, word in SUBJECTS:
        print(f"\n>>>> Valutazione Fusione: {subject_name} (token: '{word}') <<<<")
        m = evaluate_fusion(data_root, subject_name, word, output_dir)
        if m:
            all_metrics[subject_name] = m
            print(f"\n--- Metriche Fusione: {subject_name} ---")
            print(f"{"Metodo":<26} | {"Core Mean":<10} | {"Core Min":<10} | {"BG Leak":<10} | {"Zero %":<8} | {"Max Grad":<8}")
            print("-" * 84)
            for name, vals in m.items():
                print(f"{name:<26} | {vals["core_mean"]:<10.3f} | {vals["core_min"]:<10.3f} | {vals["bg_leak"]:<10.5f} | {vals["zero_pct"]:<7.1f}% | {vals["max_grad"]:<8.3f}")

    json_path = output_dir / "metrics_fusion.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[COMPLETATO] Tutte le metriche salvate in: {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

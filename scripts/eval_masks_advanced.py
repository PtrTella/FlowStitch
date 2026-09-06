#!/usr/bin/env python3
"""
FlowStitch Advanced Mask Benchmark: Hybrid vs. SAM-Flow vs. Kinetic-Semantic Hull
===================================================================================
Script di valutazione e benchmarking offline su Mac (senza GPU L40):
Confronta le 4 formulazioni chiave per l'isolamento semantico:
1. Pure Attention           : Baseline naturale FLUX.1 (Layer 10)
2. Hybrid (Binary Cut)      : Riferimento solido binario (Cap. 6)
3. SAM-Flow (Paper SOTA)    : Metodo ufficiale di SAM-Flow (arXiv:2606.06228)
4. Core-Fused (Plateau)     : Soft Trimap Bilaterale C^1 precedente
5. Kinetic-Semantic Hull    : Metodo Proposto Intelligente (Chiusura Morfologica + Coerenza Cinetica + Plateau C^1)

Genera:
- Griglie comparative multi-pannello ad alta risoluzione (2x4)
- Profilo 1D orizzontale (sezione centro geometrico)
- Profilo 1D verticale (sezione asse di simmetria: punta -> base)
- Metriche quantitative: Core Mean, Core Min, BG Leak, Zero %, Max Gradient, Tip Preservation
- Salvataggio in outputs/eval_masks_advanced/
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

BENCHMARK_METHODS = [
    ("1. Pure Attention", "attention"),
    ("2. Hybrid (Binary Cut)", "hybrid"),
    ("3. SAM-Flow (Paper)", "sam_flow"),
    ("4. Core-Fused Plateau", "core_fused"),
    ("5. Kinetic-Semantic Hull", "kinetic_hull"),
]


def evaluate_subject_advanced(data_root: Path, subject_name: str, word: str, output_dir: Path):
    subject_dir = data_root / subject_name
    if not subject_dir.exists():
        print(f"[ERRORE] Cartella non trovata: {subject_dir}")
        return None

    orig_img_path = subject_dir / "final_image.png"
    orig_img = Image.open(orig_img_path).convert("RGB") if orig_img_path.exists() else None

    masks = {}
    for disp_name, method_key in BENCHMARK_METHODS:
        m = compile_mask_for_scene(
            db_path=str(subject_dir),
            word_to_isolate=word,
            method=method_key,
        )
        masks[disp_name] = m.view(64, 64).float().cpu()

    # Riferimento oggettivo dalla Pure Attention
    m_base = masks["1. Pure Attention"]
    core_region = (m_base > 0.40)
    bg_region = (m_base < 0.10)

    # Identifica il baricentro e il punto apicale (tip)
    ys, xs = torch.where(core_region)
    if len(ys) > 0:
        y_center = int(ys.float().mean().item())
        x_center = int(xs.float().mean().item())
        y_tip = int(ys.min().item())
        x_tip = int(xs[ys == y_tip].float().mean().item())
    else:
        y_center, x_center = 32, 32
        y_tip, x_tip = 16, 32

    metrics = {}
    for name, m in masks.items():
        core_mean = m[core_region].mean().item() if core_region.any() else 0.0
        core_min = m[core_region].min().item() if core_region.any() else 0.0
        bg_leak = m[bg_region].mean().item() if bg_region.any() else 0.0
        zero_pct = (m == 0.0).float().mean().item() * 100.0

        # Misura la preservazione della punta / apice
        tip_val = m[y_tip, x_tip].item()

        # Lipschitz gradient metric: max |grad|
        dx = torch.abs(m[:, 1:] - m[:, :-1])
        dy = torch.abs(m[1:, :] - m[:-1, :])
        max_grad = max(dx.max().item(), dy.max().item())

        metrics[name] = {
            "core_mean": core_mean,
            "core_min": core_min,
            "bg_leak": bg_leak,
            "zero_pct": zero_pct,
            "max_grad": max_grad,
            "tip_preservation": tip_val,
        }

    # --- Plotting Comparativo 2x4 (8 Pannelli) ---
    fig = plt.figure(figsize=(22, 11), dpi=150)
    plt.suptitle(
        f"FlowStitch Advanced Benchmark: Hybrid vs. SAM-Flow vs. Kinetic Hull — {subject_name} ('{word}')",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    # Pannello 1: Originale FLUX.1
    ax_img = plt.subplot(2, 4, 1)
    if orig_img is not None:
        ax_img.imshow(orig_img)
    ax_img.set_title("Originale FLUX.1", fontsize=12, fontweight="bold")
    ax_img.axhline(y_center * 16, color="cyan", linestyle="--", alpha=0.8, label=f"Orizz. Y={y_center}")
    ax_img.axvline(x_center * 16, color="yellow", linestyle=":", alpha=0.8, label=f"Vert. X={x_center}")
    ax_img.scatter([x_tip * 16], [y_tip * 16], color="red", s=50, zorder=5, label=f"Apice ({x_tip},{y_tip})")
    ax_img.legend(loc="upper right", fontsize=8)
    ax_img.axis("off")

    # Pannelli 2-6: Le 5 Maschere
    panel_positions = [2, 3, 4, 5, 6]
    for idx, (disp_name, _) in enumerate(BENCHMARK_METHODS):
        ax = plt.subplot(2, 4, panel_positions[idx])
        m_t = masks[disp_name].numpy()
        cmap = "gray" if "Hybrid" in disp_name else "magma"
        im = ax.imshow(m_t, cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(disp_name, fontsize=12, fontweight="bold")
        ax.axhline(y_center, color="cyan", linestyle="--", alpha=0.4)
        ax.axvline(x_center, color="yellow", linestyle=":", alpha=0.4)
        ax.scatter([x_tip], [y_tip], color="red", s=30, zorder=5)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Pannello 7: Profilo 1D Orizzontale (Centro Y=y_center)
    ax_h = plt.subplot(2, 4, 7)
    x_axis = np.arange(64)
    colors = {
        "1. Pure Attention": "#7f7f7f",
        "2. Hybrid (Binary Cut)": "#2ca02c",
        "3. SAM-Flow (Paper)": "#d62728",
        "4. Core-Fused Plateau": "#1f77b4",
        "5. Kinetic-Semantic Hull": "#ff7f0e",
    }
    for disp_name, _ in BENCHMARK_METHODS:
        vals = masks[disp_name][y_center, :].numpy()
        lw = 2.6 if "Hull" in disp_name else (2.0 if "Plateau" in disp_name or "SAM" in disp_name else 1.4)
        ls = "-" if "Hull" in disp_name or "Hybrid" in disp_name else "--"
        ax_h.plot(x_axis, vals, label=disp_name.split(". ")[-1], color=colors[disp_name], linewidth=lw, linestyle=ls)

    ax_h.set_title(f"Profilo 1D Orizzontale (Y={y_center})", fontsize=11, fontweight="bold")
    ax_h.set_xlabel("Coordinata X (Patch)")
    ax_h.set_ylabel("M(x)")
    ax_h.set_ylim(-0.05, 1.05)
    ax_h.grid(True, linestyle=":", alpha=0.6)
    ax_h.legend(fontsize=7, loc="upper right")

    # Pannello 8: Profilo 1D Verticale (Asse Centrale X=x_center)
    ax_v = plt.subplot(2, 4, 8)
    y_axis = np.arange(64)
    for disp_name, _ in BENCHMARK_METHODS:
        vals = masks[disp_name][:, x_center].numpy()
        lw = 2.6 if "Hull" in disp_name else (2.0 if "Plateau" in disp_name or "SAM" in disp_name else 1.4)
        ls = "-" if "Hull" in disp_name or "Hybrid" in disp_name else "--"
        ax_v.plot(y_axis, vals, label=disp_name.split(". ")[-1], color=colors[disp_name], linewidth=lw, linestyle=ls)

    ax_v.set_title(f"Profilo 1D Verticale (X={x_center}) [Apice Y={y_tip}]", fontsize=11, fontweight="bold")
    ax_v.set_xlabel("Coordinata Y (Top -> Bottom)")
    ax_v.set_ylabel("M(y)")
    ax_v.set_ylim(-0.05, 1.05)
    ax_v.axvline(y_tip, color="red", linestyle=":", alpha=0.7, label=f"Apice (Y={y_tip})")
    ax_v.grid(True, linestyle=":", alpha=0.6)
    ax_v.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_fig = output_dir / f"eval_adv_{subject_name}.png"
    plt.savefig(out_fig, bbox_inches="tight")
    plt.close()

    print(f"[SUCCESS] Salvata figura 2x4 in: {out_fig}")
    return metrics


def main():
    data_root = PROJECT_ROOT / "data/dataset_v1"
    output_dir = PROJECT_ROOT / "outputs/eval_masks_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FLOWSTITCH ADVANCED MASK BENCHMARK: HYBRID vs. SAM-FLOW vs. KINETIC HULL")
    print(f"Data root:   {data_root}")
    print(f"Output dir:  {output_dir}")
    print("=" * 90)

    all_metrics = {}
    for subject_name, word in SUBJECTS:
        print(f"\n>>>> Benchmark Avanzato: {subject_name} (token: '{word}') <<<<")
        m = evaluate_subject_advanced(data_root, subject_name, word, output_dir)
        if m:
            all_metrics[subject_name] = m
            print(f"\n--- Metriche: {subject_name} ---")
            header = f"{'Metodo':<26} | {'Core Mean':<9} | {'Core Min':<9} | {'Apice (Tip)':<11} | {'BG Leak':<9} | {'Zero %':<7} | {'Max Grad':<8}"
            print(header)
            print("-" * len(header))
            for name, vals in m.items():
                print(
                    f"{name:<26} | {vals['core_mean']:<9.3f} | {vals['core_min']:<9.3f} | {vals['tip_preservation']:<11.3f} | {vals['bg_leak']:<9.5f} | {vals['zero_pct']:<6.1f}% | {vals['max_grad']:<8.3f}"
                )

    json_path = output_dir / "metrics_advanced.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 90)
    print(f"[COMPLETATO] Tutte le metriche salvate in: {json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()

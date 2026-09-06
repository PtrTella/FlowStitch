import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from transformers import T5Tokenizer

from flowstitch.core.serialization import load_tensors
from flowstitch.core.tokenizer_utils import find_token_indices
from flowstitch.extraction.attention_mask import extract_attention_mask

def smoothstep(edge0, edge1, x):
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def evaluate_sample_masks(data_dir: Path, target_word: str):
    with open(data_dir / "metadata.json", "r") as f:
        prompt = json.load(f)["prompt"]

    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True)
    t_idx = find_token_indices(tokenizer, prompt, target_word)

    # 1. Attention Map
    attn_dict = load_tensors(str(data_dir / "attention_maps.pt"))
    layer_key = sorted([k for k in attn_dict.keys() if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))[-1]
    attn = extract_attention_mask(attn_dict[layer_key], t_idx, normalize=True).view(64, 64).float()

    # 2. Velocity field & Cosine Alignment
    v0 = load_tensors(str(data_dir / "v0_velocity.pt"))
    if isinstance(v0, dict): v0 = list(v0.values())[0]
    v0_normed = F.normalize(v0.view(4096, -1).float(), dim=-1)

    # Reference target vector: attention-weighted mean velocity
    mean_v = F.normalize((v0_normed * attn.view(4096, 1)).sum(dim=0, keepdim=True), dim=-1)
    cos_raw = (v0_normed @ mean_v.T).view(64, 64)
    cos_pos = torch.clamp(cos_raw, min=0.0, max=1.0)

    # --- 6 Mathematical Formulations ---

    # F1. Prodotto Diretto (Baseline Fuzzy AND)
    m_prod = attn * cos_pos
    m_prod = m_prod / (m_prod.max() + 1e-8)

    # F2. Soft-Calibrated Linear Clamp (Nostra proposta iniziale)
    tau_bg = 0.12
    m_calib_lin = torch.clamp((attn * cos_pos - tau_bg) / (1.0 - tau_bg), min=0.0, max=1.0)
    if m_calib_lin.max() > 0: m_calib_lin /= m_calib_lin.max()

    # F3. Smoothstep Hermite C^1 (Regolarita differenziale per Picard-Lindelof)
    # Transizione morbida con derivata prima nulla a inizio e fine
    m_smooth = smoothstep(0.10, 0.65, attn * cos_pos)
    if m_smooth.max() > 0: m_smooth /= m_smooth.max()

    # F4. Cosine-Sharpened Power Law (Attn * Cos^2 con calibrazione)
    prod_sharp = attn * (cos_pos ** 2)
    m_sharp = torch.clamp((prod_sharp - 0.08) / (1.0 - 0.08), min=0.0, max=1.0)
    if m_sharp.max() > 0: m_sharp /= m_sharp.max()

    # F5. Bilateral Sigmoidal Gate (Alpha-Gated continuo)
    tau_gate = 0.50
    gate = torch.sigmoid(10.0 * (cos_raw - tau_gate))
    m_gated = attn * gate
    m_gated = torch.clamp((m_gated - 0.05) / 0.95, min=0.0, max=1.0)
    if m_gated.max() > 0: m_gated /= m_gated.max()

    # F6. Geometric Mean Matte (sqrt(Attn * Cos_pos))
    m_geom = torch.sqrt(torch.clamp(attn * cos_pos, min=0.0))
    m_geom = torch.clamp((m_geom - 0.25) / 0.75, min=0.0, max=1.0)
    if m_geom.max() > 0: m_geom /= m_geom.max()

    ref_img = Image.open(data_dir / "final_image.png").convert("RGB") if (data_dir / "final_image.png").exists() else None

    masks = {
        "1. Prodotto Diretto (A * C)": m_prod,
        "2. Soft Calibrated Linear": m_calib_lin,
        "3. Hermite Smoothstep (C^1)": m_smooth,
        "4. Cosine-Sharpened (A * C^2)": m_sharp,
        "5. Bilateral Sigmoidal Gate": m_gated,
        "6. Geometric Mean Matte": m_geom,
    }

    # --- Quantitative Metrics ---
    core_region = attn > 0.40
    bg_region = attn < 0.10

    metrics = {}
    for name, m in masks.items():
        bg_mean = m[bg_region].mean().item() if bg_region.any() else 0.0
        core_mean = m[core_region].mean().item() if core_region.any() else 0.0
        zero_pct = (m == 0.0).float().mean().item() * 100.0
        dx = torch.abs(m[:, 1:] - m[:, :-1])
        dy = torch.abs(m[1:, :] - m[:-1, :])
        max_grad = max(dx.max().item(), dy.max().item())
        mean_grad = 0.5 * (dx.mean().item() + dy.mean().item())

        metrics[name] = {
            "bg_leakage": bg_mean,
            "core_density": core_mean,
            "zero_sparsity_pct": zero_pct,
            "max_step_grad": max_grad,
            "mean_grad": mean_grad,
        }

    return {
        "ref_img": ref_img,
        "attn": attn,
        "cos_raw": cos_raw,
        "masks": masks,
        "metrics": metrics,
    }

def run_experiment(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ("a_blue_sphere", "sphere"),
        ("a_red_cube", "cube"),
        ("a_yellow_pyramid", "pyramid"),
    ]

    all_metrics = {}

    for sample_name, target_word in samples:
        data_dir = Path("data/dataset_v1") / sample_name
        res = evaluate_sample_masks(data_dir, target_word)
        all_metrics[sample_name] = res["metrics"]

        # Plot 2x4 Grid
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        
        # Row 1: FLUX ref, Attention, Cosine, F1
        axes[0, 0].imshow(res["ref_img"])
        axes[0, 0].set_title("Originale FLUX", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        im1 = axes[0, 1].imshow(res["attn"].cpu().numpy(), cmap="magma")
        axes[0, 1].set_title("Cross-Attention Pura A(x)", fontsize=11, fontweight="bold")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        axes[0, 1].axis("off")

        im2 = axes[0, 2].imshow(res["cos_raw"].cpu().numpy(), cmap="coolwarm")
        axes[0, 2].set_title("Cosine Direction C(x)", fontsize=11, fontweight="bold")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        axes[0, 2].axis("off")

        im3 = axes[0, 3].imshow(res["masks"]["1. Prodotto Diretto (A * C)"].cpu().numpy(), cmap="magma")
        axes[0, 3].set_title("1. Prodotto Diretto (A * C)", fontsize=11, fontweight="bold")
        plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)
        axes[0, 3].axis("off")

        # Row 2: F2, F3, F4, F5
        row2_names = [
            "2. Soft Calibrated Linear",
            "3. Hermite Smoothstep (C^1)",
            "4. Cosine-Sharpened (A * C^2)",
            "5. Bilateral Sigmoidal Gate",
        ]
        for col_idx, m_name in enumerate(row2_names):
            ax = axes[1, col_idx]
            im = ax.imshow(res["masks"][m_name].cpu().numpy(), cmap="magma")
            ax.set_title(m_name, fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis("off")

        plt.suptitle(f"Studio Formulazioni Attention-Cosine Fusion: '{sample_name}'", fontsize=14, fontweight="bold")
        plt.tight_layout()
        save_path = output_dir / f"fusion_suite_{sample_name}.png"
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Salvato plot: {save_path}")

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Salvato riassunto metriche in: {summary_path}")

    # Markdown Table
    print("\n" + "="*85)
    print(f"{'Formulazione':<32} | {'BG Leakage':<12} | {'Core Density':<12} | {'Zero %':<10} | {'Max Step Grad':<12}")
    print("="*85)
    
    method_names = list(list(all_metrics.values())[0].keys())
    for m in method_names:
        avg_bg = np.mean([all_metrics[s][m]["bg_leakage"] for s in all_metrics])
        avg_core = np.mean([all_metrics[s][m]["core_density"] for s in all_metrics])
        avg_zero = np.mean([all_metrics[s][m]["zero_sparsity_pct"] for s in all_metrics])
        avg_grad = np.mean([all_metrics[s][m]["max_step_grad"] for s in all_metrics])
        print(f"{m:<32} | {avg_bg:<12.4f} | {avg_core:<12.4f} | {avg_zero:<9.1f}% | {avg_grad:<12.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    out_dir = Path("outputs/exp_attention_cosine")
    run_experiment(out_dir)

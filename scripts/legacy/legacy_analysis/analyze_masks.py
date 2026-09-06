"""
FlowStitch PoC Step 1: Systematic Mask & Velocity Field Analysis
================================================================
Compares all semantic extraction methods on actual dataset samples:
- Raw reference image
- Raw Cross-Attention (Layer 10, target token)
- Otsu Binarization
- Kinetic Energy ||v_0|| (Demonstrating the Thermodynamic Void / Vuoto Termodinamico)
- Cosine Direction Similarity (Semantic direction alignment with object centroid)
- Fiedler Vector (Spectral Graph Cut)
- Hybrid (Fiedler + Attention compass)
- Soft Calibrated Attention (Continuous C-inf, zero outside, unit peak)
"""

import os
import json
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

from flowstitch.core.serialization import load_tensors
from flowstitch.core.tokenizer_utils import find_token_indices
from flowstitch.extraction.attention_mask import extract_attention_mask, otsu_threshold
from flowstitch.extraction.spectral_mask import compute_fiedler_mask
from flowstitch.extraction.energy_mask import compute_chebyshev_threshold
from flowstitch.extraction.hybrid_mask import hybrid_semantic_decomposition
from transformers import T5Tokenizer


def analyze_scene(db_path: Path, word_to_isolate: str, output_path: Path):
    print(f"\n{'='*70}\nAnalyzing: {db_path.name} ('{word_to_isolate}')\n{'='*70}")

    # 1. Carica metadati e immagine di riferimento
    with open(db_path / "metadata.json", "r") as f:
        meta = json.load(f)
    prompt = meta["prompt"]
    print(f"Prompt: '{prompt}'")

    ref_img_path = db_path / "final_image.png"
    ref_img = Image.open(ref_img_path).convert("RGB") if ref_img_path.exists() else None

    # 2. Tokenizer per trovare gli indici
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True)
    token_indices = find_token_indices(tokenizer, prompt, word_to_isolate)
    print(f"Token indices for '{word_to_isolate}': {token_indices}")

    # 3. Carica attention_maps e v0
    attn_dict = load_tensors(str(db_path / "attention_maps.pt"))
    layer_key = sorted([k for k in attn_dict.keys() if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))[-1]
    layer_10 = attn_dict[layer_key]

    v0 = load_tensors(str(db_path / "v0_velocity.pt"))
    if isinstance(v0, dict):
        v0 = list(v0.values())[0]

    # --- Calcolo di tutte le rappresentazioni ---

    # A. Cross-Attention pura normalizzata [0, 1]
    attn_raw = extract_attention_mask(layer_10, token_indices, normalize=True)
    attn_2d = attn_raw.view(64, 64).float().cpu().numpy()

    # B. Otsu Binarization (Hard cut)
    otsu_mask = otsu_threshold(attn_raw).view(64, 64).float().cpu().numpy()

    # C. Energia cinetica ||v_0||
    energy_2d = torch.norm(v0.float(), p=2, dim=-1).view(64, 64)
    energy_norm = ((energy_2d - energy_2d.min()) / (energy_2d.max() - energy_2d.min() + 1e-8)).float().cpu().numpy()

    # D. Chebyshev Energy Gating (Soglia statistica mu + sigma)
    cheb_mask = compute_chebyshev_threshold(torch.norm(v0.float(), p=2, dim=-1, keepdim=True), k=1.0).view(64, 64).float().cpu().numpy()

    # E. Cosine Direction Similarity rispetto al baricentro semantico
    # Troviamo il baricentro dell'oggetto tramite l'attenzione
    v0_flat = v0.view(4096, -1).to(torch.float32)  # [4096, dim]
    v0_normed = F.normalize(v0_flat, dim=-1)
    weights = attn_raw.view(4096, 1).to(torch.float32)
    mean_target_vector = F.normalize((v0_normed * weights).sum(dim=0, keepdim=True), dim=-1)  # [1, dim]
    cosine_sim = (v0_normed @ mean_target_vector.T).view(64, 64)  # [-1, 1]
    cosine_sim_norm = ((cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min() + 1e-8)).float().cpu().numpy()

    # F. Vettore di Fiedler (Partizionamento Spettrale)
    fiedler = compute_fiedler_mask(v0, target_resolution=32).view(64, 64).float().cpu().numpy()

    # G. Ibrido (Fiedler + Attention)
    hybrid = hybrid_semantic_decomposition(v0, attn_raw, target_resolution=32).view(64, 64).float().cpu().numpy()

    # H. Soft Calibrated Attention (La nostra proposta pulita: zero background, morbida, picco 1.0)
    tau = 0.15
    calibrated = torch.clamp((attn_raw.float() - tau) / (1.0 - tau), min=0.0, max=1.0)
    calibrated_2d = calibrated.view(64, 64).float().cpu().numpy()

    # --- Plotting Comparativo 2x4 ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    panels = [
        ("1. Originale FLUX (Dataset)", ref_img, "image"),
        ("2. Cross-Attention Pura (Layer 10)", attn_2d, "magma"),
        ("3. Otsu Threshold (Hard Cut)", otsu_mask, "gray"),
        ("4. Energia Cinetica ||v_0|| (Vuoto Termodinamico)", energy_norm, "plasma"),
        ("5. Cosine Direction Similarity", cosine_sim_norm, "coolwarm"),
        ("6. Vettore di Fiedler (Spettrale 32x32)", fiedler, "gray"),
        ("7. Hybrid (Fiedler + Attn)", hybrid, "gray"),
        ("8. Soft Calibrated Attention (Clean C-inf)", calibrated_2d, "magma"),
    ]

    for ax, (title, data, cmap) in zip(axes.flat, panels):
        if cmap == "image":
            if data is not None:
                ax.imshow(data)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        else:
            im = ax.imshow(data, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.suptitle(
        f"FlowStitch PoC Step 1 — Studio Comparativo delle Maschere: '{word_to_isolate}' in '{db_path.name}'",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs" / "poc_masks"
    
    # 1. Sfera Blu
    db_sphere = root / "data" / "dataset_v1" / "a_blue_sphere"
    if db_sphere.exists():
        analyze_scene(db_sphere, "sphere", out_dir / "poc_step1_sphere_masks.png")
        
    # 2. Cubo Rosso
    db_cube = root / "data" / "dataset_v1" / "a_red_cube"
    if db_cube.exists():
        analyze_scene(db_cube, "cube", out_dir / "poc_step1_cube_masks.png")

    # 3. Piramide Gialla
    db_pyramid = root / "data" / "dataset_v1" / "a_yellow_pyramid"
    if db_pyramid.exists():
        analyze_scene(db_pyramid, "pyramid", out_dir / "poc_step1_pyramid_masks.png")

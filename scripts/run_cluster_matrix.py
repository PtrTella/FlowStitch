#!/usr/bin/env python3
"""
FlowStitch Cluster Matrix Execution Script.

Esegue la matrice sperimentale completa per la tesi:
- 4 Maschere: pure_attention, calibrated_attention, fused_linear, hermite
- 3 Comportamenti del Rumore Iniziale x0: spherical_weld, hard_mosaic, zero_inpainting
- 2 Regimi di Damping Velocità D(t): kts, constant

Progettato sia per essere importato dal master runner (scripts/main.py),
sia per girare standalone.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from diffusers import FluxPipeline

from flowstitch.core.cluster_utils import setup_cluster_network, load_flux_pipeline
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.mask_compilation import compile_mask_for_scene
from flowstitch.pipelines.latent_stitching import run_latent_stitching

# Inizializza automaticamente la configurazione di rete se su cluster
setup_cluster_network()

logger = logging.getLogger("cluster_matrix")

DEFAULT_SUBJECTS = [
    ("a_blue_sphere", "sphere"),
    ("a_red_cube", "cube"),
    ("a_yellow_pyramid", "pyramid"),
]

MASK_METHODS = {
    "kinetic_hull": "kinetic_hull",
    "sam_flow": "sam_flow",
    "core_fused": "core_fused",
    "core_fused_soft": "core_fused_soft",
    "hermite": "hermite",
    "calibrated_attention": "calibrated_attention",
    "pure_attention": "attention",
    "fused_linear": "fused_linear",
    "bilateral_sigmoid": "bilateral_sigmoid",
    "otsu": "otsu",
    "chebyshev": "chebyshev",
    "spectral": "spectral",
    "hybrid": "hybrid",
    "tda": "tda",
}

DEFAULT_MASKS = [
    "core_fused",          # Default Canonico (Plateau Trimap M=1 nel core)
    "core_fused_soft",     # Ablation (Soft Trimap senza plateau)
    "hermite",             # Hermite Product C^1 (mostra il crollo/dropout sui riflessi)
    "calibrated_attention",# Attention con piedistallo sottratto
    "pure_attention",      # Baseline grezza
]

X0_MODES = ["spherical_weld", "hard_mosaic", "zero_inpainting"]
DAMPING_MODES = ["kts", "constant"]


def create_comparison_grid(
    subject_dir: Path,
    subject_name: str,
    masks: list[str],
    x0_modes: list[str],
    damping_modes: list[str],
    output_path: Path,
):
    """Crea una griglia visuale con tutti i risultati della matrice per il soggetto."""
    try:
        images = {}
        for x0 in x0_modes:
            for damp in damping_modes:
                for mask in masks:
                    img_file = subject_dir / f"{mask}_{x0}_{damp}.png"
                    if img_file.exists():
                        images[(mask, x0, damp)] = Image.open(img_file).resize((256, 256))

        if not images:
            return

        rows = len(x0_modes) * len(damping_modes)
        cols = len(masks)
        
        cell_size = 256
        header_height = 60
        left_label_width = 180
        
        grid_w = left_label_width + cols * cell_size
        grid_h = header_height + rows * cell_size
        
        grid_img = Image.new("RGB", (grid_w, grid_h), color=(245, 245, 245))
        draw = ImageDraw.Draw(grid_img)
        
        # Column headers (Masks)
        for c_idx, mask in enumerate(masks):
            x = left_label_width + c_idx * cell_size + 10
            y = 20
            draw.text((x, y), mask, fill=(0, 0, 0))
            
        # Rows (x0_mode + damping)
        row_labels = []
        for x0 in x0_modes:
            for damp in damping_modes:
                row_labels.append((x0, damp))
                
        for r_idx, (x0, damp) in enumerate(row_labels):
            y_pos = header_height + r_idx * cell_size
            label = f"x0: {x0}\nKTS: {damp}"
            draw.text((10, y_pos + 100), label, fill=(0, 0, 0))
            
            for c_idx, mask in enumerate(masks):
                x_pos = left_label_width + c_idx * cell_size
                img = images.get((mask, x0, damp))
                if img is not None:
                    grid_img.paste(img, (x_pos, y_pos))
                draw.rectangle([x_pos, y_pos, x_pos + cell_size, y_pos + cell_size], outline=(200, 200, 200))
                
        grid_img.save(output_path)
        logger.info(f"[GRID] Salvata griglia comparativa in: {output_path}")
    except Exception as e:
        logger.warning(f"Impossibile generare la griglia per {subject_name}: {e}")


def run_matrix(
    pipe,
    output_dir: str,
    project_root: Path = None,
    data_root: str = None,
    subjects: list[str] = None,
    masks: list[str] = None,
    x0_modes: list[str] = None,
    damping_modes: list[str] = None,
    ambient_prompt: str = "a crystal clear lake, calm water surface, highly detailed",
    steps: int = 4,
    seed: int = 42,
    create_grids: bool = True,
):
    """
    Esegue la matrice sperimentale riusando pipe già caricata in VRAM.
    Compatibile con l'architettura dei moduli scripts/exp*.py.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    data_dir = Path(data_root) if data_root else project_root / "data/dataset_v1"
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Risolvi soggetti
    all_subjects = DEFAULT_SUBJECTS
    if subjects:
        matched = [s for s in DEFAULT_SUBJECTS if s[0] in subjects]
        all_subjects = matched if matched else [(s, s.split("_")[-1]) for s in subjects]

    active_masks = masks or list(MASK_METHODS.keys())
    active_x0 = x0_modes or X0_MODES
    active_damping = damping_modes or DAMPING_MODES

    logger.info("=" * 70)
    logger.info("ESPERIMENTO MATRIX: 4 Maschere x 3 x0 x 2 Damping")
    logger.info(f"Soggetti: {[s[0] for s in all_subjects]}")
    logger.info(f"Maschere: {active_masks}")
    logger.info(f"x0 Modes: {active_x0}")
    logger.info(f"Damping:  {active_damping}")
    total_runs = len(all_subjects) * len(active_masks) * len(active_x0) * len(active_damping)
    logger.info(f"Totale Generazioni Previste: {total_runs}")
    logger.info("=" * 70)

    device = pipe.device
    dtype = pipe.transformer.dtype

    run_counter = 0
    t0_matrix = time.time()

    for folder_name, target_word in all_subjects:
        subject_data_dir = data_dir / folder_name
        if not subject_data_dir.exists():
            logger.error(f"Dati non trovati per '{folder_name}' in {subject_data_dir}. Skip.")
            continue

        subject_out_dir = out_root / folder_name
        subject_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n>>>> INIZIO SOGGETTO: {folder_name} (token: '{target_word}') <<<<")

        for mask_name in active_masks:
            pipeline_method = MASK_METHODS[mask_name]
            logger.info(f"Compilazione maschera '{mask_name}' ({pipeline_method})...")
            compile_mask_for_scene(
                db_path=str(subject_data_dir),
                word_to_isolate=target_word,
                method=pipeline_method,
            )

            for x0 in active_x0:
                for damp in active_damping:
                    run_counter += 1
                    out_filename = f"{mask_name}_{x0}_{damp}.png"
                    out_path = subject_out_dir / out_filename

                    if out_path.exists():
                        logger.info(f"[{run_counter}/{total_runs}] Skip (già presente): {out_filename}")
                        continue

                    logger.info(f"[{run_counter}/{total_runs}] Esecuzione: Mask={mask_name} | x0={x0} | Damp={damp}")
                    t0_run = time.time()

                    cfg = FlowStitchConfig(
                        model_id=pipe.config._name_or_path if hasattr(pipe, "config") else "black-forest-labs/FLUX.1-schnell",
                        device=str(device),
                        dtype=dtype,
                        steps=steps,
                        seed=seed,
                        ambient_prompt=ambient_prompt,
                        stitching_mode="dual",
                        x0_mode=x0,
                        damping_mode=damp,
                        use_attention_grafting=False,
                        output_root=str(subject_out_dir),
                        output_filename=out_filename,
                    )

                    run_latent_stitching(config=cfg, db_path=str(subject_data_dir), pipe=pipe)
                    logger.info(f"Completato in {time.time() - t0_run:.2f}s -> {out_filename}")

        if create_grids:
            grid_out = subject_out_dir / "matrix_comparison_grid.png"
            create_comparison_grid(
                subject_dir=subject_out_dir,
                subject_name=folder_name,
                masks=active_masks,
                x0_modes=active_x0,
                damping_modes=active_damping,
                output_path=grid_out,
            )

    elapsed_total = time.time() - t0_matrix
    logger.info("=" * 70)
    logger.info(f"[SUCCESS] Matrice completata in {elapsed_total / 60.0:.2f} minuti!")
    logger.info(f"Risultati in: {out_root}")
    logger.info("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Esegui la matrice sperimentale FlowStitch")
    parser.add_argument("--data_root", type=str, default="data/dataset_v1")
    parser.add_argument("--output_root", type=str, default="outputs/cluster_matrix")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--ambient_prompt", type=str, default="a crystal clear lake, calm water surface, highly detailed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--masks", nargs="+", choices=list(MASK_METHODS.keys()), default=DEFAULT_MASKS)
    parser.add_argument("--x0_modes", nargs="+", choices=X0_MODES, default=X0_MODES)
    parser.add_argument("--damping_modes", nargs="+", choices=DAMPING_MODES, default=DAMPING_MODES)
    parser.add_argument("--create_grids", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    pipe = load_flux_pipeline(model_id=args.model_id, device=args.device)

    run_matrix(
        pipe=pipe,
        output_dir=args.output_root,
        data_root=args.data_root,
        subjects=args.subjects,
        masks=args.masks,
        x0_modes=args.x0_modes,
        damping_modes=args.damping_modes,
        ambient_prompt=args.ambient_prompt,
        steps=args.steps,
        seed=args.seed,
        create_grids=args.create_grids,
    )


if __name__ == "__main__":
    main()

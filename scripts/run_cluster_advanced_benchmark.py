#!/usr/bin/env python3
"""
FlowStitch Cluster Advanced Benchmark Runner
==============================================
Esegue lo stitching e la generazione FLUX.1 sul cluster Giano per il quartetto di confronto:
1. hybrid       : Riferimento binario spettrale Fiedler (Cap. 6)
2. sam_flow     : SOTA del paper SAM-Flow (arXiv:2606.06228)
3. core_fused   : Soft Trimap Bilaterale C^1 precedente
4. kinetic_hull : Metodo Proposto Intelligente (Chiusura Morfologica + Coerenza Cinetica + Plateau C^1)

Combina ciascuna maschera con i 2 regimi di stitching più stabili e puliti:
- (spherical_weld, kts)
- (spherical_weld, constant)
Generando una griglia comparativa 4x2 ad alta definizione per ciascun soggetto.
"""
from __future__ import annotations
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flowstitch.core.cluster_utils import setup_cluster_network, load_flux_pipeline
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.mask_compilation import compile_mask_for_scene
from flowstitch.pipelines.latent_stitching import run_latent_stitching

setup_cluster_network()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cluster_advanced_benchmark")

BENCHMARK_MASKS = [
    ("1_hybrid", "hybrid"),
    ("2_sam_flow", "sam_flow"),
    ("3_core_fused", "core_fused"),
    ("4_kinetic_hull", "kinetic_hull"),
]

BENCHMARK_CONFIGS = [
    ("spherical_weld", "kts"),
    ("spherical_weld", "constant"),
]

SUBJECTS = [
    ("a_blue_sphere", "sphere"),
    ("a_red_cube", "cube"),
    ("a_yellow_pyramid", "pyramid"),
]


def create_benchmark_grid(
    subject_dir: Path,
    subject_name: str,
    output_path: Path,
):
    """Crea una griglia visuale comparativa 4 Colonne x 2 Righe."""
    try:
        images = {}
        for mask_label, mask_key in BENCHMARK_MASKS:
            for x0, damp in BENCHMARK_CONFIGS:
                img_file = subject_dir / f"{mask_key}_{x0}_{damp}.png"
                if img_file.exists():
                    images[(mask_key, x0, damp)] = Image.open(img_file).resize((384, 384))

        if not images:
            return

        rows = len(BENCHMARK_CONFIGS)
        cols = len(BENCHMARK_MASKS)
        cell_w, cell_h = 384, 384
        header_h = 70
        label_w = 190

        grid_w = label_w + cols * cell_w
        grid_h = header_h + rows * cell_h

        grid_img = Image.new("RGB", (grid_w, grid_h), (20, 20, 25))
        draw = ImageDraw.Draw(grid_img)

        # Header colonne (Maschere)
        for c_idx, (disp_label, mask_key) in enumerate(BENCHMARK_MASKS):
            x = label_w + c_idx * cell_w + cell_w // 2
            draw.text((x, 25), disp_label, fill=(255, 255, 255), anchor="mm")

        # Righe (Configurazioni ODE)
        for r_idx, (x0, damp) in enumerate(BENCHMARK_CONFIGS):
            y = header_h + r_idx * cell_h + cell_h // 2
            row_label = f"x0: {x0}\nD(t): {damp}"
            draw.text((label_w // 2, y), row_label, fill=(200, 200, 200), anchor="mm")

            for c_idx, (_, mask_key) in enumerate(BENCHMARK_MASKS):
                key = (mask_key, x0, damp)
                if key in images:
                    px = label_w + c_idx * cell_w
                    py = header_h + r_idx * cell_h
                    grid_img.paste(images[key], (px, py))

        grid_img.save(output_path, quality=95)
        logger.info(f"[GRID] Griglia benchmark salvata in: {output_path}")

    except Exception as e:
        logger.error(f"Errore generazione griglia per {subject_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="FlowStitch Cluster Advanced Benchmark Runner")
    parser.add_argument("--data-root", type=str, default="data/dataset_v1")
    parser.add_argument("--ambient", type=str, default="a_crystal_clear_lake")
    parser.add_argument("--output-dir", type=str, default="outputs/cluster_advanced_benchmark")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ambient_dir = data_root / args.ambient
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    if not ambient_dir.exists():
        logger.error(f"Cartella sfondo non trovata: {ambient_dir}")
        sys.exit(1)

    ambient_meta = ambient_dir / "metadata.json"
    if ambient_meta.exists():
        with open(ambient_meta) as f:
            ambient_prompt = json.load(f).get("prompt", "a crystal clear lake")
    else:
        ambient_prompt = "a crystal clear lake"

    logger.info("=" * 80)
    logger.info("FLOWSTITCH CLUSTER ADVANCED BENCHMARK")
    logger.info(f"Target subjects: {[s[0] for s in SUBJECTS]}")
    logger.info(f"Maschere a confronto: {[m[1] for m in BENCHMARK_MASKS]}")
    logger.info(f"Configurazioni ODE: {BENCHMARK_CONFIGS}")
    logger.info("=" * 80)

    # 1. Caricamento Modello FLUX.1
    pipe = load_flux_pipeline()

    # 2. Loop di Esecuzione
    total_runs = len(SUBJECTS) * len(BENCHMARK_MASKS) * len(BENCHMARK_CONFIGS)
    run_idx = 0
    start_time = time.time()

    for subject_name, word in SUBJECTS:
        subject_dir = data_root / subject_name
        subj_out_dir = output_base / subject_name
        subj_out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n>>>> Inizio Benchmark per: {subject_name} (token: '{word}') <<<<")

        for _, mask_method in BENCHMARK_MASKS:
            # Compila e salva la maschera A_target.pt
            logger.info(f"--- Compilazione maschera [{mask_method}] per {subject_name} ---")
            compile_mask_for_scene(
                db_path=str(subject_dir),
                word_to_isolate=word,
                method=mask_method,
            )

            for x0_mode, damping in BENCHMARK_CONFIGS:
                run_idx += 1
                logger.info(
                    f"[{run_idx}/{total_runs}] Esecuzione Stitch: mask={mask_method} | x0={x0_mode} | D(t)={damping}"
                )

                img_name = f"{mask_method}_{x0_mode}_{damping}.png"
                config = FlowStitchConfig(
                    model_id=pipe.config._name_or_path if hasattr(pipe, "config") else "black-forest-labs/FLUX.1-schnell",
                    device=str(pipe.device),
                    dtype=pipe.transformer.dtype,
                    steps=args.steps,
                    ambient_prompt=ambient_prompt,
                    stitching_mode="dual",
                    x0_mode=x0_mode,
                    damping_mode=damping,
                    use_attention_grafting=False,
                    output_root=str(subj_out_dir),
                    output_filename=img_name,
                )

                # Esegui stitching
                out_img = run_latent_stitching(config=config, db_path=str(subject_dir), pipe=pipe)

        # Genera griglia per il soggetto
        grid_file = subj_out_dir / f"benchmark_grid_{subject_name}.png"
        create_benchmark_grid(subj_out_dir, subject_name, grid_file)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info(f"[COMPLETATO] Benchmark cluster terminato in {elapsed / 60:.2f} minuti!")
    logger.info(f"Tutte le immagini e le griglie salvate in: {output_base}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

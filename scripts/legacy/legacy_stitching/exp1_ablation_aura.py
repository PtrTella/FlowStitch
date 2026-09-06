"""
FlowStitch Experiment 1: Ablation Study (Hard Masking vs Continuous Soft Aura)
=============================================================================
Confronto sperimentale tra taglio netto a gradino (mosaico discontinuo)
e la formulazione Continuous Soft Aura con Blending Sferico.

Output:
- exp1_hard_mosaico.png      -> Evidenzia i blocchi a scacchiera (patch artifacts)
- exp1_continuous_aura.png   -> Risolve la continuita' C-infinito eliminando i difetti
"""

import logging
from pathlib import Path
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.latent_stitching import run_latent_stitching
from flowstitch.pipelines.mask_compilation import compile_mask_for_scene

logger = logging.getLogger("Exp1Ablation")


def run_ablation(pipe, output_dir: str, project_root: Path = None):
    """Esegue l'ablation tra hard cut e continuous soft aura riusando pipe in VRAM."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    db_sphere = str(project_root / "data/dataset_v1/a_blue_sphere")

    logger.info("=" * 70)
    logger.info("ESPERIMENTO 1: Ablation Study (Hard Cut vs Continuous Soft Aura)")
    logger.info(f"Target DB: {db_sphere}")
    logger.info("=" * 70)

    # 1. Compila la maschera ibrida di riferimento
    logger.info("Step 1: Compilazione maschera ibrida...")
    compile_mask_for_scene(db_path=db_sphere, word_to_isolate="sphere", method="hybrid")

    # 1.A: Baseline con Hard Cut (Mosaico Lineare Discontinuo)
    logger.info("Step 2: Generazione Baseline Hard Cut (Mosaico)...")
    config_hard = FlowStitchConfig(
        model_id="black-forest-labs/FLUX.1-schnell",
        stitching_mode="mosaico",
        ambient_prompt="a crystal clear lake",
        output_root=output_dir,
        subject_scale=1.0,
        output_filename="exp1_hard_mosaico.png",
    )
    run_latent_stitching(config_hard, db_sphere, pipe=pipe)

    # 1.B: Nostro Metodo con Continuous Soft Aura e Blending Sferico
    logger.info("Step 3: Generazione FlowStitch Continuous Soft Aura...")
    config_soft = FlowStitchConfig(
        model_id="black-forest-labs/FLUX.1-schnell",
        stitching_mode="dual",
        ambient_prompt="a crystal clear lake",
        output_root=output_dir,
        subject_scale=1.0,
        t_cutoff=0.5,
        gamma_kts=4.0,
        output_filename="exp1_continuous_aura.png",
    )
    run_latent_stitching(config_soft, db_sphere, pipe=pipe)

    logger.info("[EXP 1 COMPLETATO] Salvate: exp1_hard_mosaico.png e exp1_continuous_aura.png\n")


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).parent / "main.py"), "--exp", "ablation", *sys.argv[1:]])

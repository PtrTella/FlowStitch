"""
FlowStitch Experiment 3: Geometric Invariance (Polyhedral Shapes — Red Cube)
=============================================================================
Ipotesi Scientifica:
Mentre una sfera possiede simmetria radiale e curvatura gaussiana positiva costante,
un cubo solido presenta facce piane a curvatura nulla raccordate da discontinuita'
ortogonali (spigoli vivi a 90 gradi).

Questo esperimento dimostra la capacita' del framework di preservare geometrie
non radiali. La combinazione di maschera spettrale di Fiedler, proiezione continua
e KTS mantiene la planarita' delle facce e la nitidezza degli spigoli del cubo
senza introdurre deformazioni sferiche spurie o bleeding sul fondo acquatico.

Output generato:
- exp3_geometric_cube_routed.png -> Cubo rosso proporzionato, posato sulla superficie del lago
"""

import logging
from pathlib import Path
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.latent_stitching import run_latent_stitching
from flowstitch.pipelines.mask_compilation import compile_mask_for_scene

logger = logging.getLogger("Exp3Geometric")


def run_geometric_invariance(pipe, output_dir: str, project_root: Path = None):
    """Esegue l'esperimento di invarianza geometrica su un cubo solido a spigoli vivi."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    db_cube = str(project_root / "data/dataset_v1/a_red_cube")

    logger.info("=" * 70)
    logger.info("ESPERIMENTO 3: Invarianza Geometrica (Cubo Rosso sul Lago)")
    logger.info(f"Target DB: {db_cube}")
    logger.info("=" * 70)

    # 1. Compila la maschera per il cubo
    logger.info("Step 1: Compilazione maschera ibrida per 'cube'...")
    compile_mask_for_scene(db_path=db_cube, word_to_isolate="cube", method="hybrid")

    # 2. Configurazione con Routing Spaziale per il Cubo
    logger.info("Step 2: Esecuzione Latent Stitching per il Cubo Rosso...")
    config_cube = FlowStitchConfig(
        model_id="black-forest-labs/FLUX.1-schnell",
        stitching_mode="dual",
        ambient_prompt="a crystal clear lake, snowy mountains in background, reflection on water",
        output_root=output_dir,
        subject_scale=0.45,            # Scala al 45%
        subject_offset=(0.15, 0.0),    # Posizionato a filo d'acqua
        t_cutoff=0.5,
        gamma_kts=4.0,
        output_filename="exp3_geometric_cube_routed.png",
    )
    run_latent_stitching(config_cube, db_cube, pipe=pipe)

    logger.info("[EXP 3 COMPLETATO] Salvata: exp3_geometric_cube_routed.png\n")


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).parent / "main.py"), "--exp", "shapes", *sys.argv[1:]])

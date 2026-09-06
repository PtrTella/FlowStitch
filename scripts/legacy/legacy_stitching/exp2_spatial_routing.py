"""
FlowStitch Experiment 2: Spatial Latent Routing (Proportion & Water Surface Injection)
=====================================================================================
Ipotesi Scientifica:
I campioni di dataset estratti da scatti ravvicinati (close-up) occupano una frazione
eccessiva del dominio latente (>80%), lasciando solo una sottile cornice ambientale.

Il modulo di Spatial Latent Routing applica una trasformazione affine continua
(scala s=0.50x, traslazione verticale dy=+0.18) sui tensori latenti a risoluzione 64x64.
La varianza del rumore iniziale viene preservata a N(0, I) tramite rinormalizzazione
statistica, mentre il fattore di damping cinetico KTS decade a t -> 0, permettendo
a FLUX di generare riflessi realistici, increspature dell'acqua e coerenza luminosa.

Output generato:
- exp2_spatial_routed_sphere.png -> Sfera ridotta proporzionata, adagiata sul lago con riflessi
"""

import logging
from pathlib import Path
from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.latent_stitching import run_latent_stitching
from flowstitch.pipelines.mask_compilation import compile_mask_for_scene

logger = logging.getLogger("Exp2Routing")


def run_spatial_routing(pipe, output_dir: str, project_root: Path = None):
    """Esegue l'esperimento di Spatial Latent Routing riusando pipe in VRAM."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    db_sphere = str(project_root / "data/dataset_v1/a_blue_sphere")

    logger.info("=" * 70)
    logger.info("ESPERIMENTO 2: Spatial Latent Routing (Scala 0.50x + Offset Lago)")
    logger.info(f"Target DB: {db_sphere}")
    logger.info("=" * 70)

    # 1. Compila la maschera ibrida di riferimento
    logger.info("Step 1: Compilazione maschera ibrida...")
    compile_mask_for_scene(db_path=db_sphere, word_to_isolate="sphere", method="hybrid")

    # 2. Configurazione con Routing Spaziale e Damping KTS
    logger.info("Step 2: Esecuzione Latent Stitching con Routing Spaziale...")
    config_routed = FlowStitchConfig(
        model_id="black-forest-labs/FLUX.1-schnell",
        stitching_mode="dual",
        ambient_prompt="a crystal clear lake, snowy mountains in background, cinematic morning light",
        output_root=output_dir,
        subject_scale=0.50,            # Dimezza scala (occupa ~20% dell'area)
        subject_offset=(0.18, 0.0),    # Posiziona il baricentro a filo d'acqua
        t_cutoff=0.5,                  # Iniezione strutturale nei primi step (t in [1.0, 0.5])
        gamma_kts=4.0,                 # Decadimento verso t=0 per consentire la sintesi dei riflessi
        output_filename="exp2_spatial_routed_sphere.png",
    )
    run_latent_stitching(config_routed, db_sphere, pipe=pipe)

    logger.info("[EXP 2 COMPLETATO] Salvata: exp2_spatial_routed_sphere.png\n")


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).parent / "main.py"), "--exp", "routing", *sys.argv[1:]])

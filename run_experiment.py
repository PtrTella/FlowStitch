import os
import torch
import logging

# Configurazione del logging per l'HPC
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

from flowstitch.core.config import FlowStitchConfig
from flowstitch.pipelines.latent_stitching import run_latent_stitching


def main():
    # 1. Inizializza la configurazione per la nuova pipeline modulare
    config = FlowStitchConfig(
        model_id="black-forest-labs/FLUX.1-schnell",
        stitching_mode="dual",  # Usa la modalità "dual" (maschera sfumata + KTS)
        ambient_prompt="a crystal clear lake",  # Il lago ambientale
        output_root="outputs/experiment_cluster",
    )

    # 2. Definisci il path da cui prelevare i latenti isolati dell'oggetto (es. la sfera)
    db_path = "data/dataset_v1/a_blue_sphere"

    print(f"--- Avvio Latent Stitching Sperimentale su Nodo HPC ---")
    print(f"Ambiente Target: {config.ambient_prompt}")
    print(f"Concetto Inject: {db_path}")
    print(f"Modalita': {config.stitching_mode}")

    # 3. Richiama il modulo centrale della tua pipeline
    run_latent_stitching(config, db_path)


if __name__ == "__main__":
    main()

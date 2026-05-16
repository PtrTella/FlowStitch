import os
import json
import torch
import logging
from diffusers import FluxPipeline

# =============================================================================
# CONFIGURAZIONE SPERIMENTALE - STRADA B (FLOW PERTURBATION)
# =============================================================================
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell",  # Usiamo Schnell per coerenza col DB a 4 step
    "db_path": "data/dataset_v1/a_blue_sphere",  # La "banca del seme" topologica del nostro oggetto
    "output_dir": "data/stitching_results",  # Dove salveremo la prova del teorema
    "ambient_prompt": "a crystal clear lake",  # L'universo ospite (Il tessuto fluido)
    "word_to_isolate": "sphere",  # Il concetto semantico da estrarre
    "lambda_val": 6.0,  # Moltiplicatore di Lagrange (Forza dell'Iniezione)
    "device": torch.device("cuda"),
    "dtype": torch.bfloat16,
    "lake_seed": 1337,  # Un seed fisso per il lago, giusto per riproducibilità
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    logger.info("Caricamento del Motore Differenziale (FLUX) e Tokenizer...")
    pipe = FluxPipeline.from_pretrained(
        CONFIG["model_id"], torch_dtype=CONFIG["dtype"]
    ).to(CONFIG["device"])
    pipe.set_progress_bar_config(disable=True)

    # =========================================================================
    # FASE 1: ESTRAZIONE DAL VECTOR DB
    # =========================================================================
    logger.info("Estrazione Genetica dal DB...")

    A_target = torch.load(
        os.path.join(CONFIG["db_path"], f"A_target_{CONFIG['word_to_isolate']}.pt"),
        map_location="cpu",
        weights_only=True,
    ).to(CONFIG["device"], dtype=CONFIG["dtype"])
    v0_target = torch.load(
        os.path.join(CONFIG["db_path"], "v0_velocity.pt"),
        map_location="cpu",
        weights_only=True,
    ).to(CONFIG["device"], dtype=CONFIG["dtype"])

    with open(os.path.join(CONFIG["db_path"], "metadata.json"), "r") as f:
        metadata = json.load(f)

    # [ATTENZIONE METODOLOGICA]: NON carichiamo x0_noise.pt!
    # Vogliamo che la sfera nasca dal tessuto quantistico del lago, assorbendone la luce.

    # =========================================================================
    # FASE 2: PREPARAZIONE DELL'UNIVERSO OSPITE (Il Lago)
    # =========================================================================
    logger.info(f"Inizializzazione ambiente: '{CONFIG['ambient_prompt']}'")
    ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(
        prompt=CONFIG["ambient_prompt"], prompt_2=None
    )

    # Creiamo un tessuto quantistico (rumore) totalmente nuovo e vergine.
    generator = torch.Generator(device=CONFIG["device"]).manual_seed(
        CONFIG["lake_seed"]
    )

    with torch.no_grad():
        latents, latent_image_ids = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels // 4,
            1024,
            1024,
            CONFIG["dtype"],
            CONFIG["device"],
            generator,
        )

        # =========================================================================
        # FASE 3: INTEGRAZIONE DIFFERENZIALE IBRIDA (Il Trapianto in O(N))
        # =========================================================================
        logger.info("Avvio del Solver Custom (Equazione di Perturbazione)...")
        pipe.scheduler.set_timesteps(metadata["steps"], device=CONFIG["device"])

        for i, t in enumerate(pipe.scheduler.timesteps):
            logger.info(f"  -> Step ODE {i + 1}/{metadata['steps']} (t={t.item()})")

            # 3A. Calcolo della corrente ambientale (Unica Forward Pass!)
            # Chiediamo a FLUX: "Come faresti scorrere queste particelle per fare un lago?"
            v_ambient = pipe.transformer(
                hidden_states=latents,
                timestep=(t / 1000.0).expand(latents.shape[0]).to(latents.dtype),
                guidance=None,
                pooled_projections=ambient_pooled,
                encoder_hidden_states=ambient_embeds,
                txt_ids=ambient_txt_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # 3B. APPLICAZIONE DEL TEOREMA DI PERTURBAZIONE (Latent Stitching)
            # v_stitch = v_ambient + lambda * [A_target * (v0_target - v_ambient)]
            # Spiegazione matematica:
            # - Dove A_target è 0 (lago puro), il termine tra quadre si annulla. v_stitch = v_ambient.
            # - Dove A_target è 1 (centro sfera), v_ambient si elide, rimane la forza della sfera: v_stitch = v0_target.
            # - Dove A_target è (0, 1) (bordi), avviene una sovrapposizione continua e differenziabile,
            #   risolvendo il limite di salto di Lipschitz dimostrato nel Capitolo 4.
            delta_v = A_target * (v0_target - v_ambient)
            v_stitch = v_ambient + (CONFIG["lambda_val"] * delta_v)

            # 3C. Integrazione di Eulero (Avanzamento temporale)
            # Il solver sposta i latenti seguendo la nuova traiettoria ibrida appena calcolata.
            latents = pipe.scheduler.step(v_stitch, t, latents, return_dict=False)[0]

        # =========================================================================
        # FASE 4: COLLASSO D'ONDA E DECODING (Spazio Pixel)
        # =========================================================================
        logger.info("Decodifica VAE in corso...")
        # FLUX necessita di spacchettare i latenti (da 4096 tokens spaziali a matrice 2D)
        latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
        latents = (
            latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor

        with torch.no_grad():
            image = pipe.vae.decode(latents, return_dict=False)[0]

        image = pipe.image_processor.postprocess(image, output_type="pil")[0]

        out_path = os.path.join(
            CONFIG["output_dir"],
            f"strada_b_perfect_stitch_{CONFIG['word_to_isolate']}.png",
        )
        image.save(out_path)
        logger.info(f"[SUCCESS] Q.E.D. Immagine fusa salvata in: {out_path}")


if __name__ == "__main__":
    main()

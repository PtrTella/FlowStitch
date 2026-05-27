import os
import torch
import logging
from diffusers import FluxPipeline
import torchvision.transforms.functional as TF
import warnings


class SemanticGraftingProcessor:
    """
    Custom Attention Processor per il Latent Stitching.
    Avvolge i processori di attenzione di FLUX per iniettare l'attrattore semantico (A_target).
    """

    def __init__(self, original_processor, A_target, injection_strength=0.8):
        # Salviamo il processore originale per eseguire la matematica complessa di FLUX
        self.original_processor = original_processor

        # L'attrattore spaziale [1, 4096, 1] che definisce il perimetro dell'oggetto
        self.A_target = A_target

        # Quanta "forza" semantica applicare (1.0 = blocco totale, 0.0 = nessuna iniezione)
        self.injection_strength = injection_strength

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        # 1. Lasciamo che FLUX calcoli l'attenzione standard nel suo ecosistema
        # Questo garantisce che tutte le posizioni spaziali e i RoPE embeddings siano corretti.
        out_hidden_states = self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # 2. INIEZIONE SEMANTICA (Feature Blending post-attenzione)
        # FLUX usa tensori sequenziali. Nei layer 'Single' immagine e testo sono concatenati.
        # Dobbiamo isolare solo i token dell'immagine (che in FLUX sono sempre all'inizio della sequenza).

        num_image_tokens = self.A_target.shape[1]  # 4096

        if out_hidden_states.shape[1] >= num_image_tokens:
            # Estraiamo le features calcolate dall'attenzione per la zona immagine
            img_features = out_hidden_states[:, :num_image_tokens, :]

            # IL CAMPO DI FORZA:
            # Dove A_target è 1 (centro della sfera), respingiamo i cambiamenti
            # imposti dal prompt "lago" e forziamo il mantenimento dello stato originale
            # (che è guidato dal nostro v0 e x0).
            # Dove A_target è 0 (lago), lasciamo l'output intatto.

            # Applicazione della maschera termodinamica
            protected_features = hidden_states[:, :num_image_tokens, :]
            blended_features = img_features * (
                1.0 - (self.A_target * self.injection_strength)
            ) + protected_features * (self.A_target * self.injection_strength)

            # Sostituiamo le features modificate nel tensore di output
            out_hidden_states[:, :num_image_tokens, :] = blended_features

        return out_hidden_states


def inject_semantic_processors(
    pipe, A_target, injection_strength=0.8, target_blocks="single"
):
    """
    Inietta il Custom Processor navigando direttamente l'albero dei moduli PyTorch,
    bypassando i wrapper mancanti nelle versioni più vecchie di diffusers.
    """
    # Creiamo un dizionario di backup agganciato al transformer per conservare i processori "vanilla"
    if not hasattr(pipe.transformer, "original_processors_cache"):
        pipe.transformer.original_processors_cache = {}

    for name, module in pipe.transformer.named_modules():
        # Identifichiamo i layer di attenzione
        if name.endswith("attn") and hasattr(module, "processor"):
            # Salviamo il processore originale la prima volta che passiamo di qui
            if name not in pipe.transformer.original_processors_cache:
                pipe.transformer.original_processors_cache[name] = module.processor

            original_proc = pipe.transformer.original_processors_cache[name]

            # Applichiamo l'innesto semantico in base al tipo di blocco
            if "single_transformer_blocks" in name and target_blocks in [
                "single",
                "all",
            ]:
                module.processor = SemanticGraftingProcessor(
                    original_proc, A_target, injection_strength
                )
            elif (
                "transformer_blocks" in name
                and "single" not in name
                and target_blocks in ["double", "all"]
            ):
                module.processor = SemanticGraftingProcessor(
                    original_proc, A_target, injection_strength
                )

    return pipe


def remove_semantic_processors(pipe):
    """
    Ripristina i processori originali estraendoli dalla nostra cache custom,
    riportando il modello allo stato vergine.
    """
    if hasattr(pipe.transformer, "original_processors_cache"):
        for name, module in pipe.transformer.named_modules():
            if name.endswith("attn") and hasattr(module, "processor"):
                if name in pipe.transformer.original_processors_cache:
                    module.processor = pipe.transformer.original_processors_cache[name]


# --- CONFIGURAZIONE ---
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell",
    "db_path": "data/dataset_v1/a_yellow_pyramid",
    "output_dir": "data/stitching_results",
    "ambient_prompt": "a crystal clear lake",
    "word_to_isolate": "sphere",
    # PARAMETRI DEL TEOREMA
    "lambda_v0": 1.0,  # Forza dello stitching cinematico (Fisica)
    "injection_strength": 0.85,  # Forza della restrizione semantica (Attenzione)
    "device": "cuda",
    "dtype": torch.bfloat16,
}

torch_device = torch.device(CONFIG["device"])
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pipe = FluxPipeline.from_pretrained(
        CONFIG["model_id"], torch_dtype=CONFIG["dtype"]
    ).to(torch_device)
    pipe.set_progress_bar_config(disable=True)

    # =========================================================================
    # 1. ESTRAZIONE DAL VECTOR DB
    # =========================================================================
    logger.info("Estrazione Genetica dal DB...")

    # Assicurati di aver precedentemente compilato la maschera con lo script del notebook!
    A_target = torch.load(
        os.path.join(CONFIG["db_path"], f"A_target.pt"),
        map_location="cpu",
        weights_only=True,
    ).to(torch_device, dtype=CONFIG["dtype"])

    # FIX: GAUSSIAN BLUR SULL'ATTRATTORE (Smussamento della Scogliera)
    logger.info("Ammorbidimento dei bordi quantistici (Gaussian Blur)...")
    b, seq, c = A_target.shape  # Dovrebbe essere [1, 4096, 1]
    h = w = int(seq**0.5)  # 64x64

    # Rimodelliamo in formato immagine [Batch, Canali, Altezza, Larghezza] per il filtro
    A_target_2d = A_target.view(b, c, h, w)

    # Applichiamo il blur.
    # kernel_size=5 e sigma=2.0 sono ottimi per iniziare a fondere i bordi senza perdere la forma.
    # Riduciamo il blur da kernel=[5,5] sigma=2.0 a qualcosa di quasi impercettibile
    A_target_blurred = TF.gaussian_blur(
        A_target_2d, kernel_size=[3, 3], sigma=[2.5, 2.5]
    )
    # --- MODIFICA CRITICA: RIPRISTINO DEL PICCO A 1.0 ---
    # Il blur spalma l'energia abbassando il picco. Dividendo per il max, il core torna solido (1.0)
    max_val = A_target_blurred.max()
    A_target_blurred = A_target_blurred / (max_val + 1e-8)
    # -----------------------------------------------------
    # Riportiamo al formato sequenziale [1, 4096, 1] per il Transformer
    A_target = A_target_blurred.view(b, seq, c)

    v0_db = torch.load(
        os.path.join(CONFIG["db_path"], "v0_velocity.pt"),
        map_location="cpu",
        weights_only=True,
    ).to(torch_device, dtype=CONFIG["dtype"])
    x0_db = torch.load(
        os.path.join(CONFIG["db_path"], "x0_noise.pt"),
        map_location="cpu",
        weights_only=True,
    ).to(torch_device, dtype=CONFIG["dtype"])

    # =========================================================================
    # 2. INNESTO SEMANTICO (Hacking dell'Attenzione)
    # =========================================================================
    logger.info("Iniezione del Custom Attention Processor...")
    # Applichiamo il processore sui blocchi 'single' per proteggere l'identità della sfera
    pipe = inject_semantic_processors(
        pipe, A_target, injection_strength=CONFIG["injection_strength"]
    )

    # =========================================================================
    # 3. IL MOSAICO QUANTISTICO E ODE LOOP
    # =========================================================================
    logger.info("Preparazione Mosaico Iniziale...")
    with torch.no_grad():
        ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(
            CONFIG["ambient_prompt"], prompt_2=None
        )

        generator = torch.Generator(device=torch_device)
        latents_lake, latent_image_ids = pipe.prepare_latents(
            1,
            pipe.transformer.config.in_channels // 4,
            1024,
            1024,
            CONFIG["dtype"],
            torch_device,
            generator,
        )

        # Mosaico Iniziale (Saldatura di x0)
        latents = latents_lake * (1.0 - A_target) + x0_db * torch.sqrt(A_target)

        logger.info("Integrazione Differenziale (ODE)...")
        pipe.scheduler.set_timesteps(4, device=torch_device)

        for t in pipe.scheduler.timesteps:
            timestep_1d = (t / 1000.0).expand(latents.shape[0]).to(latents.dtype)

            # Il modello calcola le correnti per il lago, ma ora la SUA ATTENZIONE È HACKERATA!
            v_ambient = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_1d,
                guidance=None,
                pooled_projections=ambient_pooled,
                encoder_hidden_states=ambient_embeds,
                txt_ids=ambient_txt_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]

            # Forza l'inserimento geometrico della sfera (Cinematica)
            delta_v = A_target * (v0_db - v_ambient)
            v_stitch = v_ambient + (CONFIG["lambda_v0"] * delta_v)

            latents = pipe.scheduler.step(v_stitch, t, latents, return_dict=False)[0]

    # =========================================================================
    # 4. PULIZIA E DECODIFICA (FIX VAE OVERFLOW)
    # =========================================================================
    remove_semantic_processors(pipe)
    logger.info("Decodifica VAE (in Float32 per prevenire i buchi neri)...")

    with torch.no_grad():
        # Scompattamento standard
        latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
        latents = (
            latents / pipe.vae.config.scaling_factor
        ) + pipe.vae.config.shift_factor

        # --- FIX CRITICO: CAST A FLOAT32 ---
        # Spostiamo temporaneamente VAE e latenti a 32-bit per evitare NaN
        pipe.vae.to(dtype=torch.float32)
        latents_f32 = latents.to(torch.float32)

        image = pipe.vae.decode(latents_f32, return_dict=False)[0]

        # Rimettiamo il VAE in bfloat16 per pulizia
        pipe.vae.to(dtype=CONFIG["dtype"])

    image = image.detach()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]

    out_path = os.path.join(
        CONFIG["output_dir"], "stitching_completo_fisica_semantica_pyr.png"
    )
    image.save(out_path)
    logger.info(f"[SUCCESS] Immagine salvata in: {out_path}")


if __name__ == "__main__":
    main()

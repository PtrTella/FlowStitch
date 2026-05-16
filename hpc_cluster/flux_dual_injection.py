import os
import json
import torch
import logging
from diffusers import FluxPipeline
from transformers import T5Tokenizer

# =============================================================================
# CONFIGURAZIONE SPERIMENTALE - DUAL INJECTION (FISICA + SEMANTICA)
# =============================================================================
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell", 
    "db_path": "data/dataset_v1/a_blue_sphere",     
    "output_dir": "data/stitching_results",         
    "ambient_prompt": "a crystal clear lake",       
    "word_to_isolate": "sphere",                    
    
    # I DUE MOTORI DEL TRAIPIANTO
    "lambda_phys": 0.8,  # Forza strutturale dal DB (Forma geometrica perfetta)
    "lambda_sem": 0.5,   # Consapevolezza live (Texture, ombre, no trasparenza)
    
    "device": torch.device("cuda"),
    "dtype": torch.bfloat16,
    "lake_seed": 1337                               
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    logger.info("Caricamento del Motore Differenziale (FLUX) e Tokenizer...")
    pipe = FluxPipeline.from_pretrained(CONFIG["model_id"], torch_dtype=CONFIG["dtype"]).to(CONFIG["device"])
    pipe.set_progress_bar_config(disable=True)
    tokenizer = T5Tokenizer.from_pretrained(
        "google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True
    )
    
    # =========================================================================
    # FASE 1: ESTRAZIONE DAL VECTOR DB (Topologia Pura)
    # =========================================================================
    logger.info(f"Apertura del Vector DB: {CONFIG['db_path']}")
    
    with open(os.path.join(CONFIG["db_path"], "metadata.json"), "r") as f:
        metadata = json.load(f)
    target_prompt = metadata["prompt"]
    
    # 1A. COSTRUZIONE DELL'ATTRATTORE SEMANTICO (A_target)
    attn_target = torch.load(os.path.join(CONFIG["db_path"], "attention_maps.pt"), map_location="cpu", weights_only=True)
    layer_10 = attn_target['layer_10'] 
    
    tokens = tokenizer(target_prompt, return_tensors="pt").input_ids[0]
    token_indices = [i for i, token in enumerate(tokens) if tokenizer.decode([token]).strip().lower() in CONFIG["word_to_isolate"] and len(tokenizer.decode([token]).strip().lower()) > 0]
            
    if not token_indices:
        raise ValueError(f"Anomalia: Parola '{CONFIG['word_to_isolate']}' non trovata.")
        
    attn_maps_list = [layer_10[0, :, :, idx].mean(dim=0) for idx in token_indices]
    attn_map = torch.clamp(torch.stack(attn_maps_list).sum(dim=0), min=0.0, max=1.0).to(CONFIG["device"], dtype=CONFIG["dtype"])
    
    attn_min, attn_max = attn_map.min(), attn_map.max()
    A_target = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
    A_target = A_target.unsqueeze(0).unsqueeze(-1) # [1, 4096, 1]
    
    # 1B. CARICAMENTO FORZA FISICA (v0_db)
    v0_db = torch.load(os.path.join(CONFIG["db_path"], "v0_velocity.pt"), map_location="cpu", weights_only=True).to(CONFIG["device"], dtype=CONFIG["dtype"])

    # =========================================================================
    # FASE 2: PREPARAZIONE DELLE DUE MENTI (Lago e Sfera)
    # =========================================================================
    logger.info("Encoding delle istruzioni semantiche...")
    
    # Mente 1: L'Ambiente
    ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(
        prompt=CONFIG["ambient_prompt"], prompt_2=None
    )
    
    # Mente 2: L'Oggetto (Consapevolezza Live)
    target_embeds, target_pooled, target_txt_ids = pipe.encode_prompt(
        prompt=target_prompt, prompt_2=None
    )
    
    generator = torch.Generator(device=CONFIG["device"]).manual_seed(CONFIG["lake_seed"])

    with torch.no_grad():
        latents, latent_image_ids = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, 1024, 1024, CONFIG["dtype"], CONFIG["device"], generator
        )
        
        # =========================================================================
        # FASE 3: INTEGRAZIONE DIFFERENZIALE A DOPPIA INIEZIONE
        # =========================================================================
        logger.info("Avvio Dual Injection Solver (Fisica DB + Semantica Live)...")
        pipe.scheduler.set_timesteps(metadata["steps"], device=CONFIG["device"])
        
        for i, t in enumerate(pipe.scheduler.timesteps):
            logger.info(f"  -> Step ODE {i+1}/{metadata['steps']} (t={t.item()})")
            timestep_1d = (t / 1000.0).expand(latents.shape[0]).to(latents.dtype)

            # 3A. FLUSSO AMBIENTALE (Cosa farebbe il lago da solo?)
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
            
            # 3B. FLUSSO SEMANTICO (Cosa farebbe la sfera se fosse qui ora?)
            # Questa è la vera "Consapevolezza". FLUX guarda le onde del lago e prova a renderizzare "a blue sphere"
            v_live_target = pipe.transformer(
                hidden_states=latents,
                timestep=timestep_1d,
                guidance=None,
                pooled_projections=target_pooled,
                encoder_hidden_states=target_embeds,
                txt_ids=target_txt_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            
            # 3C. L'EQUAZIONE DI SINTESI (Attention Grafting Tardivo)
            # 1. Calcoliamo lo strappo fisico verso il modello ideale (DB)
            delta_phys = A_target * (v0_db - v_ambient)
            
            # 2. Calcoliamo lo strappo semantico verso il materiale/texture corretto (Live)
            delta_sem = A_target * (v_live_target - v_ambient)
            
            # 3. Fondiamo tutto! La sfera vince sul lago proporzionalmente ai nostri pesi.
            v_stitch = v_ambient + (CONFIG["lambda_phys"] * delta_phys) + (CONFIG["lambda_sem"] * delta_sem)
            
            # Integrazione
            latents = pipe.scheduler.step(v_stitch, t, latents, return_dict=False)[0]

        # =========================================================================
        # FASE 4: DECODING
        # =========================================================================
        logger.info("Decodifica VAE in corso...")
        latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        out_path = os.path.join(CONFIG["output_dir"], f"strada_b_dual_injection_{CONFIG['word_to_isolate']}.png")
        image.save(out_path)
        logger.info(f"[SUCCESS] Iniezione ibrida completata. File: {out_path}")

if __name__ == "__main__":
    main()
import os
import json
import torch
import logging
from diffusers import FluxPipeline
import warnings

# --- CONFIGURAZIONE ---
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell", 
    "db_path": "data/dataset_v1/a_blue_sphere",     
    "output_dir": "data/stitching_results",         
    "ambient_prompt": "a crystal clear lake",       
    "lambda_val": 1.0,  
    "device": "cuda",
    "dtype": torch.bfloat16,
    "lake_seed": 42                               
}

torch_device = torch.device(CONFIG["device"])
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pipe = FluxPipeline.from_pretrained(CONFIG["model_id"], torch_dtype=CONFIG["dtype"]).to(torch_device)
    pipe.set_progress_bar_config(disable=True)
    
    # 1. CARICAMENTO DAL DB (La Trinità: Maschera, Vettore, Rumore)
    logger.info("Estrazione Genetica dal DB...")
    attn_target = torch.load(os.path.join(CONFIG["db_path"], "attention_maps.pt"), map_location="cpu", weights_only=True)
    A_target = attn_target['layer_10'] # Supponendo tu abbia già la maschera 2D pronta in formato [1, 4096, 1]
    
    # Nel tuo DB dovresti avere la maschera già collassata a [1, 4096, 1] normalizzata.
    # Per semplicità qui assumiamo che A_target sia già il tensore maschera [0,1].
    # (Se hai il tensore grezzo, ripeti il codice di estrazione token dei messaggi precedenti)
    
    v0_db = torch.load(os.path.join(CONFIG["db_path"], "v0_velocity.pt"), map_location="cpu", weights_only=True).to(torch_device, dtype=CONFIG["dtype"])
    x0_db = torch.load(os.path.join(CONFIG["db_path"], "x0_noise.pt"), map_location="cpu", weights_only=True).to(torch_device, dtype=CONFIG["dtype"])

    # 2. IL MOSAICO QUANTISTICO
    logger.info("Preparazione Mosaico Iniziale...")
    with torch.no_grad():
        ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(CONFIG["ambient_prompt"], prompt_2=None)
        
        # Generiamo il rumore del lago
        generator = torch.Generator(device=torch_device).manual_seed(CONFIG["lake_seed"])
        latents_lake, latent_image_ids = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, 1024, 1024, CONFIG["dtype"], torch_device, generator
        )
        
        # TEOREMA DEL MOSAICO: Sostituiamo i pixel quantistici usando la maschera
        latents = latents_lake * (1.0 - A_target) + x0_db * A_target
        
        # 3. L'ODE LOOP 
        logger.info("Integrazione Differenziale...")
        pipe.scheduler.set_timesteps(4, device=torch_device)
        
        for t in pipe.scheduler.timesteps:
            timestep_1d = (t / 1000.0).expand(latents.shape[0]).to(latents.dtype)

            # Il modello calcola le correnti per il lago
            v_ambient = pipe.transformer(
                hidden_states=latents, timestep=timestep_1d, guidance=None,
                pooled_projections=ambient_pooled, encoder_hidden_states=ambient_embeds,
                txt_ids=ambient_txt_ids, img_ids=latent_image_ids, return_dict=False
            )[0]
            
            # Forza l'inserimento geometrico della sfera
            delta_v = A_target * (v0_db - v_ambient)
            v_stitch = v_ambient + (CONFIG["lambda_val"] * delta_v)
            
            latents = pipe.scheduler.step(v_stitch, t, latents, return_dict=False)[0]

        # 4. DECODIFICA
        logger.info("Decodifica VAE...")
        latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image.save(os.path.join(CONFIG["output_dir"], "mosaico_fisico.png"))

if __name__ == "__main__":
    main()
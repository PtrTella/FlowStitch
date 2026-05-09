import os
import json
import torch
import logging
import traceback
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from diffusers import FluxPipeline
from transformers import T5Tokenizer

# Importiamo le classi di hook già sviluppate per l'estrazione dati
from flux_dataset_generator import FluxDataCapturer, AttnProcessorWrapper

# --- CONFIGURAZIONE ---
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell",
    "output_dir": "data/stitching_experiment",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": torch.bfloat16,
    "seed": 42,
    "steps": 4,
    "source_prompt": "a glowing neon sphere",
    "target_prompt": "a calm mirror lake at night",
    # La parola o le parole da fondere (per estrarre la maschera)
    "source_target_words": ["glowing", "neon", "sphere"] 
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_token_indices(tokenizer, prompt, target_words):
    """Trova gli indici dei token corrispondenti alle parole target."""
    tokens = tokenizer.encode(prompt)
    target_indices = []
    
    # Metodo euristico semplice per trovare i token
    # (T5 aggiunge spesso ' ' davanti alle parole)
    for i, token_id in enumerate(tokens):
        decoded = tokenizer.decode([token_id]).strip().lower()
        for word in target_words:
            if decoded in word.lower() and decoded != "":
                target_indices.append(i)
                break
    
    if not target_indices:
        logger.warning(f"Nessun token trovato per le parole {target_words}. Uso i primi 10 token testuali.")
        target_indices = list(range(1, 10))
    return target_indices

def create_hard_mask(cross_attn, target_indices):
    """Trasforma la cross-attention grezza in una Hard Mask topologica."""
    # cross_attn: [1, 24, 4096, 512]
    attn_mean = cross_attn.mean(dim=1).squeeze(0) # [4096, 512]
    
    # 1. Semantic Fusion
    attn_map = attn_mean[:, target_indices].sum(dim=1) # [4096]
    
    # 2. Reshape & Normalization
    attn_map = attn_map.view(64, 64).numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    # 3. Gaussian Smoothing
    blurred = gaussian_filter(attn_map, sigma=1.5)
    
    # 4. Binarization
    mask_2d = (blurred > 0.25).astype(np.float32)
    
    # Prepariamola per il tensore latente FLUX [1, 4096, 1]
    mask_flat = mask_2d.flatten()
    mask_tensor = torch.tensor(mask_flat, dtype=CONFIG["dtype"], device=CONFIG["device"])
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1) # [1, 4096, 1]
    
    return mask_2d, mask_tensor

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger.info(f"Avvio esperimento FlowStitch. Output dir: {CONFIG['output_dir']}")
    
    # 1. Caricamento Modelli
    pipe = FluxPipeline.from_pretrained(
        CONFIG["model_id"],
        torch_dtype=CONFIG["dtype"],
    ).to(CONFIG["device"])
    
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_id"], subfolder="tokenizer_2")
    
    generator = torch.Generator(device=CONFIG["device"]).manual_seed(CONFIG["seed"])
    
    # Variabili per memorizzare i dati intermedi
    storage = {
        "source_latent_step1": None,
        "mask_tensor": None,
        "mask_2d": None
    }
    
    # --- FASE 1: GENERAZIONE SORGENTE E ESTRAZIONE MASCHERA ---
    logger.info(f"Fase 1: Generazione Sorgente -> '{CONFIG['source_prompt']}'")
    capturer = FluxDataCapturer(pipe.transformer, [10])
    capturer.attach()
    capturer.reset()
    
    def source_callback(pipe, step_index, timestep, callback_kwargs):
        if step_index == 0:
            storage["source_latent_step1"] = callback_kwargs["latents"].detach().clone()
            logger.info("   -> Latente Sorgente allo Step 1 catturato.")
        return callback_kwargs

    source_image = pipe(
        prompt=CONFIG["source_prompt"],
        num_inference_steps=CONFIG["steps"],
        generator=generator,
        callback_on_step_end=source_callback,
        output_type="pil"
    ).images[0]
    source_image.save(os.path.join(CONFIG["output_dir"], "01_source_pure.png"))
    capturer.remove() # Sganciamo l'estrattore di attenzione
    
    # Estrazione Maschera
    if "layer_10" in capturer.attn_maps:
        target_indices = get_token_indices(tokenizer, CONFIG["source_prompt"], CONFIG["source_target_words"])
        logger.info(f"   -> Calcolo maschera sui token indici: {target_indices}")
        mask_2d, mask_tensor = create_hard_mask(capturer.attn_maps["layer_10"], target_indices)
        storage["mask_2d"] = mask_2d
        storage["mask_tensor"] = mask_tensor
        
        # Salvataggio visuale maschera
        plt.imsave(os.path.join(CONFIG["output_dir"], "02_extracted_mask.png"), mask_2d, cmap='gray')
    else:
        logger.error("Mappa di attenzione non trovata. Impossibile procedere.")
        return
        
    # --- FASE 2: GENERAZIONE TARGET (CONTROLLO) ---
    logger.info(f"Fase 2: Generazione Target di Controllo -> '{CONFIG['target_prompt']}'")
    # Resettiamo il generatore per avere lo stesso rumore iniziale
    generator.manual_seed(CONFIG["seed"])
    target_image_pure = pipe(
        prompt=CONFIG["target_prompt"],
        num_inference_steps=CONFIG["steps"],
        generator=generator,
        output_type="pil"
    ).images[0]
    target_image_pure.save(os.path.join(CONFIG["output_dir"], "03_target_pure.png"))

    # --- FASE 3: LATENT STITCHING ---
    logger.info("Fase 3: Esecuzione Latent Stitching...")
    generator.manual_seed(CONFIG["seed"])
    
    def stitching_callback(pipe, step_index, timestep, callback_kwargs):
        if step_index == 0:
            # Recuperiamo il latente target attuale
            target_latent = callback_kwargs["latents"]
            source_latent = storage["source_latent_step1"]
            M = storage["mask_tensor"]
            
            # THE MAGIC EQUATION: x_stitch = M * x_source + (1 - M) * x_target
            stitched_latent = (M * source_latent) + ((1.0 - M) * target_latent)
            
            # Sovrascriviamo il latente per i successivi step
            callback_kwargs["latents"] = stitched_latent
            logger.info("   -> Stitching Matematico Applicato. Modello forzato ad armonizzare.")
            
        return callback_kwargs

    stitched_image = pipe(
        prompt=CONFIG["target_prompt"],
        num_inference_steps=CONFIG["steps"],
        generator=generator,
        callback_on_step_end=stitching_callback,
        output_type="pil"
    ).images[0]
    
    stitched_image.save(os.path.join(CONFIG["output_dir"], "04_final_stitched_image.png"))
    logger.info(f"Esperimento completato con successo. Risultati salvati in {CONFIG['output_dir']}/")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Errore fatale: {e}")
        logger.error(traceback.format_exc())

import os
import json
import torch
import logging
import traceback
from diffusers import FluxPipeline

from ..core.config import FlowStitchConfig
from ..core.flux_hooking import FluxDataCapturer
from ..core.serialization import save_tensors

logger = logging.getLogger(__name__)

def slugify(text: str) -> str:
    """Trasforma un prompt in un nome cartella sicuro."""
    return text.lower().replace(" ", "_").replace(",", "").replace(".", "")[:50]

def generate_dataset(config: FlowStitchConfig, prompts: list[str]):
    """
    Core dataset generation pipeline using the unified config and hooking system.
    """
    os.makedirs(config.output_root, exist_ok=True)
    
    logger.info(f"Caricamento {config.model_id} in VRAM...")
    try:
        pipe = FluxPipeline.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
        ).to(config.device)
    except Exception as e:
        logger.error(f"[FATAL] Errore nel caricamento del modello: {e}")
        return
    
    with FluxDataCapturer(pipe.transformer, config.target_layers, max_capture_step=1) as capturer:
        for prompt in prompts:
            slug = slugify(prompt)
            path = os.path.join(config.output_root, slug)
            os.makedirs(path, exist_ok=True)
            
            logger.info(f"--- Processando: '{prompt}' ---")
            capturer.reset()
            generator = torch.Generator(device="cpu").manual_seed(config.seed)
            
            try:
                output = pipe(
                    prompt=prompt,
                    num_inference_steps=config.steps,
                    generator=generator,
                    output_type="pil"
                )
                
                output.images[0].save(os.path.join(path, "final_image.png"))
                
                if capturer.x0 is not None and capturer.v0 is not None:
                    save_tensors(capturer.x0, os.path.join(path, "x0_noise.pt"))
                    save_tensors(capturer.v0, os.path.join(path, "v0_velocity.pt"))
                    
                    x_pred = capturer.x0 + capturer.v0
                    save_tensors(x_pred, os.path.join(path, "x_pred.pt"))
                    
                    if capturer.attn_maps:
                        save_tensors(capturer.attn_maps, os.path.join(path, "attention_maps.pt"))
                    
                    metadata = {
                        "prompt": prompt,
                        "seed": config.seed,
                        "model": config.model_id,
                        "steps": config.steps,
                        "layers_captured": list(capturer.attn_maps.keys()),
                        "math_proof": "x_pred = x0 + v0 computed successfully"
                    }
                    with open(os.path.join(path, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=4)
                        
                    logger.info(f"[SUCCESS] Dati matematici e mappe salvati in {path}")
                else:
                    logger.error(f"[ERROR] Hook non scattato per: {prompt}")

            except Exception as e:
                logger.error(f"[CRASH PROMPT] Errore imprevisto su '{prompt}': {e}")
                logger.error(traceback.format_exc())
                logger.info("Pulisco la VRAM e passo al prossimo prompt...")
                
            finally:
                torch.cuda.empty_cache()

    logger.info("Generazione Dataset completata!")

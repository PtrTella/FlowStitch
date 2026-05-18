import os
import torch
import logging
from diffusers import FluxPipeline
import warnings
import torchvision.transforms.functional as TF

# =============================================================================
# CORE CLASSES & HOOKS (L'Aura Semantica)
# =============================================================================

class SemanticGraftingProcessor:
    def __init__(self, original_processor, A_luce, injection_strength=0.8):
        self.original_processor = original_processor
        self.A_luce = A_luce 
        self.injection_strength = injection_strength

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        out_hidden_states = self.original_processor(
            attn, hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, image_rotary_emb=image_rotary_emb,
        )
        num_image_tokens = self.A_luce.shape[1] 
        if out_hidden_states.shape[1] >= num_image_tokens:
            img_features = out_hidden_states[:, :num_image_tokens, :]
            protected_features = hidden_states[:, :num_image_tokens, :] 
            blended_features = img_features * (1.0 - (self.A_luce * self.injection_strength)) + \
                               protected_features * (self.A_luce * self.injection_strength)
            out_hidden_states[:, :num_image_tokens, :] = blended_features
        return out_hidden_states

def inject_semantic_processors(pipe, A_luce, injection_strength=0.8):
    if not hasattr(pipe.transformer, "original_processors_cache"):
        pipe.transformer.original_processors_cache = {}
    for name, module in pipe.transformer.named_modules():
        if name.endswith("attn") and hasattr(module, "processor"):
            if name not in pipe.transformer.original_processors_cache:
                pipe.transformer.original_processors_cache[name] = module.processor
            if "single_transformer_blocks" in name:
                module.processor = SemanticGraftingProcessor(pipe.transformer.original_processors_cache[name], A_luce, injection_strength)
    pipe.transformer.semantic_hook_active = True
    return pipe

def remove_semantic_processors(pipe):
    if hasattr(pipe.transformer, "original_processors_cache"):
        for name, module in pipe.transformer.named_modules():
            if name.endswith("attn") and hasattr(module, "processor"):
                if name in pipe.transformer.original_processors_cache:
                    module.processor = pipe.transformer.original_processors_cache[name]
    pipe.transformer.semantic_hook_active = False


# =============================================================================
# CONFIGURAZIONE ESPERIMENTO "SINGLE KICK"
# =============================================================================
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell", 
    "db_path": "data/dataset_v1/a_red_cube",     
    "output_dir": "data/stitching_results",         
    "ambient_prompt": "a crystal clear lake",       
    
    # PARAMETRI DEL TEOREMA
    "lambda_v0": 1.0,           
    "injection_strength": 0.85, 
    
    # La soglia semantica: quanto a lungo l'attenzione tiene unita la materia (es. 0.75 = 3 step su 4)
    "semantic_threshold": 1.0,   
    
    "lake_seed": 42,
    "device": "cuda",
    "dtype": torch.bfloat16                         
}

torch_device = torch.device(CONFIG["device"])
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    pipe = FluxPipeline.from_pretrained(CONFIG["model_id"], torch_dtype=CONFIG["dtype"]).to(torch_device)
    pipe.set_progress_bar_config(disable=True)
    
    # =========================================================================
    # 1. CARICAMENTO ARTEFATTI OFFLINE
    # =========================================================================
    logger.info("Caricamento Artefatti dal Vector DB...")
    
    # A_luce è già stata sfumata e ricalibrata offline!
    A_luce = torch.load(os.path.join(CONFIG["db_path"], "A_target.pt"), map_location="cpu", weights_only=True).to(torch_device, dtype=CONFIG["dtype"])
    
    # Assicurati di aver precedentemente compilato la maschera con lo script del notebook!
    A_target = torch.load(os.path.join(CONFIG["db_path"], f"A_target.pt"), map_location="cpu", weights_only=True).to(torch_device, dtype=CONFIG["dtype"])

    # FIX: GAUSSIAN BLUR SULL'ATTRATTORE (Smussamento della Scogliera)
    logger.info("Ammorbidimento dei bordi quantistici (Gaussian Blur)...")
    b, seq, c = A_target.shape  # Dovrebbe essere [1, 4096, 1]
    h = w = int(seq ** 0.5)     # 64x64
    
    # Rimodelliamo in formato immagine [Batch, Canali, Altezza, Larghezza] per il filtro
    A_target_2d = A_target.view(b, c, h, w)
    
    # Applichiamo il blur. 
    # kernel_size=5 e sigma=2.0 sono ottimi per iniziare a fondere i bordi senza perdere la forma.
    # Riduciamo il blur da kernel=[5,5] sigma=2.0 a qualcosa di quasi impercettibile
    A_target_blurred = TF.gaussian_blur(A_target_2d, kernel_size=[3, 3], sigma=[2.5, 2.5])
    
    # Riportiamo al formato sequenziale [1, 4096, 1] per il Transformer
    A_materia = A_target_blurred.view(b, seq, c)

    # Ricostruzione della Destinazione Telemetrica (x1)
    v0_db = torch.load(os.path.join(CONFIG["db_path"], "v0_velocity.pt"), map_location="cpu", weights_only=True).to(torch_device, dtype=CONFIG["dtype"])
    x0_db = torch.load(os.path.join(CONFIG["db_path"], "x0_noise.pt"), map_location="cpu", weights_only=True).to(torch_device, dtype=CONFIG["dtype"])
    x1_db = x0_db + v0_db 

    # =========================================================================
    # 2. ODE LOOP CON SPINTA SINGOLA E MANTENIMENTO SEMANTICO
    # =========================================================================
    logger.info("Avvio Integrazione Differenziale (Mosaico Quantistico)...")
    
    with torch.no_grad():
        ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(CONFIG["ambient_prompt"], prompt_2=None)
        generator = torch.Generator(device=torch_device).manual_seed(CONFIG["lake_seed"])
        
        # Generiamo il rumore del lago
        latents_lake, latent_image_ids = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, 1024, 1024, CONFIG["dtype"], torch_device, generator
        )
        
        # ASSIOMA 1: La Nascita della Materia. Saldatura sferica del rumore iniziale
        latents = latents_lake * torch.sqrt(1.0 - A_materia) + x0_db * torch.sqrt(A_materia)
        
        pipe.scheduler.set_timesteps(4, device=torch_device)
        total_steps = len(pipe.scheduler.timesteps)
        
        # Inizializziamo l'hook semantico
        pipe = inject_semantic_processors(pipe, A_luce, injection_strength=CONFIG["injection_strength"])
        
        for i, t in enumerate(pipe.scheduler.timesteps):
            t_val = t.item() if torch.is_tensor(t) else float(t)
            timestep_1d = torch.tensor([t_val / 1000.0] * latents.shape[0], device=torch_device, dtype=latents.dtype)

            # --- FORWARD PASS DEL MODELLO ---
            v_ambient = pipe.transformer(
                hidden_states=latents, timestep=timestep_1d, guidance=None,
                pooled_projections=ambient_pooled, encoder_hidden_states=ambient_embeds,
                txt_ids=ambient_txt_ids, img_ids=latent_image_ids, return_dict=False
            )[0]
            
            t_norm = t_val / 1000.0 
            v_shifted_target = (x1_db - latents) / (t_norm + 1e-8)
            
            # Applichiamo la spinta solo all'interno del bisturi
            delta_v = A_materia * (v_shifted_target - v_ambient)
            v_stitch = v_ambient + (CONFIG["lambda_v0"] * delta_v)

            # Avanzamento
            latents = pipe.scheduler.step(v_stitch, t, latents, return_dict=False)[0]

    # =========================================================================
    # 3. DECODIFICA SICURA
    # =========================================================================
    logger.info("Decodifica VAE...")
    if pipe.transformer.semantic_hook_active:
        remove_semantic_processors(pipe)
        
    with torch.no_grad():
        latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        
        pipe.vae.to(dtype=torch.float32)
        latents_f32 = latents.to(torch.float32)
        image = pipe.vae.decode(latents_f32, return_dict=False)[0]
        pipe.vae.to(dtype=CONFIG["dtype"])
    
    image = image.detach()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    out_path = os.path.join(CONFIG["output_dir"], "test_single_kick_semantico.png")
    image.save(out_path)
    logger.info(f"[SUCCESS] Salvataggio completato in: {out_path}")

if __name__ == "__main__":
    main()
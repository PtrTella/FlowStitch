import os
import torch
import logging
import torchvision.transforms.functional as TF
from diffusers import FluxPipeline

from ..core.config import FlowStitchConfig
from ..core.serialization import load_tensors
from ..stitching.semantic_processor import inject_semantic_processors, remove_semantic_processors
from ..stitching.ode_perturbation import perform_ode_step

logger = logging.getLogger(__name__)

def run_latent_stitching(config: FlowStitchConfig, db_path: str):
    """
    Unified Latent Stitching pipeline.
    Supports mode: "mosaico" or "dual" (from config.stitching_mode).
    
    Note: "full" mode was removed — it was identical to "dual".
    See notebooks/02_Injection_Thermodynamics for the motivation.
    """
    logger.info(f"--- Latent Stitching [{config.stitching_mode}] ---")
    
    device = torch.device(config.device)
    pipe = FluxPipeline.from_pretrained(config.model_id, torch_dtype=config.dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # 1. Caricamento Dati
    A_target = load_tensors(os.path.join(db_path, "A_target.pt")).to(device, dtype=config.dtype)
    v0_db = load_tensors(os.path.join(db_path, "v0_velocity.pt")).to(device, dtype=config.dtype)
    x0_db = load_tensors(os.path.join(db_path, "x0_noise.pt")).to(device, dtype=config.dtype)
    
    # Gaussian Blur su A_target per "Dual" e "Full" modes per sfumare i bordi
    if config.stitching_mode == "dual":
        b, seq, c = A_target.shape
        h = w = int(seq ** 0.5)
        assert h * w == seq, f"seq_len must be a perfect square, got {seq}"
        A_target_2d = A_target.view(b, c, h, w)
        A_target_blurred = TF.gaussian_blur(A_target_2d, kernel_size=[3, 3], sigma=[2.5, 2.5])
        max_val = A_target_blurred.max()
        A_target_blurred = A_target_blurred / (max_val + 1e-8)
        A_target = A_target_blurred.view(b, seq, c)
        
    A_fisica = (A_target > 0.1).to(config.dtype) if config.stitching_mode == "dual" else A_target
    
    # 2. Setup Modello e Hook Semantico
    if config.stitching_mode == "dual":
        logger.info("Iniezione del Custom Attention Processor (Semantic Grafting)...")
        pipe = inject_semantic_processors(pipe, A_target, injection_strength=config.injection_strength, target_blocks="single")
        
    # 3. Mosaico Quantistico (Nascita della Materia)
    with torch.no_grad():
        ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(config.ambient_prompt, prompt_2=None)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        latents_lake, latent_image_ids = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, 1024, 1024, config.dtype, device, generator
        )
        
        # Mosaico Iniziale
        if config.stitching_mode == "mosaico":
            latents = latents_lake * (1.0 - A_fisica) + x0_db * A_fisica
        else:
            # NOTE: A_fisica is binary (thresholded), so sqrt would be a no-op.
            # For spherical blending with continuous masks, use: sqrt(1-α)·x + sqrt(α)·y
            latents = latents_lake * (1.0 - A_fisica) + x0_db * A_fisica
            
        logger.info("Integrazione Differenziale (ODE)...")
        pipe.scheduler.set_timesteps(config.steps, device=device)
        
        smoother = None
        if getattr(config, "use_ema", False):
            from ..stitching.ema_smoothing import TrajectoryEMA
            smoother = TrajectoryEMA(decay=config.ema_decay)
            logger.info(f"Trajectory EMA attivato con decay={config.ema_decay}")
        
        for t in pipe.scheduler.timesteps:
            latents = perform_ode_step(
                pipe=pipe, latents=latents, t=t,
                ambient_pooled=ambient_pooled, ambient_embeds=ambient_embeds,
                ambient_txt_ids=ambient_txt_ids, latent_image_ids=latent_image_ids,
                A_target=A_fisica, v0_target=v0_db, lambda_v0=config.lambda_v0,
                kts_t_cutoff=config.t_cutoff, kts_gamma=config.gamma_kts,
                mode=config.stitching_mode, smoother=smoother
            )

    # 4. Decodifica e Pulizia
    if config.stitching_mode == "dual":
        remove_semantic_processors(pipe)
        
    logger.info("Decodifica VAE (float32)...")
    with torch.no_grad():
        latents = pipe._unpack_latents(latents, 1024, 1024, pipe.vae_scale_factor)
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        
        pipe.vae.to(dtype=torch.float32)
        latents_f32 = latents.to(torch.float32)
        image = pipe.vae.decode(latents_f32, return_dict=False)[0]
        pipe.vae.to(dtype=config.dtype)
        
    image = image.detach()
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    os.makedirs(config.output_root, exist_ok=True)
    out_path = os.path.join(config.output_root, f"stitching_{config.stitching_mode}.png")
    image.save(out_path)
    logger.info(f"[SUCCESS] Immagine salvata in: {out_path}")
    return image

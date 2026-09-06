import os
import torch
import logging
import torchvision.transforms.functional as TF
from diffusers import FluxPipeline

from ..core.config import FlowStitchConfig
from ..core.serialization import load_tensors
from ..stitching.semantic_processor import inject_semantic_processors, remove_semantic_processors
from ..stitching.ode_perturbation import perform_ode_step
from ..stitching.spatial_routing import apply_spatial_routing

logger = logging.getLogger(__name__)

def run_latent_stitching(config: FlowStitchConfig, db_path: str, pipe=None):
    """
    Unified Latent Stitching pipeline.
    
    Supports:
    - x0_mode: "spherical_weld" (S^D variance preserving), "hard_mosaic", "zero_inpainting"
    - damping_mode: "kts" (exponential late decay), "constant" (D=1.0), "none" (D=0.0)
    - optional spatial routing & optional continuous aura smoothing
    """
    x0_mode = getattr(config, "x0_mode", "spherical_weld" if config.stitching_mode == "dual" else "hard_mosaic")
    damping_mode = getattr(config, "damping_mode", "kts")
    logger.info(f"--- Latent Stitching [mode={config.stitching_mode}, x0_mode={x0_mode}, damping={damping_mode}] ---")
    
    device = torch.device(config.device)
    if pipe is None:
        pipe = FluxPipeline.from_pretrained(config.model_id, torch_dtype=config.dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # 1. Caricamento Dati
    A_target = load_tensors(os.path.join(db_path, "A_target.pt")).to(device, dtype=config.dtype)
    v0_db = load_tensors(os.path.join(db_path, "v0_velocity.pt")).to(device, dtype=config.dtype)
    x0_db = load_tensors(os.path.join(db_path, "x0_noise.pt")).to(device, dtype=config.dtype)

    # 2. Routing Spaziale (Affine Scaling e Posizionamento del soggetto)
    if config.subject_scale < 0.999 or any(abs(o) > 1e-4 for o in config.subject_offset):
        logger.info(
            f"Applicazione Spatial Routing: scale={config.subject_scale:.2f}, "
            f"offset={config.subject_offset}"
        )
        A_target, v0_db, x0_db = apply_spatial_routing(
            mask=A_target,
            v0=v0_db,
            x0=x0_db,
            scale=config.subject_scale,
            offset=config.subject_offset,
        )
    
    # 3. Smussamento Continuo Opzionale dei Bordi
    if config.stitching_mode == "dual" and config.gaussian_kernel_size > 1:
        b, seq, c = A_target.shape
        h = w = int(seq ** 0.5)
        assert h * w == seq, f"seq_len must be a perfect square, got {seq}"
        
        # Reshape in formato immagine 2D [B, 1, H, W]
        A_target_2d = A_target.transpose(1, 2).view(b, c, h, w).to(torch.float32)
        
        k_size = config.gaussian_kernel_size
        if k_size % 2 == 0:
            k_size += 1
        A_target_blurred = TF.gaussian_blur(
            A_target_2d,
            kernel_size=[k_size, k_size],
            sigma=[config.gaussian_sigma, config.gaussian_sigma],
        )
        
        max_val = A_target_blurred.max()
        if max_val > 1e-6:
            A_target_blurred = A_target_blurred / max_val
            
        A_target = A_target_blurred.view(b, c, seq).transpose(1, 2).to(config.dtype)
        logger.info(
            f"Continuous Soft Mask post-blur: range=[{A_target.min():.3f}, {A_target.max():.3f}]"
        )
        
    # 4. Setup Modello e Hook Semantico (Opzionale, disattivato di default per ODE puro)
    use_grafting = getattr(config, "use_attention_grafting", False)
    if use_grafting:
        logger.info("Iniezione del Custom Attention Processor (Semantic Grafting)...")
        pipe = inject_semantic_processors(
            pipe, A_target, injection_strength=config.injection_strength, target_blocks="single"
        )
        
    # 5. Inizializzazione Termodinamica del Rumore (Step 2) ed Integrazione ODE
    with torch.no_grad():
        ambient_embeds, ambient_pooled, ambient_txt_ids = pipe.encode_prompt(
            config.ambient_prompt, prompt_2=None
        )
        generator = torch.Generator(device=device).manual_seed(config.seed)
        latents_lake, latent_image_ids = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels // 4, 1024, 1024, config.dtype, device, generator
        )
        
        # Dispatch x0 modes:
        if x0_mode == "spherical_weld":
            logger.info("Saldatura Termodinamica Sferica del Rumore Iniziale: Var(x0) = 1 ovunque...")
            alpha_ambient = torch.clamp(1.0 - A_target, min=0.0, max=1.0)
            alpha_target = torch.clamp(A_target, min=0.0, max=1.0)
            latents = torch.sqrt(alpha_ambient) * latents_lake + torch.sqrt(alpha_target) * x0_db
        elif x0_mode == "zero_inpainting":
            logger.info("Zero Inpainting del Rumore Iniziale (Pura emergenza differenziale guidata solo da v0)...")
            latents = latents_lake
        elif x0_mode == "hard_mosaic":
            logger.info("Mosaico Lineare del Rumore Iniziale (Baseline classica di contrasto)...")
            latents = latents_lake * (1.0 - A_target) + x0_db * A_target
        else:
            raise ValueError(f"Unknown x0_mode: '{x0_mode}'. Choose 'spherical_weld', 'zero_inpainting', or 'hard_mosaic'.")
            
        logger.info(f"Integrazione Differenziale ODE (steps={config.steps}, damping={damping_mode})...")
        pipe.scheduler.set_timesteps(config.steps, device=device)
        
        smoother = None
        if getattr(config, "use_ema", False):
            from ..stitching.ema_smoothing import TrajectoryEMA
            smoother = TrajectoryEMA(decay=config.ema_decay)
            logger.info(f"Trajectory EMA attivato con decay={config.ema_decay}")
        
        for t in pipe.scheduler.timesteps:
            latents = perform_ode_step(
                pipe=pipe,
                latents=latents,
                t=t,
                ambient_pooled=ambient_pooled,
                ambient_embeds=ambient_embeds,
                ambient_txt_ids=ambient_txt_ids,
                latent_image_ids=latent_image_ids,
                A_target=A_target,
                v0_target=v0_db,
                lambda_v0=config.lambda_v0,
                kts_t_cutoff=config.t_cutoff,
                kts_gamma=config.gamma_kts,
                mode=config.stitching_mode,
                damping_mode=damping_mode,
                smoother=smoother,
            )

    # 6. Pulizia Hook se attivati
    if use_grafting:
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
    out_filename = getattr(config, "output_filename", None) or f"stitching_{x0_mode}_{damping_mode}.png"
    out_path = os.path.join(config.output_root, out_filename)
    image.save(out_path)
    logger.info(f"[SUCCESS] Immagine salvata in: {out_path}")
    return image

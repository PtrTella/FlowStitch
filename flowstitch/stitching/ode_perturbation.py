import torch
from .kts import apply_kts
import logging

logger = logging.getLogger(__name__)

def perform_ode_step(
    pipe,
    latents,
    t,
    ambient_pooled,
    ambient_embeds,
    ambient_txt_ids,
    latent_image_ids,
    A_target,
    v0_target,
    lambda_v0,
    kts_t_cutoff=0.5,
    kts_gamma=4.0,
    mode="dual",
    damping_mode="kts",
    smoother=None,
):
    """
    Esegue un passo di integrazione ODE applicando KTS e perturbazione se richiesto.
    """
    device = latents.device
    dtype = latents.dtype
    
    t_val = t.item() if torch.is_tensor(t) else float(t)
    # FluxPipeline scheduler produces timesteps in [0, 1000]; the transformer
    # expects normalized [0, 1] values (matching official FluxPipeline: timestep/1000)
    t_norm = t_val / 1000.0
    timestep_1d = torch.tensor([t_norm] * latents.shape[0], device=device, dtype=dtype)
    
    # 1. Calcolo del campo di velocità ambientale
    v_ambient = pipe.transformer(
        hidden_states=latents, 
        timestep=timestep_1d, 
        guidance=None,
        pooled_projections=ambient_pooled, 
        encoder_hidden_states=ambient_embeds,
        txt_ids=ambient_txt_ids, 
        img_ids=latent_image_ids, 
        return_dict=False
    )[0]
    
    # 2. Perturbazione Tangenziale con KTS
    if mode in ["dual", "full", "mosaico"]:
        v_stitch = apply_kts(
            v_ambient=v_ambient,
            v_target=v0_target,
            mask=A_target,
            t_norm=t_norm,
            lambda_val=lambda_v0,
            t_cutoff=kts_t_cutoff,
            gamma=kts_gamma,
            damping_mode=damping_mode,
        )
        if smoother is not None:
            v_stitch = smoother.update(v_stitch)
    else:
        v_stitch = v_ambient
        
    # 3. Avanzamento Euleriano
    latents = pipe.scheduler.step(v_stitch, t, latents, return_dict=False)[0]
    
    return latents

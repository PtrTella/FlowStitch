import torch
import math

def compute_damping_factor(
    t_norm: float,
    t_cutoff: float = 0.5,
    gamma: float = 4.0,
    damping_mode: str = "kts",
) -> float:
    """
    Kinetic Trajectory Shaping (KTS) damping factor D(t).
    
    In FLUX Flow Matching, integration runs backward in time:
      t_norm = 1.0 (pure noise) ---> t_norm = 0.0 (clean image).
      
    Regimes:
    - "kts":
        - Macro Injection Phase (t_norm >= t_cutoff):
            D(t) = 1.0 (full steering towards the target object geometry)
        - Harmonization & Convergence Phase (t_norm < t_cutoff):
            D(t) decays smoothly towards 0.0 as t_norm -> 0.0:
            D(t) = exp(-gamma * (t_cutoff - t_norm) / t_cutoff)
    - "constant":
        D(t) = 1.0 throughout the entire trajectory (no late-stage harmonization)
    - "none":
        D(t) = 0.0 (pure ambient flow, no velocity steering)
    """
    if damping_mode == "constant":
        return 1.0
    elif damping_mode == "none":
        return 0.0
    elif damping_mode == "kts":
        if t_norm >= t_cutoff:
            return 1.0
        # Smooth exponential decay towards t_norm = 0
        progress_past_cutoff = (t_cutoff - t_norm) / max(t_cutoff, 1e-6)
        return math.exp(-gamma * progress_past_cutoff)
    else:
        raise ValueError(f"Unknown damping_mode: '{damping_mode}'. Choose from 'kts', 'constant', 'none'.")


def apply_kts(
    v_ambient: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor,
    t_norm: float,
    lambda_val: float,
    t_cutoff: float = 0.5,
    gamma: float = 4.0,
    damping_mode: str = "kts",
) -> torch.Tensor:
    """
    Applies damped ODE tangent-space perturbation:
        v_stitch = v_ambient + D(t) * lambda * [mask * (v_target - v_ambient)]
    """
    D_t = compute_damping_factor(t_norm, t_cutoff=t_cutoff, gamma=gamma, damping_mode=damping_mode)
    delta_v = mask * (v_target - v_ambient)
    return v_ambient + (D_t * lambda_val * delta_v)

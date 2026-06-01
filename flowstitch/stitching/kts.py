import torch
import math

def compute_damping_factor(t: float, t_cutoff: float = 0.8, gamma: float = 5.0) -> float:
    """
    Kinetic Trajectory Shaping (KTS).
    D(t) = exp(-gamma * max(0.0, t - t_cutoff))
    Prevents terminal singularity as t -> 1.0 (or t_val -> 1000)
    """
    return math.exp(-gamma * max(0.0, t - t_cutoff))

def apply_kts(v_ambient: torch.Tensor, v_target: torch.Tensor, mask: torch.Tensor, t_norm: float, lambda_val: float, t_cutoff: float = 0.8, gamma: float = 5.0) -> torch.Tensor:
    """
    Applies damped ODE perturbation.
    v_stitch = v_ambient + D(t) * lambda * [M * (v_target - v_ambient)]
    """
    D_t = compute_damping_factor(t_norm, t_cutoff, gamma)
    delta_v = mask * (v_target - v_ambient)
    return v_ambient + (D_t * lambda_val * delta_v)

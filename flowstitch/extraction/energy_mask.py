import torch

def compute_chebyshev_threshold(energy_map: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """
    Statistical energy gating based on Chebyshev's inequality.
    tau = mu + k * sigma
    Note: May lead to 'Thermodynamic Voids' (annular masks) on uniform surfaces
    due to energy concentration at the object boundaries.
    """
    mu = energy_map.mean()
    sigma = energy_map.std()
    
    threshold = mu + k * sigma
    return (energy_map > threshold).to(energy_map.dtype)

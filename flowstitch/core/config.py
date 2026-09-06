from __future__ import annotations
from dataclasses import dataclass, field
import torch

@dataclass
class FlowStitchConfig:
    """Centralized configuration for FlowStitch pipelines."""
    
    # Model parameters
    model_id: str = "black-forest-labs/FLUX.1-schnell"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    
    # Dataset generation parameters
    output_root: str = "data/dataset_v1"
    seed: int = 42
    steps: int = 4
    target_layers: list[int] = field(default_factory=lambda: [0, 10])
    
    # Stitching parameters
    ambient_prompt: str = "a crystal clear lake"
    lambda_v0: float = 1.0
    injection_strength: float = 0.85
    t_cutoff: float = 0.5
    gamma_kts: float = 4.0
    ema_decay: float = 0.3
    use_ema: bool = True
    stitching_mode: str = "dual" # "mosaico" or "dual"
    
    # Step 2: Initial Noise (x0) Behaviors
    # "spherical_weld" : Variance-preserving S^D Riemannian manifold interpolation (sqrt(1-M)*x_amb + sqrt(M)*x_tgt)
    # "hard_mosaic"    : Standard linear interpolation ((1-M)*x_amb + M*x_tgt)
    # "zero_inpainting" : Pure differential velocity steering without noise injection (x_amb)
    x0_mode: str = "spherical_weld"
    
    # Step 3: Velocity Damping Regimes
    # "kts"      : Exponential decay for t < t_cutoff towards 0.0 (harmonization)
    # "constant" : Full injection D(t) = 1.0 until terminal step
    # "none"     : No velocity steering D(t) = 0.0
    damping_mode: str = "kts"
    
    # Attention grafting toggle (default False for pure differential ODE stitching)
    use_attention_grafting: bool = False
    
    # Mask method ("hermite", "fused_linear", "calibrated_attention", "attention", etc.)
    mask_method: str = "hermite"
    
    # Spatial Routing & Continuous Aura parameters
    subject_scale: float = 1.0
    subject_offset: tuple[float, float] = (0.0, 0.0)
    gaussian_kernel_size: int = 3
    gaussian_sigma: float = 2.0
    output_filename: str | None = None
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

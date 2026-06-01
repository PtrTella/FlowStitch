from dataclasses import dataclass, field
from typing import List, Optional
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
    target_layers: List[int] = field(default_factory=lambda: [0, 10])
    
    # Stitching parameters
    ambient_prompt: str = "a crystal clear lake"
    lambda_v0: float = 1.0
    injection_strength: float = 0.85
    t_cutoff: float = 0.8
    gamma_kts: float = 5.0
    ema_decay: float = 0.3
    use_ema: bool = True
    stitching_mode: str = "dual" # "mosaico", "dual", "full"
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

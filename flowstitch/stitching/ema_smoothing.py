import torch

class TrajectoryEMA:
    """
    Exponential Moving Average smoother for ODE trajectory fields.
    Applies temporal smoothing: x̄_t = γ · x_t + (1 - γ) · x̄_{t-1}
    
    Originally designed for attention maps (Look-Back Flows), now used
    on velocity fields to add inertia to the stitching trajectory.
    """
    def __init__(self, decay: float = 0.3):
        self.decay = decay
        self.running_avg = None
    
    def update(self, field: torch.Tensor) -> torch.Tensor:
        if self.running_avg is None:
            self.running_avg = field.clone()
        else:
            self.running_avg = self.decay * field + (1.0 - self.decay) * self.running_avg
        return self.running_avg

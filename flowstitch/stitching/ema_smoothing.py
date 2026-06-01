import torch

class AttentionEMA:
    """
    Look-Back Flows (EMA Temporale) per smussare le mappe di attenzione.
    A_bar_t = gamma * A_t + (1 - gamma) * A_bar_{t-1}
    """
    def __init__(self, decay: float = 0.3):
        self.decay = decay
        self.running_avg = None
    
    def update(self, attention_map: torch.Tensor) -> torch.Tensor:
        if self.running_avg is None:
            self.running_avg = attention_map.clone()
        else:
            self.running_avg = self.decay * attention_map + (1.0 - self.decay) * self.running_avg
        return self.running_avg

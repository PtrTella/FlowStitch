import torch
import torch.nn.functional as F

def extract_attention_mask(layer_10_attn: torch.Tensor, token_indices: list, normalize: bool = True) -> torch.Tensor:
    """
    Extracts a basic semantic mask from cross-attention weights.
    layer_10_attn: [1, heads, 4096, text_seq_len]
    Returns shape: [1, 4096, 1]
    """
    # Average across the selected tokens
    attn_maps_list = [layer_10_attn[0, :, :, idx].mean(dim=0) for idx in token_indices]
    attn_map = torch.clamp(torch.stack(attn_maps_list).sum(dim=0), min=0.0, max=1.0)
    
    if normalize:
        attn_min, attn_max = attn_map.min(), attn_map.max()
        attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)
        
    return attn_map.unsqueeze(0).unsqueeze(-1).to(torch.float32)

def otsu_threshold(mask: torch.Tensor, bins: int = 256) -> torch.Tensor:
    """Applies Otsu's thresholding to a 1D/2D continuous mask to make it binary."""
    orig_dtype = mask.dtype
    mask_f32 = mask.to(torch.float32)
    m_flat = mask_f32.flatten()
    hist = torch.histc(m_flat, bins=bins, min=0.0, max=1.0)
    total = hist.sum()
    
    sum_total = torch.dot(torch.arange(bins, dtype=torch.float32, device=mask.device), hist)
    
    weight_b = 0.0
    sum_b = 0.0
    var_max = 0.0
    threshold_idx = 0
    
    for t in range(bins):
        weight_b += hist[t].item()
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
            
        sum_b += t * hist[t].item()
        
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        
        var_between = weight_b * weight_f * ((mean_b - mean_f) ** 2)
        
        if var_between > var_max:
            var_max = var_between
            threshold_idx = t
            
    optimal_thresh = threshold_idx / bins
    return (mask_f32 > optimal_thresh).to(orig_dtype)

import torch
from .spectral_mask import compute_fiedler_mask
from .attention_mask import otsu_threshold

def hybrid_semantic_decomposition(keys_img: torch.Tensor, attn_map: torch.Tensor, target_resolution: int = 32) -> torch.Tensor:
    """
    Combines Spectral Graph Partitioning with Attention-based Semantic Prior.
    1. Extracts spectral structural boundaries (Fiedler vector)
    2. Uses attention map as semantic prior to select the correct partition
    """
    # Get the raw spectral partitions (binary mask)
    spectral_mask = compute_fiedler_mask(keys_img, target_resolution=target_resolution)
    
    # Get the semantic core from attention
    semantic_core = otsu_threshold(attn_map)
    
    # Check overlap to determine if the spectral mask needs inversion
    # Fiedler vector partition might label object as 0 and background as 1
    overlap_normal = (spectral_mask * semantic_core).sum()
    overlap_inverted = ((1.0 - spectral_mask) * semantic_core).sum()
    
    if overlap_inverted > overlap_normal:
        spectral_mask = 1.0 - spectral_mask
        
    return spectral_mask

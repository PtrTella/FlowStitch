import numpy as np
import torch
import logging
from .attention_mask import otsu_threshold

logger = logging.getLogger(__name__)

def extract_tda_mask(
    v0: torch.Tensor,
    attn_mask: torch.Tensor,
    threshold_metric: float = 0.5,
    min_pixels: int = 10
) -> torch.Tensor:
    """
    Topological Data Analysis (TDA) based mask extraction.
    Resolves the 'Thermodynamic Void' by grouping latent velocity vectors 
    using single-linkage clustering (homology H0 filtration) on cosine distance.
    
    v0: [1, seq_len, dim] or [seq_len, dim]
    attn_mask: [1, seq_len, 1] or [seq_len]
    threshold_metric: Cosine distance threshold for single-linkage clustering.
    """
    device = v0.device
    dtype = v0.dtype
    
    # Ensure numpy arrays
    if isinstance(v0, torch.Tensor):
        v0_np = v0.detach().cpu().to(torch.float32).numpy()
    else:
        v0_np = np.array(v0)
        
    if isinstance(attn_mask, torch.Tensor):
        attn_mask_np = attn_mask.detach().cpu().to(torch.float32).numpy()
    else:
        attn_mask_np = np.array(attn_mask)
        
    if len(v0_np.shape) == 3:
        v0_np = v0_np[0] # [seq_len, dim]
        
    seq_len = v0_np.shape[0]
    
    # 1. Normalize attention mask to [0, 1] range to ensure Otsu thresholding works correctly
    attn_min, attn_max = attn_mask_np.min(), attn_mask_np.max()
    if (attn_max - attn_min) > 1e-8:
        attn_mask_np = (attn_mask_np - attn_min) / (attn_max - attn_min)
    else:
        attn_mask_np = np.zeros_like(attn_mask_np)

    # Compute Semantic Core from Attention via Otsu threshold
    semantic_core = otsu_threshold(torch.tensor(attn_mask_np)).numpy()
    semantic_core_flat = semantic_core.flatten()
    
    mask_indices = np.where(semantic_core_flat > 0.5)[0]
    
    if len(mask_indices) < min_pixels:
        logger.warning(f"Troppi pochi pixel nel Semantic Core ({len(mask_indices)}). Ritorno la maschera di attenzione originale.")
        return torch.tensor(semantic_core, device=device, dtype=dtype).view(1, seq_len, 1)
        
    # 2. Extract core vectors
    core_vectors = v0_np[mask_indices]
    
    try:
        from sklearn.metrics.pairwise import cosine_distances
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        import ripser
        
        # 3. Compute cosine distances
        dist_matrix = cosine_distances(core_vectors)
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)
        
        # 4. Optional: persistent homology diagram for validation/logging
        # We compute H0 using ripser to verify topological features
        result = ripser.ripser(dist_matrix, distance_matrix=True, maxdim=0)
        dgms = result['dgms'][0]
        finite_deaths = dgms[np.isfinite(dgms[:, 1])][:, 1]
        max_persistence = np.max(finite_deaths) if len(finite_deaths) > 0 else 0.0
        logger.info(f"TDA H0 calcolato. Max persistenza finita: {max_persistence:.4f}")
        
        # 5. Group vectors using single linkage clustering (equivalent to H0 filtration)
        condensed_dist = squareform(dist_matrix, checks=False)
        Z = linkage(condensed_dist, method='single')
        clusters = fcluster(Z, t=threshold_metric, criterion='distance')
        
        # Find the largest cluster (the persistent topological component)
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        largest_cluster_id = unique_clusters[np.argmax(counts)]
        
        # Build the final mask
        final_mask_flat = np.zeros(seq_len, dtype=np.float32)
        # Set pixels in the largest cluster to 1.0
        final_mask_flat[mask_indices[clusters == largest_cluster_id]] = 1.0
        
        logger.info(f"Maschera TDA estratta con successo. Pixel isolati: {int(final_mask_flat.sum())}/{len(mask_indices)}")
        
        return torch.tensor(final_mask_flat, device=device, dtype=dtype).view(1, seq_len, 1)
        
    except ImportError as e:
        logger.error(f"Errore di importazione librerie scientifiche per TDA: {e}. Ritorno fallback Otsu.")
        return torch.tensor(semantic_core, device=device, dtype=dtype).view(1, seq_len, 1)
    except Exception as e:
        logger.error(f"Errore durante l'estrazione TDA: {e}. Ritorno fallback Otsu.")
        return torch.tensor(semantic_core, device=device, dtype=dtype).view(1, seq_len, 1)

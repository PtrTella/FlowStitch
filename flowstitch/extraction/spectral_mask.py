import torch
import torch.nn.functional as F

def compute_fiedler_mask(keys_img: torch.Tensor, target_resolution: int = 32) -> torch.Tensor:
    """
    Adattamento di DiffCut ai Single-Stream blocks di Flux MMDiT.
    1. Decimazione bilineare dei Key tokens (64x64 -> 32x32)
    2. Matrice di affinità coseno: S = (K*K^T) / ||K||^2
    3. Laplaciano non normalizzato: L = D - S
    4. Autovettore di Fiedler (2° autovalore più piccolo)
    5. Zero-crossing per Normalized Cut
    6. Upsampling guidato alla risoluzione nativa
    """
    # Cast to float32 for CPU support of linear algebra operations
    orig_dtype = keys_img.dtype
    keys_img_f32 = keys_img.to(torch.float32)
    
    # Assuming keys_img shape: [1, seq_len, dim]
    b, seq_len, dim = keys_img_f32.shape
    h = w = int(seq_len ** 0.5)
    
    # Reshape and downsample to avoid OOM
    keys_2d = keys_img_f32.view(b, h, w, dim).permute(0, 3, 1, 2)
    keys_down = F.interpolate(keys_2d, size=(target_resolution, target_resolution), mode='bilinear', align_corners=False)
    
    # Flatten back
    keys_flat = keys_down.permute(0, 2, 3, 1).view(b, target_resolution * target_resolution, dim)
    keys_flat = F.normalize(keys_flat, dim=-1)
    
    # Cosine affinity matrix W
    W = torch.bmm(keys_flat, keys_flat.transpose(1, 2))
    # Remove negative affinities and diagonal self-loops
    W = torch.clamp(W, min=0.0)
    W.diagonal(dim1=-2, dim2=-1).zero_()
    
    # Degree matrix and Laplacian
    D = torch.diag_embed(W.sum(dim=-1))
    L = D - W
    
    # Compute Fiedler vector for the first item in batch
    L_matrix = L[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(L_matrix)
    fiedler_vector = eigenvectors[:, 1]
    
    # Zero-crossing for partition
    mask_down = (fiedler_vector > 0).float()
    mask_down_2d = mask_down.view(1, 1, target_resolution, target_resolution)
    
    # Upsample back to original resolution
    mask_up = F.interpolate(mask_down_2d, size=(h, w), mode='nearest')
    mask_up_seq = mask_up.view(1, seq_len, 1)
    
    return mask_up_seq.to(orig_dtype)

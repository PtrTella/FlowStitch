import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def apply_spatial_routing(
    mask: torch.Tensor,
    v0: torch.Tensor,
    x0: torch.Tensor,
    scale: float = 1.0,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies spatial routing (affine scaling and translation) to latent concepts.
    
    Transforms the latent concept tensors on the 64x64 token manifold:
    - mask: [B, seq_len, 1]
    - v0:   [B, seq_len, C]
    - x0:   [B, seq_len, C]
    
    Args:
        mask: Spatial attention / semantic attractor mask.
        v0: Velocity vector field at t=0.
        x0: Initial latent noise field at t=0.
        scale: Scaling factor s in (0.0, 1.0]. E.g. 0.5 halves width and height.
        offset: Normalized translation (dy, dx) in [-1.0, 1.0].
                E.g. (0.2, 0.0) translates 20% downwards toward the water surface.
                
    Returns:
        Tuple of (routed_mask, routed_v0, routed_x0) with identical shapes.
    """
    if scale >= 0.999 and abs(offset[0]) < 1e-4 and abs(offset[1]) < 1e-4:
        return mask, v0, x0

    b, seq, c_v = v0.shape
    h = w = int(seq ** 0.5)
    assert h * w == seq, f"seq_len ({seq}) must be a square grid (e.g. 64x64=4096)"

    device = v0.device
    dtype = v0.dtype

    # 1. Unpack to 2D image representations [B, Channels, H, W]
    # mask: [B, 1, H, W]
    mask_2d = mask.transpose(1, 2).contiguous().view(b, 1, h, w).to(torch.float32)
    # v0:   [B, C, H, W]
    v0_2d = v0.transpose(1, 2).contiguous().view(b, c_v, h, w).to(torch.float32)
    # x0:   [B, C, H, W]
    x0_2d = x0.transpose(1, 2).contiguous().view(b, c_v, h, w).to(torch.float32)

    # Pixel displacements
    dy_px = int(round(offset[0] * h))
    dx_px = int(round(offset[1] * w))
    translate = [dx_px, dy_px]

    logger.info(
        f"Routing semantico: scale={scale:.2f}, offset=({dy_px}px, {dx_px}px)"
    )

    # 2. Continuous Affine Transform with Bilinear Interpolation
    mask_routed_2d = TF.affine(
        mask_2d,
        angle=0.0,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    v0_routed_2d = TF.affine(
        v0_2d,
        angle=0.0,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    x0_routed_2d = TF.affine(
        x0_2d,
        angle=0.0,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    # 3. Peak restoration and energy preservation
    # Bilinear interpolation dampens peak amplitudes; restore max to 1.0 for the mask
    max_mask = mask_routed_2d.max()
    if max_mask > 1e-4:
        mask_routed_2d = mask_routed_2d / max_mask

    # Preserve unit variance N(0, I) for interpolated latent noise x0 where mask is active
    active_region = mask_routed_2d > 0.05
    if active_region.any():
        active_x0 = x0_routed_2d[active_region.expand_as(x0_routed_2d)]
        std = active_x0.std()
        if std > 1e-4:
            x0_routed_2d = x0_routed_2d / std

    # 4. Repack back to sequence format [B, seq_len, Channels]
    mask_out = mask_routed_2d.contiguous().view(b, 1, seq).transpose(1, 2).to(dtype)
    v0_out = v0_routed_2d.contiguous().view(b, c_v, seq).transpose(1, 2).to(dtype)
    x0_out = x0_routed_2d.contiguous().view(b, c_v, seq).transpose(1, 2).to(dtype)

    return mask_out, v0_out, x0_out

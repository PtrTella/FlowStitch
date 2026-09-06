"""
FlowStitch Cluster Utilities
============================
Helper functions for execution on HPC environments (e.g. NVIDIA L40 cluster at unibo).
Provides network proxy configuration and unified single-allocation FLUX.1 pipeline loading.
"""
from __future__ import annotations
import os
import socket
import logging
from typing import Optional

logger = logging.getLogger("FlowStitchClusterUtils")


def setup_cluster_network(force_ipv4: bool = False) -> None:
    """
    Configures network resolution to bypass HPC proxy/firewall issues.
    Forces IPv4 resolution if requested or if FORCE_IPV4 environment variable is "1".
    """
    if force_ipv4 or os.environ.get("FORCE_IPV4", "0") == "1":
        old_getaddrinfo = socket.getaddrinfo

        def new_getaddrinfo(*args, **kwargs):
            res = old_getaddrinfo(*args, **kwargs)
            return [r for r in res if r[0] == socket.AF_INET]

        socket.getaddrinfo = new_getaddrinfo
        logger.info("[CLUSTER NET] Forzata risoluzione IPv4 per bypass firewall/proxy HPC.")


def load_flux_pipeline(
    model_id: str = "black-forest-labs/FLUX.1-schnell",
    device: Optional[str] = None,
    dtype: Optional[any] = None,
):
    """
    Loads FLUX.1 pipeline into GPU VRAM once with optimal memory settings.
    """
    import torch
    from diffusers import FluxPipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    logger.info(f"Caricamento pipeline '{model_id}' su {device} [{dtype}]...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    logger.info(f"[SUCCESS] FLUX.1 caricato con successo in VRAM su {device} [{dtype}].")
    return pipe

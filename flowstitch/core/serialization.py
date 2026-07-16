import os
import torch
import safetensors.torch

def load_tensors(path: str):
    """
    Safely loads tensors from either a .safetensors file or a legacy .pt file.
    If path has a specific extension, it also checks the alternative extension.
    """
    base, ext = os.path.splitext(path)
    if ext in [".safetensors", ".pt"]:
        candidates = [path]
        # Also check other extension
        other_ext = ".pt" if ext == ".safetensors" else ".safetensors"
        candidates.append(base + other_ext)
    else:
        candidates = [path + ".safetensors", path + ".pt", path]

    for p in candidates:
        if not os.path.exists(p):
            continue
        
        # Try loading via safetensors
        try:
            data = safetensors.torch.load_file(p, device="cpu")
            if len(data) == 1 and "__single_tensor__" in data:
                return data["__single_tensor__"]
            return data
        except Exception:
            continue

    raise FileNotFoundError(f"Could not load tensor data from: {path}. Tried candidates: {candidates}. Only .safetensors format is supported (legacy .pt pickle loading is disabled for security).")

def save_tensors(data, path: str):
    """
    Safely saves tensors to a .safetensors file.
    If data is a single torch.Tensor, wraps it in a dict with key '__single_tensor__'.
    """
    base, ext = os.path.splitext(path)
    # Force saving as .safetensors
    out_path = base + ".safetensors"

    if isinstance(data, torch.Tensor):
        payload = {"__single_tensor__": data.contiguous()}
    elif isinstance(data, dict):
        skipped = [k for k, v in data.items() if not isinstance(v, torch.Tensor)]
        if skipped:
            import logging
            logging.getLogger(__name__).warning(f"save_tensors: skipping non-tensor keys: {skipped}")
        payload = {k: v.contiguous() for k, v in data.items() if isinstance(v, torch.Tensor)}
    else:
        raise TypeError("data must be a torch.Tensor or a dictionary of torch.Tensors")

    safetensors.torch.save_file(payload, out_path)

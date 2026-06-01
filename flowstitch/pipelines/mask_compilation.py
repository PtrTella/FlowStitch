import os
import json
import torch
import logging
from transformers import T5Tokenizer
from ..core.tokenizer_utils import find_token_indices
from ..extraction.attention_mask import extract_attention_mask
from ..core.serialization import load_tensors, save_tensors

logger = logging.getLogger(__name__)

def compile_mask_for_scene(db_path: str, word_to_isolate: str, manual_tokens: list = None):
    """
    Compiles a semantic mask from cross-attention maps for a specific scene folder.
    """
    logger.info(f"--- Compilazione Attrattore Semantico per '{word_to_isolate}' in {db_path} ---")

    metadata_path = os.path.join(db_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    target_prompt = metadata["prompt"]

    tokenizer = T5Tokenizer.from_pretrained(
        "google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True
    )

    if manual_tokens is not None:
        token_indices = manual_tokens
    else:
        token_indices = find_token_indices(tokenizer, target_prompt, word_to_isolate)

    if not token_indices:
        raise ValueError(f"CRITICO: Parola '{word_to_isolate}' non trovata nel prompt '{target_prompt}'.")

    attn_target = load_tensors(
        os.path.join(db_path, "attention_maps.pt"),
        map_location="cpu",
    )
    layer_10 = attn_target["layer_10"] 

    A_target = extract_attention_mask(layer_10, token_indices, normalize=True)

    out_file = os.path.join(db_path, "A_target.pt")
    save_tensors(A_target, out_file)
    logger.info(f"[SUCCESS] Maschera compilata e salvata in: {out_file}")

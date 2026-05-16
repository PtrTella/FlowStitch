import os
import json
import torch
import logging
from transformers import T5Tokenizer
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def find_token_indices(tokenizer, prompt: str, target_word: str) -> list:
    """
    Algoritmo di Sliding Window Minimale:
    Ricompone i token per trovare la sequenza esatta (e più corta possibile)
    che contiene la parola cercata. Supera il problema dei sub-token di T5.
    """
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
    target_word = target_word.strip().lower()

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens) + 1):
            # Decodifica il blocco di token corrente
            chunk = tokenizer.decode(tokens[i:j]).strip().lower()

            if target_word in chunk:
                # Verifica che sia il chunk "minimo" per evitare di prendere mezza frase.
                # Se togliendo il primo o l'ultimo token la parola scompare,
                # significa che abbiamo isolato i bordi perfetti.
                chunk_minus_first = (
                    tokenizer.decode(tokens[i + 1 : j]).strip().lower()
                    if i + 1 < j
                    else ""
                )
                chunk_minus_last = (
                    tokenizer.decode(tokens[i : j - 1]).strip().lower()
                    if i < j - 1
                    else ""
                )

                if (
                    target_word not in chunk_minus_first
                    and target_word not in chunk_minus_last
                ):
                    logger.info(
                        f"Ricostruzione semantica trovata: '{chunk}' -> Indici: {list(range(i, j))}"
                    )
                    return list(range(i, j))
    return []


def compile_semantic_mask(
    db_path: str, word_to_isolate: str, manual_tokens: list = None
):
    logger.info(f"--- Compilazione Attrattore Semantico per '{word_to_isolate}' ---")

    # 1. Carica metadati
    metadata_path = os.path.join(db_path, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    target_prompt = metadata["prompt"]

    # 2. Inizializza Tokenizer su CPU
    logger.info(f"Analisi del prompt: '{target_prompt}'")
    tokenizer = T5Tokenizer.from_pretrained(
        "google/t5-v1_1-xxl", legacy=False, clean_up_tokenization_spaces=True
    )

    # 3. Risoluzione degli indici (Automatica o Manuale)
    if manual_tokens is not None:
        logger.warning(
            f"OVERRIDE MANUALE: Utilizzo gli indici token forzati: {manual_tokens}"
        )
        token_indices = manual_tokens
    else:
        token_indices = find_token_indices(tokenizer, target_prompt, word_to_isolate)

    if not token_indices:
        raise ValueError(
            f"CRITICO: Parola '{word_to_isolate}' non trovata nel prompt '{target_prompt}'. Prova con manual_tokens."
        )

    # Per debug, stampiamo cosa stiamo prendendo per buono
    tokens_tensor = tokenizer(target_prompt, return_tensors="pt").input_ids[0]
    extracted_text = tokenizer.decode(tokens_tensor[token_indices])
    logger.info(f"Testo estratto dai token confermati: '{extracted_text}'")

    # 4. Estrazione Fisica della Mappa
    logger.info("Elaborazione del tensore grezzo dell'attenzione...")
    attn_target = torch.load(
        os.path.join(db_path, "attention_maps.pt"),
        map_location="cpu",
        weights_only=True,
    )
    layer_10 = attn_target["layer_10"]  # Dimensioni: [1, 24, 4096, 512]

    # 5. Collasso e Media dei Sub-Token
    # Se il cubo è diviso in 2 token, facciamo la media delle loro mappe spaziali
    attn_maps_list = [layer_10[0, :, :, idx].mean(dim=0) for idx in token_indices]

    # Se la parola è formata da più token, li sommiamo per avere la "sagoma" completa dell'oggetto
    attn_map = torch.clamp(torch.stack(attn_maps_list).sum(dim=0), min=0.0, max=1.0)

    # Normalizzazione probabilistica [0, 1]
    attn_min, attn_max = attn_map.min(), attn_map.max()
    A_target = (attn_map - attn_min) / (attn_max - attn_min + 1e-8)

    # Reshape topologico [1, 4096, 1] e cast a float32
    A_target = A_target.unsqueeze(0).unsqueeze(-1).to(torch.float32)

    # 6. Salvataggio
    out_file = os.path.join(db_path, f"A_target_{word_to_isolate}.pt")
    torch.save(A_target, out_file)
    logger.info(f"[SUCCESS] Maschera compilata e salvata in: {out_file}")
    logger.info(
        f"Shape finale: {A_target.shape} | Range: [{A_target.min():.2f}, {A_target.max():.2f}]"
    )


if __name__ == "__main__":
    # Esempio: Estrazione automatica avanzata (risolve il problema dei sub-token)
    compile_semantic_mask(db_path="data/dataset_v1/a_red_cube", word_to_isolate="cube")

    compile_semantic_mask(
        db_path="data/dataset_v1/a_blue_sphere", word_to_isolate="sphere"
    )
    compile_semantic_mask(
        db_path="data/dataset_v1/a_yellow_pyramid", word_to_isolate="pyramid"
    )

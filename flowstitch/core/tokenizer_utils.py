import logging

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
    
    logger.warning(f"Parola '{target_word}' non trovata nel prompt.")
    return []

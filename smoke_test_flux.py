import os
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig

# 1. Forza il risparmio memoria massimo
os.environ["DIFFUSERS_NO_FLASH_ATTN"] = "1"

model_id = "black-forest-labs/FLUX.1-schnell"

try:
    print("--- CARICAMENTO FLUX OTTIMIZZATO PER 11GB ---")

    # Configurazione per caricare i pezzi pesanti in 4-bit (richiede bitsandbytes)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Carichiamo il Transformer (il pezzo da 23GB) in 4-bit -> scende a circa 6-8GB
    print("Caricamento Transformer in 4-bit...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id, 
        subfolder="transformer", 
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16
    )

    # Carichiamo la pipeline usando il transformer quantizzato
    pipe = FluxPipeline.from_pretrained(
        model_id, 
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    # OFF-LOAD ESTREMO: Sposta layer per layer invece che blocchi interi
    print("Attivazione Sequential CPU Offload...")
    pipe.enable_sequential_cpu_offload()

    print("Inizio generazione (sarà lenta ma dovrebbe farcela)...")
    image = pipe(
        "A futuristic robot in a library",
        num_inference_steps=2, 
        guidance_scale=0.0,
        max_sequence_length=256
    ).images[0]

    image.save("test_flux_2080_success.png")
    print("SUCCESSO ASSOLUTO! Immagine salvata.")

except Exception as e:
    print(f"\nERRORE: {e}")
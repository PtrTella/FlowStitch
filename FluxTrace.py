import torch
from diffusers import FluxPipeline

# 1. Carica il modello Flow Matching sul Cluster (assicurati di avere VRAM a sufficienza)
model_id = "black-forest-labs/FLUX.1-schnell"
pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() # Utile se il cluster è condiviso o ha poca VRAM

# Variabili globali per salvare i nostri tensori "rubati"
v0_tensor = None
x0_tensor = None
step_counter = 0

# 2. Creiamo la nostra "Cimice" (Il Forward Hook)
def hook_fn(module, input, output):
    global v0_tensor, x0_tensor, step_counter
    
    # Il DiT di FLUX prende in input (hidden_states, encoder_hidden_states, pooled_projections, timestep, ...)
    # input[0] è il nostro x_t attuale. Allo step 0, questo è il nostro rumore ancorato x_0.
    if step_counter == 0:
        x0_tensor = input[0].detach().clone().cpu()
        
        # output[0] nel Flow Matching è il campo vettoriale predetto (v_t).
        # Allo step 0, questo è esattamente il nostro v_0!
        v0_tensor = output[0].detach().clone().cpu()
    
    step_counter += 1

# 3. Attacchiamo l'hook al Transformer (DiT)
hook_handle = pipe.transformer.register_forward_hook(hook_fn)

# 4. Impostiamo il Seed Canonico (Fondamentale per canonizzare la geometria!)
generator = torch.Generator(device="cpu").manual_seed(42)
prompt = "A red sports car parked on a green meadow, highly detailed"

print(f"Generazione in corso per il prompt: '{prompt}'...")

# 5. Facciamo partire l'inferenza (FLUX-schnell usa 4 step di default)
image = pipe(
    prompt=prompt,
    output_type="pil",
    num_inference_steps=4,
    generator=generator
).images[0]

# Rimuoviamo l'hook per tenere pulito il modello
hook_handle.remove()

# 6. La Magia Matematica: Calcoliamo x_pred
# Nel Flow Matching (Rectified Flows): v_0 = x_1 - x_0  -->  x_pred = v_0 + x_0
if v0_tensor is not None and x0_tensor is not None:
    x_pred = v0_tensor + x0_tensor
    print("\n[SUCCESS] Estrazione completata!")
    print(f"Forma del Rumore Iniziale (x_0): {x0_tensor.shape}")
    print(f"Forma del Campo Vettoriale (v_0): {v0_tensor.shape}")
    print(f"Forma della Bozza Predetta (x_pred): {x_pred.shape}")
    
    # 7. Salviamo il "Blueprint" su disco (Questo andrà in FAISS in futuro)
    torch.save(x_pred, "x_pred_blueprint.pt")
    print("Tensore x_pred salvato come 'x_pred_blueprint.pt'")
    
    # Opzionale: fai un Average Pooling per vedere quanto diventa piccolo
    # x_pred_pooled = torch.nn.functional.adaptive_avg_pool2d(x_pred, (16, 16))
else:
    print("\n[ERRORE] L'hook non ha catturato i tensori.")
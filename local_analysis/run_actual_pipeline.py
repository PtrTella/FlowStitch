import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys

# Aggiungi la root del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flowstitch.extraction.spectral_mask import compute_fiedler_mask
from flowstitch.stitching.kts import apply_kts
from flowstitch.stitching.ema_smoothing import AttentionEMA

def process_dataset_sample(sample_dir):
    print(f"\n--- Elaborazione Campione: {os.path.basename(sample_dir)} ---")
    
    # 1. Caricamento dati
    attn_path = os.path.join(sample_dir, "attention_maps.pt")
    v0_path = os.path.join(sample_dir, "v0_velocity.pt")
    img_path = os.path.join(sample_dir, "final_image.png")
    
    if not os.path.exists(attn_path) or not os.path.exists(img_path):
        print(f"File mancanti in {sample_dir}, salto...")
        return
        
    print("Caricamento Tensori...")
    attn_dict = torch.load(attn_path, map_location="cpu")
    
    # Scegliamo un layer profondo se disponibile
    layer_key = "layer_10" if "layer_10" in attn_dict else list(attn_dict.keys())[0]
    attn_map = attn_dict[layer_key] # Forma attesa: [1, heads, img_tokens, text_tokens]
    
    print(f"Mappa Attenzione {layer_key} caricata: {attn_map.shape}")
    
    # 2. Estrazione Semantica a Grafi (DiffCut)
    print("Calcolo Spectral Matting (Fiedler Vector)...")
    # Facciamo la media sulle teste dell'attenzione e cast a float32 per linalg su CPU
    # Forma: [1, 4096, 512] (b, seq_len, dim_testo)
    attn_features = attn_map.mean(dim=1).to(torch.float32)
    
    # Calcoliamo la maschera topologica
    # compute_fiedler_mask scala a 32x32 per il calcolo Laplaciano, e poi fa upsample.
    # L'output ha forma [1, seq_len, 1]
    fiedler_mask = compute_fiedler_mask(attn_features, target_resolution=32)
    
    # La seq_len è 4096, quindi l'immagine latente è 64x64
    mask_2d = fiedler_mask.view(64, 64).numpy()
    
    # 3. Dimostrazione Termodinamica (KTS + EMA)
    if os.path.exists(v0_path):
        v0 = torch.load(v0_path, map_location="cpu").to(torch.float32)
        print(f"Velocità ODE v0 caricata: {v0.shape}")
        
        v_target = v0 + (torch.randn_like(v0) * 0.05) # Simulazione deviazione
        
        t_step = 0.90 # Simulazione fase terminale (singolarità)
        
        # Riapplicazione della maschera sul vettore
        # Mask shape è [1, 4096, 1], v0 shape è es. [1, 4096, 64]
        mask_expanded = fiedler_mask.expand_as(v0)
        
        # Damping e Smoothing
        v_damped = apply_kts(v0, v_target, mask_expanded, t_norm=t_step, lambda_val=1.0, t_cutoff=0.8, gamma=5.0)
        smoother = AttentionEMA(decay=0.3)
        v_smoothed = smoother.update(v_damped)
        
        diff_damped = torch.abs(v_damped - v_target).mean().item()
        diff_smoothed = torch.abs(v_smoothed - v_damped).mean().item()
        print(f"Regolarizzazione completata. Diff Damped: {diff_damped:.5f}, Smoothing Delta: {diff_smoothed:.5f}")
    
    # 4. Salvataggio Visualizzazioni
    print("Salvataggio Risultati Visivi...")
    orig_img = Image.open(img_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Immagine Generata (Flux)")
    axes[0].axis("off")
    
    axes[1].imshow(mask_2d, cmap="viridis")
    axes[1].set_title(f"Maschera Zero-Shot\n(DiffCut Fiedler su {layer_key})")
    axes[1].axis("off")
    
    out_img_path = os.path.join(sample_dir, "validation_result.png")
    plt.tight_layout()
    plt.savefig(out_img_path)
    plt.close()
    print(f"Risultato salvato in: {out_img_path}")


def main():
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset_v1"))
    
    if not os.path.exists(dataset_dir):
        print(f"Cartella Dataset non trovata: {dataset_dir}")
        return
        
    samples = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) 
               if os.path.isdir(os.path.join(dataset_dir, d))]
               
    for sample in samples:
        process_dataset_sample(sample)

if __name__ == "__main__":
    main()

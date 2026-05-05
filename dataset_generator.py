import os
import json
import torch
import logging
import traceback
from diffusers import FluxPipeline
from typing import List

# --- CONFIGURAZIONE GLOBALE ---
# Modifica solo questa sezione per i tuoi esperimenti
CONFIG = {
    "model_id": "black-forest-labs/FLUX.1-schnell",
    "cache_dir": "/scratch.hpc/pietro.tellarini2/huggingface_cache",
    "output_root": "/scratch.hpc/pietro.tellarini2/dataset_v1",
    "device": "cuda",
    "dtype": torch.bfloat16,
    "seed": 42,
    "steps": 4,
    "target_layers": [0, 10] # Quali blocchi Transformer spiare per la Cross-Attention
}

PROMPTS = [
    "a red cube",
    "a blue sphere",
    "a yellow pyramid",
    "a red cube and a blue sphere",
    "a blue cube and a red sphere",
    "a red cube on the left, a blue sphere on the right",
    "a blue sphere on the left, a red cube on the right",
    "a majestic mountain",
    "a crystal clear lake",
    "a majestic mountain reflecting in a crystal clear lake"
]

# Configurazione Logging per Slurm
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FUNZIONI DI SUPPORTO ---
def slugify(text: str) -> str:
    """Trasforma un prompt in un nome cartella sicuro."""
    return text.lower().replace(" ", "_").replace(",", "").replace(".", "")[:50]

# --- CORE MATEMATICO: PATCH DELL'ATTENZIONE ---
class AttnProcessorWrapper:
    """
    Involucro che intercetta il calcolo dell'attenzione originale, 
    ne estrae i pesi matematici, e poi fa proseguire il calcolo normale.
    """
    def __init__(self, original_processor, layer_idx: int, capturer):
        self.original_processor = original_processor
        self.layer_idx = layer_idx
        self.capturer = capturer

    def __call__(
        self, 
        attn, 
        hidden_states, 
        encoder_hidden_states=None, 
        attention_mask=None, 
        image_rotary_emb=None,
        **kwargs
    ):
        if image_rotary_emb is not None:
            kwargs['image_rotary_emb'] = image_rotary_emb

        # Catturiamo SOLO allo step 0 (il "Sopralluogo")
        if self.capturer.is_capturing and self.capturer.step <= 1:
            
            # PROTEZIONE VRAM: Calcoliamo la mappa senza tracciare i gradienti
            with torch.no_grad():
                batch_size = hidden_states.shape[0]
                
                # 1. Calcolo Proiezioni Immagine
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                
                inner_dim = key.shape[-1]
                head_dim = inner_dim // attn.heads
                
                query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                
                # 2. Calcolo Proiezioni Testo
                encoder_query = attn.add_q_proj(encoder_hidden_states)
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                
                encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                
                if attn.norm_added_q is not None:
                    encoder_query = attn.norm_added_q(encoder_query)
                if attn.norm_added_k is not None:
                    encoder_key = attn.norm_added_k(encoder_key)
                
                # 3. Joint Attention (Unione Testo + Immagine) sull'asse della sequenza
                query = torch.cat([encoder_query, query], dim=2)
                key = torch.cat([encoder_key, key], dim=2)
                
                # Applica Rotary Embeddings se presenti (Cruciale per la coerenza spaziale!)
                image_rotary_emb = kwargs.get("image_rotary_emb", None)
                if image_rotary_emb is not None:
                    from diffusers.models.attention_processor import apply_rope
                    query, key = apply_rope(query, key, image_rotary_emb)
                
                # 4. Calcolo Pesi Attenzione
                attn_weights = torch.matmul(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
                attn_weights = torch.softmax(attn_weights, dim=-1)
                
                # 5. Estrazione Cross-Attention (Immagine -> Testo)
                n_text = encoder_hidden_states.shape[1]
                cross_attn = attn_weights[:, :, n_text:, :n_text].detach().clone().cpu()
                
                self.capturer.attn_maps[f"layer_{self.layer_idx}"] = cross_attn
            
        # Prosegui col calcolo standard (non modificato)
        return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

# --- MANAGER DEGLI HOOK ---
class FluxDataCapturer:
    """Gestisce il ciclo di vita degli hook sul modello (Attacco, Reset, Rimozione)."""
    def __init__(self, transformer, target_layers: List[int]):
        self.transformer = transformer
        self.target_layers = target_layers
        self.x0 = None
        self.v0 = None
        self.attn_maps = {} 
        self.step = 0
        self.handles = []
        self.original_processors = {} # Fondamentale per pulire la memoria
        self.is_capturing = False

    def _transformer_hook(self, module, args, kwargs, output):
        if self.is_capturing and self.step == 0:
            hidden_states = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
            self.x0 = hidden_states.detach().clone().cpu()
            
            # Gestione output (diffusers restituisce spesso una tupla o un oggetto)
            v0_tensor = output[0] if isinstance(output, (tuple, list)) else output
            # Nelle versioni recenti di diffusers potrebbe essere un BaseOutput con .sample
            if hasattr(v0_tensor, 'sample'):
                v0_tensor = v0_tensor.sample
                
            self.v0 = v0_tensor.detach().clone().cpu()
            logger.info("--> Hook Base: x0 e v0 estratti con successo.")
        self.step += 1

    def attach(self):
        # 1. Attacca hook generale
        self.handles.append(self.transformer.register_forward_hook(self._transformer_hook, with_kwargs=True))
        
        # 2. Inietta i processori di attenzione "spia"
        for idx in self.target_layers:
            block = self.transformer.transformer_blocks[idx]
            self.original_processors[idx] = block.attn.processor # Salva l'originale
            block.attn.processor = AttnProcessorWrapper(block.attn.processor, idx, self) # Metti la spia
        
        logger.info(f"Hook e Spie di Attenzione attivate sui layer: {self.target_layers}")

    def reset(self):
        self.x0 = None
        self.v0 = None
        self.attn_maps = {}
        self.step = 0
        self.is_capturing = True

    def remove(self):
        # 1. Rimuovi hook generale
        for h in self.handles:
            h.remove()
        self.handles = []
        
        # 2. Ripristina processori attenzione originali (Cruciale!)
        for idx, original_proc in self.original_processors.items():
            self.transformer.transformer_blocks[idx].attn.processor = original_proc
        self.original_processors.clear()
        
        logger.info("Hook e Spie rimossi. Modello ripristinato allo stato originale.")

# --- PIPELINE PRINCIPALE ---
def main():
    os.makedirs(CONFIG["output_root"], exist_ok=True)
    
    logger.info(f"Caricamento {CONFIG['model_id']} in VRAM...")
    try:
        pipe = FluxPipeline.from_pretrained(
            CONFIG["model_id"],
            torch_dtype=CONFIG["dtype"],
            cache_dir=CONFIG["cache_dir"],
            # Rimuoviamo local_files_only per evitare crash se manca un meta-file
        ).to(CONFIG["device"])
    except Exception as e:
        logger.error(f"[FATAL] Errore nel caricamento del modello: {e}")
        return # Se non carica il modello, è inutile continuare
    
    capturer = FluxDataCapturer(pipe.transformer, CONFIG["target_layers"])
    capturer.attach()
    
    for prompt in PROMPTS:
        slug = slugify(prompt)
        path = os.path.join(CONFIG["output_root"], slug)
        os.makedirs(path, exist_ok=True)
        
        logger.info(f"--- Processando: '{prompt}' ---")
        capturer.reset()
        generator = torch.Generator(device="cpu").manual_seed(CONFIG["seed"])
        
        # RETE DI SALVATAGGIO: Se un prompt esplode, non muore l'intero script
        try:
            # Generazione (fa scattare gli hook)
            output = pipe(
                prompt=prompt,
                num_inference_steps=CONFIG["steps"],
                generator=generator,
                output_type="pil"
            )
            
            # Salvataggio Immagine
            output.images[0].save(os.path.join(path, "final_image.png"))
            
            # Validazione e Salvataggio Tensori
            if capturer.x0 is not None and capturer.v0 is not None:
                torch.save(capturer.x0, os.path.join(path, "x0_noise.pt"))
                torch.save(capturer.v0, os.path.join(path, "v0_velocity.pt"))
                
                # LA MAGIA DEL FLOW MATCHING
                x_pred = capturer.x0 + capturer.v0
                torch.save(x_pred, os.path.join(path, "x_pred.pt"))
                
                # Salvataggio Mappe Attenzione
                if capturer.attn_maps:
                    torch.save(capturer.attn_maps, os.path.join(path, "attention_maps.pt"))
                
                # Metadati
                metadata = {
                    "prompt": prompt,
                    "seed": CONFIG["seed"],
                    "model": CONFIG["model_id"],
                    "steps": CONFIG["steps"],
                    "layers_captured": list(capturer.attn_maps.keys()),
                    "math_proof": "x_pred = x0 + v0 computed successfully"
                }
                with open(os.path.join(path, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=4)
                    
                logger.info(f"[SUCCESS] Dati matematici e mappe salvati in {path}")
            else:
                logger.error(f"[ERROR] Hook non scattato per: {prompt}")

        except Exception as e:
            # Se la generazione o il salvataggio falliscono, logga l'errore ma CONTINUA
            logger.error(f"[CRASH PROMPT] Errore imprevisto su '{prompt}': {e}")
            logger.error(traceback.format_exc())
            logger.info("Pulisco la VRAM e passo al prossimo prompt...")
            
        finally:
            # Questo blocco viene eseguito SEMPRE, sia che vada bene, sia che fallisca
            # Svuotiamo la cache di PyTorch per evitare OOM (Out Of Memory) a valanga
            torch.cuda.empty_cache()

    # Ripuliamo il modello prima di chiudere
    capturer.remove()
    logger.info("Generazione Dataset completata!")

if __name__ == "__main__":
    main()
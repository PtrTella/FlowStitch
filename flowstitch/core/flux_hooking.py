import torch
import logging
from typing import List

logger = logging.getLogger(__name__)

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

        # Catturiamo le mappe se siamo nel timestep giusto
        if self.capturer.is_capturing and (self.capturer.step <= self.capturer.max_capture_step):
            
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
                
                # Supporto per cattura multi-step (EMA)
                if self.capturer.max_capture_step > 1:
                    key_name = f"layer_{self.layer_idx}_step_{self.capturer.step}"
                else:
                    key_name = f"layer_{self.layer_idx}"
                    
                self.capturer.attn_maps[key_name] = cross_attn
            
        # Prosegui col calcolo standard (non modificato)
        return self.original_processor(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)


class FluxDataCapturer:
    """
    Gestisce il ciclo di vita degli hook sul modello.
    Supporta l'utilizzo come Context Manager.
    """
    def __init__(self, transformer, target_layers: List[int], max_capture_step: int = 1):
        self.transformer = transformer
        self.target_layers = target_layers
        self.max_capture_step = max_capture_step
        self.x0 = None
        self.v0 = None
        self.attn_maps = {} 
        self.step = 0
        self.handles = []
        self.original_processors = {} 
        self.is_capturing = False

    def _transformer_hook(self, module, args, kwargs, output):
        """Post-forward hook on the transformer.
        
        NOTE: AttnProcessorWrapper captures attention maps DURING the forward pass,
        while this hook fires AFTER. Both share self.step, but since AttnProcessors
        execute before this hook increments step, the semantics are consistent only
        when max_capture_step == 1. For multi-step capture, the processors see step N
        while this hook also processes step N then increments to N+1.
        """
        if self.is_capturing and self.step == 0:
            hidden_states = kwargs.get('hidden_states', args[0] if len(args) > 0 else None)
            if hidden_states is not None:
                self.x0 = hidden_states.detach().clone().cpu()
            
            # Gestione output (diffusers restituisce spesso una tupla o un oggetto)
            v0_tensor = output[0] if isinstance(output, (tuple, list)) else output
            if hasattr(v0_tensor, 'sample'):
                v0_tensor = v0_tensor.sample
                
            if v0_tensor is not None:
                self.v0 = v0_tensor.detach().clone().cpu()
            logger.debug("Hook Base: x0 e v0 estratti.")
            
        self.step += 1

    def attach(self):
        # 1. Attacca hook generale
        self.handles.append(self.transformer.register_forward_hook(self._transformer_hook, with_kwargs=True))
        
        # 2. Inietta i processori di attenzione "spia"
        for idx in self.target_layers:
            block = self.transformer.transformer_blocks[idx]
            self.original_processors[idx] = block.attn.processor
            block.attn.processor = AttnProcessorWrapper(block.attn.processor, idx, self)
        
        self.is_capturing = True
        logger.info(f"Hook e Spie attivate sui layer: {self.target_layers}")

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
        
        # 2. Ripristina processori attenzione originali
        for idx, original_proc in self.original_processors.items():
            self.transformer.transformer_blocks[idx].attn.processor = original_proc
        self.original_processors.clear()
        self.is_capturing = False
        logger.info("Hook e Spie rimossi.")

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

import torch
import logging

logger = logging.getLogger(__name__)

class SemanticGraftingProcessor:
    """
    Custom Attention Processor unificato per il Latent Stitching.
    Avvolge i processori di attenzione di FLUX per iniettare l'attrattore semantico.
    """
    def __init__(self, original_processor, A_target: torch.Tensor, injection_strength: float = 0.8):
        self.original_processor = original_processor
        self.A_target = A_target 
        self.injection_strength = injection_strength

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        out_hidden_states = self.original_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        
        num_image_tokens = self.A_target.shape[1] 
        
        if out_hidden_states.shape[1] >= num_image_tokens:
            img_features = out_hidden_states[:, :num_image_tokens, :]
            protected_features = hidden_states[:, :num_image_tokens, :] 
            
            blended_features = img_features * (1.0 - (self.A_target * self.injection_strength)) + \
                               protected_features * (self.A_target * self.injection_strength)
            
            out_hidden_states = out_hidden_states.clone()
            out_hidden_states[:, :num_image_tokens, :] = blended_features

        return out_hidden_states

def inject_semantic_processors(pipe, A_target: torch.Tensor, injection_strength: float = 0.8, target_blocks: str = "single"):
    """
    Inietta il Custom Processor navigando direttamente l'albero dei moduli PyTorch.
    target_blocks: 'single', 'double', o 'all'
    """
    if not hasattr(pipe.transformer, "original_processors_cache"):
        pipe.transformer.original_processors_cache = {}

    for name, module in pipe.transformer.named_modules():
        if name.endswith("attn") and hasattr(module, "processor"):
            if name not in pipe.transformer.original_processors_cache:
                pipe.transformer.original_processors_cache[name] = module.processor
            
            original_proc = pipe.transformer.original_processors_cache[name]
            
            if "single_transformer_blocks" in name and target_blocks in ["single", "all"]:
                module.processor = SemanticGraftingProcessor(original_proc, A_target, injection_strength)
            elif "transformer_blocks" in name and "single" not in name and target_blocks in ["double", "all"]:
                module.processor = SemanticGraftingProcessor(original_proc, A_target, injection_strength)
                
    pipe.transformer.semantic_hook_active = True
    return pipe

def remove_semantic_processors(pipe):
    """
    Ripristina i processori originali.
    """
    if hasattr(pipe.transformer, "original_processors_cache"):
        for name, module in pipe.transformer.named_modules():
            if name.endswith("attn") and hasattr(module, "processor"):
                if name in pipe.transformer.original_processors_cache:
                    module.processor = pipe.transformer.original_processors_cache[name]
    pipe.transformer.semantic_hook_active = False
    return pipe

import torch

def dice_coefficient(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    r"""
    Computes the Sørensen-Dice coefficient.
    Formula: (2 * |X \cap Y|) / (|X| + |Y|)
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = (pred_flat * gt_flat).sum()
    return ((2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)).item()

def iou_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, smooth: float = 1e-6) -> float:
    r"""
    Computes Intersection over Union (Jaccard index).
    Formula: |X \cap Y| / |X \cup Y|
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    intersection = (pred_flat * gt_flat).sum()
    union = pred_flat.sum() + gt_flat.sum() - intersection
    
    return ((intersection + smooth) / (union + smooth)).item()

def clip_score(image, text_prompt: str, clip_model=None, clip_processor=None) -> float:
    """
    Computes CLIP score between the generated image and the target text prompt.
    If model/processor are None, imports and loads them locally.
    """
    if clip_model is None or clip_processor is None:
        try:
            from transformers import CLIPProcessor, CLIPModel
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except ImportError:
            import logging
            logging.error("transformers library required for CLIPScore")
            return 0.0

    inputs = clip_processor(text=[text_prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        
    logits_per_image = outputs.logits_per_image 
    return logits_per_image.item()

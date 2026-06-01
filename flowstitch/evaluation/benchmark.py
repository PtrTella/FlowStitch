import logging
from .metrics import dice_coefficient, iou_score, clip_score
import torch

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Runner for automated evaluation of different masking and stitching pipelines.
    """
    def __init__(self, ground_truth_masks: dict = None):
        self.ground_truth_masks = ground_truth_masks or {}
        self.results = []
        
    def evaluate_mask(self, method_name: str, pred_mask: torch.Tensor, prompt_id: str):
        """Evaluates a single predicted mask against its ground truth (if available)."""
        if prompt_id not in self.ground_truth_masks:
            logger.warning(f"No ground truth mask found for '{prompt_id}'. Skipping mask evaluation.")
            return
            
        gt_mask = self.ground_truth_masks[prompt_id]
        
        dice = dice_coefficient(pred_mask, gt_mask)
        iou = iou_score(pred_mask, gt_mask)
        
        logger.info(f"[{method_name} - {prompt_id}] DICE: {dice:.4f} | IoU: {iou:.4f}")
        
        self.results.append({
            "method": method_name,
            "prompt_id": prompt_id,
            "metric": "DICE",
            "value": dice
        })
        self.results.append({
            "method": method_name,
            "prompt_id": prompt_id,
            "metric": "IoU",
            "value": iou
        })
        
    def evaluate_image(self, method_name: str, image, target_prompt: str, prompt_id: str):
        """Evaluates CLIPScore for a generated image."""
        score = clip_score(image, target_prompt)
        logger.info(f"[{method_name} - {prompt_id}] CLIPScore: {score:.4f}")
        
        self.results.append({
            "method": method_name,
            "prompt_id": prompt_id,
            "metric": "CLIPScore",
            "value": score
        })
        
    def summary(self):
        """Prints a summary of all benchmark results."""
        logger.info("=== Benchmark Summary ===")
        # Basic aggregation
        aggregated = {}
        for r in self.results:
            key = f"{r['method']}_{r['metric']}"
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(r['value'])
            
        for key, values in aggregated.items():
            mean_val = sum(values) / len(values)
            logger.info(f"{key}: {mean_val:.4f} (N={len(values)})")
            
        return aggregated

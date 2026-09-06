import unittest

class TestFlowStitchImports(unittest.TestCase):
    def test_core_imports(self):
        from flowstitch.core.config import FlowStitchConfig
        config = FlowStitchConfig()
        
    def test_extraction_imports(self):
        from flowstitch.extraction import (
            compute_hermite_fused_mask,
            compute_linear_fused_mask,
            compute_calibrated_attention_mask,
            extract_attention_mask,
            compute_cosine_field,
            compute_fiedler_mask,
            extract_tda_mask,
        )
        self.assertTrue(callable(compute_hermite_fused_mask))
        self.assertTrue(callable(compute_linear_fused_mask))
        self.assertTrue(callable(compute_calibrated_attention_mask))
        self.assertTrue(callable(extract_attention_mask))
        self.assertTrue(callable(compute_cosine_field))
        self.assertTrue(callable(compute_fiedler_mask))
        self.assertTrue(callable(extract_tda_mask))

    def test_stitching_imports(self):
        from flowstitch.stitching.kts import apply_kts
        from flowstitch.stitching.ema_smoothing import TrajectoryEMA
        self.assertTrue(callable(apply_kts))
        self.assertTrue(callable(TrajectoryEMA))

    def test_evaluation_imports(self):
        from flowstitch.evaluation.metrics import dice_coefficient, iou_score
        self.assertTrue(callable(dice_coefficient))
        self.assertTrue(callable(iou_score))

    def test_tda_mask_functionality(self):
        import torch
        from flowstitch.extraction.tda_mask import extract_tda_mask
        v0 = torch.randn(1, 100, 8)
        attn_mask = torch.rand(1, 100, 1)
        mask = extract_tda_mask(v0, attn_mask, min_pixels=5)
        self.assertEqual(mask.shape, (1, 100, 1))

if __name__ == "__main__":
    unittest.main()

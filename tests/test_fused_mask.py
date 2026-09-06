import unittest
import torch
from flowstitch.extraction.fused_mask import (
    compute_cosine_field,
    compute_calibrated_attention_mask,
    compute_linear_fused_mask,
    compute_hermite_fused_mask,
)
from flowstitch.extraction.attention_mask import extract_attention_mask


class TestFusedMaskSuite(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.seq_len = 4096
        self.dim = 64
        self.heads = 4
        self.text_seq_len = 77

        # Mock velocity field with a coherent target direction in the center
        self.v0 = torch.randn(1, self.seq_len, self.dim)
        target_dir = torch.randn(1, 1, self.dim)
        target_dir = target_dir / target_dir.norm(dim=-1, keepdim=True)
        # Implant coherent velocity in first 500 tokens
        self.v0[:, :500, :] += 5.0 * target_dir

        # Mock attention maps
        self.layer_attn = torch.rand(1, self.heads, self.seq_len, self.text_seq_len)
        # Emphasize first 500 tokens for token index 3
        self.layer_attn[:, :, :500, 3] += 10.0
        self.token_indices = [3]

    def test_extract_attention_mask(self):
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        self.assertEqual(attn.shape, (1, self.seq_len, 1))
        self.assertAlmostEqual(attn.min().item(), 0.0, places=4)
        self.assertAlmostEqual(attn.max().item(), 1.0, places=4)

    def test_compute_cosine_field(self):
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        cos = compute_cosine_field(self.v0, attn, normalize_positive=True)
        self.assertEqual(cos.shape, (1, self.seq_len, 1))
        self.assertGreaterEqual(cos.min().item(), 0.0)
        self.assertLessEqual(cos.max().item(), 1.0)
        # The coherent tokens should have higher cosine alignment than background
        self.assertGreater(cos[:, :500, :].mean().item(), cos[:, 500:, :].mean().item())

    def test_compute_calibrated_attention_mask(self):
        calib = compute_calibrated_attention_mask(
            self.layer_attn, self.token_indices, pedestal_quantile=0.15
        )
        self.assertEqual(calib.shape, (1, self.seq_len, 1))
        self.assertAlmostEqual(calib.min().item(), 0.0, places=4)
        self.assertAlmostEqual(calib.max().item(), 1.0, places=4)
        # Background tokens should have strict zeros due to pedestal subtraction
        zero_pct = (calib == 0.0).float().mean().item()
        self.assertGreater(zero_pct, 0.10)

    def test_compute_linear_fused_mask(self):
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        m_lin = compute_linear_fused_mask(attn, self.v0, tau_bg=0.12)
        self.assertEqual(m_lin.shape, (1, self.seq_len, 1))
        self.assertGreaterEqual(m_lin.min().item(), 0.0)
        self.assertAlmostEqual(m_lin.max().item(), 1.0, places=4)
        # Background tokens should be strictly 0
        zero_pct = (m_lin == 0.0).float().mean().item()
        self.assertGreater(zero_pct, 0.40)

    def test_compute_hermite_fused_mask_properties(self):
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        edge0 = 0.10
        edge1 = 0.65
        m_hermite = compute_hermite_fused_mask(attn, self.v0, edge0=edge0, edge1=edge1)
        self.assertEqual(m_hermite.shape, (1, self.seq_len, 1))
        self.assertAlmostEqual(m_hermite.min().item(), 0.0, places=4)
        self.assertAlmostEqual(m_hermite.max().item(), 1.0, places=4)

        # Mathematical verification of C^1 Hermite Smoothstep:
        # S(u) = 3u^2 - 2u^3
        # dS/du = 6u(1 - u) -> exactly 0 at u=0 and u=1!
        u_vals = torch.linspace(0.0, 1.0, 1001, requires_grad=True)
        s_vals = u_vals * u_vals * (3.0 - 2.0 * u_vals)
        s_vals.backward(torch.ones_like(s_vals))
        grad = u_vals.grad

        # Check grad at u=0 and u=1
        self.assertAlmostEqual(grad[0].item(), 0.0, places=5)
        self.assertAlmostEqual(grad[-1].item(), 0.0, places=5)
        # Maximum gradient should be at u=0.5: 6*(0.5)*(0.5) = 1.5
        self.assertAlmostEqual(grad.max().item(), 1.5, places=4)

    def test_compute_core_fused_mask(self):
        from flowstitch.extraction.fused_mask import compute_core_fused_mask
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        m_core = compute_core_fused_mask(attn, self.v0)
        self.assertEqual(m_core.shape, (1, self.seq_len, 1))
        self.assertAlmostEqual(m_core.min().item(), 0.0, places=4)
        self.assertAlmostEqual(m_core.max().item(), 1.0, places=4)
        # Core tokens (first 500) must have high density without zero holes
        core_tokens = m_core[:, :500, :]
        self.assertGreater(core_tokens.mean().item(), 0.60)
        self.assertEqual((core_tokens == 0).sum().item(), 0)

    def test_compute_sam_flow_mask(self):
        from flowstitch.extraction.sam_flow_mask import compute_sam_flow_mask
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        m_sam = compute_sam_flow_mask(attn)
        self.assertEqual(m_sam.shape, (1, self.seq_len, 1))
        self.assertAlmostEqual(m_sam.min().item(), 0.0, places=4)
        self.assertAlmostEqual(m_sam.max().item(), 1.0, places=4)
        core_tokens = m_sam[:, :500, :]
        self.assertGreater(core_tokens.mean().item(), 0.70)

    def test_compute_kinetic_semantic_hull_mask(self):
        from flowstitch.extraction.fused_mask import compute_kinetic_semantic_hull_mask
        attn = extract_attention_mask(self.layer_attn, self.token_indices, normalize=True)
        m_hull = compute_kinetic_semantic_hull_mask(attn, self.v0, plateau=True)
        self.assertEqual(m_hull.shape, (1, self.seq_len, 1))
        self.assertAlmostEqual(m_hull.min().item(), 0.0, places=4)
        self.assertAlmostEqual(m_hull.max().item(), 1.0, places=4)
        core_tokens = m_hull[:, :500, :]
        # Check that core tokens have solid plateau (mean > 0.85) and zero holes
        self.assertGreater(core_tokens.mean().item(), 0.85)
        self.assertEqual((core_tokens == 0).sum().item(), 0)


if __name__ == "__main__":
    unittest.main()



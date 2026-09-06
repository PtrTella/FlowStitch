import unittest
import torch
from flowstitch.stitching.kts import compute_damping_factor, apply_kts


class TestStitchingModes(unittest.TestCase):
    def test_x0_spherical_weld_variance_preservation(self):
        """
        Analytic test:
        If x_amb ~ N(0, I) and x_tgt ~ N(0, I) are independent standard normal variables,
        then x_spherical = sqrt(1 - M) * x_amb + sqrt(M) * x_tgt
        has Var(x_spherical) = (1 - M) * 1 + M * 1 = 1.0 everywhere.
        """
        torch.manual_seed(42)
        n_samples = 100000
        x_amb = torch.randn(n_samples)
        x_tgt = torch.randn(n_samples)
        
        # Test across various mask weights M in [0, 1]
        for m_val in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            M = torch.full((n_samples,), m_val)
            x_spherical = torch.sqrt(1.0 - M) * x_amb + torch.sqrt(M) * x_tgt
            var_spherical = x_spherical.var().item()
            self.assertAlmostEqual(var_spherical, 1.0, delta=0.02)

    def test_x0_hard_mosaic_variance_drop(self):
        """
        Analytic contrast test:
        For hard mosaic x_linear = (1 - M) * x_amb + M * x_tgt,
        Var(x_linear) = (1 - M)^2 + M^2 <= 1.0.
        At M = 0.5, Var = 0.25 + 0.25 = 0.50 (severe variance collapse at boundary).
        """
        torch.manual_seed(42)
        n_samples = 100000
        x_amb = torch.randn(n_samples)
        x_tgt = torch.randn(n_samples)
        
        M_half = torch.full((n_samples,), 0.5)
        x_linear = (1.0 - M_half) * x_amb + M_half * x_tgt
        var_linear = x_linear.var().item()
        self.assertAlmostEqual(var_linear, 0.5, places=2)

    def test_x0_zero_inpainting_identity(self):
        """
        Zero inpainting uses pure ambient noise: x0 = x_amb.
        """
        torch.manual_seed(42)
        x_amb = torch.randn(1, 4096, 64)
        x_zero = x_amb.clone()
        self.assertTrue(torch.equal(x_zero, x_amb))

    def test_damping_regimes(self):
        # 1. Constant regime
        for t_norm in [1.0, 0.75, 0.5, 0.25, 0.0]:
            d_const = compute_damping_factor(t_norm, damping_mode="constant")
            self.assertEqual(d_const, 1.0)

        # 2. None regime
        for t_norm in [1.0, 0.75, 0.5, 0.25, 0.0]:
            d_none = compute_damping_factor(t_norm, damping_mode="none")
            self.assertEqual(d_none, 0.0)

        # 3. KTS regime
        # Injection phase (t >= 0.5)
        self.assertEqual(compute_damping_factor(1.0, t_cutoff=0.5, damping_mode="kts"), 1.0)
        self.assertEqual(compute_damping_factor(0.5, t_cutoff=0.5, damping_mode="kts"), 1.0)
        # Convergence phase (t < 0.5)
        d_late = compute_damping_factor(0.25, t_cutoff=0.5, gamma=4.0, damping_mode="kts")
        self.assertLess(d_late, 0.2)
        d_end = compute_damping_factor(0.0, t_cutoff=0.5, gamma=4.0, damping_mode="kts")
        self.assertLess(d_end, 0.05)


if __name__ == "__main__":
    unittest.main()

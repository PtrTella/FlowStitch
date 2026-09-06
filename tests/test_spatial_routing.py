import unittest
import torch
import math
from flowstitch.stitching.spatial_routing import apply_spatial_routing
from flowstitch.stitching.kts import compute_damping_factor, apply_kts


class TestSpatialRoutingAndKTS(unittest.TestCase):
    def test_spatial_routing_identity(self):
        mask = torch.ones(1, 4096, 1)
        v0 = torch.randn(1, 4096, 64)
        x0 = torch.randn(1, 4096, 64)
        
        m_out, v_out, x_out = apply_spatial_routing(mask, v0, x0, scale=1.0, offset=(0.0, 0.0))
        self.assertTrue(torch.allclose(m_out, mask))
        self.assertTrue(torch.allclose(v_out, v0))
        self.assertTrue(torch.allclose(x_out, x0))

    def test_spatial_routing_scaling(self):
        # Create a centered circle in a 64x64 grid
        h = w = 64
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        r = torch.sqrt(x**2 + y**2)
        circle_mask = (r < 0.8).float().view(1, 4096, 1)
        v0 = torch.randn(1, 4096, 64)
        x0 = torch.randn(1, 4096, 64)

        # Scale down to 50%
        m_scaled, v_scaled, x_scaled = apply_spatial_routing(
            circle_mask, v0, x0, scale=0.5, offset=(0.0, 0.0)
        )

        self.assertEqual(m_scaled.shape, (1, 4096, 1))
        self.assertEqual(v_scaled.shape, (1, 4096, 64))
        self.assertEqual(x_scaled.shape, (1, 4096, 64))

        # Scaled area should be significantly smaller than original area
        orig_area = (circle_mask > 0.5).sum().item()
        scaled_area = (m_scaled > 0.5).sum().item()
        self.assertLess(scaled_area, orig_area * 0.4)
        self.assertGreater(scaled_area, 0)

        # Peak of mask must remain 1.0
        self.assertAlmostEqual(m_scaled.max().item(), 1.0, places=3)

    def test_kts_damping_reverse_time(self):
        # t_norm = 1.0 (start, pure noise): D = 1.0
        d_start = compute_damping_factor(t_norm=1.0, t_cutoff=0.5, gamma=4.0)
        self.assertEqual(d_start, 1.0)

        # t_norm = 0.75 (early ODE phase): D = 1.0
        d_early = compute_damping_factor(t_norm=0.75, t_cutoff=0.5, gamma=4.0)
        self.assertEqual(d_early, 1.0)

        # t_norm = 0.5 (cutoff boundary): D = 1.0
        d_cutoff = compute_damping_factor(t_norm=0.5, t_cutoff=0.5, gamma=4.0)
        self.assertEqual(d_cutoff, 1.0)

        # t_norm = 0.25 (final step before clean image): D must decay smoothly
        d_late = compute_damping_factor(t_norm=0.25, t_cutoff=0.5, gamma=4.0)
        expected_d_late = math.exp(-4.0 * (0.5 - 0.25) / 0.5)  # exp(-2) ≈ 0.1353
        self.assertAlmostEqual(d_late, expected_d_late, places=4)
        self.assertLess(d_late, 0.2)

        # t_norm = 0.0 (terminal clean image): D must be near 0
        d_end = compute_damping_factor(t_norm=0.0, t_cutoff=0.5, gamma=4.0)
        expected_d_end = math.exp(-4.0)  # exp(-4) ≈ 0.0183
        self.assertAlmostEqual(d_end, expected_d_end, places=4)
        self.assertLess(d_end, 0.05)


if __name__ == "__main__":
    unittest.main()

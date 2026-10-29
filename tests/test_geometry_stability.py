"""Regression coverage for contaminated/degenerate rendered-depth patches."""
import unittest
from unittest.mock import patch

import torch
from utils.geometry_adaptive import fit_local_quadric, _safe_symmetric_eigh


class GeometryStabilityTests(unittest.TestCase):
    def devices(self):
        return ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    def surface(self, device):
        axis = torch.linspace(-.2, .2, 13, device=device)
        u, v = torch.meshgrid(axis, axis, indexing='ij')
        return torch.stack((u, v, 1-torch.sqrt(1-u*u-v*v)), -1).reshape(1, -1, 3)

    def test_zero_weight_nonfinite_and_extreme_points_do_not_poison_fit(self):
        for device in self.devices():
            with self.subTest(device=device):
                points = self.surface(device)
                weights = torch.ones(points.shape[:2], device=device)
                anchor, radius = points.new_zeros((1, 3)), points.new_ones(1)
                expected, _, _ = fit_local_quadric(points, weights, anchor, radius)
                junk = points.new_tensor([[[1e30, -1e30, 1e30], [float('inf'), 0, 0],
                                          [float('nan'), 0, 0], [1e19, 1e19, -1e19]]])
                contaminated = torch.cat((points, junk), 1)
                # A positive-weight outlier must also respect the local radius.
                weight = torch.cat((weights, weights.new_tensor([[0., 0., 1., 1.]])), 1)
                actual, normal, good = fit_local_quadric(contaminated, weight, anchor, radius)
                self.assertTrue(good.item())
                self.assertTrue(torch.isfinite(normal).all())
                torch.testing.assert_close(actual.abs(), expected.abs(), atol=1e-5, rtol=1e-5)

    def test_mixed_batch_invalid_row_113_does_not_destroy_neighbors(self):
        for device in self.devices():
            with self.subTest(device=device):
                points = self.surface(device).expand(256, -1, -1).clone()
                weights = points.new_ones(points.shape[:2])
                anchors, radii = points.new_zeros((256, 3)), points.new_ones(256)
                points[113] = 1e30
                points[114] = float('nan')
                weights[115] = 0
                points[116, :, 1:] = 0  # Collinear.
                radii[117:121] = radii.new_tensor([0, -1, float('nan'), float('inf')])
                anchors[121] = float('nan')
                weights[122] = float('inf')
                weights[123] = -1
                weights[124, 11:] = 0  # Fewer than min_points.
                kappa, normal, good = fit_local_quadric(points, weights, anchors, radii)
                self.assertFalse(good[113:125].any())
                self.assertTrue(good[:113].all() and good[125:].all())
                self.assertTrue(torch.isfinite(kappa).all() and torch.isfinite(normal).all())
                self.assertEqual(kappa[~good].abs().sum().item(), 0)
                self.assertEqual(normal[~good].abs().sum().item(), 0)
                torch.testing.assert_close(kappa[good].abs(), torch.ones_like(kappa[good]), rtol=.08, atol=.02)

    def test_scale_rotation_and_weight_scale_invariance(self):
        for device in self.devices():
            with self.subTest(device=device):
                points = self.surface(device)
                rotation = points.new_tensor([[.8, 0, -.6], [0, 1, 0], [.6, 0, .8]])
                points = points @ rotation
                scale = points.new_tensor([1e-7, 1., 1e7])
                anchor = points.new_tensor([[0., 0., 0.], [100., -30., 20.], [0., 0., 0.]])
                world = points*scale[:, None, None] + anchor[:, None]
                weights = points.new_ones(world.shape[:2]) * points.new_tensor([1e-30, 1., 1e30])[:, None]
                kappa, normal, good = fit_local_quadric(world, weights, anchor, scale)
                self.assertTrue(good.all())
                torch.testing.assert_close(kappa.abs()*scale[:, None], torch.ones_like(kappa), rtol=.08, atol=.02)
                torch.testing.assert_close(normal.norm(dim=-1), torch.ones_like(scale))

    def test_eigensolver_failure_is_isolated_and_nonfinite_input_is_excluded(self):
        for device in self.devices():
            with self.subTest(device=device):
                matrices = torch.eye(3, dtype=torch.float64, device=device).repeat(4, 1, 1)
                matrices[1, 0, 0] = 7  # Simulated non-converging matrix.
                matrices[2, 0, 0] = float('nan')
                original = torch.linalg.eigh

                def fail_one(matrix):
                    self.assertTrue(torch.isfinite(matrix).all())
                    if (matrix[:, 0, 0] == 7).any():
                        raise torch.linalg.LinAlgError('simulated convergence failure')
                    return original(matrix)

                with patch('torch.linalg.eigh', side_effect=fail_one):
                    values, vectors, good = _safe_symmetric_eigh(matrices)
                self.assertEqual(good.tolist(), [True, False, False, True])
                self.assertTrue(torch.isfinite(values).all() and torch.isfinite(vectors).all())
                torch.testing.assert_close(values[good], torch.ones_like(values[good]))

    def test_failed_linear_solve_returns_invalid_fit(self):
        points = self.surface('cpu')
        original = torch.linalg.solve_ex

        def fail_solve(*args, **kwargs):
            result, info = original(*args, **kwargs)
            result[:] = float('nan')
            info[:] = 1
            return result, info

        with patch('torch.linalg.solve_ex', side_effect=fail_solve):
            kappa, normal, good = fit_local_quadric(points, torch.ones(points.shape[:2]), torch.zeros(1, 3), torch.ones(1))
        self.assertFalse(good.any())
        self.assertTrue(torch.isfinite(kappa).all() and torch.isfinite(normal).all())

    def test_unrelated_runtime_errors_are_not_swallowed(self):
        with patch('torch.linalg.eigh', side_effect=RuntimeError('unrelated failure')):
            with self.assertRaisesRegex(RuntimeError, 'unrelated failure'):
                _safe_symmetric_eigh(torch.eye(3)[None])


if __name__ == '__main__':
    unittest.main()

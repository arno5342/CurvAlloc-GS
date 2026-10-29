import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from utils.geometry_adaptive import (GeometryConfig, GeometryState, fit_local_quadric,
    edge_evidence, depth_layers, sample_depth_patch, robust_normalize, multiview_consensus,
    update_geometry)


class GeometryMathTests(unittest.TestCase):
    def test_plane_curvature_sphere_and_scale_invariance(self):
        uv = torch.linspace(-.2, .2, 13)
        u, v = torch.meshgrid(uv, uv, indexing='ij')
        z = 1-torch.sqrt(1-u*u-v*v)
        points = torch.stack((u, v, z), -1).reshape(1, -1, 3)
        plane = points.clone(); plane[..., 2] = 0
        weights = torch.ones(points.shape[:2])
        k, _, good = fit_local_quadric(plane, weights, torch.zeros(1, 3), torch.ones(1), 12)
        self.assertTrue(good.item())
        self.assertLess(k.abs().max().item(), 1e-5)
        k, _, good = fit_local_quadric(points, weights, torch.zeros(1, 3), torch.ones(1), 12)
        self.assertTrue(good.item())
        torch.testing.assert_close(k.abs(), torch.ones_like(k), rtol=.08, atol=.02)
        scaled, _, _ = fit_local_quadric(points*10, weights, torch.zeros(1, 3), torch.tensor([10.]), 12)
        torch.testing.assert_close(scaled.abs()*10, k.abs(), rtol=.001, atol=.001)

    def test_degenerate_and_missing_geometry(self):
        points = torch.zeros(3, 20, 3)
        points[1, :, 0] = torch.linspace(-1, 1, 20)
        weights = torch.ones(3, 20); weights[2] = 0
        k, _, good = fit_local_quadric(points, weights, torch.zeros(3, 3), torch.ones(3))
        self.assertFalse(good.any())
        self.assertTrue(torch.isfinite(k).all())

    def test_layers_do_not_mix_foreground_background(self):
        depth = torch.tensor([[2., 2., 2., 2., 4., 4., 4., 4.]])
        foreground, _, _ = depth_layers(depth, torch.ones_like(depth, dtype=torch.bool), torch.tensor([2.]), .04)
        self.assertEqual(foreground.tolist(), [[True]*4+[False]*4])
        background, _, _ = depth_layers(depth, torch.ones_like(depth, dtype=torch.bool), torch.tensor([4.]), .04)
        self.assertEqual(background.tolist(), [[False]*4+[True]*4])

    def test_edge_plane_step_outlier_and_invalid_pixels(self):
        cfg = GeometryConfig()
        depth = torch.full((4, 7, 7), 2.)
        depth[1, :, 3:] = 4.
        depth[2, 0, 0] = 10.
        depth[3] = 0
        valid = depth > 0
        jump, layer, proxy, good = edge_evidence(depth, valid, cfg)
        self.assertLess(layer[0], .001)
        self.assertGreater(layer[1], .7)
        self.assertLess(layer[2], .01)
        self.assertFalse(good[3])
        self.assertTrue(torch.isfinite(jump).all())
        self.assertEqual(proxy[0], 0)
        scores = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
        result = multiview_consensus(scores, torch.ones_like(scores))
        self.assertLess(result[0], result[1])
        self.assertEqual(result[1], 1.)

    def test_projection_depth_holes_and_occlusion(self):
        cfg = GeometryConfig(patch=7)
        cam = SimpleNamespace(FoVx=math.pi/2, FoVy=math.pi/2, world_view_transform=torch.eye(4))
        depth = torch.full((15, 15), 2.)
        anchors = torch.tensor([[0., 0., 2.], [0., 0., 4.], [0., 0., -1.]])
        obs = sample_depth_patch(cam, depth, torch.ones_like(depth), torch.zeros_like(depth), anchors, torch.full((3,), .5), cfg)
        self.assertGreater(obs['weights'][0].sum(), 0)
        self.assertEqual(obs['weights'][1:].sum(), 0)
        torch.testing.assert_close(obs['points'][0, 24], anchors[0])
        depth[:] = float('nan')
        obs = sample_depth_patch(cam, depth, torch.ones_like(depth), torch.zeros_like(depth), anchors, torch.full((3,), .5), cfg)
        self.assertEqual(obs['view_weight'].sum(), 0)

    def test_count_and_edges_are_separate_with_finite_gradients(self):
        cfg = GeometryConfig(enabled=True, min_offsets=2)
        state = GeometryState(cfg, torch.zeros(3, 3), 6)
        d = state.data
        d['valid'][:] = True; d['views'][:] = 2; d['fit_valid'][:] = True
        d['curvature'][:] = torch.tensor([0., 1., 0.]); d['photo'][:] = torch.tensor([0., 1., 0.])
        d['radius'][:] = 1; d['normal'][:, 2] = 1
        d['edge'][2] = 1; d['view_weight'][2, 0] = 1
        d['direction'][2, 0, 2] = 1; d['bounds'][2, 0] = torch.tensor([-.01, .01])
        opacity = torch.ones(3, 6, requires_grad=True)
        offsets = torch.full((3, 6, 3), .2, requires_grad=True)
        result, constrained = state.generation(opacity, offsets, torch.ones(3, dtype=torch.bool))
        self.assertEqual((result > 0).sum(-1).tolist(), [2, 6, 2])
        self.assertLessEqual(constrained[2, :, 2].abs().max(), .010001)
        (result.sum()+constrained.sum()).backward()
        self.assertTrue(torch.isfinite(offsets.grad).all())
        self.assertTrue(torch.isfinite(opacity.grad).all())
        d['views'][:] = 1
        fallback, original = state.generation(opacity, offsets, torch.ones(3, dtype=torch.bool))
        torch.testing.assert_close(fallback, opacity)
        torch.testing.assert_close(original, offsets)

    def test_cache_lifecycle_and_alignment(self):
        anchor = torch.randn(3, 3)
        state = GeometryState(GeometryConfig(enabled=True), anchor, 4)
        state.data['curvature'][:] = torch.tensor([.1, .2, .3])
        state.grow(2)
        self.assertFalse(state.data['valid'][3:].any())
        keep = torch.tensor([True, False, True, False, True])
        state.prune(keep)
        torch.testing.assert_close(state.data['curvature'], torch.tensor([.1, .3, 0.]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'geometry.pt'
            state.save(path, anchor)
            loaded = GeometryState.load(path, anchor, 4)
            torch.testing.assert_close(loaded.data['curvature'], state.data['curvature'])
            with self.assertRaises(ValueError):
                GeometryState.load(path, anchor.flip(0), 4)
        normalized = robust_normalize(torch.tensor([0., 1., 2., 10000.]), torch.tensor([True, True, True, False]))
        self.assertEqual(normalized[-1], 0.)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration requires a GPU and the installed project extensions')
class GeometryCudaTests(unittest.TestCase):
    def make_model(self):
        from scene.gaussian_model import GaussianModel
        from torch import nn
        model = GaussianModel(n_offsets=4, appearance_dim=0, progressive=False)
        grid = torch.linspace(-.3, .3, 7, device='cuda')
        x, y = torch.meshgrid(grid, grid, indexing='ij')
        anchor = torch.stack((x.flatten(), y.flatten(), torch.full((49,), 2., device='cuda')), -1)
        model._anchor = nn.Parameter(anchor)
        model._offset = nn.Parameter(torch.zeros(49, 4, 3, device='cuda'))
        model._anchor_feat = nn.Parameter(torch.zeros(49, model.feat_dim, device='cuda'))
        model._opacity = nn.Parameter(torch.zeros(49, 1, device='cuda'))
        model._scaling = nn.Parameter(torch.full((49, 6), math.log(.2), device='cuda'))
        model._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]]*49, device='cuda'))
        model._level = torch.zeros(49, 1, device='cuda', dtype=torch.int32)
        model._extra_level = torch.zeros(49, device='cuda')
        model._anchor_mask = torch.ones(49, device='cuda', dtype=torch.bool)
        model.voxel_size=.1; model.standard_dist=10.; model.spatial_lr_scale=1.
        model.init_level=0; model.levels=1; model.init_pos=torch.zeros(3, device='cuda')
        model.coarse_intervals=[]
        with torch.no_grad():
            for module in (model.mlp_opacity, model.mlp_cov):
                for param in module.parameters(): param.zero_()
            model.mlp_opacity[-2].bias.fill_(1.)
            model.mlp_cov[-1].bias.view(4, 7)[:, 3] = 1.
        model.geometry = GeometryState(GeometryConfig(enabled=True, warmup=0, interval=1, min_points=8, max_anchors=100), anchor, 4)
        return model

    def make_cameras(self):
        from scene.cameras import Camera
        import numpy as np
        return [Camera(i, np.eye(3), np.array([shift, 0., 0.]), .6, .6,
                       torch.zeros(3, 48, 48), None, 'test'+str(i), 1., i)
                for i, shift in enumerate((-.03, .03))]

    def test_render_update_backward_export_and_restore(self):
        from gaussian_renderer import render, prefilter_voxel
        from scene.gaussian_model import GaussianModel
        from arguments import OptimizationParams
        from argparse import ArgumentParser
        model = self.make_model()
        cams = self.make_cameras()
        scene = SimpleNamespace(gaussians=model, getTrainCameras=lambda: cams)
        pipe = SimpleNamespace(debug=False, compute_cov3D_python=False)
        bg = torch.zeros(3, device='cuda')
        model.eval()
        package = render(cams[0], model, pipe, bg, geometry_depth=True)
        covered = package['alpha'] > .7
        self.assertTrue(covered.any())
        torch.testing.assert_close(package['depth'][covered], torch.full_like(package['depth'][covered], 2.), atol=1e-4, rtol=1e-4)
        for camera in cams:
            camera.original_image = render(camera, model, pipe, bg)['render'].detach()
        report = update_geometry(scene, pipe, bg, 1, render, prefilter_voxel)
        self.assertGreater(report['reliable'], 0)
        self.assertGreater(report['quadric'], 0)
        self.assertLess(report['curvature_mean'], .05)
        # Exercise the original two-output rasterizer's auxiliary depth fallback.
        import gaussian_renderer as renderer
        original_rasterizer = renderer.GaussianRasterizer
        class TwoOutput:
            def __init__(self, **kwargs): self.inner = original_rasterizer(**kwargs)
            def __call__(self, **kwargs): return self.inner(**kwargs)[:2]
        try:
            renderer.GaussianRasterizer = TwoOutput
            fallback = render(cams[0], model, pipe, bg, geometry_depth=True)
            valid = fallback['alpha'] > .7
            torch.testing.assert_close(fallback['depth'][valid], torch.full_like(fallback['depth'][valid], 2.), atol=1e-4, rtol=1e-4)
        finally:
            renderer.GaussianRasterizer = original_rasterizer
        model.train()
        opt = OptimizationParams(ArgumentParser())
        model.training_setup(opt)
        model.set_anchor_mask(cams[0].camera_center, 2, 1.)
        visible = prefilter_voxel(cams[0], model, pipe, bg)
        package = render(cams[0], model, pipe, bg, visible_mask=visible, retain_grad=True)
        self.assertEqual(package['neural_opacity'].shape, (int(visible.sum())*4, 1))
        loss = package['render'].square().mean() + package['scaling'].mean()*.001
        loss.backward()
        model.training_statis(package['viewspace_points'], package['neural_opacity'], package['visibility_filter'], package['selection_mask'], visible)
        for param in (model._anchor, model._offset, model._anchor_feat):
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
        model.optimizer.step(); model.optimizer.zero_grad(set_to_none=True)
        payload = model.capture()
        restored = self.make_model()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)/'checkpoint.pth'
            torch.save((payload, 2), checkpoint)
            disk_payload, iteration = torch.load(checkpoint, weights_only=False)
            self.assertEqual(iteration, 2)
            restored.restore(disk_payload, opt)
        torch.testing.assert_close(restored.geometry.data['curvature'], model.geometry.data['curvature'])
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'point_cloud.ply')
            model.save_ply(path)
            loaded = GaussianModel(n_offsets=4, appearance_dim=0, progressive=False)
            loaded.load_ply_sparse_gaussian(path)
            loaded.configure_geometry(SimpleNamespace(), path)
            torch.testing.assert_close(loaded.geometry.data['curvature'], model.geometry.data['curvature'])
        # Real optimizer prune must keep geometry rows in the same order.
        keep_curve = model.geometry.data['curvature'][1:].clone()
        mask = torch.zeros(49, dtype=torch.bool, device='cuda'); mask[0] = True
        with torch.no_grad():
            model.prune_anchor(mask)
        torch.testing.assert_close(model.geometry.data['curvature'], keep_curve)
        torch.cuda.synchronize()

    def test_real_anchor_growth_keeps_cache_aligned(self):
        from arguments import OptimizationParams
        from argparse import ArgumentParser
        model = self.make_model()
        model.cam_infos = torch.tensor([[0., 0., 0., 1.]], device='cuda')
        model.visible_threshold = 0.
        model.training_setup(OptimizationParams(ArgumentParser()))
        old_count = len(model.get_anchor)
        with torch.no_grad():
            model._offset[:, :, 0] = 1.
            model.geometry.data['curvature'][:] = .3
            model.anchor_growing(2000, torch.full((old_count*4,), .00021, device='cuda'),
                                 .0002, .2, .25, .01,
                                 torch.ones(old_count*4, dtype=torch.bool, device='cuda'))
        self.assertGreater(len(model.get_anchor), old_count)
        self.assertEqual(len(model.get_anchor), len(model.geometry.data['valid']))
        torch.testing.assert_close(model.geometry.data['curvature'][:old_count], torch.full((old_count,), .3, device='cuda'))
        self.assertFalse(model.geometry.data['valid'][old_count:].any())


if __name__ == '__main__':
    unittest.main()

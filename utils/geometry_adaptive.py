"""Detached multi-view geometry estimates and bounded Gaussian generation.

All depths are positive camera-Z in scene units, never inverse depth. Curvature
is fitted only within one depth-connected surface; edge evidence uses all layers.
"""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import math
import torch


@dataclass
class GeometryConfig:
    enabled: bool = False
    warmup: int = 1000
    interval: int = 500
    views: int = 4
    max_anchors: int = 16384
    chunk: int = 256
    patch: int = 7
    radius_factor: float = 3.0
    min_points: int = 12
    min_views: int = 2
    alpha_min: float = 0.7
    jump_threshold: float = 0.02
    layer_threshold: float = 0.04
    layer_softness: float = 0.01
    edge_threshold: float = 0.3
    consensus_eta: float = 0.25
    jump_weight: float = 0.5
    curvature_weight: float = 0.7
    photo_threshold: float = 0.02
    min_offsets: int = 2
    normal_fraction: float = 0.15

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, 'geometry_adaptive' if f.name == 'enabled' else 'geometry_' + f.name, f.default) for f in fields(cls)})

    def validate(self, n_offsets):
        for key in ('interval', 'views', 'max_anchors', 'chunk', 'min_points', 'min_views'):
            if getattr(self, key) < 1:
                raise ValueError('geometry_' + key + ' must be positive')
        if self.patch < 3 or self.patch % 2 != 1 or self.min_points < 6:
            raise ValueError('Geometry requires an odd patch >= 3 and min_points >= 6')
        if self.warmup < 0 or self.radius_factor <= 0 or self.layer_softness <= 0:
            raise ValueError('Invalid geometry warmup/radius/softness')
        if min(self.jump_threshold, self.layer_threshold, self.photo_threshold) < 0:
            raise ValueError('Geometry noise thresholds must be nonnegative')
        if not 1 <= self.min_offsets <= n_offsets:
            raise ValueError('geometry_min_offsets must be within [1, n_offsets]')
        for key in ('alpha_min', 'consensus_eta', 'jump_weight', 'curvature_weight', 'normal_fraction', 'edge_threshold'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError('geometry_' + key + ' must be in [0, 1]')


def masked_quantile(x, valid, q, dim=-1):
    """Finite zero for rows with no observations, without treating holes as zero."""
    safe = torch.where(valid & torch.isfinite(x), x, torch.full_like(x, float('nan')))
    return torch.nan_to_num(torch.nanquantile(safe, q, dim=dim), nan=0., posinf=0., neginf=0.)


def robust_normalize(values, valid, floor=1e-5):
    result = torch.zeros_like(values)
    mask = valid & torch.isfinite(values) & (values >= 0)
    if mask.any():
        scale = torch.quantile(values[mask], .95).clamp_min(floor)
        result[mask] = (values[mask] / scale).clamp(0, 1)
    return result


def depth_layers(depth, valid, reference, threshold):
    """Select the contiguous sorted-depth layer nearest the anchor's camera Z.

    Splitting at every significant gap avoids fitting foreground and background
    together. The reference must already have passed a center visibility test.
    """
    ordered, order = torch.where(valid, depth, torch.full_like(depth, float('inf'))).sort(-1)
    finite = torch.isfinite(ordered)
    gaps = ordered[:, 1:] - ordered[:, :-1]
    breaks = (gaps > threshold * ordered[:, :-1].clamp_min(1e-6)) & finite[:, 1:]
    labels = torch.cat((torch.zeros_like(ordered[:, :1], dtype=torch.long), breaks.long().cumsum(-1)), -1)
    nearest = (ordered - reference[:, None]).abs().argmin(-1)
    target = labels.gather(1, nearest[:, None])
    selected_sorted = (labels == target) & finite
    selected = torch.zeros_like(valid).scatter(1, order, selected_sorted)
    low = torch.where(selected, depth, torch.full_like(depth, float('inf'))).amin(-1)
    high = torch.where(selected, depth, torch.full_like(depth, -float('inf'))).amax(-1)
    return selected, torch.nan_to_num(low, posinf=0.), torch.nan_to_num(high, neginf=0.)


def edge_evidence(depth, valid, cfg):
    """Q90 adjacent relative jumps, balanced largest-gap layer evidence, proxy."""
    b, h, w = depth.shape
    d1 = torch.cat((depth[:, :, :-1].reshape(b, -1), depth[:, :-1, :].reshape(b, -1)), 1)
    d2 = torch.cat((depth[:, :, 1:].reshape(b, -1), depth[:, 1:, :].reshape(b, -1)), 1)
    pairs = torch.cat(((valid[:, :, :-1] & valid[:, :, 1:]).reshape(b, -1), (valid[:, :-1, :] & valid[:, 1:, :]).reshape(b, -1)), 1)
    jumps = (d1-d2).abs() / torch.minimum(d1, d2).clamp_min(1e-6)
    jump = masked_quantile(jumps, pairs, .9)
    flat, mask = depth.flatten(1), valid.flatten(1)
    sorted_d = torch.where(mask, flat, torch.full_like(flat, float('inf'))).sort(-1).values
    count = mask.sum(-1)
    gap = sorted_d[:, 1:] - sorted_d[:, :-1]
    positions = torch.arange(1, flat.shape[1], device=depth.device)[None]
    # Both candidate layers need >= 3 samples, not isolated outliers.
    eligible = (positions >= 3) & ((count[:, None]-positions) >= 3) & torch.isfinite(gap)
    gap = torch.where(eligible, gap, torch.zeros_like(gap))
    gap_max, split = gap.max(-1)
    balance = 2 * torch.minimum(split+1, count-split-1).clamp_min(0) / count.clamp_min(1)
    relative_gap = gap_max / masked_quantile(flat, mask, .5).clamp_min(1e-6)
    layer = balance * torch.sigmoid((relative_gap-cfg.layer_threshold) / cfg.layer_softness)
    layer = torch.where(eligible.any(-1) & (relative_gap > cfg.jump_threshold), layer, torch.zeros_like(layer))
    center = depth[:, 1:-1, 1:-1]
    neighbors = [depth[:, :-2, 1:-1], depth[:, 2:, 1:-1], depth[:, 1:-1, :-2], depth[:, 1:-1, 2:]]
    proxy_valid = valid[:, 1:-1, 1:-1].clone()
    for neighbor, vm in zip(neighbors, [valid[:, :-2, 1:-1], valid[:, 2:, 1:-1], valid[:, 1:-1, :-2], valid[:, 1:-1, 2:]]):
        proxy_valid &= vm & ((neighbor-center).abs() / center.clamp_min(1e-6) < cfg.jump_threshold)
    laplace = (sum(neighbors)-4*center).abs() / center.clamp_min(1e-6)
    proxy = masked_quantile(laplace.flatten(1), proxy_valid.flatten(1), .9)
    reliable = (count >= cfg.min_points) & (pairs.sum(-1) >= cfg.min_points)
    return jump, layer, proxy, reliable


@torch.no_grad()
def _safe_symmetric_eigh(matrix):
    """Reject nonfinite rows and isolate linalg failures without losing good fits.

    The normal path is one batched call. Only convergence failures are caught;
    OOMs and other programming/runtime errors must still reach the caller.
    """
    count, size, _ = matrix.shape
    values = matrix.new_zeros((count, size))
    vectors = torch.eye(size, device=matrix.device, dtype=matrix.dtype).expand(count, -1, -1).clone()
    good = torch.zeros(count, device=matrix.device, dtype=torch.bool)
    ids = torch.isfinite(matrix).all(dim=(-1, -2)).nonzero(as_tuple=True)[0]

    def decompose(index):
        if index.numel() == 0:
            return
        try:
            val, vec = torch.linalg.eigh(matrix[index])
        except torch.linalg.LinAlgError:
            if index.numel() > 1:
                left, right = index.tensor_split(2)
                decompose(left)
                decompose(right)
            return
        finite = torch.isfinite(val).all(-1) & torch.isfinite(vec).all(dim=(-1, -2))
        accepted = index[finite]
        values[accepted], vectors[accepted], good[accepted] = val[finite], vec[finite], True

    decompose(ids)
    return values, vectors, good


@torch.no_grad()
def fit_local_quadric(points, weights, anchor, radius, min_points=12):
    """Weighted PCA + Huber IRLS with bounded, double-precision local inputs.

    Coordinates are radius-normalized. Returned principal curvatures have units
    1/world-length. Invalid fits return finite zeros with fit_valid=False, leaving
    the existing depth-proxy/photometric fallback available to the caller.
    """
    count = points.shape[0]
    curvature = points.new_zeros((count, 2))
    normals = points.new_zeros((count, 3))
    fit_valid = torch.zeros(count, dtype=torch.bool, device=points.device)
    if count == 0 or points.shape[1] < min_points:
        return curvature, normals, fit_valid

    # Mask BEFORE polynomial products: even zero * inf would otherwise be NaN.
    # Subtract the anchor before the weighted mean to avoid large world origins.
    p, a, r = points.double(), anchor.double(), radius.double()
    frame_valid = torch.isfinite(a).all(-1) & torch.isfinite(r) & (r > 0)
    safe_radius = torch.where(frame_valid, r, torch.ones_like(r))
    relative = (p-a[:, None]) / safe_radius[:, None, None]
    valid_points = (frame_valid[:, None] & torch.isfinite(relative).all(-1)
                    & torch.isfinite(weights) & (weights > 0)
                    & (torch.linalg.vector_norm(relative, dim=-1) <= 1.000001))
    relative = torch.where(valid_points[..., None], relative, torch.zeros_like(relative))
    w = torch.where(valid_points, weights.double(), torch.zeros_like(weights, dtype=torch.float64))
    # Weight scaling does not change the fit and prevents sum overflow/underflow.
    w = w / w.amax(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
    total = w.sum(-1)
    ids = ((valid_points.sum(-1) >= min_points) & (total > 0)).nonzero(as_tuple=True)[0]
    if ids.numel() == 0:
        return curvature, normals, fit_valid
    relative, w, total, valid_points = relative[ids], w[ids], total[ids], valid_points[ids]
    center = (relative*w[..., None]).sum(1) / total[:, None]
    local = torch.where(valid_points[..., None], relative-center[:, None], torch.zeros_like(relative))
    cov = torch.einsum('bni,bnj,bn->bij', local, local, w) / total[:, None, None]
    eigen, basis, good = _safe_symmetric_eigh(cov)
    good &= eigen[:, 1] > 1e-4
    if not good.any():
        return curvature, normals, fit_valid
    ids, local, w, total = ids[good], local[good], w[good], total[good]
    valid_points, basis = valid_points[good], basis[good]
    frame = torch.stack((basis[:, :, 2], basis[:, :, 1], basis[:, :, 0]), -1)
    coords = local @ frame
    u, v, z = coords.unbind(-1)
    design = torch.stack((.5*u*u, u*v, .5*v*v, u, v, torch.ones_like(u)), -1)
    work_weight = w.clone()
    eye = torch.eye(6, device=points.device, dtype=torch.float64)[None]
    good = torch.ones(len(ids), device=points.device, dtype=torch.bool)
    for _ in range(3):
        gram = torch.einsum('bni,bnj,bn->bij', design, design, work_weight) / total[:, None, None]
        rhs = torch.einsum('bni,bn,bn->bi', design, z, work_weight) / total[:, None]
        spectrum, _, spectral_ok = _safe_symmetric_eigh(gram)
        good &= spectral_ok & (spectrum[:, 0] > 1e-7) & torch.isfinite(rhs).all(-1)
        system = torch.where(good[:, None, None], gram + 1e-6*eye, eye)
        rhs = torch.where(good[:, None], rhs, torch.zeros_like(rhs))
        coeff, info = torch.linalg.solve_ex(system, rhs, check_errors=False)
        good &= (info == 0) & torch.isfinite(coeff).all(-1)
        coeff = torch.where(good[:, None], coeff, torch.zeros_like(coeff))
        residual = z - (design * coeff[:, None]).sum(-1)
        sigma = masked_quantile(residual.abs(), valid_points, .5).clamp_min(1e-4)
        work_weight = w * (1.345*sigma[:, None] / residual.abs().clamp_min(1e-8)).clamp_max(1)
    a, b, c, d, e, _ = coeff.unbind(-1)
    # metric = I + g g^T has an analytic inverse square root. This avoids the
    # failing eigensolver entirely, including its repeated-eigenvalue plane case.
    g = torch.stack((d, e), -1)
    length = torch.sqrt(1 + g.square().sum(-1))
    invsqrt = (torch.eye(2, device=points.device, dtype=torch.float64)[None]
               - (g[:, :, None]*g[:, None, :]) / (length*(length+1))[:, None, None])
    second = torch.stack((a, b, b, c), -1).reshape(-1, 2, 2) / length[:, None, None]
    shape = invsqrt @ second @ invsqrt
    mean = .5*(shape[:, 0, 0]+shape[:, 1, 1])
    spread = torch.hypot(.5*(shape[:, 0, 0]-shape[:, 1, 1]), .5*(shape[:, 0, 1]+shape[:, 1, 0]))
    kappa = (torch.stack((mean-spread, mean+spread), -1) / r[ids, None]).to(points.dtype)
    rms = ((residual.square()*w).sum(-1)/total).sqrt()
    good &= (rms < .1) & torch.isfinite(kappa).all(-1)
    accepted = ids[good]
    curvature[accepted] = kappa[good]
    normals[accepted] = frame[good, :, 2].to(points.dtype)
    fit_valid[accepted] = True
    return curvature, normals, fit_valid


def multiview_consensus(scores, weights, threshold=.3, eta=.25):
    total = weights.sum(-1)
    mean = (scores*weights).sum(-1) / total.clamp_min(1e-8)
    support = ((scores > threshold)*weights).sum(-1) / total.clamp_min(1e-8)
    return mean*(eta+(1-eta)*support)


@torch.no_grad()
def sample_depth_patch(camera, depth, alpha, residual, anchor, radius, cfg):
    """Rasterizer pixel centers are (W-1)/2, (H-1)/2, with row-vector W2C."""
    h, w = depth.shape
    view = camera.world_view_transform.to(anchor)
    pos = torch.cat((anchor, torch.ones_like(anchor[:, :1])), -1) @ view
    fx = w/(2*math.tan(camera.FoVx/2)); fy = h/(2*math.tan(camera.FoVy/2))
    xy = torch.stack((pos[:, 0]/pos[:, 2].clamp_min(1e-6)*fx+(w-1)/2,
                      pos[:, 1]/pos[:, 2].clamp_min(1e-6)*fy+(h-1)/2), -1)
    half = cfg.patch//2
    # Adjacent pixel samples are essential for jump semantics; never use strided pixels.
    dy, dx = torch.meshgrid(torch.arange(-half, half+1, device=anchor.device), torch.arange(-half, half+1, device=anchor.device), indexing='ij')
    pixels = xy.round().long()[:, None, :] + torch.stack((dx.flatten(), dy.flatten()), -1)[None]
    inside = (pixels[..., 0] >= 0) & (pixels[..., 0] < w) & (pixels[..., 1] >= 0) & (pixels[..., 1] < h)
    px = pixels[..., 0].clamp(0, w-1); py = pixels[..., 1].clamp(0, h-1)
    patch = depth[py, px]
    confidence = torch.nan_to_num(alpha[py, px], nan=0., posinf=0., neginf=0.).clamp(0, 1)
    valid = inside & torch.isfinite(patch) & (patch > 0) & (confidence >= cfg.alpha_min)
    mid = cfg.patch**2//2
    # Occluded anchors must not borrow foreground geometry from the center pixel.
    visible = (pos[:, 2] > .01) & valid[:, mid] & ((patch[:, mid]-pos[:, 2]).abs() <= radius)
    valid &= visible[:, None]
    patch = torch.nan_to_num(patch)
    selected, low, high = depth_layers(patch, valid, pos[:, 2], cfg.layer_threshold)
    camera_points = torch.stack(((pixels[..., 0]-(w-1)/2)/fx*patch,
                                  (pixels[..., 1]-(h-1)/2)/fy*patch, patch, torch.ones_like(patch)), -1)
    world = (camera_points @ torch.linalg.inv(view))[..., :3]
    distance = torch.linalg.vector_norm(world-anchor[:, None], dim=-1)
    weights = selected * (distance < radius[:, None]) * confidence.clamp(0, 1) * torch.exp(-2*(distance/radius[:, None]).square())
    jump, layer, proxy, reliable = edge_evidence(patch.reshape(-1, cfg.patch, cfg.patch), valid.reshape(-1, cfg.patch, cfg.patch), cfg)
    view_weight = torch.where(reliable & visible, (confidence*valid).sum(-1)/valid.sum(-1).clamp_min(1), torch.zeros_like(jump))
    photo = (residual[py, px]*selected).sum(-1)/selected.sum(-1).clamp_min(1)
    # A narrow slab around the selected surface, with margin capped below layer gaps.
    margin = torch.minimum(.1*radius, .2*cfg.layer_threshold*pos[:, 2].clamp_min(0))
    bounds = torch.stack((low-margin-pos[:, 2], high+margin-pos[:, 2]), -1)
    direction = view[:3, 2].expand_as(anchor)
    return dict(points=world, weights=weights, jump=jump, layer=layer, proxy=proxy,
                view_weight=view_weight, photo=photo, direction=direction, bounds=bounds)


class GeometryState:
    """Per-anchor detached cache. Unknown anchors use the original full capacity."""
    def __init__(self, cfg, anchor, n_offsets):
        cfg.validate(n_offsets)
        self.cfg, self.n_offsets = cfg, n_offsets
        self.suspended = False
        self.cursor = self.view_cursor = self.last_update = 0
        self.data = self.empty(len(anchor), anchor.device)

    def empty(self, count, device):
        shapes = {'curvature': (), 'edge': (), 'photo': (), 'curvature_proxy': (),
                  'raw_curvature': (), 'raw_proxy': (), 'raw_photo': (), 'radius': (),
                  'normal': (3,), 'views': (), 'fit_valid': (), 'valid': (),
                  'jump': (self.cfg.views,), 'layer': (self.cfg.views,),
                  'view_weight': (self.cfg.views,), 'direction': (self.cfg.views, 3),
                  'bounds': (self.cfg.views, 2)}
        return {k: torch.zeros((count,)+shape, device=device, dtype=torch.bool if k in ('fit_valid', 'valid') else torch.float32) for k, shape in shapes.items()}

    @property
    def active(self):
        return self.cfg.enabled and not self.suspended

    def grow(self, count):
        extra = self.empty(count, self.data['valid'].device)
        self.data = {k: torch.cat((v, extra[k])) for k, v in self.data.items()}

    def prune(self, keep):
        self.data = {k: v[keep] for k, v in self.data.items()}

    def generation(self, opacity, offsets, visible):
        """Discrete quota (not a learned MLP), then differentiable spatial clamps.

        Budget is an upper bound: the original positive-opacity gate still applies.
        Edge never increases the count. Geometry-unknown anchors are untouched.
        """
        if not self.active:
            return opacity, offsets
        d = {k: v[visible] for k, v in self.data.items()}
        cfg = self.cfg
        reliable = d['valid'] & (d['views'] >= cfg.min_views)
        curve = torch.where(d['fit_valid'], d['curvature'], .25*d['curvature_proxy'])
        demand = cfg.curvature_weight*curve + (1-cfg.curvature_weight)*d['photo']
        budget = (cfg.min_offsets + torch.round((self.n_offsets-cfg.min_offsets)*demand)).long()
        budget = torch.where(reliable, budget, torch.full_like(budget, self.n_offsets))
        rank = opacity.detach().argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
        allowed = rank < budget[:, None]
        radius = d['radius'].clamp_min(1e-8)
        normal = d['normal'][:, None]
        height = (offsets*normal).sum(-1, keepdim=True)
        tangent = offsets-height*normal
        tangent_limit = radius*(1-.5*curve)
        tangent = tangent * (tangent_limit[:, None, None]/torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-8)).clamp_max(1)
        normal_limit = radius*(cfg.normal_fraction+.5*curve)*(1-.75*d['edge'])
        constrained = tangent + normal*height.clamp(-normal_limit[:, None, None], normal_limit[:, None, None])
        fit = reliable & d['fit_valid']
        offsets = torch.where(fit[:, None, None], constrained, offsets)
        # For high-edge anchors enforce every supported view's selected depth slab.
        edge_on = reliable & (d['edge'] > cfg.edge_threshold)
        for j in range(cfg.views):
            direction = d['direction'][:, j, None]
            position = (offsets*direction).sum(-1)
            clipped = position.clamp(d['bounds'][:, j, 0, None], d['bounds'][:, j, 1, None])
            use = edge_on & (d['view_weight'][:, j] > 0)
            corrected = offsets+(clipped-position)[..., None]*direction
            offsets = torch.where(use[:, None, None], corrected, offsets)
        # Intersecting slabs can be inconsistent: never silently cross an earlier slab.
        for j in range(cfg.views):
            position = (offsets*d['direction'][:, j, None]).sum(-1)
            in_layer = (position >= d['bounds'][:, j, 0, None]-1e-6) & (position <= d['bounds'][:, j, 1, None]+1e-6)
            use = edge_on & (d['view_weight'][:, j] > 0)
            allowed &= ~use[:, None] | in_layer
        return torch.where(allowed, opacity, torch.zeros_like(opacity)), offsets

    def payload(self, anchors):
        return {'version': 1, 'config': asdict(self.cfg), 'n_offsets': self.n_offsets,
                'anchors': anchors.detach().cpu(), 'data': {k: v.cpu() for k, v in self.data.items()},
                'cursor': self.cursor, 'view_cursor': self.view_cursor, 'last_update': self.last_update}

    def save(self, path, anchors):
        if self.cfg.enabled:
            torch.save(self.payload(anchors), path)

    @classmethod
    def load(cls, path, anchors, n_offsets):
        payload = torch.load(path, map_location=anchors.device, weights_only=True)
        return cls.from_payload(payload, anchors, n_offsets)

    @classmethod
    def from_payload(cls, payload, anchors, n_offsets):
        payload = dict(payload)
        payload['anchors'] = payload['anchors'].to(anchors.device)
        payload['data'] = {k: v.to(anchors.device) for k, v in payload['data'].items()}
        if payload['version'] != 1 or payload['n_offsets'] != n_offsets or not torch.equal(payload['anchors'], anchors):
            raise ValueError('Geometry cache does not match anchors/capacity')
        result = cls(GeometryConfig(**payload['config']), anchors, n_offsets)
        for key, value in result.data.items():
            candidate = payload['data'][key]
            if candidate.shape != value.shape or candidate.dtype != value.dtype or not torch.isfinite(candidate).all():
                raise ValueError('Invalid geometry cache tensor: ' + key)
        result.data = payload['data']
        for key in ('cursor', 'view_cursor', 'last_update'):
            setattr(result, key, int(payload[key]))
        return result


@torch.no_grad()
def update_geometry(scene, pipe, background, iteration, render_fn, prefilter_fn):
    pc = scene.gaussians
    state = pc.geometry
    if state is None or not state.cfg.enabled:
        return None
    cfg = state.cfg
    if iteration < cfg.warmup or (state.last_update and iteration-state.last_update < cfg.interval):
        return None
    count = len(pc.get_anchor)
    if count == 0:
        return None
    # Deduplicate multi-resolution copies of the same image; they are not independent views.
    cameras = list({c.image_name: c for c in scene.getTrainCameras()}.values())
    if not cameras:
        return None
    views = min(cfg.views, len(cameras))
    selected = [cameras[(state.view_cursor+j) % len(cameras)] for j in range(views)]
    state.view_cursor = (state.view_cursor+views) % len(cameras)
    ids = (torch.arange(min(count, cfg.max_anchors), device=pc.get_anchor.device)+state.cursor) % count
    state.cursor = (state.cursor+len(ids)) % count
    was_training = pc.mlp_color.training
    snapshots = []
    state.suspended = True
    pc.eval()
    try:
        for camera in selected:
            pc.set_anchor_mask(camera.camera_center, iteration, camera.resolution_scale)
            mask = prefilter_fn(camera, pc, pipe, background)
            package = render_fn(camera, pc, pipe, background, visible_mask=mask, geometry_depth=True)
            residual = (package['render']-camera.original_image.to(background.device)).abs().mean(0)
            snapshots.append((camera, package['depth'].squeeze(0), package['alpha'].squeeze(0), residual))
    finally:
        state.suspended = False
        pc.train() if was_training else pc.eval()
    for start in range(0, len(ids), cfg.chunk):
        index = ids[start:start+cfg.chunk]
        anchor = pc.get_anchor[index].detach()
        radius = (cfg.radius_factor*torch.as_tensor(pc.voxel_size, device=anchor.device)
                  / float(pc.fork)**pc.get_level[index].flatten()).clamp_min(1e-7)
        observations = [sample_depth_patch(cam, dep, alp, res, anchor, radius, cfg) for cam, dep, alp, res in snapshots]
        points = torch.cat([o['points'] for o in observations], 1)
        weights = torch.cat([o['weights'] for o in observations], 1)
        kappa, normal, good = fit_local_quadric(points, weights, anchor, radius, cfg.min_points)
        d = state.data
        for key in d:
            d[key][index] = 0
        vw = torch.stack([o['view_weight'] for o in observations], -1)
        d['view_weight'][index, :views] = vw
        d['views'][index] = (vw > 0).sum(-1).to(d['views'].dtype)
        d['valid'][index] = vw.sum(-1) > 0
        d['fit_valid'][index] = good
        d['raw_curvature'][index] = radius * torch.linalg.vector_norm(kappa, dim=-1)
        d['normal'][index] = normal
        d['radius'][index] = radius
        for key in ('jump', 'layer', 'direction', 'bounds'):
            d[key][index, :views] = torch.stack([o[key] for o in observations], 1)
        for key in ('photo', 'proxy'):
            values = torch.stack([o[key] for o in observations], -1)
            if key == 'photo':
                d['raw_photo'][index] = multiview_consensus(values, vw, cfg.photo_threshold, cfg.consensus_eta)
            else:
                d['raw_proxy'][index] = (values*vw).sum(-1)/vw.sum(-1).clamp_min(1e-8)
    d = state.data
    reliable = d['valid'] & (d['views'] >= cfg.min_views)
    d['curvature'] = robust_normalize(d['raw_curvature'], reliable & d['fit_valid'], floor=1e-3)
    d['curvature_proxy'] = robust_normalize(d['raw_proxy'], reliable & ~d['fit_valid'], floor=1e-3)
    d['photo'] = robust_normalize(d['raw_photo'], reliable, floor=.01)
    valid_jump = d['view_weight'] > 0
    if valid_jump.any():
        scale = torch.quantile(d['jump'][valid_jump], .95)
        normalized = ((d['jump']-cfg.jump_threshold)/(scale-cfg.jump_threshold).clamp_min(1e-3)).clamp(0, 1)
        evidence = cfg.jump_weight*normalized+(1-cfg.jump_weight)*d['layer']
        d['edge'] = multiview_consensus(evidence, d['view_weight'], cfg.edge_threshold, cfg.consensus_eta)
    state.last_update = iteration
    return {'updated': len(ids), 'reliable': int(reliable.sum()), 'quadric': int((reliable & d['fit_valid']).sum()),
            'curvature_mean': float(d['curvature'][reliable].mean()) if reliable.any() else 0.,
            'edge_mean': float(d['edge'][reliable].mean()) if reliable.any() else 0.}

"""Geometry shared by grounding, planning and motion conditioning."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation


def panorama_points(distance):
    """Unproject radial distance in the mesh's x-forward, y-up, z-right frame."""
    distance = np.asarray(distance)
    h, w = distance.shape
    lon = ((np.arange(w) + .5) / w - .5) * 2 * np.pi
    lat = ((np.arange(h) + .5) / h - .5) * np.pi
    return distance[..., None] * np.stack(np.broadcast_arrays(
        np.cos(lat)[:, None] * np.cos(lon)[None],
        -np.sin(lat)[:, None] * np.ones((1, w)),
        np.cos(lat)[:, None] * np.sin(lon)[None]), axis=-1)


def radial_boundary(distance):
    """Equatorial radial boundary, with periodic interpolation of invalid samples."""
    distance = np.asarray(distance)
    h, w = distance.shape
    # For an even-height raster, the equator lies between the central two rows.
    rows = distance[(h - 1) // 2:h // 2 + 1]
    valid = np.isfinite(rows) & (rows > 1e-5)
    counts = valid.sum(axis=0)
    radii = np.where(valid, rows, 0).sum(axis=0) / np.maximum(counts, 1)
    phi = ((np.arange(w) + .5) / w - .5) * 2 * np.pi
    if not np.any(counts):
        raise ValueError('No valid equatorial depth for a collision-free trajectory')
    radii = np.interp(phi, phi[counts > 0], radii[counts > 0], period=2*np.pi)
    return phi, radii


def plan_trajectory(distance, centers, boxes, n_views=24, gamma=.4, safety_ratio=.6):
    """Plan a periodic cubic spline with a fixed view budget.

    safety_ratio limits the trajectory radius relative to the observed surface.
    Focus identities are inherited from the deformation knot, never reassigned
    after spline fitting.
    """
    phi, radius = radial_boundary(distance)
    angles = np.linspace(-np.pi, np.pi, n_views, endpoint=False)
    r = gamma * np.interp(angles, phi, radius, period=2*np.pi)
    base = np.stack((r*np.cos(angles), np.zeros(n_views), r*np.sin(angles)), -1)
    centers = np.asarray(centers).reshape(-1, 3)
    boxes = np.asarray(boxes).reshape(-1, 8, 3)
    if len(centers) == 0:
        return base, [None] * n_views, base
    focus = np.linalg.norm(base[:, None, [0, 2]] - centers[None, :, [0, 2]], axis=-1).argmin(1)
    selected = centers[focus]
    extents = np.ptp(boxes[focus], axis=1)
    scale = np.linalg.norm(extents[:, [0, 2]], axis=-1)
    gain = np.clip(.25*scale, .05, .60)
    sigma = np.clip(.60*scale, .20, 1.20)
    delta = (angles - np.arctan2(selected[:, 2], selected[:, 0]) + np.pi) % (2*np.pi) - np.pi
    delta = np.where(delta == -np.pi, np.pi, delta)
    weight = delta*np.exp(-delta**2/(2*sigma**2))
    view = selected-base
    view[:, 1] = 0
    tangent = np.cross(view, [0., 1., 0.])
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-12)

    def safe(points):
        az = np.arctan2(points[:, 2], points[:, 0])
        limit = safety_ratio*np.interp(az, phi, radius, period=2*np.pi)
        rho = np.linalg.norm(points[:, [0, 2]], axis=-1)
        points[:, [0, 2]] *= np.minimum(1, limit/np.maximum(rho, 1e-12))[:, None]
        return points

    knots = safe(base-gain[:, None]*weight[:, None]*tangent)
    spline = CubicSpline(np.arange(n_views+1), np.vstack((knots, knots[0])), bc_type='periodic')
    # Arc-length sampling makes spline fitting effective, rather than evaluating
    # it only at its original knots. Recheck safety because cubics can overshoot.
    t = np.linspace(0, n_views, n_views*128+1)
    dense = safe(spline(t))
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    samples = np.interp(np.linspace(0, length[-1], n_views, endpoint=False), length, t)
    points = safe(spline(samples))
    focus = focus[np.floor(samples+.5).astype(int) % n_views].tolist()
    return points, focus, knots


def relative_motion(source_w2c, target_w2c, distance):
    """Six DoF: source-frame camera displacement / vertical extent, then rotvec."""
    distance = np.asarray(distance)
    top = distance[:5]
    bottom = distance[-5:]
    top = top[np.isfinite(top) & (top > 1e-5)]
    bottom = bottom[np.isfinite(bottom) & (bottom > 1e-5)]
    if not len(top) or not len(bottom):
        raise ValueError('Valid zenith and nadir depth required for motion normalization')
    extent = np.median(top)+np.median(bottom)
    src = np.linalg.inv(source_w2c)
    tgt = np.linalg.inv(target_w2c)
    displacement = source_w2c[:3, :3] @ (tgt[:3, 3]-src[:3, 3]) / extent
    rotation = source_w2c[:3, :3] @ tgt[:3, :3]
    return np.r_[displacement, Rotation.from_matrix(rotation).as_rotvec()]


def support_pose(world_to_camera):
    """Mesh world-to-camera -> SDF/support camera-to-world, including all axes."""
    change = np.eye(4)
    change[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    return change @ np.linalg.inv(world_to_camera) @ change.T


def evaluation_poses(distance, mode='geometry', seed=42, vertical_jitter=.02):
    """Generate evaluation views as mesh world-to-camera matrices.

    vertical_jitter controls perturbation amplitude as a fraction of the circle radius.
    """
    if mode not in ('appearance', 'geometry'):
        raise ValueError('Evaluation mode must be appearance or geometry')
    _, boundary = radial_boundary(distance)
    radius = .3*boundary.min()
    n = 20 if mode == 'appearance' else 360
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    centers = np.stack((radius*np.cos(angles), np.zeros(n), radius*np.sin(angles)), -1)
    if mode == 'appearance':
        centers[:, 1] = rng.uniform(-vertical_jitter*radius, vertical_jitter*radius, n)
        directions = rng.normal(size=(n, 3))
    else:
        directions = centers.copy()
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    left = np.cross([0., 1., 0.], directions)
    left /= np.linalg.norm(left, axis=-1, keepdims=True)
    up = np.cross(directions, left)
    poses = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    poses[:, :3, :3] = np.stack((left, up, directions), axis=1)
    poses[:, :3, 3] = -np.einsum('nij,nj->ni', poses[:, :3, :3], centers)
    return poses

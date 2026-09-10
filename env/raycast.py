"""Batched ray / segment geometry in numpy.

All functions are vectorised over an arbitrary leading batch; nothing loops over
rays.  Conventions:
  * origins  ``o``  : [..., 2]
  * directions ``d``: [..., 2] unit vectors (same leading shape as ``o``)
  * boxes           : [B, M, 4] as (xmin, ymin, xmax, ymax) with a [B, M] mask
  * circles         : [B, N, 2] centres with a per-ray [..., N] mask
where the leading batch of ``o``/``d`` is [B, K] (K rays per env).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-9
INF = np.float32(1e9)


def _safe_inv(d: np.ndarray) -> np.ndarray:
    d = np.where(np.abs(d) < EPS, np.where(d < 0, -EPS, EPS), d)
    return 1.0 / d


def ray_box_t(o: np.ndarray, d: np.ndarray, boxes: np.ndarray, box_mask: np.ndarray) -> np.ndarray:
    """Distance along each ray to the first intersection with each box (slab method).

    o, d : [B, K, 2];  boxes : [B, M, 4];  box_mask : [B, M]
    returns t : [B, K, M]  (INF where no hit). A ray whose origin lies inside a box
    does not register that box (t would be 0); collision resolution keeps agents
    outside boxes so this only matters for degenerate inputs.
    """
    inv = _safe_inv(d.astype(np.float32))
    ix, iy = inv[:, :, None, 0], inv[:, :, None, 1]                # [B,K,1]
    ox, oy = o[:, :, None, 0], o[:, :, None, 1]
    bx0, by0 = boxes[:, None, :, 0], boxes[:, None, :, 1]           # [B,1,M]
    bx1, by1 = boxes[:, None, :, 2], boxes[:, None, :, 3]
    tx1 = (bx0 - ox) * ix
    tx2 = (bx1 - ox) * ix
    ty1 = (by0 - oy) * iy
    ty2 = (by1 - oy) * iy
    tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))   # [B,K,M]
    tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= tmin) & (tmin >= 0.0) & box_mask[:, None, :]
    return np.where(hit, tmin, INF).astype(np.float32)


def ray_boundary_t(o: np.ndarray, d: np.ndarray, size: float) -> np.ndarray:
    """Distance to exit the [0,size]^2 arena. o, d: [B, K, 2] -> [B, K]."""
    inv = _safe_inv(d)
    t_hi = (size - o) * inv
    t_lo = (0.0 - o) * inv
    t = np.where(d > 0, t_hi, np.where(d < 0, t_lo, INF))
    return np.maximum(t.min(-1), 0.0).astype(np.float32)


def ray_circle_t(o: np.ndarray, d: np.ndarray, centers: np.ndarray, radius: float,
                 circle_mask: np.ndarray) -> np.ndarray:
    """Distance along each ray to each circle.  o, d: [B,K,2]; centers: [B,N,2];
    circle_mask: [B,K,N] -> t: [B,K,N] (INF where no hit)."""
    fx = centers[:, None, :, 0] - o[:, :, None, 0]                  # [B,K,N]
    fy = centers[:, None, :, 1] - o[:, :, None, 1]
    dx, dy = d[:, :, None, 0], d[:, :, None, 1]
    tc = fx * dx + fy * dy
    l2 = fx * fx + fy * fy - tc * tc
    disc = radius * radius - l2
    hit = (disc >= 0.0) & (tc > 0.0) & circle_mask
    t = tc - np.sqrt(np.maximum(disc, 0.0))
    return np.where(hit & (t >= 0.0), t, INF).astype(np.float32)


def cast_rays(o, d, boxes, box_mask, centers, radius, circle_mask, arena_size, max_range):
    """Full ray cast against boundary, boxes and agent bodies.

    Returns (dist [B,K], kind [B,K], hit_idx [B,K]) with kind in
    {0: nothing within range, 1: wall/obstacle, 2: agent body} and hit_idx the index
    of the agent hit (or -1).  ``dist`` is clipped to ``max_range``.
    """
    t_wall = ray_boundary_t(o, d, arena_size)
    if boxes.shape[1] > 0:
        t_wall = np.minimum(t_wall, ray_box_t(o, d, boxes, box_mask).min(-1))
    t_c = ray_circle_t(o, d, centers, radius, circle_mask)        # [B,K,N]
    idx = t_c.argmin(-1)
    t_agent = np.take_along_axis(t_c, idx[..., None], -1)[..., 0]
    agent_first = t_agent < t_wall
    dist = np.where(agent_first, t_agent, t_wall)
    kind = np.where(agent_first, 2, 1).astype(np.int64)
    hit_idx = np.where(agent_first, idx, -1)
    out_of_range = dist > max_range
    kind = np.where(out_of_range, 0, kind)
    hit_idx = np.where(out_of_range, -1, hit_idx)
    dist = np.minimum(dist, max_range).astype(np.float32)
    return dist, kind, hit_idx


def segment_blocked(p, q, boxes, box_mask, centers, radius, circle_mask):
    """True where the segment p->q is interrupted by a box or a masked-in circle.

    p, q : [B, K, 2]; boxes [B,M,4]; centers [B,N,2]; circle_mask [B,K,N].
    """
    diff = q - p
    length = np.linalg.norm(diff, axis=-1)
    d = diff / np.maximum(length, EPS)[..., None]
    blocked = np.zeros(length.shape, dtype=bool)
    if boxes.shape[1] > 0:
        tb = ray_box_t(p, d, boxes, box_mask).min(-1)
        blocked |= tb < length
    tc = ray_circle_t(p, d, centers, radius, circle_mask).min(-1)
    blocked |= tc < length
    return blocked


def cone_membership(o, theta, points, fov_rad, max_range):
    """Whether each point lies inside the vision cone of each observer.

    o: [B,N,2]; theta: [B,N]; points: [B,P,2] -> (inside [B,N,P], rel_angle [B,N,P], dist [B,N,P])
    """
    rel = points[:, None, :, :] - o[:, :, None, :]
    dist = np.linalg.norm(rel, axis=-1)
    ang = np.arctan2(rel[..., 1], rel[..., 0]) - theta[:, :, None]
    ang = wrap_angle(ang)
    inside = (np.abs(ang) <= fov_rad / 2.0) & (dist <= max_range) & (dist > EPS)
    return inside, ang, dist


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def ray_directions(theta, fov_rad, num_rays):
    """Unit direction vectors for ``num_rays`` rays spanning the cone. theta: [...] -> [..., R, 2]."""
    offsets = np.linspace(-fov_rad / 2.0, fov_rad / 2.0, num_rays, dtype=np.float32)
    ang = theta[..., None] + offsets
    return np.stack([np.cos(ang), np.sin(ang)], -1).astype(np.float32)

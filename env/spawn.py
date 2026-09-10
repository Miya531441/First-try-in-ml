"""Procedural layouts parameterised by coverage fraction, with two obstacle classes,
minimum gaps, spawn clearance and a flood-fill connectivity check."""
from __future__ import annotations

from collections import deque

import numpy as np

from env.arena import CRATE, WALL


def _box_dist(a: np.ndarray, b: np.ndarray) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _point_box_dist(p: np.ndarray, b: np.ndarray) -> float:
    q = np.clip(p, b[0:2], b[2:4])
    return float(np.linalg.norm(q - p))


def connected(boxes: np.ndarray, size: float, a: np.ndarray, b: np.ndarray, radius: float, cell: float = 1.0) -> bool:
    """Flood fill on a grid with obstacles inflated by the agent radius: can a walk from a to b?"""
    n = int(np.ceil(size / cell))
    free = np.ones((n, n), bool)
    xs = (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    for bx in boxes:
        inside = (X >= bx[0] - radius) & (X <= bx[2] + radius) & (Y >= bx[1] - radius) & (Y <= bx[3] + radius)
        free &= ~inside
    # arena walls
    free[X < radius] = False
    free[X > size - radius] = False
    free[Y < radius] = False
    free[Y > size - radius] = False
    ia, ib = (int(a[0] / cell), int(a[1] / cell)), (int(b[0] / cell), int(b[1] / cell))
    if not (free[ia] and free[ib]):
        return False
    seen = np.zeros_like(free)
    seen[ia] = True
    q = deque([ia])
    while q:
        x, y = q.popleft()
        if (x, y) == ib:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = x + dx, y + dy
            if 0 <= u < n and 0 <= v < n and free[u, v] and not seen[u, v]:
                seen[u, v] = True
                q.append((u, v))
    return False


def generate_obstacles(rng: np.random.Generator, cfg, spawn_center: np.ndarray, coverage_range=None):
    """Mirrored boxes reaching a target coverage fraction.  Returns (boxes [M,4], kinds [M])."""
    S = cfg.arena_size
    lo, hi = coverage_range if coverage_range is not None else (cfg.coverage_min, cfg.coverage_max)
    target = rng.uniform(lo, hi) * S * S
    centers = np.stack([spawn_center, S - spawn_center])
    clearance = cfg.spawn_clearance + cfg.spawn_zone_radius
    for _attempt in range(12):
        boxes, kinds, area, tries = [], [], 0.0, 0
        while area < target and tries < 2000 and len(boxes) + 2 <= cfg.max_obstacles:
            tries += 1
            w = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max)
            h = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max)
            x = rng.uniform(1.0, S - w - 1.0)
            y = rng.uniform(1.0, S - h - 1.0)
            box = np.array([x, y, x + w, y + h], np.float32)
            mirror = np.array([S - box[2], S - box[3], S - box[0], S - box[1]], np.float32)
            if any(_point_box_dist(c, bx) < clearance for c in centers for bx in (box, mirror)):
                continue
            if _box_dist(box, mirror) < cfg.obstacle_min_gap and not np.allclose(box, mirror, atol=0.5):
                continue
            if any(_box_dist(box, o) < cfg.obstacle_min_gap or _box_dist(mirror, o) < cfg.obstacle_min_gap for o in boxes):
                continue
            kind = CRATE if rng.random() < cfg.crate_fraction else WALL
            if np.allclose(box, mirror, atol=0.5):      # self-mirrored central box: add once
                boxes.append(box)
                kinds.append(kind)
                area += w * h
            else:
                boxes += [box, mirror]
                kinds += [kind, kind]
                area += 2 * w * h
        arr = np.array(boxes, np.float32).reshape(-1, 4)
        if len(arr) == 0 or connected(arr, S, centers[0], centers[1], cfg.collision_radius):
            return arr, np.array(kinds, np.int64)
        target *= 0.85       # layout blocked the lanes: retry with a little less cover
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)


def sample_spawn_center(rng: np.random.Generator, cfg) -> np.ndarray:
    """Team A's spawn centre; team B is always the point mirror (S - centre).

    corners: a random diagonal corner, ``corner_margin`` from the walls, so the squads
    start in opposite corners ~S*sqrt(2) apart.  lanes: the v1 behaviour, ``spawn_distance``
    apart across the centre line."""
    S = cfg.arena_size
    if cfg.spawn_mode == "corners":
        m = cfg.corner_margin
        j = cfg.spawn_lateral_jitter * 0.5
        x = m + rng.uniform(0.0, j)
        y = (m + rng.uniform(0.0, j)) if rng.random() < 0.5 else (S - m - rng.uniform(0.0, j))
        return np.array([x, y], np.float32)
    x = S / 2.0 - cfg.spawn_distance / 2.0
    y = rng.uniform(S / 2.0 - cfg.spawn_lateral_jitter, S / 2.0 + cfg.spawn_lateral_jitter)
    return np.array([x, y], np.float32)


def sample_team_positions(rng: np.random.Generator, cfg, center: np.ndarray, boxes: np.ndarray, n: int):
    S, r = cfg.arena_size, cfg.collision_radius
    pts, tries = [], 0
    while len(pts) < n and tries < 500:
        tries += 1
        off = rng.uniform(-cfg.spawn_zone_radius * 0.7, cfg.spawn_zone_radius * 0.7, size=2)
        p = np.clip(center + off, r + 0.1, S - r - 0.1)
        if len(boxes) and (((p >= boxes[:, 0:2] - r) & (p <= boxes[:, 2:4] + r)).all(-1)).any():
            continue
        if any(np.linalg.norm(p - q) < 2.2 * r for q in pts):
            continue
        pts.append(p.astype(np.float32))
    while len(pts) < n:
        pts.append(np.array([center[0], center[1] + 2.5 * r * len(pts)], np.float32))
    return np.stack(pts)


def spawn_episode(rng: np.random.Generator, cfg, team_size: int, coverage_range=None):
    """(boxes [M,4], kinds [M], positions [2*team_size,2], headings [2*team_size])."""
    S = cfg.arena_size
    center = sample_spawn_center(rng, cfg)
    boxes, kinds = generate_obstacles(rng, cfg, center, coverage_range)
    pos_a = sample_team_positions(rng, cfg, center, boxes, team_size)
    pos_b = (S - pos_a).astype(np.float32)
    pos = np.concatenate([pos_a, pos_b], 0)
    headings = rng.uniform(-np.pi, np.pi, size=2 * team_size).astype(np.float32)
    return boxes, kinds, pos, headings

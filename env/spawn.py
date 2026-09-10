"""Procedural obstacle layouts and mirrored team spawns."""
from __future__ import annotations

import numpy as np


def generate_obstacles(rng: np.random.Generator, cfg, spawn_center: np.ndarray):
    """Return an [M, 4] float32 array of mirrored boxes for one env.

    Half of the boxes are sampled in the arena, the other half are their point
    reflections about the centre.  Boxes are rejected if they intrude into either
    spawn zone or overlap an existing box too heavily.
    """
    S = cfg.arena_size
    n_total = int(rng.integers(cfg.num_obstacles_min, cfg.num_obstacles_max + 1))
    n_half = max(1, n_total // 2)
    boxes = []
    zone_r = cfg.spawn_zone_radius
    centers = np.stack([spawn_center, S - spawn_center])
    attempts = 0
    while len(boxes) < n_half and attempts < 200:
        attempts += 1
        w = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max)
        h = rng.uniform(cfg.obstacle_size_min, cfg.obstacle_size_max)
        if rng.random() < 0.5:
            w, h = h, w
        x = rng.uniform(1.0, S - w - 1.0)
        y = rng.uniform(1.0, S - h - 1.0)
        box = np.array([x, y, x + w, y + h], dtype=np.float32)
        # keep spawn zones clear
        clear = True
        for c in centers:
            nearest = np.clip(c, box[0:2], box[2:4])
            if np.linalg.norm(nearest - c) < zone_r:
                clear = False
        if not clear:
            continue
        # reject heavy overlap with existing boxes (including mirrors)
        ok = True
        for b in boxes:
            ix = max(0.0, min(box[2], b[2]) - max(box[0], b[0]))
            iy = max(0.0, min(box[3], b[3]) - max(box[1], b[1]))
            if ix * iy > 0.25 * min(w * h, (b[2] - b[0]) * (b[3] - b[1])):
                ok = False
                break
        if not ok:
            continue
        mirror = np.array([S - box[2], S - box[3], S - box[0], S - box[1]], dtype=np.float32)
        boxes.append(box)
        boxes.append(mirror)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def sample_spawn_center(rng: np.random.Generator, cfg) -> np.ndarray:
    """Team A spawn centre; team B is the point mirror.  ~40 m apart."""
    S = cfg.arena_size
    x = S / 2.0 - cfg.spawn_distance / 2.0
    y = rng.uniform(S / 2.0 - cfg.spawn_lateral_jitter, S / 2.0 + cfg.spawn_lateral_jitter)
    return np.array([x, y], dtype=np.float32)


def sample_team_positions(rng: np.random.Generator, cfg, center: np.ndarray, boxes: np.ndarray, n: int):
    """n positions clustered within spawn_zone_radius of ``center``, outside obstacles,
    separated by at least 3 collision radii."""
    S = cfg.arena_size
    r = cfg.collision_radius
    pts = []
    tries = 0
    while len(pts) < n and tries < 500:
        tries += 1
        off = rng.uniform(-cfg.spawn_zone_radius * 0.7, cfg.spawn_zone_radius * 0.7, size=2)
        p = np.clip(center + off, r + 0.1, S - r - 0.1)
        if len(boxes) and (((p >= boxes[:, 0:2] - r) & (p <= boxes[:, 2:4] + r)).all(-1)).any():
            continue
        if any(np.linalg.norm(p - q) < 3 * r for q in pts):
            continue
        pts.append(p.astype(np.float32))
    while len(pts) < n:  # degenerate fallback: stack along y
        pts.append(np.array([center[0], center[1] + 1.5 * len(pts)], dtype=np.float32))
    return np.stack(pts)


def spawn_episode(rng: np.random.Generator, cfg, team_size: int):
    """Full episode layout: (boxes [M,4], positions [2*team_size,2], headings [2*team_size])."""
    S = cfg.arena_size
    center = sample_spawn_center(rng, cfg)
    boxes = generate_obstacles(rng, cfg, center)
    pos_a = sample_team_positions(rng, cfg, center, boxes, team_size)
    pos_b = (S - pos_a).astype(np.float32)          # mirrored positions
    pos = np.concatenate([pos_a, pos_b], 0)
    headings = rng.uniform(-np.pi, np.pi, size=2 * team_size).astype(np.float32)
    return boxes, pos, headings

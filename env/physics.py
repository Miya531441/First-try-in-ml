"""Batched kinematics and collision resolution."""
from __future__ import annotations

import numpy as np

from env.agent import AgentState, EnvConfig
from env.raycast import wrap_angle


def integrate(state: AgentState, cfg: EnvConfig, actions: np.ndarray):
    """Apply continuous controls (forward, strafe, turn) in [-1, 1] to all alive agents.

    Velocity follows a first-order lag towards the commanded velocity; the turn rate
    is applied directly.  Dead agents do not move.
    """
    alive = state.alive.astype(np.float32)
    fwd = np.clip(actions[..., 0], -1, 1)
    strafe = np.clip(actions[..., 1], -1, 1)
    turn = np.clip(actions[..., 2], -1, 1)

    omega = turn * cfg.max_turn_rate * alive
    state.omega = omega.astype(np.float32)
    state.theta = wrap_angle(state.theta + omega * cfg.dt).astype(np.float32)

    # local thrust -> clip magnitude to 1 -> world frame (x forward, y left)
    thrust = np.stack([fwd, strafe], -1)
    mag = np.linalg.norm(thrust, axis=-1, keepdims=True)
    thrust = thrust / np.maximum(mag, 1.0)
    c, s = np.cos(state.theta), np.sin(state.theta)
    vx = c * thrust[..., 0] - s * thrust[..., 1]
    vy = s * thrust[..., 0] + c * thrust[..., 1]
    v_target = np.stack([vx, vy], -1) * cfg.max_speed * alive[..., None]
    alpha = min(1.0, cfg.dt / max(cfg.vel_tau, 1e-6))
    state.vel = (state.vel + (v_target - state.vel) * alpha).astype(np.float32)
    state.vel *= alive[..., None]
    state.pos = (state.pos + state.vel * cfg.dt).astype(np.float32)


def resolve_collisions(state: AgentState, cfg: EnvConfig, boxes: np.ndarray, box_mask: np.ndarray):
    """Push agents out of the outer walls, obstacles and each other.  Alive agents only."""
    r = cfg.collision_radius
    S = cfg.arena_size
    pos = state.pos
    alive = state.alive

    # outer walls
    pos[..., 0] = np.clip(pos[..., 0], r, S - r)
    pos[..., 1] = np.clip(pos[..., 1], r, S - r)

    # obstacles: loop over the (few) obstacle slots, vectorised over envs x agents
    for m in range(boxes.shape[1]):
        bm = box_mask[:, m]
        if not bm.any():
            continue
        lo = boxes[:, m, 0:2][:, None, :]
        hi = boxes[:, m, 2:4][:, None, :]
        closest = np.clip(pos, lo, hi)
        diff = pos - closest
        dist = np.linalg.norm(diff, axis=-1)
        inside = (dist < 1e-6) & (pos >= lo).all(-1) & (pos <= hi).all(-1)
        touching = (dist < r) & ~inside
        active = bm[:, None] & alive
        # outside but overlapping: push along the normal
        n = diff / np.maximum(dist, 1e-6)[..., None]
        push = (r - dist)[..., None] * n
        pos += np.where((touching & active)[..., None], push, 0.0)
        # inside: exit through the nearest face
        if (inside & active).any():
            d_lo = pos - lo + r
            d_hi = hi - pos + r
            cand = np.concatenate([d_lo, d_hi], -1)           # [E,N,4]
            k = cand.argmin(-1)
            delta = np.take_along_axis(cand, k[..., None], -1)[..., 0]
            move = np.zeros_like(pos)
            ax = k % 2
            sign = np.where(k < 2, -1.0, 1.0)
            np.put_along_axis(move, ax[..., None], (sign * delta)[..., None], -1)
            pos += np.where((inside & active)[..., None], move, 0.0)

    # agent-agent separation (two Jacobi sweeps)
    for _ in range(2):
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        pair = (dist < 2 * r) & alive[:, :, None] & alive[:, None, :]
        pair &= ~np.eye(pos.shape[1], dtype=bool)[None]
        if not pair.any():
            break
        n = diff / np.maximum(dist, 1e-6)[..., None]
        # coincident agents: deterministic separation axis
        n = np.where((dist < 1e-6)[..., None], np.array([1.0, 0.0], np.float32), n)
        push = 0.5 * (2 * r - dist)[..., None] * n * pair[..., None]
        pos += push.sum(2)
        pos[..., 0] = np.clip(pos[..., 0], r, S - r)
        pos[..., 1] = np.clip(pos[..., 1], r, S - r)

    state.pos = pos.astype(np.float32)


def local_frame(state: AgentState, vec: np.ndarray) -> np.ndarray:
    """Rotate world vectors [E, N, ..., 2] into each agent's local frame (x forward, y left)."""
    c, s = np.cos(state.theta), np.sin(state.theta)
    while c.ndim < vec.ndim - 1:
        c, s = c[..., None], s[..., None]
    x = c * vec[..., 0] + s * vec[..., 1]
    y = -s * vec[..., 0] + c * vec[..., 1]
    return np.stack([x, y], -1)

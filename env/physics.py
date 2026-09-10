"""Batched kinematics (asymmetric speeds, rate-limited turning) and collision resolution."""
from __future__ import annotations

import numpy as np

from env.agent import AgentState, EnvConfig
from env.raycast import wrap_angle


def integrate(state: AgentState, cfg: EnvConfig, actions: np.ndarray):
    """Controls in [-1, 1]: forward/backward, strafe, turn.  Dead agents do not move.

    Turn: the command sets a target angular velocity (+/- max_turn_rate); the change per
    step is capped by turn_accel.  Thrust: the (forward, strafe) command is normalised to
    the unit disc then scaled per axis (forward 4, backward 1.5, strafe 2 m/s); velocity
    follows it with a first-order lag.
    """
    alive = state.alive.astype(np.float32)
    fwd = np.clip(actions[..., 0], -1, 1)
    strafe = np.clip(actions[..., 1], -1, 1)
    turn = np.clip(actions[..., 2], -1, 1)

    omega_target = turn * cfg.max_turn_rate * alive
    d_omega = np.clip(omega_target - state.omega, -cfg.turn_accel * cfg.dt, cfg.turn_accel * cfg.dt)
    state.omega = ((state.omega + d_omega) * alive).astype(np.float32)
    state.theta = wrap_angle(state.theta + state.omega * cfg.dt).astype(np.float32)

    cmd = np.stack([fwd, strafe], -1)
    cmd = cmd / np.maximum(np.linalg.norm(cmd, axis=-1, keepdims=True), 1.0)
    lx = cmd[..., 0] * np.where(cmd[..., 0] >= 0, cfg.speed_forward, cfg.speed_backward)
    ly = cmd[..., 1] * cfg.speed_strafe
    c, s = np.cos(state.theta), np.sin(state.theta)
    v_target = np.stack([c * lx - s * ly, s * lx + c * ly], -1) * alive[..., None]
    alpha = min(1.0, cfg.dt / max(cfg.vel_tau, 1e-6))
    state.vel = ((state.vel + (v_target - state.vel) * alpha) * alive[..., None]).astype(np.float32)
    state.pos = (state.pos + state.vel * cfg.dt).astype(np.float32)


def resolve_collisions(state: AgentState, cfg: EnvConfig, boxes: np.ndarray, box_mask: np.ndarray):
    """Push agents out of the outer walls, all obstacles (walls and crates) and each other."""
    r = cfg.collision_radius
    S = cfg.arena_size
    pos = state.pos
    alive = state.alive
    pos[..., 0] = np.clip(pos[..., 0], r, S - r)
    pos[..., 1] = np.clip(pos[..., 1], r, S - r)

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
        n = diff / np.maximum(dist, 1e-6)[..., None]
        pos += np.where((touching & active)[..., None], (r - dist)[..., None] * n, 0.0)
        if (inside & active).any():
            cand = np.concatenate([pos - lo + r, hi - pos + r], -1)
            k = cand.argmin(-1)
            delta = np.take_along_axis(cand, k[..., None], -1)[..., 0]
            move = np.zeros_like(pos)
            np.put_along_axis(move, (k % 2)[..., None], (np.where(k < 2, -1.0, 1.0) * delta)[..., None], -1)
            pos += np.where((inside & active)[..., None], move, 0.0)

    for _ in range(2):
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        pair = (dist < 2 * r) & alive[:, :, None] & alive[:, None, :] & ~np.eye(pos.shape[1], dtype=bool)[None]
        if not pair.any():
            break
        n = diff / np.maximum(dist, 1e-6)[..., None]
        n = np.where((dist < 1e-6)[..., None], np.array([1.0, 0.0], np.float32), n)
        pos += (0.5 * (2 * r - dist)[..., None] * n * pair[..., None]).sum(2)
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

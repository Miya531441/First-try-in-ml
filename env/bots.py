"""Scripted baseline opponents.  Each bot maps a ``SquadVecEnv.bot_view`` dict to
continuous actions [E, T, 4] and only uses information the agents could observe
(visible enemies, teammates' broadcast contacts, own state, obstacle layout)."""
from __future__ import annotations

import numpy as np

from env.raycast import wrap_angle


def _turn_towards(theta, target_angle, gain=3.0):
    return np.clip(wrap_angle(target_angle - theta) * gain, -1.0, 1.0)


def _local_dir(theta, world_vec):
    c, s = np.cos(theta), np.sin(theta)
    x = c * world_vec[..., 0] + s * world_vec[..., 1]
    y = -s * world_vec[..., 0] + c * world_vec[..., 1]
    v = np.stack([x, y], -1)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)


def _best_target(view):
    """Nearest visible enemy per agent; falls back to own then allies' last-known contact.
    Returns (target_pos [E,T,2], has_target [E,T], visible [E,T])."""
    pos = view["pos"]
    epos = view["enemy_pos"]                                     # [E,T,T,2] nan where invisible
    d = np.linalg.norm(np.nan_to_num(epos - pos[:, :, None], nan=1e6), axis=-1)
    d = np.where(view["enemy_visible"], d, np.inf)
    j = d.argmin(-1)
    visible = np.isfinite(d.min(-1))
    vis_target = np.take_along_axis(epos, j[..., None, None].repeat(2, -1), 2)[:, :, 0]
    # team-wide freshest contact
    valid = view["contact_valid"]
    ctime = np.where(valid, view["contact_time"], -np.inf)
    k = ctime.argmax(-1)                                         # [E]
    any_valid = valid.any(-1)
    team_contact = view["contact_pos"][np.arange(pos.shape[0]), k]  # [E,2]
    own_ok = valid
    fallback = np.where(own_ok[..., None], view["contact_pos"], team_contact[:, None])
    has_fb = own_ok | any_valid[:, None]
    target = np.where(visible[..., None], np.nan_to_num(vis_target), fallback)
    return target, visible | has_fb, visible


class Bot:
    name = "bot"

    def __init__(self, cfg, num_envs: int, rng: np.random.Generator):
        self.cfg, self.E, self.rng = cfg, num_envs, rng

    def reset(self, idx):
        pass

    def act(self, view) -> np.ndarray:
        raise NotImplementedError


class RandomBot(Bot):
    name = "random"

    def act(self, view):
        E, T = view["pos"].shape[:2]
        a = self.rng.uniform(-1, 1, size=(E, T, 4)).astype(np.float32)
        a[..., 3] = (self.rng.random((E, T)) < 0.3)
        return a


class SpinnerBot(Bot):
    """Rotates continuously and fires whenever an enemy is in the cone."""
    name = "spinner"

    def __init__(self, cfg, num_envs, rng):
        super().__init__(cfg, num_envs, rng)
        self.dir = np.ones((num_envs, cfg.team_size), np.float32)

    def reset(self, idx):
        self.dir[idx] = self.rng.choice([-1.0, 1.0], size=(len(idx), self.cfg.team_size))

    def act(self, view):
        E, T = view["pos"].shape[:2]
        target, has, visible = _best_target(view)
        a = np.zeros((E, T, 4), np.float32)
        ang = np.arctan2(*(target - view["pos"])[..., ::-1].transpose(2, 0, 1))
        a[..., 2] = np.where(visible, _turn_towards(view["theta"], ang), 0.6 * self.dir)
        a[..., 3] = visible & (np.abs(wrap_angle(ang - view["theta"])) < np.deg2rad(4))
        return a


class ChargerBot(Bot):
    """Moves toward the last known enemy position (own or teammates'), fires on sight."""
    name = "charger"

    def act(self, view):
        E, T = view["pos"].shape[:2]
        target, has, visible = _best_target(view)
        S = self.cfg.arena_size
        # with no contact at all, head for the mirrored spawn (enemy side)
        default = (S - view["pos"])
        goal = np.where(has[..., None], target, default)
        rel = goal - view["pos"]
        ang = np.arctan2(rel[..., 1], rel[..., 0])
        a = np.zeros((E, T, 4), np.float32)
        a[..., 2] = _turn_towards(view["theta"], ang)
        ld = _local_dir(view["theta"], rel)
        dist = np.linalg.norm(rel, axis=-1)
        move = np.where((dist > 6.0) | ~visible, 1.0, 0.0)          # close in until ~6 m
        a[..., 0] = ld[..., 0] * move
        a[..., 1] = ld[..., 1] * move
        aligned = np.abs(wrap_angle(ang - view["theta"])) < np.deg2rad(4)
        a[..., 3] = visible & aligned
        # simple wall/obstacle avoidance: if blocked ahead, strafe
        a[..., 1] += self._avoid(view) * 0.8
        return a

    def _avoid(self, view):
        pos, theta = view["pos"], view["theta"]
        ahead = pos + np.stack([np.cos(theta), np.sin(theta)], -1) * 1.5
        boxes, mask = view["boxes"], view["box_mask"]
        lo = boxes[:, None, :, 0:2] - 0.5
        hi = boxes[:, None, :, 2:4] + 0.5
        inside = ((ahead[:, :, None] >= lo).all(-1) & (ahead[:, :, None] <= hi).all(-1) & mask[:, None]).any(-1)
        side = np.where(np.sin(theta * 3.0) > 0, 1.0, -1.0)
        return np.where(inside, side, 0.0)


class HolderBot(Bot):
    """Takes cover beside the obstacle closest to its spawn-side lane, then sweeps its
    cone across the approach toward the enemy side.  Fires on sight."""
    name = "holder"

    def __init__(self, cfg, num_envs, rng):
        super().__init__(cfg, num_envs, rng)
        self.goal = np.zeros((num_envs, cfg.team_size, 2), np.float32)
        self.phase = np.zeros((num_envs, cfg.team_size), np.float32)

    def reset(self, idx):
        self.goal[idx] = np.nan  # lazily chosen on first act
        self.phase[idx] = self.rng.uniform(0, 2 * np.pi, size=(len(idx), self.cfg.team_size))

    def _pick_goals(self, view, need):
        boxes, mask = view["boxes"], view["box_mask"]
        S = self.cfg.arena_size
        pos = view["pos"]
        centre = np.array([S / 2, S / 2], np.float32)
        for e, i in zip(*np.nonzero(need)):
            bm = boxes[e][mask[e]]
            if len(bm) == 0:
                self.goal[e, i] = pos[e, i] + (centre - pos[e, i]) * 0.3
                continue
            bc = (bm[:, 0:2] + bm[:, 2:4]) / 2
            score = np.linalg.norm(bc - pos[e, i], axis=-1) + 0.5 * np.linalg.norm(bc - centre, axis=-1)
            score += self.rng.uniform(0, 4, size=len(bm))
            b = bm[score.argmin()]
            # stand on the spawn side of the box, offset by 1 m
            toward = pos[e, i] - (b[0:2] + b[2:4]) / 2
            toward /= max(np.linalg.norm(toward), 1e-6)
            self.goal[e, i] = np.clip((b[0:2] + b[2:4]) / 2 + toward * (0.5 * max(b[2] - b[0], b[3] - b[1]) + 1.2),
                                      1.0, S - 1.0)

    def act(self, view):
        E, T = view["pos"].shape[:2]
        need = np.isnan(self.goal[..., 0])
        if need.any():
            self._pick_goals(view, need)
        target, has, visible = _best_target(view)
        pos, theta = view["pos"], view["theta"]
        a = np.zeros((E, T, 4), np.float32)
        rel_goal = self.goal - pos
        dist = np.linalg.norm(rel_goal, axis=-1)
        S = self.cfg.arena_size
        sweep_centre = np.arctan2(*((S - pos) - pos)[..., ::-1].transpose(2, 0, 1))
        t = view["t"][:, None] * self.cfg.dt
        sweep = sweep_centre + np.deg2rad(45) * np.sin(1.5 * t + self.phase)
        ang_t = np.arctan2(*(target - pos)[..., ::-1].transpose(2, 0, 1))
        desired = np.where(visible, ang_t, sweep)
        a[..., 2] = _turn_towards(theta, desired)
        moving = (dist > 0.8) & ~visible
        ld = _local_dir(theta, rel_goal)
        a[..., 0] = ld[..., 0] * moving
        a[..., 1] = ld[..., 1] * moving
        a[..., 3] = visible & (np.abs(wrap_angle(ang_t - theta)) < np.deg2rad(4))
        return a


BOTS = {b.name: b for b in (RandomBot, SpinnerBot, ChargerBot, HolderBot)}


def make_bot(name: str, cfg, num_envs: int, rng: np.random.Generator) -> Bot:
    return BOTS[name](cfg, num_envs, rng)

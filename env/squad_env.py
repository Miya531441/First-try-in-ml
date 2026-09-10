"""Batched squad-combat environment (spec v2) plus a PettingZoo ``ParallelEnv`` wrapper.

``SquadVecEnv`` simulates ``num_envs`` arenas in lock-step with numpy.  Team 0 occupies
agent slots ``[0, T)`` and team 1 ``[T, 2T)``; per episode a squad size pair is sampled
from ``cfg.team_sizes`` and unused slots stay inactive.  Finished episodes auto-reset and
report a summary in ``info["episode"]``.

Observation layout (per agent, egocentric):
  vision   R x [dist, nothing, wall, ally, enemy]          (walls occlude; crates do not)
  proprio  hp, ammo, cooldown, reload, vx, vy, omega, time
  allies   (T-1) x 8   blackboard ally slots
  enemies  T x 11      blackboard enemy tracks (persistent ids, staleness, confidence)
  map      64-ray 360-degree scan of static geometry (walls + crates), heading-rotated
  role one-hot, previous action, plan token one-hot (optional)
``obs_layout`` describes the slices so the encoder and mirror augmentation can find them.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from env.agent import AgentState, EnvConfig
from env.arena import Arena
from env.physics import integrate, local_frame, resolve_collisions
from env.raycast import cast_rays, cone_membership, ray_boundary_t, ray_box_t, segment_blocked, wrap_angle
from env.spawn import spawn_episode

DISCRETE_TABLE = [
    np.array([-1.0, 0.0, 1.0], np.float32),
    np.array([-1.0, 0.0, 1.0], np.float32),
    np.array([-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0], np.float32),
    np.array([0.0, 1.0], np.float32),
]
DISCRETE_NVEC = [len(t) for t in DISCRETE_TABLE]
ALLY_DIM, ENEMY_DIM = 8, 11
COVERAGE_BINS = 12


def decode_discrete(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    return np.stack([DISCRETE_TABLE[k][a[..., k]] for k in range(4)], -1).astype(np.float32)


def make_env_config(d: Optional[Dict[str, Any]] = None) -> EnvConfig:
    return EnvConfig.from_dict(d or {})


class SquadVecEnv:
    def __init__(self, cfg: EnvConfig, num_envs: int = 1, seed: int = 0):
        self.cfg = cfg
        self.E, self.T, self.N, self.R, self.M = int(num_envs), cfg.team_size, 2 * cfg.team_size, cfg.num_rays, cfg.max_obstacles
        self.K = cfg.map_rays
        self.rng = np.random.default_rng(seed)
        self.arena = Arena(self.E, cfg.arena_size, self.M)
        self.state = AgentState(self.E, self.N, cfg)
        self.t = np.zeros(self.E, np.int64)
        self.coverage_range = (cfg.coverage_min, cfg.coverage_max)     # curriculum sets this
        self.corner_lerp = 1.0                                          # curriculum sets this
        self.plan_token = np.zeros((self.E, 2), np.int64)

        E, N, R, T, K = self.E, self.N, self.R, self.T, self.K
        self.ray_dist = np.full((E, N, R), cfg.vision_range, np.float32)
        self.ray_kind = np.zeros((E, N, R), np.int64)
        self.ray_hit = np.full((E, N, R), -1, np.int64)
        self.map_scan = np.full((E, N, K), cfg.vision_range, np.float32)
        self.vis = np.zeros((E, N, N), bool)
        self.shot_fired = np.zeros((E, N), bool)
        self.shot_aim = np.zeros((E, N), np.float32)
        self.shot_dist = np.zeros((E, N), np.float32)
        self.shot_kind = np.zeros((E, N), np.int64)
        self.shot_victim = np.full((E, N), -1, np.int64)
        self.dmg_by = np.zeros((E, N, N), np.float32)
        # crossfire bookkeeping: last hit time / bearing on victim v by shooter s
        self.last_hit_time = np.full((E, N, N), -1e9, np.float32)
        self.last_hit_bearing = np.zeros((E, N, N), np.float32)

        self.allies = np.array([[j for j in range(N) if j != i and (j // T) == (i // T)] for i in range(N)],
                               dtype=np.int64).reshape(N, T - 1)
        self.enemies = np.array([[j for j in range(N) if (j // T) != (i // T)] for i in range(N)], dtype=np.int64)  # [N,T]
        self.gs_order = np.stack([np.arange(N), np.concatenate([np.arange(T, N), np.arange(T)])])
        self.slot = np.arange(N) % T

        # observation layout
        self.obs_layout: Dict[str, Any] = {}
        o = 0
        def add(name, size, shape=None):
            nonlocal o
            self.obs_layout[name] = {"start": o, "size": size, "shape": shape or (size,)}
            o += size
        add("vision", R * 5, (R, 5))
        add("proprio", 8)
        add("allies", (T - 1) * ALLY_DIM, (T - 1, ALLY_DIM))
        add("enemies", T * ENEMY_DIM, (T, ENEMY_DIM))
        add("map", K)
        add("role", cfg.num_roles)
        if cfg.include_prev_action:
            add("prev_action", 4)
        if cfg.plan_tokens:
            add("plan", cfg.plan_tokens)
        self.obs_dim = o
        self.plan_dim = cfg.plan_tokens
        self.plan_slice = (slice(self.obs_layout["plan"]["start"], self.obs_layout["plan"]["start"] + cfg.plan_tokens)
                           if cfg.plan_tokens else None)
        self.gs_core_dim = 11 * N + 6 * self.M + 1
        self.gs_dim = self.gs_core_dim + T + cfg.num_roles
        self.aux_dim = (T, 3)                  # per enemy slot: rel x, rel y (local, /arena), mask

        self._init_stats()
        self.reset()

    # ------------------------------------------------------------------ stats
    def _init_stats(self):
        E, N = self.E, self.N
        z = lambda *s: np.zeros(s, np.float32)
        self.ep_shots, self.ep_hits_enemy, self.ep_hits_ally = z(E, N), z(E, N), z(E, N)
        self.ep_damage_dealt, self.ep_damage_taken = z(E, N), z(E, N)
        self.ep_dist, self.ep_centroid_sum, self.ep_nearest_enemy_sum, self.ep_alive_steps = z(E, N), z(E, N), z(E, N), z(E, N)
        self.ep_pair_sum, self.ep_pair_steps = z(E, 2), z(E, 2)
        self.ep_cov_ent_sum, self.ep_cov_steps = z(E, 2), z(E, 2)
        self.ep_crossfire = z(E, 2)
        self.ep_flank_hits, self.ep_spot_steps, self.ep_close_damage = z(E, N), z(E, N), z(E, N)
        self.ep_reward = z(E, N)
        self.ep_first_contact = np.full(E, -1, np.int64)
        self.ep_engage_angles = [[] for _ in range(E)]

    def _reset_stats(self, idx):
        for arr in (self.ep_shots, self.ep_hits_enemy, self.ep_hits_ally, self.ep_damage_dealt, self.ep_damage_taken,
                    self.ep_dist, self.ep_centroid_sum, self.ep_nearest_enemy_sum, self.ep_alive_steps, self.ep_pair_sum,
                    self.ep_pair_steps, self.ep_cov_ent_sum, self.ep_cov_steps, self.ep_crossfire, self.ep_reward,
                    self.ep_flank_hits, self.ep_spot_steps, self.ep_close_damage):
            arr[idx] = 0
        self.ep_first_contact[idx] = -1
        for e in idx:
            self.ep_engage_angles[e] = []

    # ------------------------------------------------------------------ reset
    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def set_coverage_range(self, lo: float, hi: float):
        self.coverage_range = (float(lo), float(hi))

    def set_corner_lerp(self, k: float):
        """1.0 = full corner spawns, smaller = closer to the arena centre (curriculum)."""
        self.corner_lerp = float(k)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.seed(seed)
        idx = np.arange(self.E)
        self._reset_envs(idx)
        self._perceive(idx)
        return self._build_obs(), self._build_global_state()

    def _sample_roles(self) -> np.ndarray:
        cfg = self.cfg
        roles = np.zeros(self.N, np.int64)
        for k in range(2):
            perm = self.rng.permutation(cfg.num_roles)
            roles[k * self.T:(k + 1) * self.T] = np.tile(perm, int(np.ceil(self.T / cfg.num_roles)))[: self.T]
        return roles

    def _sample_active(self) -> np.ndarray:
        cfg = self.cfg
        sizes = cfg.team_sizes or [[self.T, self.T]]
        p = np.array(cfg.team_size_probs[: len(sizes)] if cfg.team_size_probs else [1.0] * len(sizes), np.float64)
        na, nb = sizes[self.rng.choice(len(sizes), p=p / p.sum())]
        active = np.zeros(self.N, bool)
        active[: min(na, self.T)] = True
        active[self.T: self.T + min(nb, self.T)] = True
        return active

    def _reset_envs(self, idx: np.ndarray):
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        boxes_l, kinds_l, pos, theta, roles, active = [], [], [], [], [], []
        for _ in idx:
            b, k, p, h = spawn_episode(self.rng, self.cfg, self.T, self.coverage_range, self.corner_lerp)
            boxes_l.append(b)
            kinds_l.append(k)
            pos.append(p)
            theta.append(h)
            roles.append(self._sample_roles())
            active.append(self._sample_active())
        self.arena.set(idx, boxes_l, kinds_l)
        self.state.reset_envs(idx, np.stack(pos), np.stack(theta), np.stack(roles), np.stack(active))
        self.t[idx] = 0
        self.plan_token[idx] = 0
        self.dmg_by[idx] = 0.0
        self.last_hit_time[idx] = -1e9
        self.shot_fired[idx] = False
        self._reset_stats(idx)

    # ------------------------------------------------------------- perception
    def _perceive(self, idx: np.ndarray):
        """Vision rays (walls + bodies), pairwise LOS and the 360-degree static map scan."""
        cfg, st = self.cfg, self.state
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        e, N, R, K = idx.size, self.N, self.R, self.K
        pos, theta, alive = st.pos[idx], st.theta[idx], st.alive[idx]
        boxes, wall_mask, all_mask = self.arena.boxes[idx], self.arena.wall_mask[idx], self.arena.mask[idx]
        r = cfg.collision_radius

        # vision cone rays: walls occlude, crates do not
        offsets = np.linspace(-cfg.fov / 2, cfg.fov / 2, R, dtype=np.float32)
        ang = theta[..., None] + offsets
        d = np.stack([np.cos(ang), np.sin(ang)], -1).reshape(e, N * R, 2).astype(np.float32)
        o = np.repeat(pos, R, axis=1)
        owner = np.repeat(np.arange(N), R)
        cmask = alive[:, None, :] & (owner[None, :, None] != np.arange(N)[None, None, :])
        dist, kind, hit = cast_rays(o, d, boxes, wall_mask, pos, r, cmask, cfg.arena_size, cfg.vision_range)
        dist, kind, hit = dist.reshape(e, N, R), kind.reshape(e, N, R), hit.reshape(e, N, R)
        dead = ~alive
        dist[dead], kind[dead], hit[dead] = cfg.vision_range, 0, -1
        self.ray_dist[idx], self.ray_kind[idx], self.ray_hit[idx] = dist, kind, hit

        # pairwise visibility: i sees j (cone, range, not blocked by a wall or a body)
        inside, _, _ = cone_membership(pos, theta, pos, cfg.fov, cfg.vision_range)
        p = np.repeat(pos, N, axis=1)
        q = np.tile(pos, (1, N, 1))
        ii, jj = np.repeat(np.arange(N), N), np.tile(np.arange(N), N)
        kmask = alive[:, None, :] & (ii[None, :, None] != np.arange(N)) & (jj[None, :, None] != np.arange(N))
        blocked = segment_blocked(p, q, boxes, wall_mask, pos, r, kmask).reshape(e, N, N)
        vis = inside & ~blocked & alive[:, :, None] & alive[:, None, :] & ~np.eye(N, dtype=bool)[None]
        self.vis[idx] = vis

        # static map scan: K rays over 360 degrees against walls, crates and the boundary
        mang = theta[..., None] + np.arange(K, dtype=np.float32) * (2 * np.pi / K)
        md = np.stack([np.cos(mang), np.sin(mang)], -1).reshape(e, N * K, 2).astype(np.float32)
        mo = np.repeat(pos, K, axis=1)
        t_wall = ray_boundary_t(mo, md, cfg.arena_size)
        if boxes.shape[1] > 0:
            t_wall = np.minimum(t_wall, ray_box_t(mo, md, boxes, all_mask).min(-1))
        self.map_scan[idx] = np.minimum(t_wall, cfg.vision_range).reshape(e, N, K)

    def _update_tracks(self, now: np.ndarray):
        """Each agent's own tracks are refreshed for every enemy it currently sees."""
        st, T, N = self.state, self.T, self.N
        en = self.enemies                                                    # [N,T]
        seen = np.take_along_axis(self.vis, en[None].repeat(self.E, 0), 2)  # [E,N,T]
        if not seen.any():
            return
        e_pos = st.pos[:, en]                                               # [E,N,T,2]
        st.track_pos = np.where(seen[..., None], e_pos, st.track_pos)
        st.track_vel = np.where(seen[..., None], st.vel[:, en], st.track_vel)
        st.track_theta = np.where(seen, st.theta[:, en], st.track_theta)
        st.track_hp = np.where(seen, st.hp[:, en], st.track_hp)
        st.track_time = np.where(seen, now[:, None, None], st.track_time)
        st.track_valid |= seen

    def _blackboard(self):
        """Team-merged tracks: per agent, the freshest track among allies (comms full or
        contacts_only) or its own (comms none).  Returns dict of [E,N,T,...] arrays."""
        st, cfg = self.state, self.cfg
        if cfg.comms == "none":
            return {k: getattr(st, "track_" + k) for k in ("pos", "vel", "theta", "hp", "time", "valid")}
        E, N, T = self.E, self.N, self.T
        team_members = np.concatenate([np.arange(N)[:, None], self.allies], 1)          # [N,T]
        tt = np.where(st.track_valid, st.track_time, -1.0)[:, team_members]              # [E,N,T(member),T(enemy)]
        best = tt.argmax(2)                                                              # [E,N,T(enemy)]
        e_idx = np.arange(E)[:, None, None]
        n_idx = np.arange(N)[None, :, None]
        en_idx = np.arange(T)[None, None, :]
        member = team_members[n_idx, best]                                               # global agent index [E,N,T]
        pick = lambda a: a[e_idx, member, en_idx]
        return {"pos": pick(st.track_pos), "vel": pick(st.track_vel), "theta": pick(st.track_theta),
                "hp": pick(st.track_hp), "time": pick(st.track_time), "valid": pick(st.track_valid)}

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        cfg, st = self.cfg, self.state
        E, N, T = self.E, self.N, self.T
        actions = np.asarray(actions, np.float32).reshape(E, N, 4).copy()
        actions[~st.alive] = 0.0
        alive_before = st.alive.copy()
        pos_before = st.pos.copy()

        integrate(st, cfg, actions)
        resolve_collisions(st, cfg, self.arena.boxes, self.arena.mask)

        # ---------------------------------------------------------- firing (hitscan along the cone bisector)
        want = actions[..., 3] > 0.5
        fire = want & st.alive & (st.cooldown <= 0.0) & (st.reload <= 0.0) & (st.ammo > 0)
        speed = np.linalg.norm(st.vel, axis=-1)
        spread = cfg.spread_rest_deg + (cfg.spread_max_deg - cfg.spread_rest_deg) * np.clip(speed / cfg.max_speed, 0, 1)
        aim = st.theta + self.rng.normal(0.0, 1.0, size=(E, N)).astype(np.float32) * np.deg2rad(spread)
        d = np.stack([np.cos(aim), np.sin(aim)], -1).astype(np.float32)
        cmask = st.alive[:, None, :] & ~np.eye(N, dtype=bool)[None]
        sdist, skind, svictim = cast_rays(st.pos, d, self.arena.boxes, self.arena.wall_mask, st.pos,
                                          cfg.collision_radius, cmask, cfg.arena_size, cfg.weapon_range)
        hit_agent = fire & (skind == 2)
        victim = np.where(hit_agent, svictim, -1)
        victim_team = np.take_along_axis(st.team, np.maximum(victim, 0), 1)
        friendly = hit_agent & (victim_team == st.team)
        enemy_hit = hit_agent & ~friendly
        apply = enemy_hit | (friendly & cfg.friendly_fire)

        dmg = np.zeros((E, N), np.float32)
        ee, ss = np.nonzero(apply)
        np.add.at(dmg, (ee, victim[ee, ss]), cfg.damage)
        ee2, ss2 = np.nonzero(enemy_hit)
        vv2 = victim[ee2, ss2]
        np.add.at(self.dmg_by, (ee2, vv2, ss2), cfg.damage)

        self.shot_fired, self.shot_aim, self.shot_dist = fire, aim.astype(np.float32), sdist
        self.shot_kind, self.shot_victim = np.where(fire, skind, 0), victim

        st.ammo -= fire.astype(np.int32)
        st.cooldown = np.where(fire, cfg.cooldown, st.cooldown).astype(np.float32)
        start_reload = (st.ammo <= 0) & (st.reload <= 0.0) & st.alive
        st.reload = np.where(start_reload, cfg.reload_time, st.reload).astype(np.float32)
        reload_before = st.reload > 0.0
        st.cooldown = np.maximum(st.cooldown - cfg.dt, 0.0).astype(np.float32)
        st.reload = np.maximum(st.reload - cfg.dt, 0.0).astype(np.float32)
        st.ammo = np.where(reload_before & (st.reload <= 0.0), cfg.magazine, st.ammo).astype(np.int32)

        # ---------------------------------------------------------- damage
        hp_before = st.hp.copy()
        st.hp = np.maximum(st.hp - dmg, 0.0).astype(np.float32)
        newly_dead = alive_before & (st.hp <= 0.0)
        st.alive = st.alive & (st.hp > 0.0)
        st.vel[~st.alive] = 0.0
        st.omega[~st.alive] = 0.0

        # ---------------------------------------------------------- rewards
        rw = cfg.reward
        dealt_enemy, dealt_ally = np.zeros((E, N), np.float32), np.zeros((E, N), np.float32)
        np.add.at(dealt_enemy, (ee2, ss2), cfg.damage)
        fe, fs = np.nonzero(friendly & cfg.friendly_fire)
        np.add.at(dealt_ally, (fe, fs), cfg.damage)
        taken = hp_before - st.hp
        reward = rw.damage_dealt * dealt_enemy + rw.damage_taken * taken + rw.friendly_damage * dealt_ally
        reward += rw.death * newly_dead + rw.step * st.alive
        if newly_dead.any():
            contrib = (self.dmg_by > 0.0) & newly_dead[:, :, None]
            if rw.kill_split:
                reward += (rw.kill * contrib / np.maximum(contrib.sum(-1, keepdims=True), 1)).sum(1)
            else:
                reward += rw.kill * contrib.sum(1)
            self.dmg_by[newly_dead] = 0.0
            # confirmed kills are broadcast: every agent's track of the victim reads hp 0
            for e_, v_ in zip(*np.nonzero(newly_dead)):
                slot = v_ % T
                obs_team = 1 - st.team[e_, v_]
                members = np.arange(obs_team * T, (obs_team + 1) * T)
                st.track_hp[e_, members, slot] = 0.0
                st.track_valid[e_, members, slot] = True
                st.track_pos[e_, members, slot] = st.pos[e_, v_]
                st.track_time[e_, members, slot] = (self.t[e_] + 1) * cfg.dt

        # ---------------------------------------------------------- perception, tracks, stats
        self._perceive(np.arange(E))
        now = (self.t + 1) * cfg.dt
        self._update_tracks(now.astype(np.float32))
        same_team = st.team_mask()
        enemy_vis = self.vis & ~same_team
        sees_enemy = enemy_vis.any(-1)
        st.prev_action = actions
        self.t += 1

        dist_mat = np.linalg.norm(st.pos[:, None, :, :] - st.pos[:, :, None, :], axis=-1)
        self.ep_first_contact = np.where((self.ep_first_contact < 0) & sees_enemy.any(-1), self.t, self.ep_first_contact)
        self.ep_shots += fire
        self.ep_hits_enemy += enemy_hit
        self.ep_hits_ally += friendly
        self.ep_damage_dealt += dealt_enemy
        self.ep_damage_taken += taken
        self.ep_dist += np.linalg.norm(st.pos - pos_before, axis=-1) * alive_before
        self.ep_alive_steps += st.alive
        alive_f = st.alive.astype(np.float32)
        for k in range(2):
            sl = slice(k * T, (k + 1) * T)
            a = alive_f[:, sl]
            n_alive = a.sum(-1)
            centroid = (st.pos[:, sl] * a[..., None]).sum(1) / np.maximum(n_alive, 1)[:, None]
            self.ep_centroid_sum[:, sl] += np.linalg.norm(st.pos[:, sl] - centroid[:, None], axis=-1) * a
            pair = a[:, :, None] * a[:, None, :] * (1 - np.eye(T))[None]
            npair = pair.sum((1, 2))
            self.ep_pair_sum[:, k] += (dist_mat[:, sl, sl] * pair).sum((1, 2)) / np.maximum(npair, 1) * (npair > 0)
            self.ep_pair_steps[:, k] += npair > 0
            # angular coverage entropy of the squad's cones
            centers = (np.arange(COVERAGE_BINS) + 0.5) * (2 * np.pi / COVERAGE_BINS)
            covered = np.abs(wrap_angle(centers[None, None, :] - st.theta[:, sl, None])) <= cfg.fov / 2   # [E,T,B]
            hist = (covered * a[..., None]).sum(1)
            pmass = hist / np.maximum(hist.sum(-1, keepdims=True), 1e-6)
            ent = -(pmass * np.log(np.maximum(pmass, 1e-9))).sum(-1) / np.log(COVERAGE_BINS)
            multi = n_alive >= 2
            self.ep_cov_ent_sum[:, k] += ent * multi
            self.ep_cov_steps[:, k] += multi
        enemy_dist = np.where(~same_team & st.alive[:, None, :], dist_mat, np.inf).min(-1)
        self.ep_nearest_enemy_sum += np.where(np.isfinite(enemy_dist), enemy_dist, cfg.arena_size) * alive_f

        eng = np.full((E, N), np.nan, np.float32)
        rr = cfg.role_rewards
        role_on = cfg.roles_enabled and rr.enabled
        if enemy_hit.any():
            rel = st.pos[ee2, ss2] - st.pos[ee2, vv2]
            bearing = np.arctan2(rel[:, 1], rel[:, 0])
            ang = np.abs(wrap_angle(bearing - st.theta[ee2, vv2]))
            hit_dist = np.linalg.norm(rel, axis=-1)
            eng[ee2, ss2] = np.rad2deg(ang)
            flank = ang > np.deg2rad(rr.flank_angle_deg)
            close = hit_dist <= rr.assault_close_range
            self.ep_flank_hits[ee2, ss2] += flank
            self.ep_close_damage[ee2, ss2] += cfg.damage * close
            for e_, a_ in zip(ee2, ang):
                self.ep_engage_angles[e_].append(float(np.rad2deg(a_)))
            # crossfire: another teammate hit the same victim within 2 s from a bearing >= 45 deg apart
            tnow = self.t * cfg.dt
            for e_, s_, v_, b_ in zip(ee2, ss2, vv2, bearing):
                mates = self.allies[s_]
                recent = self.last_hit_time[e_, v_, mates] >= tnow[e_] - 2.0
                apart = np.abs(wrap_angle(self.last_hit_bearing[e_, v_, mates] - b_)) >= np.deg2rad(45)
                cross = recent & apart
                if cross.any():
                    self.ep_crossfire[e_, st.team[e_, s_]] += 1
                    if role_on:
                        reward[e_, s_] += rr.crossfire_bonus
                        reward[e_, mates[cross]] += rr.crossfire_bonus
            self.last_hit_time[ee2, vv2, ss2] = tnow[ee2]
            self.last_hit_bearing[ee2, vv2, ss2] = bearing
            if role_on:
                role_s = st.role[ee2, ss2]
                reward[ee2, ss2] += rr.assault_close_damage * cfg.damage * close * (role_s == 0)
                reward[ee2, ss2] += rr.flank_damage * cfg.damage * flank * (role_s == 1)
                # overwatch assist: teammates' damage on an enemy this agent currently sees
                for e_, s_, v_ in zip(ee2, ss2, vv2):
                    mates = self.allies[s_]
                    watching = self.vis[e_, mates, v_] & (st.role[e_, mates] == 2)
                    reward[e_, mates[watching]] += rr.overwatch_assist_damage * cfg.damage
        # overwatch spotting: enemies only this agent sees, from range
        en = self.enemies
        sees = np.take_along_axis(self.vis, en[None].repeat(E, 0), 2)                    # [E,N,T]
        team_sees = np.zeros_like(sees)
        for j in range(T - 1):
            team_sees |= sees[:, self.allies[:, j]]
        far = np.linalg.norm(st.pos[:, en] - st.pos[:, :, None], axis=-1) >= rr.overwatch_min_range
        unique_spot = (sees & ~team_sees & far).sum(-1)
        self.ep_spot_steps += (unique_spot > 0)
        if role_on:
            reward += rr.overwatch_spot * unique_spot * (st.role == 2)
        self.ep_reward += reward

        # ---------------------------------------------------------- termination
        a_alive = st.alive[:, :T].any(-1)
        b_alive = st.alive[:, T:].any(-1)
        timeout = self.t >= cfg.max_steps
        done = ~a_alive | ~b_alive | timeout
        winner = np.full(E, -1, np.int64)
        winner[a_alive & ~b_alive] = 0
        winner[b_alive & ~a_alive] = 1
        if cfg.timeout_hp_tiebreak:
            hp_a, hp_b = st.hp[:, :T].sum(-1), st.hp[:, T:].sum(-1)
            tb = timeout & a_alive & b_alive
            winner[tb & (hp_a > hp_b)] = 0
            winner[tb & (hp_b > hp_a)] = 1
        term = np.where(winner[:, None] == st.team, rw.win, np.where(winner[:, None] < 0, rw.draw, rw.loss)) * st.active
        reward += term * done[:, None]
        self.ep_reward += term * done[:, None]
        reward *= st.active

        info: Dict[str, Any] = {"fire": fire, "hit_enemy": enemy_hit, "hit_ally": friendly, "engage_angle": eng,
                                "sees_enemy": sees_enemy, "alive": st.alive.copy(), "winner": winner, "t": self.t.copy()}
        done_idx = np.nonzero(done)[0]
        if done_idx.size:
            info["episode"] = self._episode_summary(done_idx, winner[done_idx])
            self._reset_envs(done_idx)
            self._perceive(done_idx)
        info["done_idx"] = done_idx
        return self._build_obs(), self._build_global_state(), reward.astype(np.float32), done, info

    def _episode_summary(self, idx, winner):
        cfg = self.cfg
        steps = np.maximum(self.ep_alive_steps[idx], 1)
        stats = np.stack([self.ep_centroid_sum[idx] / steps / cfg.arena_size,
                          self.ep_nearest_enemy_sum[idx] / steps / cfg.arena_size,
                          self.ep_shots[idx] / (cfg.max_steps * cfg.dt / cfg.cooldown),
                          self.ep_dist[idx] / (cfg.max_speed * cfg.max_steps * cfg.dt),
                          self.ep_flank_hits[idx] / np.maximum(self.ep_hits_enemy[idx], 1),
                          self.ep_spot_steps[idx] / steps], -1).astype(np.float32)
        hits_team = np.stack([self.ep_hits_enemy[idx, : self.T].sum(-1), self.ep_hits_enemy[idx, self.T:].sum(-1)], -1)
        return {
            "idx": idx, "winner": winner, "length": self.t[idx].copy(), "agent_stats": stats,
            "roles": self.state.role[idx].copy(), "active": self.state.active[idx].copy(),
            "shots": self.ep_shots[idx].copy(), "hits_enemy": self.ep_hits_enemy[idx].copy(),
            "hits_ally": self.ep_hits_ally[idx].copy(),
            "damage_dealt": self.ep_damage_dealt[idx].copy(), "damage_taken": self.ep_damage_taken[idx].copy(),
            "pair_dist": self.ep_pair_sum[idx] / np.maximum(self.ep_pair_steps[idx], 1),
            "coverage_entropy": self.ep_cov_ent_sum[idx] / np.maximum(self.ep_cov_steps[idx], 1),
            "crossfire_rate": self.ep_crossfire[idx] / np.maximum(hits_team, 1),
            "first_contact": self.ep_first_contact[idx].copy(), "return": self.ep_reward[idx].copy(),
            "flank_hits": self.ep_flank_hits[idx].copy(), "spot_steps": self.ep_spot_steps[idx].copy(),
            "close_damage": self.ep_close_damage[idx].copy(),
            "engage_angles": [list(self.ep_engage_angles[e]) for e in idx], "alive": self.state.alive[idx].copy(),
        }

    # ------------------------------------------------------------ observations
    def _build_obs(self) -> np.ndarray:
        cfg, st = self.cfg, self.state
        E, N, R, T = self.E, self.N, self.R, self.T
        parts = []
        nd = (self.ray_dist / cfg.vision_range)[..., None]
        hit_team = st.team[np.arange(E)[:, None, None], np.maximum(self.ray_hit, 0)]
        is_agent = self.ray_kind == 2
        own_team = st.team[:, :, None]
        onehot = np.stack([self.ray_kind == 0, self.ray_kind == 1, is_agent & (hit_team == own_team),
                           is_agent & (hit_team != own_team)], -1)
        parts.append(np.concatenate([nd, onehot.astype(np.float32)], -1).reshape(E, N, R * 5))
        lv = local_frame(st, st.vel) / cfg.max_speed
        parts.append(np.stack([st.hp / cfg.hp, st.ammo / cfg.magazine, st.cooldown / cfg.cooldown, st.reload / cfg.reload_time,
                               lv[..., 0], lv[..., 1], st.omega / cfg.max_turn_rate,
                               np.broadcast_to((self.t / cfg.max_steps)[:, None], (E, N))], -1).astype(np.float32))
        # ally slots (2 x 8) - only with comms == full
        al = self.allies
        ally_alive = st.alive[:, al].astype(np.float32)
        rel = local_frame(st, st.pos[:, al] - st.pos[:, :, None]) / cfg.arena_size
        dth = st.theta[:, al] - st.theta[:, :, None]
        allies = np.concatenate([rel, np.cos(dth)[..., None], np.sin(dth)[..., None], (st.hp[:, al] / cfg.hp)[..., None],
                                 ally_alive[..., None], (st.ammo[:, al] / cfg.magazine)[..., None],
                                 (st.reload[:, al] > 0)[..., None].astype(np.float32)], -1) * ally_alive[..., None]
        if cfg.comms != "full":
            allies = np.zeros_like(allies)
        parts.append(allies.reshape(E, N, -1).astype(np.float32))
        # enemy tracks (3 x 11)
        bb = self._blackboard()
        now = (self.t * cfg.dt)[:, None, None]
        valid = bb["valid"].astype(np.float32)
        stale = np.clip((now - bb["time"]) / cfg.track_staleness_cap, 0, 1) * valid + (1 - valid)
        conf = np.exp(-np.maximum(now - bb["time"], 0) / cfg.track_confidence_tau) * valid
        t_rel = local_frame(st, bb["pos"] - st.pos[:, :, None]) / cfg.arena_size
        t_dth = bb["theta"] - st.theta[:, :, None]
        t_vel = local_frame(st, bb["vel"]) / cfg.max_speed
        vis_now = np.take_along_axis(self.vis, self.enemies[None].repeat(E, 0), 2).astype(np.float32)
        enemies = np.concatenate([t_rel, np.cos(t_dth)[..., None], np.sin(t_dth)[..., None], t_vel,
                                  (bb["hp"] / cfg.hp)[..., None], stale[..., None], conf[..., None], valid[..., None],
                                  vis_now[..., None]], -1) * valid[..., None]
        enemies[..., 7] = stale                  # staleness reads 1 for never-seen tracks
        parts.append(enemies.reshape(E, N, -1).astype(np.float32))
        parts.append((self.map_scan / cfg.vision_range).astype(np.float32))
        role = np.eye(cfg.num_roles, dtype=np.float32)[st.role]
        parts.append(role if cfg.roles_enabled else np.zeros_like(role))
        if cfg.include_prev_action:
            parts.append(st.prev_action)
        if self.plan_dim:
            tok = self.plan_token[np.arange(E)[:, None], st.team]
            parts.append(np.eye(self.plan_dim, dtype=np.float32)[tok])
        obs = np.concatenate(parts, -1).astype(np.float32)
        obs[~st.alive] = 0.0
        return obs

    def aux_targets(self) -> np.ndarray:
        """[E, N, T, 3]: true relative position of each enemy slot (local frame, /arena) and a mask."""
        st, cfg = self.state, self.cfg
        rel = local_frame(st, st.pos[:, self.enemies] - st.pos[:, :, None]) / cfg.arena_size
        mask = (st.alive[:, self.enemies] & st.alive[:, :, None]).astype(np.float32)
        return np.concatenate([rel * mask[..., None], mask[..., None]], -1).astype(np.float32)

    def _team_global_state(self) -> np.ndarray:
        cfg, st = self.cfg, self.state
        E = self.E
        out = []
        for k in range(2):
            order = self.gs_order[k]
            pos, theta, vel = st.pos[:, order], st.theta[:, order], st.vel[:, order]
            if k == 1:
                pos, theta, vel = cfg.arena_size - pos, theta + np.pi, -vel
            feats = np.concatenate([
                pos / cfg.arena_size, np.cos(theta)[..., None], np.sin(theta)[..., None], vel / cfg.max_speed,
                (st.hp[:, order] / cfg.hp)[..., None], (st.ammo[:, order] / cfg.magazine)[..., None],
                (st.cooldown[:, order] / cfg.cooldown)[..., None], (st.reload[:, order] / cfg.reload_time)[..., None],
                st.alive[:, order].astype(np.float32)[..., None]], -1).reshape(E, -1)
            out.append(np.concatenate([feats, self.arena.encoding(mirrored=(k == 1)), (self.t / cfg.max_steps)[:, None]], -1))
        return np.stack(out, 1).astype(np.float32)

    def _build_global_state(self) -> np.ndarray:
        st, E, T = self.state, self.E, self.T
        gs = self._team_global_state()[np.arange(E)[:, None], st.team]
        slot = np.eye(T, dtype=np.float32)[self.slot][None].repeat(E, 0)
        role = np.eye(self.cfg.num_roles, dtype=np.float32)[st.role]
        return np.concatenate([gs, slot, role], -1).astype(np.float32)

    # ------------------------------------------------------------ utilities
    def set_plan_tokens(self, tokens: np.ndarray):
        self.plan_token[:] = tokens

    def bot_view(self, team: int) -> Dict[str, Any]:
        st, T = self.state, self.T
        own = slice(team * T, (team + 1) * T)
        enemy = slice((1 - team) * T, (2 - team) * T)
        vis = self.vis[:, own, enemy]
        bb = self._blackboard()
        return {
            "pos": st.pos[:, own], "theta": st.theta[:, own], "vel": st.vel[:, own], "alive": st.alive[:, own],
            "ammo": st.ammo[:, own], "reload": st.reload[:, own], "cooldown": st.cooldown[:, own],
            "enemy_visible": vis, "enemy_pos": np.where(vis[..., None], st.pos[:, None, enemy, :], np.nan),
            "track_pos": bb["pos"][:, own], "track_valid": bb["valid"][:, own] & (bb["hp"][:, own] > 0),
            "track_time": bb["time"][:, own],
            "boxes": self.arena.boxes, "box_mask": self.arena.mask, "box_kind": self.arena.kind, "t": self.t, "team": team,
        }

    def get_frame(self, e: int) -> Dict[str, Any]:
        st = self.state
        m = self.arena.mask[e]
        return {
            "t": int(self.t[e]), "pos": st.pos[e].copy(), "theta": st.theta[e].copy(), "hp": st.hp[e].copy(),
            "alive": st.alive[e].copy(), "team": st.team[e].copy(), "role": st.role[e].copy(),
            "ray_dist": self.ray_dist[e].copy(), "ray_kind": self.ray_kind[e].copy(),
            "shot_fired": self.shot_fired[e].copy(), "shot_aim": self.shot_aim[e].copy(),
            "shot_dist": self.shot_dist[e].copy(), "shot_kind": self.shot_kind[e].copy(),
            "boxes": self.arena.boxes[e][m].copy(), "box_kind": self.arena.kind[e][m].copy(),
            "plan": self.plan_token[e].copy(),
        }


# ---------------------------------------------------------------------------
try:
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
except ImportError:                                   # pragma: no cover
    ParallelEnv = object
    spaces = None


class SquadParallelEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` over a single ``SquadVecEnv``.  Agents are ``team{k}_{i}``;
    inactive slots (smaller squads) are absent from ``agents`` for that episode."""

    metadata = {"render_modes": ["rgb_array"], "name": "squad_combat_v2", "is_parallelizable": True}

    def __init__(self, cfg: Optional[EnvConfig] = None, action_mode: str = "hybrid", seed: int = 0,
                 render_mode: Optional[str] = None):
        self.cfg = cfg or EnvConfig()
        self.env = SquadVecEnv(self.cfg, num_envs=1, seed=seed)
        self.action_mode = action_mode
        self.render_mode = render_mode
        self.possible_agents = [f"team{i // self.cfg.team_size}_{i % self.cfg.team_size}" for i in range(self.env.N)]
        self.agents = list(self.possible_agents)
        self._obs_space = spaces.Box(-np.inf, np.inf, (self.env.obs_dim,), np.float32)
        self._act_space = spaces.MultiDiscrete(DISCRETE_NVEC) if action_mode == "discrete" else spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._renderer = None

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def state(self) -> np.ndarray:
        return self.env._build_global_state()[0]

    def reset(self, seed=None, options=None):
        obs, gs = self.env.reset(seed=seed)
        active = self.env.state.active[0]
        self.agents = [a for i, a in enumerate(self.possible_agents) if active[i]]
        return ({a: obs[0, i] for i, a in enumerate(self.possible_agents) if active[i]},
                {a: {"global_state": gs[0, i]} for i, a in enumerate(self.possible_agents) if active[i]})

    def step(self, actions):
        act = np.zeros((1, self.env.N, 4), np.float32)
        for i, a in enumerate(self.possible_agents):
            if a in actions:
                v = np.asarray(actions[a], np.float32)
                act[0, i] = decode_discrete(v) if self.action_mode == "discrete" else v
        alive_before = self.env.state.alive[0].copy()
        obs, gs, rew, done, info = self.env.step(act)
        done = bool(done[0])
        alive_after = info["alive"][0]
        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for i, a in enumerate(self.possible_agents):
            if a not in self.agents:
                continue
            observations[a] = obs[0, i]
            rewards[a] = float(rew[0, i])
            terminations[a] = bool(done or (alive_before[i] and not alive_after[i]))
            truncations[a] = bool(done and "episode" in info and info["episode"]["length"][0] >= self.cfg.max_steps
                                  and info["episode"]["winner"][0] < 0)
            infos[a] = {"global_state": gs[0, i], "alive": bool(alive_after[i])}
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        from scripts.render import Renderer
        if self._renderer is None:
            self._renderer = Renderer(self.cfg, headless=True)
        return self._renderer.draw(self.env.get_frame(0))

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

"""Batched 3v3 squad-combat environment plus a PettingZoo ``ParallelEnv`` wrapper.

``SquadVecEnv`` simulates ``num_envs`` arenas in lock-step with numpy; every
per-step quantity (physics, hitscan, ray casting, line-of-sight) is vectorised over
envs x agents x rays.  Team 0 occupies agent indices ``[0, T)`` and team 1
``[T, 2T)``.  The env auto-resets finished episodes and reports their summary in
``info``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from env.agent import AgentState, EnvConfig
from env.arena import Arena
from env.physics import integrate, local_frame, resolve_collisions
from env.raycast import (cast_rays, cone_membership, ray_directions, segment_blocked, wrap_angle)
from env.spawn import spawn_episode

# fully-discretised action variant: (forward, strafe, turn, fire)
DISCRETE_TABLE = [
    np.array([-1.0, 0.0, 1.0], np.float32),
    np.array([-1.0, 0.0, 1.0], np.float32),
    np.array([-1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0], np.float32),   # fine turn steps so aiming is possible
    np.array([0.0, 1.0], np.float32),
]
DISCRETE_NVEC = [len(t) for t in DISCRETE_TABLE]


def decode_discrete(a: np.ndarray) -> np.ndarray:
    """[..., 4] integer indices -> [..., 4] continuous action vector."""
    a = np.asarray(a, dtype=np.int64)
    return np.stack([DISCRETE_TABLE[k][a[..., k]] for k in range(4)], -1).astype(np.float32)


def make_env_config(d: Optional[Dict[str, Any]] = None) -> EnvConfig:
    return EnvConfig.from_dict(d or {})


class SquadVecEnv:
    """Vectorised environment.  See module docstring."""

    def __init__(self, cfg: EnvConfig, num_envs: int = 1, seed: int = 0):
        self.cfg = cfg
        self.E = int(num_envs)
        self.T = cfg.team_size
        self.N = 2 * self.T
        self.R = cfg.num_rays
        self.M = cfg.max_obstacles
        self.rng = np.random.default_rng(seed)
        self.arena = Arena(self.E, cfg.arena_size, self.M)
        self.state = AgentState(self.E, self.N, cfg)
        self.t = np.zeros(self.E, np.int64)
        self.shaping_coef = 1.0                      # annealed externally
        self.plan_token = np.zeros((self.E, 2), np.int64)

        E, N, R, T = self.E, self.N, self.R, self.T
        self.ray_dist = np.full((E, N, R), cfg.vision_range, np.float32)
        self.ray_kind = np.zeros((E, N, R), np.int64)
        self.ray_hit = np.full((E, N, R), -1, np.int64)
        self.vis = np.zeros((E, N, N), bool)          # vis[e,i,j]: i sees j (cone + clear LOS)
        # last shots (for rendering / metrics)
        self.shot_fired = np.zeros((E, N), bool)
        self.shot_aim = np.zeros((E, N), np.float32)
        self.shot_dist = np.zeros((E, N), np.float32)
        self.shot_kind = np.zeros((E, N), np.int64)
        self.shot_victim = np.full((E, N), -1, np.int64)
        self.dmg_by = np.zeros((E, N, N), np.float32)  # damage dealt to victim i by shooter j

        # ally index table: allies[i] = other members of i's team, ascending
        self.allies = np.array([[j for j in range(N) if j != i and (j // T) == (i // T)] for i in range(N)],
                               dtype=np.int64).reshape(N, T - 1)
        self.gs_order = np.stack([np.arange(N), np.concatenate([np.arange(T, N), np.arange(T)])])
        self.slot = np.arange(N) % T

        # observation layout
        self.vision_dim = R * 5
        self.proprio_dim = 8
        self.team_dim = (T - 1) * 10
        self.role_dim = cfg.num_roles
        self.prev_action_dim = 4 if cfg.include_prev_action else 0
        self.plan_dim = cfg.plan_tokens
        self.obs_dim = (self.vision_dim + self.proprio_dim + self.team_dim + self.role_dim
                        + self.prev_action_dim + self.plan_dim)
        self.plan_slice = slice(self.obs_dim - self.plan_dim, self.obs_dim) if self.plan_dim else None
        self.gs_core_dim = 11 * N + 5 * self.M + 1
        self.gs_dim = self.gs_core_dim + T + cfg.num_roles

        self._init_stats()
        self.reset()

    # ------------------------------------------------------------------ stats
    def _init_stats(self):
        E, N = self.E, self.N
        self.ep_shots = np.zeros((E, N), np.float32)
        self.ep_hits_enemy = np.zeros((E, N), np.float32)
        self.ep_hits_ally = np.zeros((E, N), np.float32)
        self.ep_damage_dealt = np.zeros((E, N), np.float32)
        self.ep_damage_taken = np.zeros((E, N), np.float32)
        self.ep_dist = np.zeros((E, N), np.float32)
        self.ep_centroid_sum = np.zeros((E, N), np.float32)
        self.ep_nearest_enemy_sum = np.zeros((E, N), np.float32)
        self.ep_alive_steps = np.zeros((E, N), np.float32)
        self.ep_pair_sum = np.zeros((E, 2), np.float32)
        self.ep_pair_steps = np.zeros((E, 2), np.float32)
        self.ep_first_contact = np.full(E, -1, np.int64)
        self.ep_reward = np.zeros((E, N), np.float32)
        self.ep_engage_angles = [[] for _ in range(E)]

    def _reset_stats(self, idx):
        for arr in (self.ep_shots, self.ep_hits_enemy, self.ep_hits_ally, self.ep_damage_dealt,
                    self.ep_damage_taken, self.ep_dist, self.ep_centroid_sum, self.ep_nearest_enemy_sum,
                    self.ep_alive_steps, self.ep_pair_sum, self.ep_pair_steps, self.ep_reward):
            arr[idx] = 0
        self.ep_first_contact[idx] = -1
        for e in idx:
            self.ep_engage_angles[e] = []

    # ------------------------------------------------------------------ reset
    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

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
            reps = int(np.ceil(self.T / cfg.num_roles))
            roles[k * self.T:(k + 1) * self.T] = np.tile(perm, reps)[: self.T]
        return roles

    def _reset_envs(self, idx: np.ndarray):
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        boxes_list, pos, theta, roles = [], [], [], []
        for _ in idx:
            b, p, h = spawn_episode(self.rng, self.cfg, self.T)
            boxes_list.append(b)
            pos.append(p)
            theta.append(h)
            roles.append(self._sample_roles())
        self.arena.set(idx, boxes_list)
        self.state.reset_envs(idx, np.stack(pos), np.stack(theta), np.stack(roles))
        self.t[idx] = 0
        self.plan_token[idx] = 0
        self.dmg_by[idx] = 0.0
        self.shot_fired[idx] = False
        self._reset_stats(idx)

    # ------------------------------------------------------------- perception
    def _perceive(self, idx: np.ndarray):
        """Ray cast + pairwise line-of-sight for the envs in ``idx``."""
        cfg, st = self.cfg, self.state
        idx = np.asarray(idx)
        if idx.size == 0:
            return
        e = idx.size
        N, R = self.N, self.R
        pos, theta, alive = st.pos[idx], st.theta[idx], st.alive[idx]
        boxes, bmask = self.arena.boxes[idx], self.arena.mask[idx]
        r = cfg.collision_radius

        # vision rays
        d = ray_directions(theta, cfg.fov, R).reshape(e, N * R, 2)
        o = np.repeat(pos, R, axis=1)                               # [e, N*R, 2]
        owner = np.repeat(np.arange(N), R)                          # [N*R]
        cmask = alive[:, None, :] & (owner[None, :, None] != np.arange(N)[None, None, :])
        dist, kind, hit = cast_rays(o, d, boxes, bmask, pos, r, cmask, cfg.arena_size, cfg.vision_range)
        dead = ~alive
        dist = dist.reshape(e, N, R)
        kind = kind.reshape(e, N, R)
        hit = hit.reshape(e, N, R)
        dist[dead] = cfg.vision_range
        kind[dead] = 0
        hit[dead] = -1
        self.ray_dist[idx], self.ray_kind[idx], self.ray_hit[idx] = dist, kind, hit

        # pairwise visibility: i sees j
        inside, _, _ = cone_membership(pos, theta, pos, cfg.fov, cfg.vision_range)   # [e,N,N]
        p = np.repeat(pos, N, axis=1)                                # [e, N*N, 2]  (i major)
        q = np.tile(pos, (1, N, 1))                                  # [e, N*N, 2]
        ii = np.repeat(np.arange(N), N)
        jj = np.tile(np.arange(N), N)
        kmask = alive[:, None, :] & (ii[None, :, None] != np.arange(N)) & (jj[None, :, None] != np.arange(N))
        blocked = segment_blocked(p, q, boxes, bmask, pos, r, kmask).reshape(e, N, N)
        vis = inside & ~blocked & alive[:, :, None] & alive[:, None, :]
        vis &= ~np.eye(N, dtype=bool)[None]
        self.vis[idx] = vis

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

        # ---------------------------------------------------------- firing
        want = actions[..., 3] > 0.5
        can = st.alive & (st.cooldown <= 0.0) & (st.reload <= 0.0) & (st.ammo > 0)
        fire = want & can
        speed = np.linalg.norm(st.vel, axis=-1)
        spread = cfg.spread_rest_deg + (cfg.spread_max_deg - cfg.spread_rest_deg) * np.clip(speed / cfg.max_speed, 0, 1)
        aim = st.theta + self.rng.normal(0.0, 1.0, size=(E, N)).astype(np.float32) * np.deg2rad(spread)
        d = np.stack([np.cos(aim), np.sin(aim)], -1).astype(np.float32)
        cmask = st.alive[:, None, :] & ~np.eye(N, dtype=bool)[None]
        sdist, skind, svictim = cast_rays(st.pos, d, self.arena.boxes, self.arena.mask, st.pos,
                                          cfg.collision_radius, cmask, cfg.arena_size, 1e6)
        hit_agent = fire & (skind == 2)
        victim = np.where(hit_agent, svictim, -1)
        shooter_team = st.team
        victim_team = np.take_along_axis(st.team, np.maximum(victim, 0), 1)
        friendly = hit_agent & (victim_team == shooter_team)
        enemy_hit = hit_agent & ~friendly
        apply = enemy_hit | (friendly & cfg.friendly_fire)

        dmg = np.zeros((E, N), np.float32)                 # damage received per victim
        ee, ss = np.nonzero(apply)
        np.add.at(dmg, (ee, victim[ee, ss]), cfg.damage)
        # kill-credit bookkeeping (enemy damage only)
        ee2, ss2 = np.nonzero(enemy_hit)
        np.add.at(self.dmg_by, (ee2, victim[ee2, ss2], ss2), cfg.damage)

        self.shot_fired = fire
        self.shot_aim = aim.astype(np.float32)
        self.shot_dist = sdist
        self.shot_kind = np.where(fire, skind, 0)
        self.shot_victim = victim

        # ammo / timers
        st.ammo -= fire.astype(np.int32)
        st.cooldown = np.where(fire, cfg.cooldown, st.cooldown).astype(np.float32)
        start_reload = (st.ammo <= 0) & (st.reload <= 0.0) & st.alive
        st.reload = np.where(start_reload, cfg.reload_time, st.reload).astype(np.float32)
        reload_before = st.reload > 0.0
        st.cooldown = np.maximum(st.cooldown - cfg.dt, 0.0).astype(np.float32)
        st.reload = np.maximum(st.reload - cfg.dt, 0.0).astype(np.float32)
        finished = reload_before & (st.reload <= 0.0)
        st.ammo = np.where(finished, cfg.magazine, st.ammo).astype(np.int32)

        # ---------------------------------------------------------- damage
        hp_before = st.hp.copy()
        st.hp = np.maximum(st.hp - dmg, 0.0).astype(np.float32)
        newly_dead = alive_before & (st.hp <= 0.0)
        st.alive = st.alive & (st.hp > 0.0)
        st.vel[~st.alive] = 0.0

        # ---------------------------------------------------------- rewards
        rw = cfg.reward
        dealt_enemy = np.zeros((E, N), np.float32)
        dealt_ally = np.zeros((E, N), np.float32)
        np.add.at(dealt_enemy, (ee2, ss2), cfg.damage)
        fe, fs = np.nonzero(friendly & cfg.friendly_fire)
        np.add.at(dealt_ally, (fe, fs), cfg.damage)
        taken = hp_before - st.hp
        reward = (rw.damage_dealt * dealt_enemy + rw.damage_taken * taken + rw.friendly_damage * dealt_ally)
        reward += rw.death * newly_dead
        reward += rw.step * st.alive
        # kill credit: every enemy who damaged the victim
        if newly_dead.any():
            contrib = (self.dmg_by > 0.0) & newly_dead[:, :, None]        # [E, victim, shooter]
            if rw.kill_split:
                n_contrib = np.maximum(contrib.sum(-1, keepdims=True), 1)
                reward += (rw.kill * contrib / n_contrib).sum(1)
            else:
                reward += rw.kill * contrib.sum(1)
            self.dmg_by[newly_dead] = 0.0

        # ---------------------------------------------------------- perception / shaping
        self._perceive(np.arange(E))
        same_team = st.team_mask()
        enemy_vis = self.vis & ~same_team                                  # [E,N,N]
        sees_enemy = enemy_vis.any(-1)
        reward += self.shaping_coef * rw.shaping_los * sees_enemy
        # last-known contacts
        dpos = st.pos[:, None, :, :] - st.pos[:, :, None, :]
        dist_mat = np.linalg.norm(dpos, axis=-1)
        dist_masked = np.where(enemy_vis, dist_mat, np.inf)
        nearest = dist_masked.argmin(-1)
        now = (self.t + 1) * cfg.dt
        st.contact_pos = np.where(sees_enemy[..., None], np.take_along_axis(st.pos, nearest[..., None].repeat(2, -1), 1),
                                  st.contact_pos).astype(np.float32)
        st.contact_time = np.where(sees_enemy, now[:, None], st.contact_time).astype(np.float32)
        st.contact_valid |= sees_enemy
        st.prev_action = actions

        # ---------------------------------------------------------- stats
        self.t += 1
        any_contact = sees_enemy.any(-1)
        self.ep_first_contact = np.where((self.ep_first_contact < 0) & any_contact, self.t, self.ep_first_contact)
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
            cd = np.linalg.norm(st.pos[:, sl] - centroid[:, None], axis=-1) * a
            self.ep_centroid_sum[:, sl] += cd
            # pairwise mean distance among alive teammates
            pair = a[:, :, None] * a[:, None, :] * (1 - np.eye(T))[None]
            pd = (dist_mat[:, sl, sl] * pair).sum((1, 2)) / np.maximum(pair.sum((1, 2)), 1)
            has_pair = pair.sum((1, 2)) > 0
            self.ep_pair_sum[:, k] += pd * has_pair
            self.ep_pair_steps[:, k] += has_pair
        enemy_dist = np.where(~same_team & st.alive[:, None, :], dist_mat, np.inf).min(-1)
        enemy_dist = np.where(np.isfinite(enemy_dist), enemy_dist, cfg.arena_size)
        self.ep_nearest_enemy_sum += enemy_dist * alive_f
        eng = np.full((E, N), np.nan, np.float32)
        if enemy_hit.any():
            # angle of attack relative to the victim's heading: 0 = frontal, 180 = from behind
            v = victim[ee2, ss2]
            rel = st.pos[ee2, ss2] - st.pos[ee2, v]
            ang = np.abs(wrap_angle(np.arctan2(rel[:, 1], rel[:, 0]) - st.theta[ee2, v]))
            eng[ee2, ss2] = np.rad2deg(ang)
            for e_, a_ in zip(ee2, ang):
                self.ep_engage_angles[e_].append(float(np.rad2deg(a_)))
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
        term = np.where(winner[:, None] == st.team, rw.win, np.where(winner[:, None] < 0, rw.draw, rw.loss))
        reward += term * done[:, None]
        self.ep_reward += term * done[:, None]

        info: Dict[str, Any] = {
            "fire": fire, "hit_enemy": enemy_hit, "hit_ally": friendly, "engage_angle": eng,
            "sees_enemy": sees_enemy, "alive": st.alive.copy(), "winner": winner, "t": self.t.copy(),
        }
        done_idx = np.nonzero(done)[0]
        if done_idx.size:
            info["episode"] = self._episode_summary(done_idx, winner[done_idx])
            self._reset_envs(done_idx)
            self._perceive(done_idx)
        info["done_idx"] = done_idx
        return self._build_obs(), self._build_global_state(), reward.astype(np.float32), done, info

    def _episode_summary(self, idx, winner):
        steps = np.maximum(self.ep_alive_steps[idx], 1)
        stats = np.stack([
            self.ep_centroid_sum[idx] / steps / self.cfg.arena_size,
            self.ep_nearest_enemy_sum[idx] / steps / self.cfg.arena_size,
            self.ep_shots[idx] / (self.cfg.max_steps * self.cfg.dt / self.cfg.cooldown),
            self.ep_dist[idx] / (self.cfg.max_speed * self.cfg.max_steps * self.cfg.dt),
        ], -1).astype(np.float32)
        return {
            "idx": idx, "winner": winner, "length": self.t[idx].copy(),
            "agent_stats": stats, "roles": self.state.role[idx].copy(),
            "shots": self.ep_shots[idx].copy(), "hits_enemy": self.ep_hits_enemy[idx].copy(),
            "hits_ally": self.ep_hits_ally[idx].copy(),
            "damage_dealt": self.ep_damage_dealt[idx].copy(), "damage_taken": self.ep_damage_taken[idx].copy(),
            "pair_dist": self.ep_pair_sum[idx] / np.maximum(self.ep_pair_steps[idx], 1),
            "first_contact": self.ep_first_contact[idx].copy(),
            "return": self.ep_reward[idx].copy(),
            "engage_angles": [list(self.ep_engage_angles[e]) for e in idx],
            "alive": self.state.alive[idx].copy(),
        }

    # ------------------------------------------------------------ observations
    def _build_obs(self) -> np.ndarray:
        cfg, st = self.cfg, self.state
        E, N, R, T = self.E, self.N, self.R, self.T
        parts = []
        # 1. vision
        nd = (self.ray_dist / cfg.vision_range)[..., None]
        hit_team = st.team[np.arange(E)[:, None, None], np.maximum(self.ray_hit, 0)]      # [E,N,R]
        is_agent = self.ray_kind == 2
        own_team = st.team[:, :, None]
        onehot = np.stack([self.ray_kind == 0, self.ray_kind == 1,
                           is_agent & (hit_team == own_team), is_agent & (hit_team != own_team)], -1)
        parts.append(np.concatenate([nd, onehot.astype(np.float32)], -1).reshape(E, N, R * 5))
        # 2. proprioception
        lv = local_frame(st, st.vel) / cfg.max_speed
        parts.append(np.stack([st.hp / cfg.hp, st.ammo / cfg.magazine, st.cooldown / cfg.cooldown,
                               st.reload / cfg.reload_time, lv[..., 0], lv[..., 1],
                               st.omega / cfg.max_turn_rate,
                               np.broadcast_to((self.t / cfg.max_steps)[:, None], (E, N))], -1).astype(np.float32))
        # 3. team channel
        if T > 1:
            al = self.allies                                                  # [N, T-1]
            ally_pos = st.pos[:, al]                                          # [E,N,T-1,2]
            ally_theta = st.theta[:, al]
            ally_alive = st.alive[:, al].astype(np.float32)
            rel = local_frame(st, ally_pos - st.pos[:, :, None]) / cfg.arena_size
            dth = ally_theta - st.theta[:, :, None]
            c_pos = st.contact_pos[:, al]
            c_rel = local_frame(st, c_pos - st.pos[:, :, None]) / cfg.arena_size
            c_valid = st.contact_valid[:, al].astype(np.float32)
            now = (self.t * cfg.dt)[:, None, None]
            stale = np.clip((now - st.contact_time[:, al]) / cfg.contact_staleness_cap, 0, 1) * c_valid + (1 - c_valid)
            team = np.concatenate([rel, np.cos(dth)[..., None], np.sin(dth)[..., None],
                                   (st.hp[:, al] / cfg.hp)[..., None], ally_alive[..., None],
                                   c_rel * c_valid[..., None], stale[..., None], c_valid[..., None]], -1)
            team = team * ally_alive[..., None]          # dead allies: all zeros (alive flag included)
            parts.append(team.reshape(E, N, -1).astype(np.float32))
        # 4. role
        role = np.eye(cfg.num_roles, dtype=np.float32)[st.role]
        if not cfg.roles_enabled:
            role = np.zeros_like(role)
        parts.append(role)
        # 5. previous action
        if cfg.include_prev_action:
            parts.append(st.prev_action)
        # 6. plan token
        if self.plan_dim:
            tok = self.plan_token[np.arange(E)[:, None], st.team]              # [E,N]
            parts.append(np.eye(self.plan_dim, dtype=np.float32)[tok])
        obs = np.concatenate(parts, -1).astype(np.float32)
        obs[~st.alive] = 0.0
        return obs

    def _team_global_state(self) -> np.ndarray:
        """[E, 2, gs_core_dim]: full state from each team's (mirrored) perspective."""
        cfg, st = self.cfg, self.state
        E, N = self.E, self.N
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
            enc = self.arena.encoding(mirrored=(k == 1))
            out.append(np.concatenate([feats, enc, (self.t / cfg.max_steps)[:, None]], -1))
        return np.stack(out, 1).astype(np.float32)

    def _build_global_state(self) -> np.ndarray:
        st = self.state
        E, N, T = self.E, self.N, self.T
        team_gs = self._team_global_state()
        gs = team_gs[np.arange(E)[:, None], st.team]                           # [E,N,core]
        slot = np.eye(T, dtype=np.float32)[self.slot][None].repeat(E, 0)
        role = np.eye(self.cfg.num_roles, dtype=np.float32)[st.role]
        return np.concatenate([gs, slot, role], -1).astype(np.float32)

    # ------------------------------------------------------------ utilities
    def set_plan_tokens(self, tokens: np.ndarray):
        self.plan_token[:] = tokens

    def set_shaping_coef(self, c: float):
        self.shaping_coef = float(c)

    def bot_view(self, team: int) -> Dict[str, Any]:
        """Everything a scripted bot on ``team`` may use: own-team state, which enemies
        are currently visible (and their positions only where visible), teammates'
        last-known contacts, and the obstacle layout."""
        st, T = self.state, self.T
        own = slice(team * T, (team + 1) * T)
        enemy = slice((1 - team) * T, (2 - team) * T)
        vis = self.vis[:, own, enemy]
        enemy_pos = np.where(vis[..., None], st.pos[:, None, enemy, :], np.nan)   # [E,T,T,2]
        return {
            "pos": st.pos[:, own], "theta": st.theta[:, own], "vel": st.vel[:, own], "alive": st.alive[:, own],
            "ammo": st.ammo[:, own], "reload": st.reload[:, own], "cooldown": st.cooldown[:, own],
            "enemy_visible": vis, "enemy_pos": enemy_pos,
            "contact_pos": st.contact_pos[:, own], "contact_valid": st.contact_valid[:, own],
            "contact_time": st.contact_time[:, own],
            "boxes": self.arena.boxes, "box_mask": self.arena.mask, "t": self.t, "team": team,
        }

    def get_frame(self, e: int) -> Dict[str, Any]:
        """Snapshot of env ``e`` for rendering / replay."""
        st = self.state
        return {
            "t": int(self.t[e]), "pos": st.pos[e].copy(), "theta": st.theta[e].copy(), "hp": st.hp[e].copy(),
            "alive": st.alive[e].copy(), "team": st.team[e].copy(), "role": st.role[e].copy(),
            "ray_dist": self.ray_dist[e].copy(), "ray_kind": self.ray_kind[e].copy(),
            "shot_fired": self.shot_fired[e].copy(), "shot_aim": self.shot_aim[e].copy(),
            "shot_dist": self.shot_dist[e].copy(), "shot_kind": self.shot_kind[e].copy(),
            "boxes": self.arena.boxes[e][self.arena.mask[e]].copy(),
            "plan": self.plan_token[e].copy(),
        }


# ---------------------------------------------------------------------------
# PettingZoo wrapper
# ---------------------------------------------------------------------------
try:
    from pettingzoo import ParallelEnv
    from gymnasium import spaces
except ImportError:                                   # pragma: no cover
    ParallelEnv = object
    spaces = None


class SquadParallelEnv(ParallelEnv):
    """PettingZoo ``ParallelEnv`` over a single ``SquadVecEnv`` instance.

    Agents are named ``team{k}_{i}``.  With ``action_mode='hybrid'`` the action is a
    Box(4) whose last entry is thresholded at 0.5; with ``'discrete'`` it is a
    MultiDiscrete over the table in ``DISCRETE_TABLE``.
    """

    metadata = {"render_modes": ["rgb_array"], "name": "squad_combat_v0", "is_parallelizable": True}

    def __init__(self, cfg: Optional[EnvConfig] = None, action_mode: str = "hybrid", seed: int = 0,
                 render_mode: Optional[str] = None):
        self.cfg = cfg or EnvConfig()
        self.env = SquadVecEnv(self.cfg, num_envs=1, seed=seed)
        self.action_mode = action_mode
        self.render_mode = render_mode
        self.possible_agents = [f"team{i // self.cfg.team_size}_{i % self.cfg.team_size}" for i in range(self.env.N)]
        self.agents = list(self.possible_agents)
        self._obs_space = spaces.Box(-np.inf, np.inf, (self.env.obs_dim,), np.float32)
        if action_mode == "discrete":
            self._act_space = spaces.MultiDiscrete(DISCRETE_NVEC)
        else:
            self._act_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._renderer = None

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def state(self) -> np.ndarray:
        return self.env._build_global_state()[0]

    def reset(self, seed=None, options=None):
        obs, gs = self.env.reset(seed=seed)
        self.agents = list(self.possible_agents)
        observations = {a: obs[0, i] for i, a in enumerate(self.agents)}
        infos = {a: {"global_state": gs[0, i]} for i, a in enumerate(self.agents)}
        return observations, infos

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

"""Environment configuration and batched agent state (spec v2)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import numpy as np


@dataclass
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    damage_dealt: float = 0.01        # per hp point dealt to an enemy
    damage_taken: float = -0.005      # per hp point taken
    friendly_damage: float = -1.0     # per hp point dealt to an ally (never annealed)
    kill: float = 0.5                 # per enemy kill, to every teammate who damaged it
    kill_split: bool = False          # split the kill bonus equally among contributors
    death: float = -0.5
    step: float = -0.0005


@dataclass
class EnvConfig:
    # world
    arena_size: float = 48.0
    dt: float = 0.05
    max_steps: int = 1200
    team_size: int = 3                                   # slots per team (max squad size)
    team_sizes: List[List[int]] = field(default_factory=lambda: [[3, 3], [2, 2], [3, 2], [2, 3]])
    team_size_probs: List[float] = field(default_factory=lambda: [0.4, 0.2, 0.2, 0.2])
    # obstacles: parameterised by coverage fraction of the arena area
    coverage_min: float = 0.12
    coverage_max: float = 0.18
    crate_fraction: float = 0.4                           # share of obstacles that are low crates
    obstacle_size_min: float = 2.0
    obstacle_size_max: float = 8.0
    obstacle_min_gap: float = 4.0
    spawn_clearance: float = 3.0
    max_obstacles: int = 32
    # spawns
    spawn_distance: float = 30.0
    spawn_lateral_jitter: float = 4.0
    spawn_zone_radius: float = 4.0
    # body & motion
    collision_radius: float = 1.0
    speed_forward: float = 4.0
    speed_strafe: float = 2.0
    speed_backward: float = 1.5
    max_turn_rate_deg: float = 120.0
    turn_accel_deg: float = 600.0
    vel_tau: float = 0.1
    # sensing
    vision_fov_deg: float = 60.0
    vision_range: float = 24.0
    num_rays: int = 32
    map_rays: int = 64
    # weapon
    hp: float = 100.0
    damage: float = 34.0
    cooldown: float = 0.4
    magazine: int = 12
    reload_time: float = 2.0
    spread_rest_deg: float = 0.5
    spread_max_deg: float = 2.0
    friendly_fire: bool = True
    timeout_hp_tiebreak: bool = False
    # team information
    comms: str = "full"                                   # full | contacts_only | none
    track_staleness_cap: float = 10.0
    track_confidence_tau: float = 3.0
    # observation extras
    include_prev_action: bool = True
    num_roles: int = 3
    roles_enabled: bool = True
    plan_tokens: int = 0
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def max_speed(self) -> float:
        return self.speed_forward

    @property
    def max_turn_rate(self) -> float:
        return np.deg2rad(self.max_turn_rate_deg)

    @property
    def turn_accel(self) -> float:
        return np.deg2rad(self.turn_accel_deg)

    @property
    def fov(self) -> float:
        return np.deg2rad(self.vision_fov_deg)

    @property
    def num_agents(self) -> int:
        return 2 * self.team_size

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EnvConfig":
        d = dict(d or {})
        rw = d.pop("reward", {}) or {}
        rw.pop("shaping_los", None)           # removed in spec v2
        d.pop("shaping_los", None)
        return EnvConfig(**d, reward=RewardConfig(**rw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentState:
    """Structure-of-arrays state for all agents in all envs: shapes [E, N, ...]."""

    def __init__(self, num_envs: int, num_agents: int, cfg: EnvConfig):
        E, N, T = num_envs, num_agents, cfg.team_size
        self.cfg = cfg
        self.pos = np.zeros((E, N, 2), np.float32)
        self.theta = np.zeros((E, N), np.float32)
        self.vel = np.zeros((E, N, 2), np.float32)
        self.omega = np.zeros((E, N), np.float32)
        self.hp = np.full((E, N), cfg.hp, np.float32)
        self.ammo = np.full((E, N), cfg.magazine, np.int32)
        self.cooldown = np.zeros((E, N), np.float32)
        self.reload = np.zeros((E, N), np.float32)
        self.alive = np.ones((E, N), bool)
        self.active = np.ones((E, N), bool)               # slot participates in this episode
        self.team = np.repeat(np.arange(2), T)[None, :].repeat(E, 0)
        self.role = np.zeros((E, N), np.int64)
        self.prev_action = np.zeros((E, N, 4), np.float32)
        # per-agent enemy tracks, indexed by enemy slot (persistent id)
        self.track_pos = np.zeros((E, N, T, 2), np.float32)
        self.track_vel = np.zeros((E, N, T, 2), np.float32)
        self.track_theta = np.zeros((E, N, T), np.float32)
        self.track_hp = np.zeros((E, N, T), np.float32)
        self.track_time = np.zeros((E, N, T), np.float32)
        self.track_valid = np.zeros((E, N, T), bool)

    def reset_envs(self, idx: np.ndarray, pos: np.ndarray, theta: np.ndarray, roles: np.ndarray, active: np.ndarray):
        cfg = self.cfg
        self.pos[idx] = pos
        self.theta[idx] = theta
        self.vel[idx] = 0.0
        self.omega[idx] = 0.0
        self.active[idx] = active
        self.alive[idx] = active
        self.hp[idx] = np.where(active, cfg.hp, 0.0)
        self.ammo[idx] = cfg.magazine
        self.cooldown[idx] = 0.0
        self.reload[idx] = 0.0
        self.role[idx] = roles
        self.prev_action[idx] = 0.0
        for a in (self.track_pos, self.track_vel, self.track_theta, self.track_hp, self.track_time):
            a[idx] = 0.0
        self.track_valid[idx] = False

    @property
    def heading_vec(self) -> np.ndarray:
        return np.stack([np.cos(self.theta), np.sin(self.theta)], -1)

    def team_mask(self) -> np.ndarray:
        return self.team[:, :, None] == self.team[:, None, :]

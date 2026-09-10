"""Environment configuration and batched agent state."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

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
    shaping_los: float = 0.001        # per step with an enemy in cone + clear LOS (annealed)


@dataclass
class EnvConfig:
    arena_size: float = 64.0
    dt: float = 0.05
    max_steps: int = 1200
    team_size: int = 3
    num_obstacles_min: int = 8
    num_obstacles_max: int = 16
    obstacle_size_min: float = 2.0
    obstacle_size_max: float = 10.0
    spawn_distance: float = 40.0
    spawn_lateral_jitter: float = 6.0
    spawn_zone_radius: float = 5.0
    collision_radius: float = 0.4
    max_speed: float = 4.0
    max_turn_rate_deg: float = 180.0
    vel_tau: float = 0.1              # first-order velocity lag (s)
    vision_fov_deg: float = 60.0
    vision_range: float = 30.0
    num_rays: int = 32
    hp: float = 100.0
    damage: float = 34.0
    cooldown: float = 0.4
    magazine: int = 12
    reload_time: float = 2.0
    spread_rest_deg: float = 0.5
    spread_max_deg: float = 2.0
    friendly_fire: bool = True
    timeout_hp_tiebreak: bool = False
    include_prev_action: bool = True
    num_roles: int = 3
    roles_enabled: bool = True
    plan_tokens: int = 0              # >0 enables the commander token slot in observations
    contact_staleness_cap: float = 10.0
    reward: RewardConfig = field(default_factory=RewardConfig)

    @property
    def max_turn_rate(self) -> float:
        return np.deg2rad(self.max_turn_rate_deg)

    @property
    def fov(self) -> float:
        return np.deg2rad(self.vision_fov_deg)

    @property
    def num_agents(self) -> int:
        return 2 * self.team_size

    @property
    def max_obstacles(self) -> int:
        return self.num_obstacles_max + (self.num_obstacles_max % 2)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EnvConfig":
        d = dict(d or {})
        rw = d.pop("reward", {}) or {}
        return EnvConfig(**d, reward=RewardConfig(**rw))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentState:
    """Structure-of-arrays state for all agents in all envs: shapes [E, N, ...]."""

    def __init__(self, num_envs: int, num_agents: int, cfg: EnvConfig):
        E, N = num_envs, num_agents
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
        self.team = np.repeat(np.arange(2), cfg.team_size)[None, :].repeat(E, 0)   # [E,N]
        self.role = np.zeros((E, N), np.int64)
        self.prev_action = np.zeros((E, N, 4), np.float32)
        # last known enemy contact per agent: world position, time of sighting, valid flag
        self.contact_pos = np.zeros((E, N, 2), np.float32)
        self.contact_time = np.zeros((E, N), np.float32)
        self.contact_valid = np.zeros((E, N), bool)

    def reset_envs(self, idx: np.ndarray, pos: np.ndarray, theta: np.ndarray, roles: np.ndarray):
        cfg = self.cfg
        self.pos[idx] = pos
        self.theta[idx] = theta
        self.vel[idx] = 0.0
        self.omega[idx] = 0.0
        self.hp[idx] = cfg.hp
        self.ammo[idx] = cfg.magazine
        self.cooldown[idx] = 0.0
        self.reload[idx] = 0.0
        self.alive[idx] = True
        self.role[idx] = roles
        self.prev_action[idx] = 0.0
        self.contact_pos[idx] = 0.0
        self.contact_time[idx] = 0.0
        self.contact_valid[idx] = False

    @property
    def heading_vec(self) -> np.ndarray:
        return np.stack([np.cos(self.theta), np.sin(self.theta)], -1)

    def team_mask(self) -> np.ndarray:
        """[E, N, N] True where agents i and j are on the same team."""
        return self.team[:, :, None] == self.team[:, None, :]

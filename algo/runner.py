"""Stateful policy driver used by evaluation and rendering: keeps hidden states,
frame stacks and commander tokens for the team(s) it controls."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from algo.networks import ActorCritic
from env.squad_env import SquadVecEnv


class PolicyRunner:
    def __init__(self, policy: ActorCritic, env: SquadVecEnv, teams=(0, 1), deterministic: bool = False,
                 plan_interval: int = 20):
        self.policy, self.env, self.det = policy, env, deterministic
        self.teams = list(teams)
        T, N = env.T, env.N
        self.agents = np.concatenate([np.arange(k * T, (k + 1) * T) for k in self.teams])
        self.E, self.n = env.E, len(self.agents)
        self.H, self.stack = policy.hidden, policy.stack
        self.h = torch.zeros(self.E, self.n, self.H)
        self.slot = torch.as_tensor(np.tile(self.agents % T, self.E))
        self.first = np.ones(self.E, bool)
        self.fs = np.zeros((self.E, self.n, self.stack, env.obs_dim), np.float32) if self.stack > 1 else None
        self.plan_interval = plan_interval
        self.policy.eval()

    def reset(self, idx, obs):
        self.h[torch.as_tensor(idx)] = 0.0
        self.first[idx] = True
        if self.fs is not None:
            self.fs[idx] = obs[idx][:, self.agents][:, :, None]

    def _input(self, obs):
        o = obs[:, self.agents]
        if self.fs is None:
            return o
        self.fs = np.roll(self.fs, -1, axis=2)
        self.fs[:, :, -1] = o
        return self.fs.reshape(self.E, self.n, -1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, gs: np.ndarray) -> np.ndarray:
        E, n, T = self.E, self.n, self.env.T
        x = self._input(obs).copy()
        if self.policy.plan_tokens:
            decide = (self.env.t % self.plan_interval) == 0
            if decide.any():
                alive = torch.as_tensor(self.env.state.alive[:, self.agents], dtype=torch.float32).view(E, len(self.teams), T, 1)
                team_h = (self.h.view(E, len(self.teams), T, self.H) * alive).sum(2) / alive.sum(2).clamp(min=1)
                logits = self.policy.commander_logits(team_h.view(-1, self.H)).view(E, len(self.teams), -1)
                tok = logits.argmax(-1) if self.det else torch.distributions.Categorical(logits=logits).sample()
                tok = tok.numpy()
                for i, k in enumerate(self.teams):
                    self.env.plan_token[decide, k] = tok[decide, i]
                onehot = np.eye(self.env.plan_dim, dtype=np.float32)[self.env.plan_token[:, self.agents // T]]
                x.reshape(E, n, self.stack, -1)[:, :, -1, self.env.plan_slice] = onehot
        obs_t = torch.as_tensor(x).view(E * n, -1)
        gs_t = torch.as_tensor(gs[:, self.agents]).reshape(E * n, -1)
        first = torch.as_tensor(np.repeat(self.first, n), dtype=torch.float32)
        a, _, _, h_new, _ = self.policy.act(obs_t, gs_t, self.h.view(E * n, self.H), self.slot, first, self.det)
        self.h = h_new.view(E, n, self.H)
        self.first[:] = False
        return self.policy.to_env_action(a).reshape(E, n, 4)

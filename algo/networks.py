"""Actor-critic networks for MAPPO.

Actor: 1D conv over the [5, 32] ray tensor -> concat scalars -> MLP 2x256 -> GRU 256
       -> Gaussian(3) + Bernoulli(1) heads (or 4 categorical heads in discrete mode).
Critic: MLP 2x256 on the centralised global state concatenated with the local obs.
``ActorCritic`` optionally holds one such network per squad slot (ablation).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical, Normal

from env.squad_env import DISCRETE_NVEC, decode_discrete


def _init(m: nn.Module, gain: float = np.sqrt(2)):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(m.weight, gain)
        nn.init.zeros_(m.bias)
    return m


def mlp(inp: int, hidden: int, n: int = 2) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(n):
        layers += [_init(nn.Linear(d, hidden)), nn.ReLU()]
        d = hidden
    return nn.Sequential(*layers)


class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int, num_rays: int, stack: int, hidden: int):
        super().__init__()
        self.obs_dim, self.R, self.stack = obs_dim, num_rays, stack
        self.vision_dim = num_rays * 5
        self.scalar_dim = obs_dim - self.vision_dim
        self.conv = nn.Sequential(
            _init(nn.Conv1d(5 * stack, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv1d(32, 32, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten(), _init(nn.Linear(32 * ((num_rays + 1) // 2), 128)), nn.ReLU())
        self.scalars = nn.Sequential(_init(nn.Linear(self.scalar_dim * stack, 128)), nn.ReLU())
        self.mlp = mlp(256, hidden, 2)
        self.out_dim = hidden

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        x = obs.view(B, self.stack, self.obs_dim)
        vis = x[:, :, : self.vision_dim].reshape(B, self.stack, self.R, 5).permute(0, 1, 3, 2).reshape(B, self.stack * 5, self.R)
        sc = x[:, :, self.vision_dim:].reshape(B, -1)
        return self.mlp(torch.cat([self.conv(vis), self.scalars(sc)], -1))


class Policy(nn.Module):
    """One actor + critic pair."""

    def __init__(self, obs_dim: int, gs_dim: int, num_rays: int, hidden: int = 256, use_gru: bool = True,
                 stack: int = 1, action_mode: str = "hybrid", plan_tokens: int = 0):
        super().__init__()
        self.obs_dim, self.hidden, self.use_gru, self.stack, self.action_mode = obs_dim, hidden, use_gru, stack, action_mode
        self.encoder = ObsEncoder(obs_dim, num_rays, stack, hidden)
        self.gru = nn.GRUCell(hidden, hidden) if use_gru else None
        if use_gru:
            for name, p in self.gru.named_parameters():
                (nn.init.orthogonal_ if "weight" in name else nn.init.zeros_)(p)
        if action_mode == "hybrid":
            self.mu = _init(nn.Linear(hidden, 3), 0.01)
            self.log_std = nn.Parameter(torch.full((3,), -0.5))
            self.fire = _init(nn.Linear(hidden, 1), 0.01)
        else:
            self.heads = nn.ModuleList([_init(nn.Linear(hidden, n), 0.01) for n in DISCRETE_NVEC])
        self.commander = _init(nn.Linear(hidden, plan_tokens), 0.01) if plan_tokens else None
        self.critic = nn.Sequential(mlp(gs_dim + obs_dim, hidden, 2), _init(nn.Linear(hidden, 1), 1.0))

    # --- core -------------------------------------------------------------
    def core_step(self, obs: torch.Tensor, h: torch.Tensor, first: Optional[torch.Tensor] = None):
        x = self.encoder(obs)
        if self.gru is None:
            return x, h
        if first is not None:
            h = h * (1.0 - first.float())[:, None]
        h = self.gru(x, h)
        return h, h

    def core_seq(self, obs: torch.Tensor, h0: torch.Tensor, first: torch.Tensor):
        """obs [L,B,D], h0 [B,H], first [L,B] -> feats [L,B,H], hidden states before each step [L,B,H]."""
        L, B = obs.shape[:2]
        x = self.encoder(obs.reshape(L * B, -1)).view(L, B, -1)
        if self.gru is None:
            return x, h0.unsqueeze(0).expand(L, B, -1)
        h = h0
        feats, hs = [], []
        for t in range(L):
            h = h * (1.0 - first[t].float())[:, None]
            hs.append(h)
            h = self.gru(x[t], h)
            feats.append(h)
        return torch.stack(feats), torch.stack(hs)

    # --- heads ------------------------------------------------------------
    def dist(self, feat: torch.Tensor):
        if self.action_mode == "hybrid":
            return Normal(self.mu(feat), self.log_std.exp().expand_as(self.mu(feat))), Bernoulli(logits=self.fire(feat).squeeze(-1))
        return [Categorical(logits=h(feat)) for h in self.heads]

    def sample(self, feat: torch.Tensor, deterministic: bool = False):
        if self.action_mode == "hybrid":
            n, b = self.dist(feat)
            cont = n.mean if deterministic else n.sample()
            fire = (b.probs > 0.5).float() if deterministic else b.sample()
            logp = n.log_prob(cont).sum(-1) + b.log_prob(fire)
            return torch.cat([cont, fire[:, None]], -1), logp
        ds = self.dist(feat)
        acts = [(d.probs.argmax(-1) if deterministic else d.sample()) for d in ds]
        logp = sum(d.log_prob(a) for d, a in zip(ds, acts))
        return torch.stack(acts, -1).float(), logp

    def log_prob_entropy(self, feat: torch.Tensor, action: torch.Tensor):
        if self.action_mode == "hybrid":
            n, b = self.dist(feat)
            logp = n.log_prob(action[..., :3]).sum(-1) + b.log_prob(action[..., 3])
            ent = n.entropy().sum(-1) + b.entropy()
            return logp, ent
        ds = self.dist(feat)
        a = action.long()
        logp = sum(d.log_prob(a[..., k]) for k, d in enumerate(ds))
        ent = sum(d.entropy() for d in ds)
        return logp, ent

    def value(self, gs: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        last = obs.view(obs.shape[0], self.stack, self.obs_dim)[:, -1]     # latest frame only
        return self.critic(torch.cat([gs, last], -1)).squeeze(-1)


class ActorCritic(nn.Module):
    """Shared policy (``num_policies=1``) or one policy per squad slot."""

    def __init__(self, obs_dim: int, gs_dim: int, num_rays: int, team_size: int, hidden: int = 256,
                 use_gru: bool = True, stack: int = 1, action_mode: str = "hybrid", num_policies: int = 1,
                 plan_tokens: int = 0):
        super().__init__()
        self.hidden, self.team_size, self.stack, self.action_mode = hidden, team_size, stack, action_mode
        self.obs_dim, self.gs_dim, self.num_rays, self.use_gru, self.plan_tokens = obs_dim, gs_dim, num_rays, use_gru, plan_tokens
        self.num_policies = num_policies
        self.policies = nn.ModuleList([Policy(obs_dim, gs_dim, num_rays, hidden, use_gru, stack, action_mode, plan_tokens)
                                       for _ in range(num_policies)])

    def _groups(self, slot: torch.Tensor):
        if self.num_policies == 1:
            return [(self.policies[0], None)]
        return [(self.policies[k], slot == k) for k in range(self.num_policies)]

    def to_env_action(self, action: torch.Tensor) -> np.ndarray:
        a = action.detach().cpu().numpy()
        if self.action_mode == "hybrid":
            a = a.copy()
            a[..., :3] = np.clip(a[..., :3], -1.0, 1.0)
            return a.astype(np.float32)
        return decode_discrete(a)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, gs: torch.Tensor, h: torch.Tensor, slot: torch.Tensor, first: torch.Tensor,
            deterministic: bool = False):
        """Single step for a flat batch.  Returns action, logp, value, new h, features."""
        B = obs.shape[0]
        action = torch.zeros(B, 4)
        logp = torch.zeros(B)
        value = torch.zeros(B)
        h_new = torch.zeros_like(h)
        feats = torch.zeros(B, self.hidden)
        for pol, m in self._groups(slot):
            idx = slice(None) if m is None else m
            if m is not None and not m.any():
                continue
            f, hn = pol.core_step(obs[idx], h[idx], first[idx])
            a, lp = pol.sample(f, deterministic)
            action[idx], logp[idx], h_new[idx], feats[idx] = a, lp, hn, f
            value[idx] = pol.value(gs[idx], obs[idx])
        return action, logp, value, h_new, feats

    def evaluate(self, obs: torch.Tensor, gs: torch.Tensor, h0: torch.Tensor, first: torch.Tensor,
                 action: torch.Tensor, slot: torch.Tensor):
        """Sequences: obs [L,B,D], gs [L,B,G], h0 [B,H], first [L,B], action [L,B,4], slot [B].
        Returns logp [L,B], entropy [L,B], value [L,B], hidden-before-step [L,B,H]."""
        L, B = obs.shape[:2]
        logp = torch.zeros(L, B)
        ent = torch.zeros(L, B)
        val = torch.zeros(L, B)
        hs = torch.zeros(L, B, self.hidden)
        for pol, m in self._groups(slot):
            if m is not None and not m.any():
                continue
            idx = slice(None) if m is None else m
            o = obs[:, idx]
            f, h_before = pol.core_seq(o, h0[idx], first[:, idx])
            lp, e = pol.log_prob_entropy(f.reshape(-1, self.hidden), action[:, idx].reshape(-1, 4))
            logp[:, idx], ent[:, idx] = lp.view(L, -1), e.view(L, -1)
            val[:, idx] = pol.value(gs[:, idx].reshape(L * o.shape[1], -1), o.reshape(L * o.shape[1], -1)).view(L, -1)
            hs[:, idx] = h_before
        return logp, ent, val, hs

    def commander_logits(self, team_h: torch.Tensor) -> torch.Tensor:
        """team_h [B, H] (mean hidden over alive teammates) -> logits [B, plan_tokens]."""
        return self.policies[0].commander(team_h)

    def initial_hidden(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden)

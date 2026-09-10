"""Actor-critic networks for MAPPO (spec v2).

Actor: entity-token encoder (per-type MLPs -> 2-layer transformer) -> GRU(256) -> heads
       (Gaussian(3) + Bernoulli(1), or 4 categorical heads), plus an auxiliary head that
       predicts every enemy's true relative position from the recurrent state.
Critic: MLP 2x256 on the centralised global state concatenated with the local obs.
``ActorCritic`` optionally holds one such network per squad slot (ablation).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
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


class EntityEncoder(nn.Module):
    """Tokens: self, ally x (T-1), enemy track x T, vision, map -> transformer -> pooled feature."""

    def __init__(self, layout: Dict[str, dict], obs_dim: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 out_dim: int = 256):
        super().__init__()
        self.layout, self.obs_dim, self.d = layout, obs_dim, d_model
        self_parts = [k for k in ("proprio", "role", "prev_action", "plan") if k in layout]
        idx = np.concatenate([np.arange(layout[k]["start"], layout[k]["start"] + layout[k]["size"]) for k in self_parts])
        self.register_buffer("self_idx", torch.as_tensor(idx, dtype=torch.long))
        self.R, C = layout["vision"]["shape"]
        self.n_ally, self.ally_dim = layout["allies"]["shape"]
        self.n_enemy, self.enemy_dim = layout["enemies"]["shape"]
        self.K = layout["map"]["shape"][0]
        self.self_mlp = mlp(len(idx), d_model)
        self.ally_mlp = mlp(self.ally_dim, d_model)
        self.enemy_mlp = mlp(self.enemy_dim, d_model)
        self.vision = nn.Sequential(_init(nn.Conv1d(C, 32, 3, padding=1)), nn.ReLU(),
                                    _init(nn.Conv1d(32, 32, 3, stride=2, padding=1)), nn.ReLU(), nn.Flatten(),
                                    _init(nn.Linear(32 * ((self.R + 1) // 2), d_model)), nn.ReLU())
        self.map = nn.Sequential(_init(nn.Conv1d(1, 16, 5, padding=2, padding_mode="circular")), nn.ReLU(),
                                 _init(nn.Conv1d(16, 16, 5, stride=4, padding=2, padding_mode="circular")), nn.ReLU(),
                                 nn.Flatten(), _init(nn.Linear(16 * ((self.K + 3) // 4), d_model)), nn.ReLU())
        self.type_emb = nn.Parameter(torch.zeros(5, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(_init(nn.Linear(2 * d_model, out_dim)), nn.ReLU())
        self.out_dim = out_dim

    def _seg(self, x, name):
        l = self.layout[name]
        return x[:, l["start"]: l["start"] + l["size"]]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        te = self.type_emb
        tok_self = self.self_mlp(obs[:, self.self_idx]) + te[0]
        allies = self._seg(obs, "allies").reshape(B, self.n_ally, self.ally_dim)
        enemies = self._seg(obs, "enemies").reshape(B, self.n_enemy, self.enemy_dim)
        tok_ally = self.ally_mlp(allies) + te[1]
        tok_enemy = self.enemy_mlp(enemies) + te[2]
        vis = self._seg(obs, "vision").reshape(B, self.R, -1).transpose(1, 2)
        tok_vis = self.vision(vis) + te[3]
        tok_map = self.map(self._seg(obs, "map").unsqueeze(1)) + te[4]
        tokens = torch.cat([tok_self[:, None], tok_ally, tok_enemy, tok_vis[:, None], tok_map[:, None]], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=obs.device),
                         allies[..., 5] <= 0.0,                       # dead / hidden ally slots
                         enemies[..., 9] <= 0.0,                      # never-seen enemy tracks
                         torch.zeros(B, 2, dtype=torch.bool, device=obs.device)], 1)
        h = self.transformer(tokens, src_key_padding_mask=pad)
        keep = (~pad).float()[..., None]
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.out(torch.cat([h[:, 0], pooled], -1))


class ConvMLPEncoder(nn.Module):
    """Baseline encoder: 1D conv over the ray tensor, MLP over everything else."""

    def __init__(self, layout: Dict[str, dict], obs_dim: int, hidden: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.R, C = layout["vision"]["shape"]
        self.vdim = self.R * C
        self.conv = nn.Sequential(_init(nn.Conv1d(C, 32, 3, padding=1)), nn.ReLU(),
                                  _init(nn.Conv1d(32, 32, 3, stride=2, padding=1)), nn.ReLU(), nn.Flatten(),
                                  _init(nn.Linear(32 * ((self.R + 1) // 2), 128)), nn.ReLU())
        self.scalars = nn.Sequential(_init(nn.Linear(obs_dim - self.vdim, 128)), nn.ReLU())
        self.mlp = mlp(256, hidden, 2)
        self.out_dim = hidden

    def forward(self, obs):
        B = obs.shape[0]
        vis = obs[:, : self.vdim].reshape(B, self.R, -1).transpose(1, 2)
        return self.mlp(torch.cat([self.conv(vis), self.scalars(obs[:, self.vdim:])], -1))


class Policy(nn.Module):
    def __init__(self, layout, obs_dim, gs_dim, n_enemy, hidden=256, use_gru=True, stack=1, action_mode="hybrid",
                 plan_tokens=0, init_log_std=-1.0, encoder="entity", d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        self.obs_dim, self.hidden, self.use_gru, self.stack, self.action_mode = obs_dim, hidden, use_gru, stack, action_mode
        self.n_enemy = n_enemy
        enc = (EntityEncoder(layout, obs_dim, d_model, n_layers, n_heads, hidden) if encoder == "entity"
               else ConvMLPEncoder(layout, obs_dim, hidden))
        self.encoder = enc
        self.stack_proj = nn.Sequential(_init(nn.Linear(stack * enc.out_dim, hidden)), nn.ReLU()) if stack > 1 else None
        self.gru = nn.GRUCell(hidden, hidden) if use_gru else None
        if use_gru:
            for name, p in self.gru.named_parameters():
                (nn.init.orthogonal_ if "weight" in name else nn.init.zeros_)(p)
        if action_mode == "hybrid":
            self.mu = _init(nn.Linear(hidden, 3), 0.01)
            self.log_std = nn.Parameter(torch.full((3,), float(init_log_std)))
            self.fire = _init(nn.Linear(hidden, 1), 0.01)
        else:
            self.heads = nn.ModuleList([_init(nn.Linear(hidden, n), 0.01) for n in DISCRETE_NVEC])
        self.commander = _init(nn.Linear(hidden, plan_tokens), 0.01) if plan_tokens else None
        self.aux = _init(nn.Linear(hidden, 2 * n_enemy), 1.0)
        self.critic = nn.Sequential(mlp(gs_dim + obs_dim, hidden, 2), _init(nn.Linear(hidden, 1), 1.0))

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if self.stack == 1:
            return self.encoder(obs)
        B = obs.shape[0]
        frames = obs.view(B * self.stack, self.obs_dim)
        return self.stack_proj(self.encoder(frames).view(B, -1))

    def core_step(self, obs, h, first=None):
        x = self.encode(obs)
        if self.gru is None:
            return x, h
        if first is not None:
            h = h * (1.0 - first.float())[:, None]
        h = self.gru(x, h)
        return h, h

    def core_seq(self, obs, h0, first):
        L, B = obs.shape[:2]
        x = self.encode(obs.reshape(L * B, -1)).view(L, B, -1)
        if self.gru is None:
            return x, h0.unsqueeze(0).expand(L, B, -1)
        h, feats, hs = h0, [], []
        for t in range(L):
            h = h * (1.0 - first[t].float())[:, None]
            hs.append(h)
            h = self.gru(x[t], h)
            feats.append(h)
        return torch.stack(feats), torch.stack(hs)

    def dist(self, feat):
        if self.action_mode == "hybrid":
            mu = self.mu(feat)
            return Normal(mu, self.log_std.exp().expand_as(mu)), Bernoulli(logits=self.fire(feat).squeeze(-1))
        return [Categorical(logits=h(feat)) for h in self.heads]

    def sample(self, feat, deterministic=False):
        if self.action_mode == "hybrid":
            n, b = self.dist(feat)
            cont = n.mean if deterministic else n.sample()
            fire = (b.probs > 0.5).float() if deterministic else b.sample()
            return torch.cat([cont, fire[:, None]], -1), n.log_prob(cont).sum(-1) + b.log_prob(fire)
        ds = self.dist(feat)
        acts = [(d.probs.argmax(-1) if deterministic else d.sample()) for d in ds]
        return torch.stack(acts, -1).float(), sum(d.log_prob(a) for d, a in zip(ds, acts))

    def log_prob_entropy(self, feat, action):
        if self.action_mode == "hybrid":
            n, b = self.dist(feat)
            return n.log_prob(action[..., :3]).sum(-1) + b.log_prob(action[..., 3]), n.entropy().sum(-1) + b.entropy()
        ds = self.dist(feat)
        a = action.long()
        return sum(d.log_prob(a[..., k]) for k, d in enumerate(ds)), sum(d.entropy() for d in ds)

    def value(self, gs, obs):
        last = obs.view(obs.shape[0], self.stack, self.obs_dim)[:, -1]
        return self.critic(torch.cat([gs, last], -1)).squeeze(-1)


class ActorCritic(nn.Module):
    """Shared policy (``num_policies=1``) or one policy per squad slot."""

    def __init__(self, layout: Dict[str, dict], obs_dim: int, gs_dim: int, team_size: int, hidden: int = 256,
                 use_gru: bool = True, stack: int = 1, action_mode: str = "hybrid", num_policies: int = 1,
                 plan_tokens: int = 0, init_log_std: float = -1.0, encoder: str = "entity", d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.layout, self.hidden, self.team_size, self.stack, self.action_mode = layout, hidden, team_size, stack, action_mode
        self.obs_dim, self.gs_dim, self.use_gru, self.plan_tokens = obs_dim, gs_dim, use_gru, plan_tokens
        self.num_policies = num_policies
        self.policies = nn.ModuleList([Policy(layout, obs_dim, gs_dim, team_size, hidden, use_gru, stack, action_mode,
                                              plan_tokens, init_log_std, encoder, d_model, n_layers, n_heads)
                                       for _ in range(num_policies)])

    def _groups(self, slot):
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
    def act(self, obs, gs, h, slot, first, deterministic=False):
        B = obs.shape[0]
        action, logp, value = torch.zeros(B, 4), torch.zeros(B), torch.zeros(B)
        h_new, feats = torch.zeros_like(h), torch.zeros(B, self.hidden)
        for pol, m in self._groups(slot):
            if m is not None and not m.any():
                continue
            idx = slice(None) if m is None else m
            f, hn = pol.core_step(obs[idx], h[idx], first[idx])
            a, lp = pol.sample(f, deterministic)
            action[idx], logp[idx], h_new[idx], feats[idx] = a, lp, hn, f
            value[idx] = pol.value(gs[idx], obs[idx])
        return action, logp, value, h_new, feats

    def evaluate(self, obs_actor, obs_critic, gs, h0, first, action, slot):
        """Sequences [L,B,...].  Returns logp, entropy, value, hidden-before-step [L,B,H],
        aux prediction [L,B,T,2] (from the actor's recurrent features)."""
        L, B = obs_actor.shape[:2]
        logp, ent, val = torch.zeros(L, B), torch.zeros(L, B), torch.zeros(L, B)
        hs = torch.zeros(L, B, self.hidden)
        aux = torch.zeros(L, B, self.team_size, 2)
        for pol, m in self._groups(slot):
            if m is not None and not m.any():
                continue
            idx = slice(None) if m is None else m
            o = obs_actor[:, idx]
            n = o.shape[1]
            f, h_before = pol.core_seq(o, h0[idx], first[:, idx])
            lp, e = pol.log_prob_entropy(f.reshape(-1, self.hidden), action[:, idx].reshape(-1, 4))
            logp[:, idx], ent[:, idx] = lp.view(L, n), e.view(L, n)
            oc = obs_critic[:, idx]
            val[:, idx] = pol.value(gs[:, idx].reshape(L * n, -1), oc.reshape(L * n, -1)).view(L, n)
            hs[:, idx] = h_before
            aux[:, idx] = pol.aux(f.reshape(-1, self.hidden)).view(L, n, self.team_size, 2)
        return logp, ent, val, hs, aux

    def commander_logits(self, team_h):
        return self.policies[0].commander(team_h)

    def initial_hidden(self, n: int) -> torch.Tensor:
        return torch.zeros(n, self.hidden)

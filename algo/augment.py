"""Mirror augmentation about the agent's forward axis (left <-> right).

Egocentric observations are reflected (rays reversed, lateral coordinates and sines
negated, map scan reversed) together with the lateral action components, so a policy
trained on (obs, action) is also trained on (mirror(obs), mirror(action))."""
from __future__ import annotations

from typing import Dict

import torch

from env.squad_env import DISCRETE_NVEC


def mirror_obs(obs: torch.Tensor, layout: Dict[str, dict], obs_dim: int) -> torch.Tensor:
    """obs [..., stack*obs_dim] -> mirrored copy."""
    lead = obs.shape[:-1]
    x = obs.reshape(-1, obs.shape[-1] // obs_dim, obs_dim).clone()

    def seg(name):
        l = layout[name]
        return l["start"], l["start"] + l["size"], l["shape"]

    s, e, (R, C) = seg("vision")
    x[:, :, s:e] = x[:, :, s:e].reshape(x.shape[0], x.shape[1], R, C).flip(2).reshape(x.shape[0], x.shape[1], -1)
    s, e, _ = seg("proprio")
    x[:, :, s + 5] = -x[:, :, s + 5]           # lateral velocity
    x[:, :, s + 6] = -x[:, :, s + 6]           # angular velocity
    s, e, (n, d) = seg("allies")
    a = x[:, :, s:e].reshape(x.shape[0], x.shape[1], n, d)
    a[..., 1] = -a[..., 1]
    a[..., 3] = -a[..., 3]
    s, e, (n, d) = seg("enemies")
    t = x[:, :, s:e].reshape(x.shape[0], x.shape[1], n, d)
    t[..., 1] = -t[..., 1]
    t[..., 3] = -t[..., 3]
    t[..., 5] = -t[..., 5]
    s, e, (K,) = seg("map")
    idx = (K - torch.arange(K, device=obs.device)) % K
    x[:, :, s:e] = x[:, :, s:e][:, :, idx]
    if "prev_action" in layout:
        s, e, _ = seg("prev_action")
        x[:, :, s + 1] = -x[:, :, s + 1]
        x[:, :, s + 2] = -x[:, :, s + 2]
    return x.reshape(*lead, -1)


def mirror_action(action: torch.Tensor, action_mode: str) -> torch.Tensor:
    a = action.clone()
    if action_mode == "hybrid":
        a[..., 1] = -a[..., 1]
        a[..., 2] = -a[..., 2]
    else:
        a[..., 1] = (DISCRETE_NVEC[1] - 1) - a[..., 1]
        a[..., 2] = (DISCRETE_NVEC[2] - 1) - a[..., 2]
    return a


def mirror_aux(aux: torch.Tensor) -> torch.Tensor:
    """aux [..., T, 3] (rel x, rel y, mask): negate rel y."""
    a = aux.clone()
    a[..., 1] = -a[..., 1]
    return a

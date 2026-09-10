"""Role discriminator q(z | tau) over trajectory summary statistics (DIAYN-style bonus)."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoleDiscriminator(nn.Module):
    STAT_NAMES = ["dist_to_centroid", "dist_to_nearest_enemy", "shots_fired", "distance_travelled",
                  "flank_fraction", "spotting_fraction"]

    def __init__(self, num_roles: int, stat_dim: int = 6, hidden: int = 64, lr: float = 1e-3,
                 buffer_size: int = 4096, beta: float = 0.05):
        super().__init__()
        self.num_roles, self.beta = num_roles, beta
        self.net = nn.Sequential(nn.Linear(stat_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, num_roles))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.data = deque(maxlen=buffer_size)
        self.last_acc = float("nan")

    def add(self, stats: np.ndarray, roles: np.ndarray):
        for s, r in zip(stats.reshape(-1, stats.shape[-1]), roles.reshape(-1)):
            self.data.append((s.astype(np.float32), int(r)))

    @torch.no_grad()
    def bonus(self, stats: np.ndarray, roles: np.ndarray) -> np.ndarray:
        """beta * (log q(z|tau) - log(1/K)), zero-centred so an uninformative q gives 0."""
        x = torch.as_tensor(stats, dtype=torch.float32).reshape(-1, stats.shape[-1])
        z = torch.as_tensor(roles).reshape(-1)
        logq = F.log_softmax(self.net(x), -1).gather(1, z[:, None]).squeeze(1)
        b = self.beta * (logq + np.log(self.num_roles))
        return b.numpy().reshape(roles.shape)

    def train_steps(self, steps: int = 20, batch_size: int = 256, rng: np.random.Generator | None = None) -> dict:
        if len(self.data) < batch_size:
            return {}
        rng = rng or np.random.default_rng()
        losses, accs = [], []
        for _ in range(steps):
            idx = rng.integers(0, len(self.data), size=batch_size)
            xs = torch.as_tensor(np.stack([self.data[i][0] for i in idx]))
            ys = torch.as_tensor([self.data[i][1] for i in idx])
            logits = self.net(xs)
            loss = F.cross_entropy(logits, ys)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == ys).float().mean().item())
        self.last_acc = float(np.mean(accs))
        return {"disc/loss": float(np.mean(losses)), "disc/accuracy": self.last_acc}

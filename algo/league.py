"""Opponent league: policy snapshots + scripted bots, Elo tracking, prioritised sampling."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

LATEST = "latest"


class League:
    """Members are either policy snapshots (state_dicts on disk) or scripted bots.

    Sampling: with probability ``p_latest`` the opponent is the current policy
    (pure self-play); otherwise a pool member is drawn with probability
    proportional to ``(win rate vs the current policy) + floor`` so opponents that
    beat us are revisited more often (PFSP-style), which counters strategy cycling
    and forgetting.
    """

    def __init__(self, save_dir: str, p_latest: float = 0.7, win_floor: float = 0.1,
                 elo_k: float = 16.0, winrate_ema: float = 0.05, max_snapshots: int = 20,
                 scripted: Optional[List[str]] = None, scripted_only: bool = False):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.p_latest, self.win_floor, self.elo_k, self.ema = p_latest, win_floor, elo_k, winrate_ema
        self.max_snapshots = max_snapshots
        self.scripted_only = scripted_only
        self.members: Dict[str, dict] = {}
        self.elo: Dict[str, float] = {LATEST: 1000.0}
        self.games: Dict[str, int] = {LATEST: 0}
        for name in scripted or []:
            self.members[f"bot:{name}"] = {"type": "bot", "bot": name, "winrate": 0.5}
            self.elo[f"bot:{name}"] = 1000.0
            self.games[f"bot:{name}"] = 0
        self.snapshot_names: List[str] = []

    # ---------------------------------------------------------------- pool
    def add_snapshot(self, policy: torch.nn.Module, update: int):
        name = f"snap:{update:06d}"
        path = os.path.join(self.save_dir, f"{name.replace(':', '_')}.pt")
        torch.save(policy.state_dict(), path)
        self.members[name] = {"type": "snapshot", "path": path, "winrate": 0.5}
        self.elo[name] = self.elo[LATEST]
        self.games[name] = 0
        self.snapshot_names.append(name)
        # evict the oldest snapshots beyond the cap (keep the very first for reference)
        while len(self.snapshot_names) > self.max_snapshots:
            old = self.snapshot_names.pop(1 if len(self.snapshot_names) > 1 else 0)
            self.members.pop(old, None)
        return name

    def pool(self) -> List[str]:
        return list(self.members.keys())

    def sample_opponent(self, rng: np.random.Generator) -> str:
        pool = self.pool()
        if not self.scripted_only and (not pool or rng.random() < self.p_latest):
            return LATEST
        if not pool:
            return LATEST
        w = np.array([self.members[n]["winrate"] + self.win_floor for n in pool])
        return pool[rng.choice(len(pool), p=w / w.sum())]

    # ------------------------------------------------------------- results
    def record(self, opponent: str, score: float):
        """``score``: 1 learner win, 0 loss, 0.5 draw (from the latest policy's view)."""
        if opponent == LATEST:
            self.games[LATEST] += 1
            return
        ra, rb = self.elo[LATEST], self.elo[opponent]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        self.elo[LATEST] = ra + self.elo_k * (score - ea)
        self.elo[opponent] = rb - self.elo_k * (score - ea)
        self.games[LATEST] += 1
        self.games[opponent] += 1
        m = self.members[opponent]
        m["winrate"] = (1 - self.ema) * m["winrate"] + self.ema * (1.0 - score)

    def summary(self) -> Dict[str, float]:
        out = {"elo/latest": self.elo[LATEST]}
        for n in self.pool():
            out[f"elo/{n}"] = self.elo[n]
            out[f"winrate_vs_latest/{n}"] = self.members[n]["winrate"]
        return out

    def save(self):
        with open(os.path.join(self.save_dir, "league.json"), "w") as f:
            json.dump({"members": self.members, "elo": self.elo, "games": self.games,
                       "snapshot_names": self.snapshot_names}, f, indent=1)

    def load(self):
        p = os.path.join(self.save_dir, "league.json")
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            self.members, self.elo, self.games = d["members"], d["elo"], d["games"]
            self.snapshot_names = d["snapshot_names"]


def elo_update(ra: float, rb: float, score_a: float, k: float = 16.0):
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb - k * (score_a - ea)

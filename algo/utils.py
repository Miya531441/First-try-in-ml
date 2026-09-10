"""Small helpers: config loading, seeding, running statistics."""
from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict

import numpy as np
import torch
import yaml


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a yaml config; an ``inherit`` key names a base config (relative to the file)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_name = cfg.pop("inherit", None)
    if base_name:
        base = load_config(os.path.join(os.path.dirname(path), base_name))
        cfg = deep_update(base, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def parse_overrides(items) -> Dict[str, Any]:
    """['a.b=1', 'c=true'] -> nested dict with yaml-parsed values."""
    out: Dict[str, Any] = {}
    for it in items or []:
        key, val = it.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RunningMeanStd:
    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape)
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m_a = self.var * self.count
        m_b = bv * bc
        self.var = (m_a + m_b + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def state(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]


class ReturnNormalizer:
    """Scales rewards by the running std of the discounted return (per env-agent stream)."""

    def __init__(self, shape, gamma: float):
        self.ret = np.zeros(shape, np.float64)
        self.rms = RunningMeanStd(())
        self.gamma = gamma

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret)
        self.ret[done] = 0.0
        return reward / np.sqrt(self.rms.var + 1e-8)

    def state(self):
        return self.rms.state()

    def load(self, s):
        self.rms.load(s)

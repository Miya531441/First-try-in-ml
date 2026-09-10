"""Rollout storage for recurrent MAPPO with chunked BPTT."""
from __future__ import annotations

from typing import Dict, Iterator

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, T: int, E: int, N: int, obs_dim: int, gs_dim: int, hidden: int, chunk_len: int,
                 gamma: float, lam: float, plan_tokens: int = 0):
        assert T % chunk_len == 0, "rollout_len must be a multiple of chunk_len"
        self.T, self.E, self.N, self.L = T, E, N, chunk_len
        self.C = T // chunk_len
        self.gamma, self.lam = gamma, lam
        self.obs = torch.zeros(T, E, N, obs_dim)
        self.gs = torch.zeros(T, E, N, gs_dim)
        self.actions = torch.zeros(T, E, N, 4)
        self.logp = torch.zeros(T, E, N)
        self.values = torch.zeros(T, E, N)
        self.rewards = torch.zeros(T, E, N)
        self.first = torch.zeros(T, E)               # step t begins a new episode
        self.done = torch.zeros(T, E)                # episode ends after step t
        self.alive = torch.zeros(T, E, N)            # agent alive when acting
        self.learner = torch.zeros(T, E, N)          # counts toward the loss
        self.h0 = torch.zeros(self.C, E, N, hidden)  # hidden state at chunk starts
        self.plan_tokens = plan_tokens
        if plan_tokens:
            self.plan_action = torch.zeros(T, E, 2, dtype=torch.long)
            self.plan_logp = torch.zeros(T, E, 2)
            self.plan_mask = torch.zeros(T, E, 2)
        self.advantages = torch.zeros(T, E, N)
        self.returns = torch.zeros(T, E, N)
        self.step = 0

    def add(self, obs, gs, action, logp, value, first, alive, learner, h):
        t = self.step
        self.obs[t], self.gs[t], self.actions[t] = obs, gs, action
        self.logp[t], self.values[t] = logp, value
        self.first[t], self.alive[t], self.learner[t] = first, alive, learner
        if t % self.L == 0:
            self.h0[t // self.L] = h
        self.step += 1

    def add_outcome(self, reward, done):
        t = self.step - 1
        self.rewards[t] = torch.as_tensor(reward)
        self.done[t] = torch.as_tensor(done, dtype=torch.float32)

    def compute_gae(self, last_value: torch.Tensor):
        adv = torch.zeros(self.E, self.N)
        next_value = last_value
        for t in reversed(range(self.T)):
            nonterminal = 1.0 - self.done[t][:, None]
            delta = self.rewards[t] + self.gamma * next_value * nonterminal - self.values[t]
            adv = delta + self.gamma * self.lam * nonterminal * adv
            self.advantages[t] = adv
            next_value = self.values[t]
        self.returns = self.advantages + self.values
        self.step = 0

    def iterate(self, num_minibatches: int, rng: np.random.Generator) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield minibatches of sequences [L, B, ...] sampled over (chunk, env) pairs.

        All N agents of a sampled env-chunk are included together (agent-major inner
        order, so B = num_pairs * N) which lets the commander head pool teammates."""
        C, E, N, L = self.C, self.E, self.N, self.L
        total = C * E
        perm = rng.permutation(total)
        size = max(1, total // num_minibatches)
        for k in range(num_minibatches):
            ids = perm[k * size:(k + 1) * size]
            c, e = np.unravel_index(ids, (C, E))
            c, e = torch.as_tensor(c), torch.as_tensor(e)
            P = len(ids)
            n = torch.arange(N).repeat(P)                       # agent-major: pair p -> [p*N, (p+1)*N)
            cc, ee = c.repeat_interleave(N), e.repeat_interleave(N)

            def seq(x):   # x: [T, E, N, ...] -> [L, P*N, ...]
                y = x.view(C, L, E, N, *x.shape[3:])[cc, :, ee, n]
                return y.transpose(0, 1).contiguous()

            def seq_env(x):  # x: [T, E, ...] -> [L, P, ...]
                y = x.view(C, L, E, *x.shape[2:])[c, :, e]
                return y.transpose(0, 1).contiguous()

            first_env = seq_env(self.first)                     # [L, P]
            batch = {
                "obs": seq(self.obs), "gs": seq(self.gs), "actions": seq(self.actions),
                "logp": seq(self.logp), "values": seq(self.values), "adv": seq(self.advantages),
                "returns": seq(self.returns), "alive": seq(self.alive), "learner": seq(self.learner),
                "first": first_env.repeat_interleave(N, dim=1), "h0": self.h0[cc, ee, n],
                "slot": n % (N // 2), "num_pairs": P,
            }
            if self.plan_tokens:
                batch["plan_action"] = seq_env(self.plan_action)   # [L, P, 2]
                batch["plan_logp"] = seq_env(self.plan_logp)
                batch["plan_mask"] = seq_env(self.plan_mask)
            yield batch

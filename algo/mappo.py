"""MAPPO (PPO with a centralised critic) with self-play league, recurrent policy,
role diversity bonus and an optional hierarchical commander."""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from algo.buffer import RolloutBuffer
from algo.discriminator import RoleDiscriminator
from algo.league import LATEST, League
from algo.networks import ActorCritic
from algo.utils import ReturnNormalizer, seed_everything
from env.agent import EnvConfig
from env.bots import BOTS, make_bot
from env.squad_env import SquadVecEnv


def build_env_config(cfg: dict) -> EnvConfig:
    ec = dict(cfg["env"])
    hier = cfg.get("hierarchical", {}) or {}
    ec["plan_tokens"] = int(hier.get("plan_tokens", 4)) if hier.get("enabled", False) else 0
    ec["roles_enabled"] = bool(cfg.get("roles", {}).get("enabled", True))
    return EnvConfig.from_dict(ec)


def build_policy(cfg: dict, env: SquadVecEnv) -> ActorCritic:
    mc = cfg["model"]
    stack = int(mc.get("frame_stack", 4)) if mc.get("memory", "gru") == "framestack" else 1
    return ActorCritic(env.obs_dim, env.gs_dim, env.R, env.T, hidden=int(mc.get("hidden", 256)),
                       use_gru=mc.get("memory", "gru") == "gru", stack=stack, action_mode=mc.get("action_mode", "hybrid"),
                       num_policies=env.T if mc.get("policy", "shared") == "per_slot" else 1,
                       plan_tokens=env.cfg.plan_tokens)


class FrameStack:
    """Keeps the last ``k`` observations per agent; the flattened stack is the policy input."""

    def __init__(self, k: int, E: int, N: int, D: int):
        self.k, self.buf = k, np.zeros((E, N, k, D), np.float32)

    def reset(self, idx, obs):
        self.buf[idx] = obs[idx][:, :, None]

    def push(self, obs):
        self.buf = np.roll(self.buf, -1, axis=2)
        self.buf[:, :, -1] = obs

    def get(self) -> np.ndarray:
        return self.buf.reshape(self.buf.shape[0], self.buf.shape[1], -1)


class MAPPOTrainer:
    def __init__(self, cfg: dict, log_dir: str):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        tc = cfg["train"]
        seed = int(cfg.get("seed", 0))
        seed_everything(seed)
        self.rng = np.random.default_rng(seed + 1)
        torch.set_num_threads(int(tc.get("torch_threads", os.cpu_count() or 4)))

        self.env_cfg = build_env_config(cfg)
        self.E = int(tc["num_envs"])
        self.env = SquadVecEnv(self.env_cfg, self.E, seed=seed + 2)
        self.N, self.T = self.env.N, self.env.T
        self.policy = build_policy(cfg, self.env)
        self.H = self.policy.hidden
        self.stack = self.policy.stack
        self.obs_in_dim = self.env.obs_dim * self.stack
        self.lr0 = float(tc["lr"])
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr0, eps=1e-5)
        self.rollout_len = int(tc["rollout_len"])
        self.buffer = RolloutBuffer(self.rollout_len, self.E, self.N, self.obs_in_dim, self.env.gs_dim, self.H,
                                    int(tc.get("chunk_len", 32)), float(tc["gamma"]), float(tc["gae_lambda"]),
                                    plan_tokens=self.env_cfg.plan_tokens)
        self.ret_norm = ReturnNormalizer((self.E, self.N), float(tc["gamma"])) if tc.get("normalize_returns", True) else None

        lc = cfg.get("league", {})
        self.league = League(os.path.join(log_dir, "league"), p_latest=float(lc.get("p_latest", 0.7)),
                             win_floor=float(lc.get("win_floor", 0.1)), elo_k=float(lc.get("elo_k", 16.0)),
                             max_snapshots=int(lc.get("max_snapshots", 20)), scripted=list(lc.get("scripted", [])),
                             scripted_only=bool(lc.get("scripted_only", False)))
        self.snapshot_every = int(lc.get("snapshot_every", 20))
        self.opp_nets: Dict[str, ActorCritic] = {}
        self.bots = {f"bot:{n}": make_bot(n, self.env_cfg, self.E, np.random.default_rng(seed + 3 + i))
                     for i, n in enumerate(BOTS) if f"bot:{n}" in self.league.members}
        for b in self.bots.values():
            b.reset(np.arange(self.E))

        dc = cfg.get("diversity", {})
        self.disc = RoleDiscriminator(self.env_cfg.num_roles, beta=float(dc.get("beta", 0.05))) \
            if dc.get("enabled", False) and self.env_cfg.roles_enabled else None
        hc = cfg.get("hierarchical", {}) or {}
        self.hier = bool(hc.get("enabled", False))
        self.plan_interval = int(hc.get("interval", 20))
        self.plan_coef = float(hc.get("loss_coef", 1.0))

        self.writer = SummaryWriter(log_dir)
        self.update_idx = 0
        self.total_updates = int(tc["total_updates"])
        self.global_step = 0
        self.recent: deque = deque(maxlen=200)
        self._reset_runtime()

    # ------------------------------------------------------------- runtime
    def _reset_runtime(self):
        self.obs, self.gs = self.env.reset()
        self.h = torch.zeros(self.E, self.N, self.H)
        self.first = np.ones(self.E, bool)
        self.slot = torch.as_tensor(np.tile(np.arange(self.N) % self.T, self.E))
        self.env_opp: List[str] = [self.league.sample_opponent(self.rng) for _ in range(self.E)]
        self.fs = FrameStack(self.stack, self.E, self.N, self.env.obs_dim) if self.stack > 1 else None
        if self.fs:
            self.fs.reset(np.arange(self.E), self.obs)
        self.plan = np.zeros((self.E, 2), np.int64)

    def _opp_net(self, name: str) -> ActorCritic:
        if name not in self.opp_nets:
            if len(self.opp_nets) >= 8:
                self.opp_nets.pop(next(iter(self.opp_nets)))
            net = build_policy(self.cfg, self.env)
            net.load_state_dict(torch.load(self.league.members[name]["path"], map_location="cpu"))
            net.eval()
            self.opp_nets[name] = net
        return self.opp_nets[name]

    def _policy_input(self) -> np.ndarray:
        return self.fs.get() if self.fs else self.obs

    def _learner_mask(self) -> np.ndarray:
        m = np.ones((self.E, self.N), np.float32)
        opp_latest = np.array([o == LATEST for o in self.env_opp])
        m[~opp_latest, self.T:] = 0.0
        return m

    # ------------------------------------------------------------ commander
    def _commander_step(self, obs_in: np.ndarray, learner: np.ndarray):
        """Emit plan tokens for envs at a decision step; returns (action, logp, mask) [E,2]."""
        E, N, T = self.E, self.N, self.T
        decide = (self.env.t % self.plan_interval) == 0
        action = self.plan.copy()
        logp = np.zeros((E, 2), np.float32)
        mask = np.zeros((E, 2), np.float32)
        if decide.any():
            alive = torch.as_tensor(self.env.state.alive, dtype=torch.float32)
            hh = self.h.view(E, 2, T, self.H)
            w = alive.view(E, 2, T, 1)
            team_h = (hh * w).sum(2) / w.sum(2).clamp(min=1.0)                     # [E,2,H]
            with torch.no_grad():
                logits = self.policy.commander_logits(team_h.view(E * 2, self.H)).view(E, 2, -1)
                # opponent snapshots run their own commander for team 1
                for name in set(self.env_opp):
                    if name.startswith("snap:"):
                        envs = np.array([i for i, o in enumerate(self.env_opp) if o == name])
                        logits[envs, 1] = self._opp_net(name).commander_logits(team_h[envs, 1])
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            d = torch.as_tensor(decide)
            action = np.where(decide[:, None], a.numpy(), self.plan)
            logp = np.where(decide[:, None], lp.numpy(), 0.0).astype(np.float32)
            team_learner = learner.reshape(E, 2, T).min(-1)                         # [E,2]
            mask = (decide[:, None] & (team_learner > 0)).astype(np.float32)
            self.plan = action
            self.env.set_plan_tokens(self.plan)
            # write the fresh token into the current observation (latest frame)
            tok = np.eye(self.env.plan_dim, dtype=np.float32)[self.plan[:, :, None].repeat(T, 2).reshape(E, N)]
            obs_in = obs_in.copy()
            view = obs_in.reshape(E, N, self.stack, self.env.obs_dim)
            view[:, :, -1, self.env.plan_slice] = tok
            obs_in = view.reshape(E, N, -1)
        return obs_in, action, logp, mask

    # -------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect_rollout(self) -> Dict[str, float]:
        E, N, T = self.E, self.N, self.T
        stats = defaultdict(list)
        self.policy.eval()
        for t in range(self.rollout_len):
            obs_in = self._policy_input()
            learner = self._learner_mask()
            if self.hier:
                obs_in, p_act, p_logp, p_mask = self._commander_step(obs_in, learner)
            obs_t = torch.as_tensor(obs_in).view(E * N, -1)
            gs_t = torch.as_tensor(self.gs).view(E * N, -1)
            first_t = torch.as_tensor(np.repeat(self.first, N), dtype=torch.float32)
            alive = self.env.state.alive.copy()
            h_flat = self.h.view(E * N, self.H)
            action, logp, value, h_new, _ = self.policy.act(obs_t, gs_t, h_flat, self.slot, first_t)
            env_act = self.policy.to_env_action(action).reshape(E, N, 4)
            h_new = h_new.view(E, N, self.H).clone()

            # opponents for team 1 in non-self-play envs
            groups = defaultdict(list)
            for e, name in enumerate(self.env_opp):
                if name != LATEST:
                    groups[name].append(e)
            for name, envs in groups.items():
                envs = np.array(envs)
                if name.startswith("bot:"):
                    a = self.bots[name].act(self.env.bot_view(1))
                    env_act[envs, T:] = a[envs]
                else:
                    net = self._opp_net(name)
                    sel = (envs[:, None] * N + np.arange(T, N)[None]).reshape(-1)
                    a, _, _, hn, _ = net.act(obs_t[sel], gs_t[sel], h_flat[sel], self.slot[sel], first_t[sel])
                    env_act[envs, T:] = net.to_env_action(a).reshape(len(envs), T, 4)
                    h_new[envs, T:] = hn.view(len(envs), T, self.H)

            self.buffer.add(torch.as_tensor(obs_in).view(E, N, -1), torch.as_tensor(self.gs), action.view(E, N, 4),
                            logp.view(E, N), value.view(E, N), torch.as_tensor(self.first, dtype=torch.float32),
                            torch.as_tensor(alive, dtype=torch.float32), torch.as_tensor(learner), self.h)
            if self.hier:
                tt = self.buffer.step - 1
                self.buffer.plan_action[tt] = torch.as_tensor(p_act)
                self.buffer.plan_logp[tt] = torch.as_tensor(p_logp)
                self.buffer.plan_mask[tt] = torch.as_tensor(p_mask)

            self.obs, self.gs, rew, done, info = self.env.step(env_act)
            self.global_step += E
            rew = rew.copy()

            # episode bookkeeping for finished envs
            if "episode" in info:
                ep = info["episode"]
                for k, e in enumerate(ep["idx"]):
                    opp = self.env_opp[e]
                    w = int(ep["winner"][k])
                    score = 1.0 if w == 0 else (0.0 if w == 1 else 0.5)
                    self.league.record(opp, score)
                    self._log_episode(ep, k, opp, stats)
                    self.env_opp[e] = self.league.sample_opponent(self.rng)
                if self.disc is not None:
                    lm = learner[ep["idx"]] > 0                                     # [n_done, N]
                    self.disc.add(ep["agent_stats"][lm], ep["roles"][lm])
                    bonus = self.disc.bonus(ep["agent_stats"], ep["roles"]) * lm
                    rew[ep["idx"]] += bonus
                    stats["disc/bonus"].append(float(bonus[lm].mean()) if lm.any() else 0.0)
                for b in self.bots.values():
                    b.reset(ep["idx"])
            done_full = np.repeat(done[:, None], N, 1)
            rew_n = self.ret_norm(rew, done_full) if self.ret_norm else rew
            self.buffer.add_outcome(torch.as_tensor(rew_n, dtype=torch.float32), done)
            h_new[torch.as_tensor(done)] = 0.0
            self.h = h_new
            self.first = done.copy()
            if self.fs:
                self.fs.push(self.obs)
                if done.any():
                    self.fs.reset(np.nonzero(done)[0], self.obs)
            self.plan[done] = 0

        # bootstrap value
        obs_in = self._policy_input()
        _, _, last_value, _, _ = self.policy.act(torch.as_tensor(obs_in).view(E * N, -1), torch.as_tensor(self.gs).view(E * N, -1),
                                                 self.h.view(E * N, self.H), self.slot,
                                                 torch.as_tensor(np.repeat(self.first, N), dtype=torch.float32))
        self.buffer.compute_gae(last_value.view(E, N))
        return {k: float(np.mean(v)) for k, v in stats.items()}

    def _log_episode(self, ep, k, opp, stats):
        T = self.T
        w = int(ep["winner"][k])
        rec = {"opp": opp, "win": float(w == 0), "loss": float(w == 1), "draw": float(w < 0),
               "length": int(ep["length"][k]), "return": float(ep["return"][k, :T].mean()),
               "first_contact": int(ep["first_contact"][k]), "pair_dist": float(ep["pair_dist"][k, 0]),
               "roles": ep["roles"][k, :T].copy(), "agent_stats": ep["agent_stats"][k, :T].copy(),
               "shots": ep["shots"][k, :T].copy(), "hits_enemy": ep["hits_enemy"][k, :T].copy(),
               "hits_ally": ep["hits_ally"][k, :T].copy(), "engage": ep["engage_angles"][k]}
        self.recent.append(rec)
        kind = "self" if opp == LATEST else ("bot" if opp.startswith("bot:") else "pool")
        stats[f"win_rate/{kind}"].append(rec["win"])
        stats[f"win_rate/vs_{opp}"].append(rec["win"])
        stats["episode/length"].append(rec["length"])
        stats["episode/return"].append(rec["return"])
        stats["episode/draw_rate"].append(rec["draw"])
        shots = rec["shots"].sum()
        stats["behaviour/accuracy"].append(rec["hits_enemy"].sum() / max(shots, 1))
        stats["behaviour/friendly_fire_rate"].append(rec["hits_ally"].sum() / max(shots, 1))
        stats["behaviour/shots_per_agent"].append(shots / T)
        stats["behaviour/pair_distance"].append(rec["pair_dist"])
        if rec["first_contact"] >= 0:
            stats["behaviour/time_to_first_contact_s"].append(rec["first_contact"] * self.env_cfg.dt)
        if rec["engage"]:
            stats["behaviour/engage_angle_mean_deg"].append(float(np.mean(rec["engage"])))
            stats["behaviour/flank_fraction"].append(float(np.mean(np.array(rec["engage"]) > 90.0)))
        if self.env_cfg.roles_enabled:
            for r, st in zip(rec["roles"], rec["agent_stats"]):
                for name, v in zip(RoleDiscriminator.STAT_NAMES, st):
                    stats[f"roles/{name}/role{int(r)}"].append(float(v))

    # --------------------------------------------------------------- update
    def update(self) -> Dict[str, float]:
        tc = self.cfg["train"]
        progress = self.update_idx / max(1, self.total_updates)
        lr = self.lr0 * (1.0 - progress) if tc.get("lr_anneal", True) else self.lr0
        for g in self.opt.param_groups:
            g["lr"] = lr
        e0, e1 = float(tc.get("entropy_coef", 0.01)), float(tc.get("entropy_coef_final", 0.001))
        ent_coef = e0 + (e1 - e0) * progress
        clip, vf_coef, max_gn = float(tc["clip"]), float(tc["value_coef"]), float(tc["max_grad_norm"])
        self.policy.train()
        agg = defaultdict(list)
        for _ in range(int(tc["epochs"])):
            for b in self.buffer.iterate(int(tc["minibatches"]), self.rng):
                logp, ent, values, hs = self.policy.evaluate(b["obs"], b["gs"], b["h0"], b["first"], b["actions"], b["slot"])
                m_act = b["learner"] * b["alive"]
                m_val = b["learner"]
                adv = b["adv"]
                mu = (adv * m_act).sum() / m_act.sum().clamp(min=1)
                sd = torch.sqrt(((adv - mu) ** 2 * m_act).sum() / m_act.sum().clamp(min=1)) + 1e-8
                adv_n = (adv - mu) / sd
                ratio = torch.exp(logp - b["logp"])
                s1 = ratio * adv_n
                s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_n
                pg_loss = -(torch.min(s1, s2) * m_act).sum() / m_act.sum().clamp(min=1)
                v_clipped = b["values"] + torch.clamp(values - b["values"], -clip, clip)
                v_loss = torch.max((values - b["returns"]) ** 2, (v_clipped - b["returns"]) ** 2)
                v_loss = 0.5 * (v_loss * m_val).sum() / m_val.sum().clamp(min=1)
                ent_loss = (ent * m_act).sum() / m_act.sum().clamp(min=1)
                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss
                if self.hier:
                    c_loss, c_ent = self._commander_loss(b, hs, adv_n, m_act, clip)
                    loss = loss + self.plan_coef * (c_loss - ent_coef * c_ent)
                    agg["loss/commander"].append(c_loss.item())
                self.opt.zero_grad()
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_gn)
                self.opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)) * m_act
                    agg["loss/policy"].append(pg_loss.item())
                    agg["loss/value"].append(v_loss.item())
                    agg["loss/entropy"].append(ent_loss.item())
                    agg["stats/approx_kl"].append((kl.sum() / m_act.sum().clamp(min=1)).item())
                    agg["stats/clipfrac"].append((((ratio - 1).abs() > clip).float() * m_act).sum().item() / max(m_act.sum().item(), 1))
                    agg["stats/grad_norm"].append(float(gn))
        out = {k: float(np.mean(v)) for k, v in agg.items()}
        out["stats/lr"] = lr
        out["stats/entropy_coef"] = ent_coef
        if self.policy.action_mode == "hybrid":
            out["stats/action_std"] = float(self.policy.policies[0].log_std.detach().exp().mean())
        return out

    def _commander_loss(self, b, hs, adv_n, m_act, clip):
        L, B, H = hs.shape
        P, N, T = b["num_pairs"], self.N, self.T
        w = b["alive"].view(L, P, 2, T, 1)
        team_h = (hs.view(L, P, 2, T, H) * w).sum(3) / w.sum(3).clamp(min=1.0)       # [L,P,2,H]
        logits = self.policy.commander_logits(team_h.view(-1, H)).view(L, P, 2, -1)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(b["plan_action"])
        ma = m_act.view(L, P, 2, T)
        adv_team = (adv_n.view(L, P, 2, T) * ma).sum(-1) / ma.sum(-1).clamp(min=1)
        ratio = torch.exp(logp - b["plan_logp"])
        s1, s2 = ratio * adv_team, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_team
        m = b["plan_mask"]
        loss = -(torch.min(s1, s2) * m).sum() / m.sum().clamp(min=1)
        ent = (dist.entropy() * m).sum() / m.sum().clamp(min=1)
        return loss, ent

    # ----------------------------------------------------------------- train
    def train(self, updates: Optional[int] = None):
        tc = self.cfg["train"]
        anneal = float(tc.get("shaping_anneal_frac", 0.3))
        log_every = int(tc.get("log_every", 1))
        ckpt_every = int(tc.get("checkpoint_every", 20))
        video_every = int(tc.get("video_every", 50))
        end = self.total_updates if updates is None else min(self.total_updates, self.update_idx + updates)
        while self.update_idx < end:
            t0 = time.time()
            progress = self.update_idx / max(1, self.total_updates)
            self.env.set_shaping_coef(max(0.0, 1.0 - progress / anneal) if anneal > 0 else 0.0)
            roll = self.collect_rollout()
            t1 = time.time()
            upd = self.update()
            if self.disc is not None:
                upd.update(self.disc.train_steps(rng=self.rng))
            self.update_idx += 1
            if self.update_idx % self.snapshot_every == 0 and not self.league.scripted_only:
                self.league.add_snapshot(self.policy, self.update_idx)
            if self.update_idx % log_every == 0:
                logs = {**roll, **upd, **self.league.summary(), "time/rollout_s": t1 - t0,
                        "time/update_s": time.time() - t1, "time/steps_per_s": self.rollout_len * self.E / (time.time() - t0),
                        "stats/shaping_coef": self.env.shaping_coef}
                for k, v in logs.items():
                    self.writer.add_scalar(k, v, self.global_step)
                self._print(logs)
            if self.update_idx % ckpt_every == 0:
                self.save(os.path.join(self.log_dir, "checkpoints", f"update_{self.update_idx:06d}.pt"))
                self.save(os.path.join(self.log_dir, "checkpoints", "latest.pt"))
            if video_every and self.update_idx % video_every == 0:
                self.record_video()
        self.save(os.path.join(self.log_dir, "checkpoints", "latest.pt"))
        self.league.save()

    def _print(self, logs):
        keys = ["win_rate/self", "win_rate/bot", "win_rate/pool", "elo/latest", "episode/return", "episode/length",
                "behaviour/accuracy", "behaviour/friendly_fire_rate", "loss/policy", "loss/value", "stats/approx_kl",
                "disc/accuracy", "time/steps_per_s"]
        parts = [f"{k.split('/')[-1]}={logs[k]:.3g}" for k in keys if k in logs]
        print(f"[update {self.update_idx}/{self.total_updates} step {self.global_step}] " + " ".join(parts), flush=True)

    def record_video(self):
        try:
            from scripts.render import record_episode
            out = os.path.join(self.log_dir, "videos", f"update_{self.update_idx:06d}.gif")
            frames = record_episode(self.cfg, self.policy, out, seed=self.update_idx, max_frames=600)
            if frames is not None and len(frames):
                idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
                strip = np.concatenate([frames[i] for i in idx], 1)
                self.writer.add_image("rollout/strip", strip.transpose(2, 0, 1), self.global_step)
        except Exception as ex:  # rendering must never kill training
            print(f"[video] skipped: {ex}", flush=True)

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(), "opt": self.opt.state_dict(), "update_idx": self.update_idx,
            "global_step": self.global_step, "cfg": self.cfg,
            "ret_norm": self.ret_norm.state() if self.ret_norm else None,
            "disc": self.disc.state_dict() if self.disc else None,
            "elo": self.league.elo,
        }, path)
        self.league.save()

    def load(self, path: str):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        self.opt.load_state_dict(ck["opt"])
        self.update_idx, self.global_step = ck["update_idx"], ck["global_step"]
        if self.ret_norm and ck.get("ret_norm"):
            self.ret_norm.load(ck["ret_norm"])
        if self.disc and ck.get("disc"):
            self.disc.load_state_dict(ck["disc"])
        self.league.load()
        self._reset_runtime()

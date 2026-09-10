"""Evaluate a checkpoint against scripted baselines, itself and league snapshots.

  python scripts/eval.py --checkpoint runs/base/checkpoints/latest.pt --episodes 100
Reports win rate, Elo (from the league file), and behavioural metrics: teammate
pairwise distance, time to first contact, accuracy, friendly-fire rate and the
distribution of engagement angles relative to the victim's heading.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.mappo import build_env_config, build_policy  # noqa: E402
from algo.runner import PolicyRunner  # noqa: E402
from env.bots import BOTS, make_bot  # noqa: E402
from env.squad_env import SquadVecEnv  # noqa: E402

ANGLE_BINS = [0, 45, 90, 135, 180.01]


def load_policy(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    env = SquadVecEnv(build_env_config(cfg), 1)
    policy = build_policy(cfg, env)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return cfg, policy


def run_matches(cfg: dict, policy, opponent: str, episodes: int, num_envs: int = 32, seed: int = 0,
                deterministic: bool = False, opp_policy=None):
    ec = build_env_config(cfg)
    env = SquadVecEnv(ec, num_envs, seed=seed)
    interval = int((cfg.get("hierarchical") or {}).get("interval", 20))
    obs, gs = env._build_obs(), env._build_global_state()
    if opponent == "self":
        runners = [PolicyRunner(policy, env, (0, 1), deterministic, interval)]
        bot = None
    elif opponent.startswith("snap:") or opp_policy is not None:
        runners = [PolicyRunner(policy, env, (0,), deterministic, interval), PolicyRunner(opp_policy, env, (1,), deterministic, interval)]
        bot = None
    else:
        runners = [PolicyRunner(policy, env, (0,), deterministic, interval)]
        bot = make_bot(opponent, ec, num_envs, np.random.default_rng(seed))
        bot.reset(np.arange(num_envs))
    for r in runners:
        r.reset(np.arange(num_envs), obs)
    T = env.T
    res = {"win": [], "loss": [], "draw": [], "length": [], "pair_dist": [], "first_contact": [], "shots": 0,
           "hits_enemy": 0, "hits_ally": 0, "engage": [], "damage_dealt": [], "damage_taken": [],
           "coverage_entropy": [], "crossfire": [], "role": {}}
    n = 0
    while n < episodes:
        a = np.zeros((num_envs, env.N, 4), np.float32)
        for r in runners:
            a[:, r.agents] = r.act(obs, gs)
        if bot is not None:
            a[:, T:] = bot.act(env.bot_view(1))
        obs, gs, _, done, info = env.step(a)
        if "episode" in info:
            ep = info["episode"]
            for k, e in enumerate(ep["idx"]):
                if n >= episodes:
                    break
                n += 1
                w = int(ep["winner"][k])
                res["win"].append(w == 0)
                res["loss"].append(w == 1)
                res["draw"].append(w < 0)
                res["length"].append(int(ep["length"][k]))
                res["pair_dist"].append(float(ep["pair_dist"][k, 0]))
                res["coverage_entropy"].append(float(ep["coverage_entropy"][k, 0]))
                if ep["hits_enemy"][k, :T].sum() > 0:
                    res["crossfire"].append(float(ep["crossfire_rate"][k, 0]))
                if ep["first_contact"][k] >= 0:
                    res["first_contact"].append(ep["first_contact"][k] * ec.dt)
                res["shots"] += float(ep["shots"][k, :T].sum())
                res["hits_enemy"] += float(ep["hits_enemy"][k, :T].sum())
                res["hits_ally"] += float(ep["hits_ally"][k, :T].sum())
                res["engage"] += list(ep["engage_angles"][k])
                res["damage_dealt"].append(float(ep["damage_dealt"][k, :T].sum()))
                for i in range(T):
                    if not ep["active"][k, i]:
                        continue
                    rr = res["role"].setdefault(int(ep["roles"][k, i]), {"shots": 0.0, "hits": 0.0, "flank": 0.0, "stats": []})
                    rr["shots"] += float(ep["shots"][k, i])
                    rr["hits"] += float(ep["hits_enemy"][k, i])
                    rr["flank"] += float(ep["flank_hits"][k, i])
                    rr["stats"].append(ep["agent_stats"][k, i])
                res["damage_taken"].append(float(ep["damage_taken"][k, :T].sum()))
            for r in runners:
                r.reset(ep["idx"], obs)
            if bot is not None:
                bot.reset(ep["idx"])
    per_role = {}
    for r, d in sorted(res["role"].items()):
        st = np.mean(d["stats"], 0)
        per_role[["assault", "flanker", "overwatch"][r] if r < 3 else f"role{r}"] = {
            "shots_per_episode": d["shots"] / max(n, 1), "accuracy": d["hits"] / max(d["shots"], 1),
            "flank_fraction": d["flank"] / max(d["hits"], 1), "dist_to_centroid_m": float(st[0] * ec.arena_size),
            "dist_to_nearest_enemy_m": float(st[1] * ec.arena_size), "distance_travelled_norm": float(st[3]),
            "unique_spotting_fraction": float(st[5])}
    hist, _ = np.histogram(res["engage"], bins=ANGLE_BINS)
    hist = hist / max(hist.sum(), 1)
    return {
        "opponent": opponent, "episodes": n,
        "win_rate": float(np.mean(res["win"])), "loss_rate": float(np.mean(res["loss"])), "draw_rate": float(np.mean(res["draw"])),
        "mean_length_s": float(np.mean(res["length"]) * ec.dt),
        "teammate_pair_distance_m": float(np.mean(res["pair_dist"])),
        "angular_coverage_entropy": float(np.mean(res["coverage_entropy"])),
        "crossfire_rate": float(np.mean(res["crossfire"])) if res["crossfire"] else float("nan"),
        "time_to_first_contact_s": float(np.mean(res["first_contact"])) if res["first_contact"] else float("nan"),
        "contact_rate": len(res["first_contact"]) / max(n, 1),
        "shot_accuracy": res["hits_enemy"] / max(res["shots"], 1),
        "friendly_fire_rate": res["hits_ally"] / max(res["shots"], 1),
        "shots_per_episode": res["shots"] / max(n, 1),
        "damage_dealt": float(np.mean(res["damage_dealt"])), "damage_taken": float(np.mean(res["damage_taken"])),
        "engage_angle_hist(0-45,45-90,90-135,135-180)": [round(float(x), 3) for x in hist],
        "flank_fraction(>90deg)": float(np.mean(np.array(res["engage"]) > 90)) if res["engage"] else float("nan"),
        "per_role": per_role,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--opponents", nargs="*", default=["random", "spinner", "charger", "holder", "self"],
                    help="bot names, 'self', 'pool' (all league snapshots) or a snapshot .pt path")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None, help="json output path")
    ap.add_argument("--save-episode", default=None, help="also record one episode vs the first opponent (gif)")
    args = ap.parse_args()
    cfg, policy = load_policy(args.checkpoint)
    league_dir = os.path.join(os.path.dirname(os.path.dirname(args.checkpoint)), "league")
    elo = {}
    if os.path.exists(os.path.join(league_dir, "league.json")):
        with open(os.path.join(league_dir, "league.json")) as f:
            elo = json.load(f)["elo"]
    opponents = []
    for o in args.opponents:
        if o == "pool":
            opponents += sorted(glob.glob(os.path.join(league_dir, "snap_*.pt")))
        else:
            opponents.append(o)
    results = []
    for o in opponents:
        opp_policy = None
        name = o
        if o.endswith(".pt"):
            opp_cfg, opp_policy = cfg, build_policy(cfg, SquadVecEnv(build_env_config(cfg), 1))
            opp_policy.load_state_dict(torch.load(o, map_location="cpu"))
            name = "snap:" + os.path.basename(o)[5:-3]
        elif o not in BOTS and o != "self":
            raise SystemExit(f"unknown opponent {o}")
        r = run_matches(cfg, policy, name, args.episodes, args.num_envs, args.seed, args.deterministic, opp_policy)
        r["opponent_elo"] = elo.get(name, elo.get(f"bot:{name}", None))
        results.append(r)
        print(json.dumps(r, indent=1))
    summary = {"checkpoint": args.checkpoint, "latest_elo": elo.get("latest"), "results": results}
    print("\n| opponent | win | draw | loss | acc | ff | pair dist | 1st contact | flank>90 | crossfire | cov. entropy |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['opponent']} | {r['win_rate']:.2f} | {r['draw_rate']:.2f} | {r['loss_rate']:.2f} | {r['shot_accuracy']:.2f} | "
              f"{r['friendly_fire_rate']:.3f} | {r['teammate_pair_distance_m']:.1f} m | {r['time_to_first_contact_s']:.1f} s | "
              f"{r['flank_fraction(>90deg)']:.2f} | {r['crossfire_rate']:.2f} | {r['angular_coverage_entropy']:.2f} |")
    roles_seen = [r for r in results if r["per_role"]]
    if roles_seen:
        print("\n| opponent | role | shots/ep | acc | flank | dist to squad | dist to enemy | spotting |")
        print("|---|---|---|---|---|---|---|---|")
        for r in roles_seen:
            for name, d in r["per_role"].items():
                print(f"| {r['opponent']} | {name} | {d['shots_per_episode']:.1f} | {d['accuracy']:.2f} | {d['flank_fraction']:.2f} | "
                      f"{d['dist_to_centroid_m']:.1f} m | {d['dist_to_nearest_enemy_m']:.1f} m | {d['unique_spotting_fraction']:.2f} |")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)
    if args.save_episode:
        from scripts.render import record_episode
        record_episode(cfg, policy, args.save_episode, seed=args.seed, opponent=opponents[0] if opponents[0] in BOTS or opponents[0] == "self" else "self")
        print(f"wrote {args.save_episode}")


if __name__ == "__main__":
    main()

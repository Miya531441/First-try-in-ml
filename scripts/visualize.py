"""Visualise battles across training stages and across the phases of a battle.

Picks checkpoints from evenly spaced training stages of a run, replays the same
battle (same seed => same layout, spawns and opponent) with each of them and writes:

  <out>/stage_<k>_<name>.gif        full battle per stage
  <out>/stage_<k>_<name>_trace.png  movement traces, shots and deaths per stage
  <out>/contact_sheet.png           rows = training stages, columns = battle phases
                                    (spawn, first contact, first hit, midpoint, end)
  <out>/side_by_side.gif            all stages playing simultaneously
  <out>/summary.md                  outcome / length / accuracy per stage

Usage:
  python scripts/visualize.py --run runs/roles --stages 4 --opponent charger --seed 3
  python scripts/visualize.py --checkpoints a.pt b.pt c.pt --opponent self --out viz/
  python scripts/visualize.py --run runs/roles --times 0 5 10 20 40     # fixed-time columns
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.render import TEAM_COLORS, Renderer, save_gif  # noqa: E402

PHASES = ["spawn", "first contact", "first hit", "midpoint", "end"]


# ------------------------------------------------------------------ checkpoints
def select_checkpoints(run_dir: str, n: int) -> List[str]:
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "update_*.pt")))
    if not cks:
        raise SystemExit(f"no checkpoints under {run_dir}/checkpoints")
    if n >= len(cks):
        return cks
    idx = np.unique(np.linspace(0, len(cks) - 1, n).round().astype(int))
    return [cks[i] for i in idx]


def stage_name(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    if base.startswith("update_"):
        return f"update {int(base[7:])}"
    return base


# --------------------------------------------------------------------- rollout
def rollout(cfg: dict, policy, opponent: str, seed: int, max_steps: int, frame_skip: int) -> Dict:
    """Play one battle; return rendered images, raw frames, per-step events and a summary."""
    from algo.mappo import build_env_config
    from algo.runner import PolicyRunner
    from env.bots import make_bot
    from env.squad_env import SquadVecEnv
    ec = build_env_config(cfg)
    env = SquadVecEnv(ec, 1, seed=seed)
    teams = (0, 1) if opponent == "self" else (0,)
    runner = PolicyRunner(policy, env, teams, False, int((cfg.get("hierarchical") or {}).get("interval", 20)))
    bot = None if opponent == "self" else make_bot(opponent, ec, 1, np.random.default_rng(seed))
    if bot:
        bot.reset(np.array([0]))
    obs, gs = env._build_obs(), env._build_global_state()
    runner.reset(np.array([0]), obs)
    rend = Renderer(ec, headless=True)
    imgs, frames, times = [], [], []
    events = {"first_contact": None, "first_hit": None, "hits": [], "deaths": [], "shots": 0, "hits_enemy": 0,
              "hits_ally": 0}
    alive_prev = env.state.alive[0].copy()
    winner, length = -1, max_steps
    for t in range(max_steps):
        fr = env.get_frame(0)
        img = rend.draw(fr)
        if t % frame_skip == 0:
            imgs.append(img)
            times.append(t)
        frames.append(fr)
        a = np.zeros((1, env.N, 4), np.float32)
        a[:, runner.agents] = runner.act(obs, gs)
        if bot:
            a[:, env.T:] = bot.act(env.bot_view(1))
        obs, gs, _, done, info = env.step(a)
        if events["first_contact"] is None and info["sees_enemy"][0].any():
            events["first_contact"] = t
        for i in np.nonzero(info["hit_enemy"][0] | info["hit_ally"][0])[0]:
            v = int(env.shot_victim[0, i])
            events["hits"].append((t, int(i), v, frames[-1]["pos"][v].copy(), bool(info["hit_ally"][0, i])))
            if events["first_hit"] is None:
                events["first_hit"] = t
        events["shots"] += int(info["fire"][0, : env.T].sum())
        events["hits_enemy"] += int(info["hit_enemy"][0, : env.T].sum())
        events["hits_ally"] += int(info["hit_ally"][0, : env.T].sum())
        died = alive_prev & ~info["alive"][0]
        for i in np.nonzero(died)[0]:
            events["deaths"].append((t, int(i), frames[-1]["pos"][i].copy()))
        alive_prev = info["alive"][0].copy()
        if done[0]:
            winner, length = int(info["winner"][0]), t + 1
            frames.append({**frames[-1], "t": t + 1})
            imgs.append(imgs[-1])
            times.append(t + 1)
            break
    rend.close()
    return {"imgs": imgs, "times": times, "frames": frames, "events": events, "winner": winner, "length": length,
            "dt": ec.dt, "boxes": frames[0]["boxes"], "box_kind": frames[0]["box_kind"], "team": frames[0]["team"],
            "role": frames[0]["role"], "active": env.state.active[0].copy() if not done[0] else frames[0]["alive"] | True,
            "arena": ec.arena_size}


# ---------------------------------------------------------------- compositions
def _label(img: Image.Image, text: str, xy=(6, 4), fill=(255, 255, 255)) -> Image.Image:
    d = ImageDraw.Draw(img)
    w = d.textlength(text) + 8
    d.rectangle([xy[0] - 3, xy[1] - 2, xy[0] + w, xy[1] + 14], fill=(0, 0, 0))
    d.text(xy, text, fill=fill)
    return img


def phase_indices(r: Dict, times_s: Optional[List[float]]) -> List[tuple]:
    """(column label, image index) for each column of the contact sheet."""
    times, dt, n = r["times"], r["dt"], len(r["imgs"])

    def nearest(step):
        return int(np.argmin(np.abs(np.array(times) - step)))

    if times_s:
        return [(f"t={s:g}s", nearest(s / dt)) for s in times_s]
    ev = r["events"]
    fc, fh = ev["first_contact"], ev["first_hit"]
    return [("spawn", 0),
            ("first contact", nearest(fc)) if fc is not None else ("no contact", n - 1),
            ("first hit", nearest(fh)) if fh is not None else ("no hit", n - 1),
            ("midpoint", nearest(r["length"] // 2)), ("end", n - 1)]


def contact_sheet(results: List[Dict], names: List[str], times_s: Optional[List[float]], scale: float = 0.5) -> Image.Image:
    cols = None
    tiles = []
    for r, name in zip(results, names):
        cols = phase_indices(r, times_s)
        row = []
        for label, idx in cols:
            im = Image.fromarray(r["imgs"][idx])
            im = im.resize((int(im.width * scale), int(im.height * scale)))
            t = r["times"][idx] * r["dt"]
            row.append(_label(im, f"{label}  t={t:.1f}s"))
        tiles.append(row)
    w, h = tiles[0][0].size
    left = 150
    sheet = Image.new("RGB", (left + w * len(cols), h * len(tiles)), (18, 18, 20))
    d = ImageDraw.Draw(sheet)
    for i, (r, name) in enumerate(zip(results, names)):
        out = {0: "WIN", 1: "LOSS", -1: "DRAW"}[r["winner"]]
        d.text((8, i * h + 8), name, fill=(230, 230, 230))
        d.text((8, i * h + 24), f"{out}  {r['length'] * r['dt']:.1f}s", fill=(200, 200, 200))
        d.text((8, i * h + 40), f"shots {r['events']['shots']}", fill=(160, 160, 160))
        d.text((8, i * h + 56), f"acc {r['events']['hits_enemy'] / max(r['events']['shots'], 1):.2f}", fill=(160, 160, 160))
        for j, im in enumerate(tiles[i]):
            sheet.paste(im, (left + j * w, i * h))
    return sheet


def trace_image(r: Dict, px: float = 8.0) -> Image.Image:
    """Movement traces (fading in over time), hit markers and deaths for one battle."""
    S = int(r["arena"] * px)
    im = Image.new("RGB", (S, S), (28, 28, 32))
    d = ImageDraw.Draw(im)

    def P(p):
        return (p[0] * px, S - p[1] * px)

    kinds = r.get("box_kind", np.zeros(len(r["boxes"]), int))
    for b, kind in zip(r["boxes"], kinds):
        d.rectangle([P((b[0], b[3])), P((b[2], b[1]))], fill=(110, 110, 118) if kind == 0 else (96, 74, 48))
    frames = r["frames"]
    N = len(r["team"])
    for i in range(N):
        col = TEAM_COLORS[int(r["team"][i])]
        pts = [(f["pos"][i], f["alive"][i]) for f in frames]
        for k in range(1, len(pts)):
            if not pts[k][1]:
                break
            a = 0.25 + 0.75 * k / len(pts)
            c = tuple(int(v * a + 28 * (1 - a)) for v in col)
            d.line([P(pts[k - 1][0]), P(pts[k][0])], fill=c, width=2)
        if not pts[0][1]:
            continue                                                   # inactive slot this episode
        p0 = P(pts[0][0])
        d.ellipse([p0[0] - 4, p0[1] - 4, p0[0] + 4, p0[1] + 4], outline=col, width=2)
        role = "AFO"[int(r["role"][i]) % 3] if "role" in r else ""
        d.text((p0[0] + 6, p0[1] - 6), f"{i}{role}", fill=col)
    for t, shooter, victim, pos, friendly in r["events"]["hits"]:
        p = P(pos)
        c = (255, 200, 0) if friendly else (255, 80, 80)
        d.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], outline=c, width=1)
    for t, i, pos in r["events"]["deaths"]:
        p = P(pos)
        d.line([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=(255, 255, 255), width=2)
        d.line([p[0] - 6, p[1] + 6, p[0] + 6, p[1] - 6], fill=(255, 255, 255), width=2)
        d.text((p[0] + 8, p[1] - 6), f"{t * r['dt']:.1f}s", fill=(255, 255, 255))
    out = {0: "WIN", 1: "LOSS", -1: "DRAW"}[r["winner"]]
    _label(im, f"{out} in {r['length'] * r['dt']:.1f}s | o start (A assault, F flanker, O overwatch)  x death  red o hit  yellow o friendly")
    return im


def side_by_side(results: List[Dict], names: List[str], scale: float = 0.5) -> List[np.ndarray]:
    n = max(len(r["imgs"]) for r in results)
    out = []
    for k in range(n):
        tiles = []
        for r, name in zip(results, names):
            im = Image.fromarray(r["imgs"][min(k, len(r["imgs"]) - 1)])
            im = im.resize((int(im.width * scale), int(im.height * scale)))
            tag = name if k < len(r["imgs"]) else f"{name} (ended)"
            tiles.append(np.asarray(_label(im, tag)))
        out.append(np.concatenate(tiles, 1))
    return out


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory containing checkpoints/update_*.pt")
    ap.add_argument("--checkpoints", nargs="*", help="explicit checkpoints instead of --run")
    ap.add_argument("--stages", type=int, default=4, help="number of training stages to pick from --run")
    ap.add_argument("--opponent", default="self", help="self | random | spinner | charger | holder")
    ap.add_argument("--seed", type=int, default=0, help="same seed => same layout for every stage")
    ap.add_argument("--times", nargs="*", type=float, default=None, help="fixed snapshot times (s) instead of battle phases")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from algo.mappo import build_env_config, build_policy
    from env.squad_env import SquadVecEnv

    cks = args.checkpoints or select_checkpoints(args.run, args.stages)
    out = args.out or os.path.join(args.run or ".", "viz")
    os.makedirs(out, exist_ok=True)
    results, names = [], []
    for ck in cks:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = c["cfg"]
        policy = build_policy(cfg, SquadVecEnv(build_env_config(cfg), 1))
        policy.load_state_dict(c["policy"])
        name = stage_name(ck)
        r = rollout(cfg, policy, args.opponent, args.seed, args.max_steps, args.frame_skip)
        results.append(r)
        names.append(name)
        k = len(results) - 1
        save_gif(os.path.join(out, f"stage_{k}_{name.replace(' ', '_')}.gif"), r["imgs"])
        trace_image(r).save(os.path.join(out, f"stage_{k}_{name.replace(' ', '_')}_trace.png"))
        print(f"{name}: {'WIN' if r['winner'] == 0 else 'LOSS' if r['winner'] == 1 else 'DRAW'} "
              f"in {r['length'] * r['dt']:.1f}s, shots {r['events']['shots']}, hits {r['events']['hits_enemy']}")
    contact_sheet(results, names, args.times).save(os.path.join(out, "contact_sheet.png"))
    save_gif(os.path.join(out, "side_by_side.gif"), side_by_side(results, names))
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(f"opponent: {args.opponent}, seed: {args.seed}\n\n| stage | outcome | length (s) | shots | hits | accuracy | "
                "friendly hits | first contact (s) | first hit (s) |\n|---|---|---|---|---|---|---|---|---|\n")
        for r, name in zip(results, names):
            ev = r["events"]
            fc = f"{ev['first_contact'] * r['dt']:.1f}" if ev["first_contact"] is not None else "-"
            fh = f"{ev['first_hit'] * r['dt']:.1f}" if ev["first_hit"] is not None else "-"
            f.write(f"| {name} | {'WIN' if r['winner'] == 0 else 'LOSS' if r['winner'] == 1 else 'DRAW'} | "
                    f"{r['length'] * r['dt']:.1f} | {ev['shots']} | {ev['hits_enemy']} | "
                    f"{ev['hits_enemy'] / max(ev['shots'], 1):.2f} | {ev['hits_ally']} | {fc} | {fh} |\n")
    print(f"wrote {out}/contact_sheet.png, side_by_side.gif, summary.md and per-stage gifs/traces")


if __name__ == "__main__":
    main()

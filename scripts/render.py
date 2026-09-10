"""pygame renderer: vision cones (with occlusion), firing rays, hit markers, HUD.

Usage:
  python scripts/render.py --episode runs/x/videos/update_000050.npz          # replay in a window
  python scripts/render.py --episode ep.npz --gif out.gif                     # export a GIF
  python scripts/render.py --checkpoint runs/x/checkpoints/latest.pt --gif out.gif [--opponent charger]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.agent import EnvConfig  # noqa: E402

TEAM_COLORS = [(66, 135, 245), (240, 80, 70)]
ROLE_NAMES = ["assault", "flanker", "overwatch"]
PLAN_NAMES = ["regroup", "split_flank", "push", "hold"]


class Renderer:
    def __init__(self, cfg: EnvConfig, px_per_m: float = 10.0, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        self.cfg, self.k = cfg, px_per_m
        self.W = int(cfg.arena_size * px_per_m)
        self.hud = 28
        self.screen = pygame.display.set_mode((self.W, self.W + self.hud)) if not headless else pygame.Surface((self.W, self.W + self.hud))
        self.font = pygame.font.SysFont(None, 20)
        self.headless = headless

    def _p(self, xy):
        return int(xy[0] * self.k), int(self.W - xy[1] * self.k)

    def draw(self, f: Dict) -> np.ndarray:
        pg, cfg, k = self.pg, self.cfg, self.k
        s = self.screen
        s.fill((28, 28, 32))
        overlay = pg.Surface((self.W, self.W), pg.SRCALPHA)
        kinds = f.get("box_kind", np.zeros(len(f["boxes"]), int))
        for b, kind in zip(f["boxes"], kinds):
            rect = pg.Rect(self._p((b[0], b[3])), (int((b[2] - b[0]) * k), int((b[3] - b[1]) * k)))
            if kind == 0:
                pg.draw.rect(s, (110, 110, 118), rect)                       # wall: blocks vision + movement
            else:
                pg.draw.rect(s, (96, 74, 48), rect)                          # low crate: movement only
                pg.draw.rect(s, (150, 120, 80), rect, 2)
        N = len(f["pos"])
        R = f["ray_dist"].shape[1]
        offsets = np.linspace(-cfg.fov / 2, cfg.fov / 2, R)
        for i in range(N):
            if not f["alive"][i]:
                pg.draw.circle(s, (70, 70, 70), self._p(f["pos"][i]), int(cfg.collision_radius * k) + 2, 1)
                continue
            col = TEAM_COLORS[int(f["team"][i])]
            ang = f["theta"][i] + offsets
            # occlusion shadow: the full unobstructed cone drawn dim, the visible part drawn brighter
            full = f["pos"][i] + np.stack([np.cos(ang), np.sin(ang)], -1) * cfg.vision_range
            pg.draw.polygon(overlay, (*col, 18), [self._p(f["pos"][i])] + [self._p(e) for e in full])
            ends = f["pos"][i] + np.stack([np.cos(ang), np.sin(ang)], -1) * f["ray_dist"][i][:, None]
            poly = [self._p(f["pos"][i])] + [self._p(e) for e in ends]
            pg.draw.polygon(overlay, (*col, 55), poly)
            for r_, e in enumerate(ends):   # mark rays that see an agent body
                if f["ray_kind"][i, r_] == 2:
                    pg.draw.circle(overlay, (255, 255, 120, 160), self._p(e), 2)
        s.blit(overlay, (0, 0))
        for i in range(N):
            if not f["alive"][i]:
                continue
            col = TEAM_COLORS[int(f["team"][i])]
            p = self._p(f["pos"][i])
            if f["shot_fired"][i]:
                d = np.array([np.cos(f["shot_aim"][i]), np.sin(f["shot_aim"][i])])
                dist = min(float(f["shot_dist"][i]), cfg.arena_size * 2)
                e = f["pos"][i] + d * dist
                pg.draw.line(s, (255, 240, 120), p, self._p(e), 1)
                if f["shot_kind"][i] == 2:
                    pg.draw.circle(s, (255, 60, 60), self._p(e), 5, 2)
                else:
                    pg.draw.circle(s, (200, 200, 200), self._p(e), 2)
            pg.draw.circle(s, col, p, max(3, int(cfg.collision_radius * k)))
            hd = f["pos"][i] + np.array([np.cos(f["theta"][i]), np.sin(f["theta"][i])]) * 1.0
            pg.draw.line(s, (255, 255, 255), p, self._p(hd), 2)
            # hp bar
            w = 14
            pg.draw.rect(s, (60, 60, 60), pg.Rect(p[0] - w // 2, p[1] - 12, w, 3))
            pg.draw.rect(s, (90, 220, 90), pg.Rect(p[0] - w // 2, p[1] - 12, int(w * f["hp"][i] / cfg.hp), 3))
            if "role" in f:
                lbl = self.font.render(ROLE_NAMES[int(f["role"][i]) % 3][0].upper(), True, (230, 230, 230))
                s.blit(lbl, (p[0] + 6, p[1] - 8))
        alive = [int(f["alive"][f["team"] == t].sum()) for t in (0, 1)]
        txt = f"t={f['t'] * cfg.dt:5.1f}s   blue alive {alive[0]}   red alive {alive[1]}"
        if "plan" in f and cfg.plan_tokens:
            txt += f"   plans: {PLAN_NAMES[int(f['plan'][0])]} / {PLAN_NAMES[int(f['plan'][1])]}"
        s.blit(self.font.render(txt, True, (230, 230, 230)), (6, self.W + 6))
        if not self.headless:
            pg.display.flip()
        return np.transpose(pg.surfarray.array3d(s), (1, 0, 2)).copy()

    def close(self):
        self.pg.quit()


# --------------------------------------------------------------------------- episode I/O
FRAME_KEYS = ["pos", "theta", "hp", "alive", "ray_dist", "ray_kind", "shot_fired", "shot_aim", "shot_dist", "shot_kind", "plan"]


def save_episode(path: str, frames: List[Dict], cfg: EnvConfig):
    data = {k: np.stack([f[k] for f in frames]) for k in FRAME_KEYS}
    data["t"] = np.array([f["t"] for f in frames])
    data["team"], data["role"], data["boxes"] = frames[0]["team"], frames[0]["role"], frames[0]["boxes"]
    data["box_kind"] = frames[0].get("box_kind", np.zeros(len(frames[0]["boxes"]), int))
    data["cfg"] = np.array([__import__("json").dumps(cfg.to_dict())])
    np.savez_compressed(path, **data)


def load_episode(path: str):
    d = np.load(path, allow_pickle=True)
    cfg = EnvConfig.from_dict(__import__("json").loads(str(d["cfg"][0])))
    frames = []
    for t in range(len(d["t"])):
        f = {k: d[k][t] for k in FRAME_KEYS}
        f.update(t=int(d["t"][t]), team=d["team"], role=d["role"], boxes=d["boxes"],
                 box_kind=d["box_kind"] if "box_kind" in d else np.zeros(len(d["boxes"]), int))
        frames.append(f)
    return frames, cfg


def save_gif(path: str, frames: List[np.ndarray], fps: int = 20):
    from PIL import Image
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)


def record_episode(cfg: dict, policy, out_path: Optional[str], seed: int = 0, max_frames: int = 1200,
                   opponent: str = "self", deterministic: bool = False, frame_skip: int = 2):
    """Roll one episode (policy vs itself or a scripted bot) and save a GIF + replayable npz."""
    from algo.mappo import build_env_config
    from algo.runner import PolicyRunner
    from env.bots import make_bot
    from env.squad_env import SquadVecEnv
    ec = build_env_config(cfg)
    env = SquadVecEnv(ec, 1, seed=seed)
    teams = (0, 1) if opponent == "self" else (0,)
    runner = PolicyRunner(policy, env, teams, deterministic, int((cfg.get("hierarchical") or {}).get("interval", 20)))
    bot = None if opponent == "self" else make_bot(opponent, ec, 1, np.random.default_rng(seed))
    if bot:
        bot.reset(np.array([0]))
    obs, gs = env._build_obs(), env._build_global_state()
    runner.reset(np.array([0]), obs)
    rend = Renderer(ec, headless=True)
    frames, imgs = [], []
    for t in range(max_frames):
        a = np.zeros((1, env.N, 4), np.float32)
        a[:, runner.agents] = runner.act(obs, gs)
        if bot:
            a[:, env.T:] = bot.act(env.bot_view(1))
        obs, gs, _, done, info = env.step(a)
        if done[0]:
            break
        if t % frame_skip == 0:
            fr = env.get_frame(0)
            frames.append(fr)
            imgs.append(rend.draw(fr))
    rend.close()
    if out_path and frames:
        save_gif(out_path, imgs)
        save_episode(os.path.splitext(out_path)[0] + ".npz", frames, ec)
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", help="saved .npz episode to replay")
    ap.add_argument("--checkpoint", help="checkpoint to roll out and render")
    ap.add_argument("--opponent", default="self", help="self | random | spinner | charger | holder")
    ap.add_argument("--gif", help="write a GIF instead of opening a window")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.checkpoint:
        import torch
        from algo.mappo import build_env_config, build_policy
        from env.squad_env import SquadVecEnv
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        env = SquadVecEnv(build_env_config(cfg), 1)
        policy = build_policy(cfg, env)
        policy.load_state_dict(ck["policy"])
        out = args.gif or "episode.gif"
        record_episode(cfg, policy, out, seed=args.seed, opponent=args.opponent)
        print(f"wrote {out} and {os.path.splitext(out)[0]}.npz")
        return
    frames, cfg = load_episode(args.episode)
    rend = Renderer(cfg, headless=bool(args.gif))
    imgs = []
    clock = rend.pg.time.Clock()
    for f in frames:
        if not args.gif:
            for ev in rend.pg.event.get():
                if ev.type == rend.pg.QUIT:
                    rend.close()
                    return
        imgs.append(rend.draw(f))
        if not args.gif:
            clock.tick(args.fps)
    if args.gif:
        save_gif(args.gif, imgs, args.fps)
        print(f"wrote {args.gif}")
    rend.close()


if __name__ == "__main__":
    main()

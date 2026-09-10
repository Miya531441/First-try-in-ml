"""Train MAPPO self-play.

  python scripts/train.py --config configs/base.yaml --log-dir runs/base
  python scripts/train.py --config configs/roles.yaml --set train.num_envs=64 train.total_updates=500
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algo.mappo import MAPPOTrainer  # noqa: E402
from algo.utils import load_config, parse_overrides  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.num_envs=64")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--updates", type=int, default=None, help="run at most this many updates now")
    args = ap.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    log_dir = args.log_dir or os.path.join("runs", os.path.splitext(os.path.basename(args.config))[0])
    trainer = MAPPOTrainer(cfg, log_dir)
    if args.resume:
        trainer.load(args.resume)
    print(f"obs_dim={trainer.env.obs_dim} gs_dim={trainer.env.gs_dim} params={sum(p.numel() for p in trainer.policy.parameters())}")
    trainer.train(args.updates)


if __name__ == "__main__":
    main()

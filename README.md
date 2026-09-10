# 3v3 squad combat with cone-limited vision (self-play MARL)

A squad of 3 agents learns, purely through self-play, to eliminate an enemy squad of 3
identical agents in a 64 x 64 m arena with mirrored cover.  Every agent only perceives the
world through a 60 degree, 30 m vision cone, so the task is a hard POMDP and coordination
(splitting, flanking, covering angles) is the skill to be learned.

See `DESIGN.md` for assumptions, spec issues and the reasoning behind each component.

## Layout

```
env/      arena.py  agent.py  physics.py  raycast.py  spawn.py  squad_env.py  bots.py
algo/     mappo.py  buffer.py  networks.py  league.py  discriminator.py  runner.py  utils.py
scripts/  train.py  eval.py  render.py
configs/  base.yaml  roles.yaml  hierarchical.yaml  1v1_scripted.yaml  ablations/*.yaml
tests/    test_env.py  test_algo.py
```

* `env/squad_env.py` - `SquadVecEnv` (batched numpy simulation of N arenas, auto-reset)
  and `SquadParallelEnv` (PettingZoo `ParallelEnv`, passes `parallel_api_test`).
* `env/raycast.py` - batched ray/box, ray/circle, boundary and segment-occlusion tests.
  All rays of all agents of all envs are cast against all obstacles at once.
* `env/bots.py` - scripted baselines: random, spinner, charger, holder.
* `algo/mappo.py` - MAPPO trainer (centralised critic on the mirrored global state,
  recurrent actor with chunked BPTT, self-play league, role diversity bonus,
  optional hierarchical commander, tensorboard logging, periodic GIF rollouts).
* `scripts/render.py` - pygame renderer: cones with occlusion, firing rays, hit markers,
  HUD; replays `.npz` episodes saved by training/eval.

## Install and test

```
pip install -r requirements.txt
python -m pytest tests -q
```

## Milestones and commands

| # | What | Command |
|---|------|---------|
| 1 | Env + renderer + bots | `python -m pytest tests`; `python scripts/render.py --checkpoint <ckpt> --opponent charger --gif out.gif` |
| 2 | 1v1 vs scripted bots, no roles | `python scripts/train.py --config configs/1v1_scripted.yaml` |
| 3 | 3v3 self-play, shared policy | `python scripts/train.py --config configs/base.yaml` |
| 4 | Roles + diversity bonus | `python scripts/train.py --config configs/roles.yaml` |
| 5 | League + ablations | `configs/ablations/{framestack,per_slot,no_friendly_fire,no_shaping,discrete}.yaml` |
| 6 | Hierarchical commander | `python scripts/train.py --config configs/hierarchical.yaml` |

Any config key can be overridden on the command line:

```
python scripts/train.py --config configs/roles.yaml --set train.num_envs=64 env.reward.friendly_damage=-0.1
```

Evaluation (win rates, Elo from the league file, behavioural metrics):

```
python scripts/eval.py --checkpoint runs/roles/checkpoints/latest.pt --episodes 200 \
    --opponents random spinner charger holder self pool --out results.json
```

Replay a recorded episode:

```
python scripts/render.py --episode runs/roles/videos/update_000050.npz          # window
python scripts/render.py --episode runs/roles/videos/update_000050.npz --gif x.gif
```

Logs go to tensorboard under `runs/<name>`: `tensorboard --logdir runs`.

## Key metrics logged

* `elo/latest`, `elo/<member>` - league Elo (primary progress metric).
* `win_rate/self|pool|bot`, `win_rate/vs_<opponent>`.
* `behaviour/pair_distance` (teammate spread), `time_to_first_contact_s`, `accuracy`,
  `friendly_fire_rate`, `engage_angle_mean_deg`, `flank_fraction` (hits from behind the
  victim's 90 degree line).
* `roles/<stat>/role<k>` - per-role behavioural statistics; `disc/accuracy` - how
  identifiable the roles are from behaviour (role collapse shows up as ~1/3).

## Throughput

The simulation runs at roughly 3.5k env-steps/s for 128 envs on one CPU core-set; the full
training loop (rollout + PPO update, CPU only, 4 cores) runs at ~700 env-steps/s, so the
default 3000-update schedule (~100M steps) is a multi-day CPU run.  A GPU for the update
phase and more cores for the rollout are the obvious levers; `train.torch_threads`
controls the CPU thread count.

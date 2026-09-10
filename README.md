# Squad combat with cone-limited vision (self-play MARL, spec v2)

Squads of 2-3 agents learn, purely through self-play, to eliminate an enemy squad in a
48 x 48 m arena with mirrored cover (full walls and low crates), starting from opposite
corners.  Each squad member is assigned a role (assault, flanker, overwatch) with its own
small incentive, a diversity bonus keeps the roles behaviourally distinct, and a
crossfire bonus rewards coordinated fire.  Every agent only perceives
the world through a 60 degree, 24 m vision cone plus a shared team blackboard (ally slots,
persistent enemy tracks with staleness/confidence) and a 64-ray static map scan, so the
task is a hard POMDP and coordination (splitting, flanking, crossfire, covering angles)
is the skill to be learned.

See `DESIGN.md` for assumptions, spec issues and the reasoning behind each component.

## Layout

```
env/      arena.py  agent.py  physics.py  raycast.py  spawn.py  squad_env.py  bots.py
algo/     mappo.py  buffer.py  networks.py  league.py  discriminator.py  runner.py  utils.py
scripts/  train.py  eval.py  render.py  visualize.py
configs/  base.yaml  roles.yaml  hierarchical.yaml  1v1_scripted.yaml  ablations/*.yaml
tests/    test_env.py  test_algo.py
```

* `env/squad_env.py` - `SquadVecEnv` (batched numpy simulation of N arenas, auto-reset,
  squad sizes sampled per episode from {3v3, 2v2, 3v2, 2v3}, team blackboard) and
  `SquadParallelEnv` (PettingZoo `ParallelEnv`, passes `parallel_api_test`).
* `env/spawn.py` - coverage-parameterised layouts (12-18% of the arena), two obstacle
  classes, 4 m minimum gaps, 3 m spawn clearance, flood-fill connectivity check.
* `env/raycast.py` - batched ray/box, ray/circle, boundary and segment-occlusion tests.
  All rays of all agents of all envs are cast against all obstacles at once.
* `env/bots.py` - scripted baselines: random, spinner, charger, holder.
* `algo/networks.py` - entity-token encoder (per-type MLPs -> 2-layer transformer) -> GRU
  -> action heads + auxiliary enemy-position head; `algo/augment.py` - mirror augmentation.
* `algo/mappo.py` - MAPPO trainer (centralised critic on the mirrored global state,
  recurrent actor with chunked BPTT, self-play league, role diversity bonus, auxiliary
  loss, mirror augmentation, obstacle-coverage curriculum, optional hierarchical
  commander, tensorboard logging, periodic GIF rollouts).
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
| 3 | Self-play without roles (ablation) | `python scripts/train.py --config configs/no_roles.yaml` |
| 4 | Roles + diversity bonus + role rewards (default) | `python scripts/train.py --config configs/base.yaml` |
| 5 | League + ablations | `configs/ablations/{comms_full,comms_contacts_only,comms_none,no_role_rewards,lane_spawns,framestack,per_slot,no_friendly_fire,discrete,conv_mlp_encoder,no_mirror_no_aux}.yaml` |
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

Visualise how battles change across training stages (same layout and opponent for every
checkpoint; writes per-stage GIFs, movement traces, a contact sheet of battle phases, a
side-by-side GIF and a summary table into `runs/<name>/viz`):

```
python scripts/visualize.py --run runs/roles --stages 4 --opponent charger --seed 3
python scripts/visualize.py --run runs/roles --times 0 5 10 20 40      # fixed-time columns
python scripts/visualize.py --checkpoints a.pt b.pt --opponent self --out viz/
```

Logs go to tensorboard under `runs/<name>`: `tensorboard --logdir runs`.

## Key metrics logged

* `elo/latest`, `elo/<member>` - league Elo (primary progress metric).
* `win_rate/self|pool|bot`, `win_rate/vs_<opponent>`.
* `behaviour/pair_distance` (teammate spread), `time_to_first_contact_s`, `accuracy`,
  `friendly_fire_rate`, `engage_angle_mean_deg`, `flank_fraction` (hits from behind the
  victim's 90 degree line), `crossfire_rate` (enemy hits where a teammate hit the same
  target within 2 s from a bearing >= 45 degrees apart), `coverage_entropy` (entropy of the
  squad's cone coverage over 12 sectors, 1 = evenly spread), `win_rate/squad_<AvB>`.
* `loss/aux_enemy_pos` - auxiliary enemy-position prediction error (drops as the
  recurrent state learns to track enemies).
* `stats/corner_lerp`, `stats/coverage_hi` - the spawn-distance and obstacle-coverage
  curricula (both ramp over the first third of training).
* `roles/<stat>/role<k>` - per-role behavioural statistics (distance to squad, distance to
  enemy, shots, distance travelled, flank fraction, unique spotting, accuracy);
  `disc/accuracy` - how identifiable the roles are from behaviour (role collapse shows up
  as ~1/3).  `scripts/eval.py` prints the same per-role table, and the trace images label
  every agent with its role letter (A/F/O).

## Throughput

The simulation runs at ~1.5k env-steps/s for 128 envs on CPU (vision rays, LOS and the
64-ray map scan are all batched numpy).  The transformer encoder makes the PPO update the
bottleneck on CPU: the full loop runs at ~120 env-steps/s on 4 cores, so the default
3000-update schedule (~100M steps) needs a GPU.  For CPU experiments use
`model.encoder=conv_mlp`, `train.epochs=2` and fewer envs; `train.torch_threads` controls
the CPU thread count.

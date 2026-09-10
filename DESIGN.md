# Design notes, assumptions and spec issues

## Spec v2 changes and the assumptions behind them

* **Body & motion.** Radius 1.0 m, arena 48 m, vision/weapon range 24 m, spawns 30 m
  apart (scaled from 40/64).  The turn command sets a target angular velocity (max 120
  deg/s) and the change per step is capped at 600 deg/s^2.  The (forward, strafe) command
  is normalised to the unit disc and scaled per axis: 4.0 forward, 1.5 backward, 2.0 strafe.
  The firing ray is the cone bisector (tested), it stops at the first wall or body
  (tested), and the renderer draws the full cone dimly under the visible polygon so the
  occluded part reads as a shadow.
* **Obstacles.** Layouts are generated to a target coverage fraction; each box and its
  point mirror are accepted only if every pairwise gap is >= 4 m and the box is >= 3 m
  from the spawn zones; a grid flood fill (obstacles inflated by the agent radius) must
  connect the two spawns or the layout is regenerated with slightly less cover.  Walls
  block vision, shots and movement; low crates block movement only (vision, shots and the
  map scan pass over them - the map scan reports both, as static geometry).  The training
  curriculum raises the upper coverage bound from 6% to 18% over the first 33% of
  training; the lower bound is two thirds of it, so the final range is 12-18%.
* **Team blackboard.** Every agent keeps its own tracks of each enemy slot (persistent id
  = enemy slot); with `comms: full` or `contacts_only` an agent reads, per enemy, the
  freshest track among its squad (dead allies' last reports persist), with `none` only its
  own.  Track features: relative position, relative heading, relative velocity, last-seen
  hp, staleness (capped at 10 s, 1 for never-seen), confidence `exp(-staleness / 3 s)`,
  valid flag, visible-now flag.  Ally slots (position, heading, hp, alive, ammo, reloading)
  are only exposed with `comms: full`.  Confirmed kills are broadcast (track hp -> 0).  The
  64-ray 360-degree map scan is the agent's own sensing and is present in every mode.
  The in-cone LOS shaping reward is removed.
* **Policy.** Tokens: self (proprio + role + previous action + plan), 2 ally slots,
  3 enemy tracks, vision (conv over the 32 rays), map (circular conv over the 64 rays); each
  type has its own MLP and a learned type embedding; a 2-layer, 4-head transformer with
  key-padding for dead allies / never-seen tracks; the self token and the masked mean are
  concatenated and fed to the GRU(256).  An auxiliary head on the recurrent state predicts
  every enemy's true relative position (masked MSE, weight 0.1, ground truth only in the
  buffer).  Mirror augmentation reflects a random half of the sequences about the agent's
  forward axis (rays reversed, lateral coordinates/sines negated, map scan reversed,
  strafe/turn actions negated); the critic keeps the unmirrored global state because
  axis-aligned obstacles cannot be reflected about an arbitrary agent axis, and the stored
  old log-prob is reused for the mirrored sample (standard approximation).
* **Squad sizes.** {3v3, 2v2, 3v2, 2v3} sampled per episode; unused slots never spawn,
  are masked out of every loss, and a team loses when all of its active agents are dead.
* **Metrics.** Crossfire rate = enemy hits where another teammate hit the same victim
  within the previous 2 s from a bearing >= 45 degrees apart.  Angular coverage entropy =
  entropy of the squad's summed cone coverage over 12 sectors of 30 degrees, normalised
  by log 12 and averaged over steps with >= 2 alive teammates.
* **Cost.** The transformer makes the chunked-BPTT update ~6x more expensive than the
  conv-MLP actor; on a 4-core CPU the full loop drops to ~120 env-steps/s.  Use a GPU, or
  the `conv_mlp_encoder` ablation for CPU experiments.

## Corner spawns and role-focused squad play

* **Spawns.** `spawn_mode: corners` (default) puts team A's spawn centre 6 m from a random
  diagonal corner and team B at the point mirror, so squads start ~49 m apart in opposite
  corners and must cross the whole arena; `lanes` restores the 30 m centre-line spawns.
  Obstacle generation keeps the usual 3 m clearance around both spawn zones.
* **Role incentives** (`roles.rewards`, all small compared with the +1 terminal reward):
  assault +0.005/hp for damage dealt from within 10 m; flanker +0.01/hp for hits landing in
  the victim's rear half (engagement angle > 90 degrees); overwatch +0.002 per step per
  enemy that only it sees from >= 12 m, and +0.005/hp for teammates' damage on an enemy
  it currently watches.  Any pair of shooters that produce a crossfire hit both get +0.1.
  These make the roles' optimal behaviours differ by construction; the discriminator
  bonus (now over six statistics, adding flank fraction and unique spotting) keeps them
  distinguishable even where the incentives overlap.  `ablations/no_role_rewards.yaml`
  isolates their effect, `no_roles.yaml` removes roles entirely.
* **Squad sizes** are weighted 70% 3v3 so full-role squads dominate; 2v2 / 3v2 / 2v3 remain
  at 10% each so the policy still handles a missing role.
* **Corner spawns make the search problem much harder, and that changes the reward
  balance.** Squads start ~49 m apart with a 24 m cone in a 48 m arena, so a random policy
  rarely meets the enemy at all.  In a 0.37M-step check every episode timed out, time to
  first contact rose from 6 s to 40 s, and the draw rate hit 94%: with a draw worth 0 and
  friendly fire worth -34 per hit, hiding beat fighting.  Two mitigations are now the
  defaults, and both are config knobs:
  - `reward.draw: -0.3` and `timeout_hp_tiebreak: true`, so a timeout is bad for both
    teams and the healthier squad still wins one.  This cut the draw rate from 94% to 19%.
  - `train.spawn_curriculum: {start: 0.45, frac: 0.33}` interpolates the spawn centres from
    mid-arena out to the true corners over the first third of training (`stats/corner_lerp`
    logs the ramp), so early episodes make contact in ~5 s instead of ~40 s.
  Neither removes the underlying cost: at full corner separation contact still takes ~30 s
  of a 60 s episode.  If contact time stays high late in training, lengthen
  `train.spawn_curriculum.frac`, raise `env.max_steps`, or fall back to
  `ablations/lane_spawns.yaml`.

## Full-arena vision and denser opaque cover

* **Sight is now limited by geometry, not by a range clip.** `vision_range` is 68 m, just
  over the 67.9 m arena diagonal, so every ray terminates on a wall, an agent, or the arena
  boundary.  Measured over random rollouts, 100% of rays end on geometry (84% before) and
  the fraction of live agents with an enemy in view rose from 0.00% to 3.2%.
* **Ray count 32 -> 48.**  A 2 m body at 68 m subtends 1.69 degrees while 32 rays over a
  60 degree cone are 1.94 degrees apart, so distant enemies fell *between* rays and were
  invisible to the vision channel.  48 rays give 1.28 degree spacing, a 1.3x margin, and
  `test_ray_spacing_resolves_a_body_at_maximum_range` pins this.
* **More opaque cover.**  Coverage is 17-22% (was 12-18%) and the low-crate share is down
  to 15%, so most obstacles block sight.  Set `crate_fraction: 0.0` to make every obstacle
  opaque.
* **The 4 m minimum gap caps coverage near 22%.**  With small 2-8 m boxes the generator
  stalls at ~12% no matter how many obstacles it is allowed, because each box needs a 4 m
  clear halo.  Larger 4-11 m blocks reach ~19-22%, and they cut long sightlines far more
  effectively than scattered small boxes, so the size range was raised rather than the gap
  lowered.  Asking for more than ~22% with a 4 m gap is geometrically impossible in a 48 m
  arena; the generator now refuses to overshoot the ceiling instead of exceeding it.
* **Long-range fire is a live balance question.**  Shots have always been unlimited range;
  what changed is that agents can now *see* far enough to aim at distant targets.  A
  stationary, aimed agent hits a 1 m radius body 99.5% of the time at 60 m, so there is no
  range falloff discouraging fire from spawn.  Movement spread is the only counterweight
  (36.7% at 60 m at full speed, 84.8% at 20 m).  `env.weapon_range` (default 68 m, i.e.
  unchanged) lowers the hitscan range independently of sight if "see far, shoot near" is
  wanted; raising `spread_max_deg` is the other lever.

## Results from the 1.5x arena run (2.58M steps, CPU, conv encoder)

600 updates were requested; the run died at update 420 on a league bug (now fixed and
covered by `test_league_snapshot_eviction_keeps_the_pool_usable`: evicting a snapshot from
the pool left environments holding a dangling opponent reference).  2.58M steps of data
survived, and Elo was still climbing at +1.55 per update when it stopped, so these are
mid-training trends, not a converged policy.  Note the run used `model.encoder=conv_mlp`
for CPU throughput, so it says nothing about the entity-token default.

**Combat skill clearly improves.**

| metric | first fifth | last fifth |
|---|---|---|
| Elo | 814 | 1104 |
| win rate vs scripted bots | 0.01 | 0.36 |
| shot accuracy | 0.070 | 0.128 |
| friendly-fire rate | 0.014 | 0.000 |
| mean episode return | -5.92 | +0.43 |

Final checkpoint win rates over 40 episodes: random 1.00, spinner 0.70, charger 0.47,
holder 0.35, self 0.55.  Friendly fire is fully eliminated, which is what the -34 per hit
penalty was for.

**Squad play collapses into a blob, and the roles do not separate at all.**

| metric | first fifth | last fifth | wanted |
|---|---|---|---|
| teammate pair distance | 8.7 m | 4.4 m | larger (splitting) |
| angular coverage entropy | 0.569 | 0.485 | larger (covering angles) |
| mean engagement angle | 85 deg | 38 deg | larger (flanking) |
| flank fraction | 0.49 | 0.18 | larger |
| crossfire rate | 0.000 | 0.003 | larger |
| discriminator accuracy | 0.377 | 0.362 | >> 0.333 (chance) |

Over the last 60 updates the three roles differ by at most 0.011 on *every* logged
statistic (distance to squad centroid, distance to nearest enemy, shots, distance
travelled, flank fraction, spotting, accuracy).  This is role collapse, and the diversity
bonus did not prevent it: with the discriminator at chance, `beta * (log q - log 1/3)` is
approximately zero, so the bonus supplies no gradient exactly when it is most needed.

**Why the blob wins, and what to try.**  Clumping and facing the same way both concentrates
fire and drives friendly fire to zero, which is worth far more than the role bonuses
(0.005-0.01 per hp) under the -1.0 per hp friendly-fire penalty.  Splitting is punished
before it can pay off.  Candidate fixes, roughly in order of expected effect:
1. Cut `reward.friendly_damage` toward -0.1 per hp so spreading out is not the only way to
   avoid catastrophic penalties (this was flagged as a spec risk from the start).
2. Raise `diversity.beta` and warm up the discriminator, or make the bonus per step rather
   than terminal, so it is non-zero before roles differ.
3. Raise the role bonuses and `crossfire_bonus` by an order of magnitude; the crossfire
   bonus never bootstraps because crossfire essentially never happens by chance.
4. Add an explicit anti-clumping term, or make friendly fire a function of proximity.

## Assumptions carried over from v1

* **Frames.** Local frame is x forward, y left.  Team B's world is point-mirrored about the
  arena centre `(x, y) -> (S - x, S - y)`, `theta -> theta + pi` for the critic's global
  state and obstacle encoding, so a single shared network sees the same problem from
  either side.  Actor observations are egocentric so they need no mirroring.
* **Spawns (lanes mode).** Team A's centre is at x = 9, y in [20, 28]; team B is the point mirror.
* **Sightings** use an exact line-of-sight test (enemy inside +/-30 degrees, within range,
  segment not blocked by a wall or another body), independent of the 32 ray samples.
* **Dead agents** stop colliding and occluding, receive zero observations, and stay in the
  trajectory until the episode ends so they are credited for the team's terminal outcome.
  Their post-death steps are masked out of the policy loss but kept in the value loss.
* **Kill credit.** Every teammate who damaged the victim gets the full +0.5 (config flag
  `reward.kill_split` to divide it instead).
* **Damage dealt** counts the full 34 even if the victim had less HP; damage taken is
  clipped to remaining HP.
* **Reloading** starts automatically when the magazine empties.
* **Timeout** is a draw by default; `env.timeout_hp_tiebreak` awards the win to the team
  with more total HP.
* **Diversity bonus** is a trajectory-level term added on the last step:
  `beta * (log q(z|tau) - log(1/3))`, zero-centred so an uninformative discriminator adds
  nothing.  Summary statistics: mean distance to squad centroid, mean distance to nearest
  enemy, shots fired, distance travelled (all normalised).
* **Roles** are a permutation of {assault, flanker, overwatch} per team per episode, so
  every role appears once per squad.
* **Commander (phase 2).** One token per team every 20 steps, sampled from a categorical
  head on the mean GRU hidden state of the alive teammates.  The token is written into the
  observation of the same step (no one-step lag) and into the env so later observations
  carry it.  It is trained with the same clipped PPO objective using the team-mean
  advantage at the decision step; opponent snapshots run their own commander.
* **Self-play data.** Both teams' trajectories are used for learning when the opponent is
  the latest policy; only team A's when the opponent is a snapshot or a bot.  Opponent
  snapshots keep their own hidden states, bots are stateless numpy functions of
  `SquadVecEnv.bot_view`, which only exposes what the agents could observe.
* **Recurrent PPO.** 256-step rollouts are split into 32-step chunks; the GRU hidden state
  is stored at chunk boundaries and re-unrolled during the update with resets at episode
  starts.  Minibatches sample whole (chunk, env) pairs so teammates stay together.

## Spec items that are underdetermined or likely to break training

1. **Friendly-fire penalty of -1.0 per HP point** is -34 per hit, 34x the win reward.
   With return normalisation the running std is dominated by these rare events, shrinking
   every other signal, and early policies learn "never fire" before they learn to aim.  It is
   implemented as specified but exposed as `env.reward.friendly_damage`; -0.1/point is a
   more workable value, or anneal the spread and keep friendly fire off until agents shoot
   competently (`configs/ablations/no_friendly_fire.yaml`).
2. **Timeout = draw with only -0.6 total time pressure** makes mutual camping almost free
   once agents learn that pushing costs HP; expect drawn-out stalemates in mid training.
   `timeout_hp_tiebreak: true` fixes the incentive.  The 1v1 config enables it.
3. **Ray sampling gap.** 32 rays over 60 degrees are 1.9 degrees apart; a 0.4 m body at 30 m
   subtends 1.5 degrees, so distant enemies can be invisible to the ray channel.  Increase
   `num_rays` or the radius if long-range engagements matter.
4. **Kill/damage scale.** A full kill yields 0.34 * 3 + 0.5 = 1.52 dense reward versus 1.0
   for winning; dense terms dominate the terminal one, which is fine early but may reward
   trading damage over winning.  Consider annealing `damage_dealt`.
5. **Return normalisation + terminal-only diversity bonus**: the bonus is tiny relative to
   the FF penalty scale and is scaled by the same running std; if role divergence does not
   show in `disc/accuracy`, raise `beta` or add the bonus per step.
6. **Spread depends on speed at the moment of firing**; strafing at full speed costs 2 degrees,
   which at 20 m is a 0.7 m miss on a 0.4 m target, so run-and-gun is strongly penalised.
7. **Spawn headings are random**, so agents frequently start facing away from the enemy;
   with the v2 blackboard a teammate's sighting is enough to orient, and the LOS shaping
   term is gone.  Watch `time_to_first_contact_s`.
8. **Elo is only updated in games against pool members**, never in pure self-play, so the
   number of Elo-informative games per update is ~30% of episodes.  Bots are members of
   the pool, which anchors the scale.
9. **Commander credit assignment.** The commander's advantage is the team-mean advantage at
   the decision step; a 20-step option-level return would be cleaner.  This is enough to
   test whether the token helps Elo, which is the stated goal.

## Reward summary (per agent)

| Term | Value |
|---|---|
| win / loss / draw | +1 / -1 / 0 |
| damage dealt to enemy | +0.01 per HP |
| damage taken | -0.005 per HP |
| damage dealt to ally | -1.0 per HP (never annealed) |
| enemy kill | +0.5 to every contributor |
| own death | -0.5 |
| per step alive | -0.0005 |
| role diversity | beta * (log q(z\|tau) - log 1/3) at episode end, beta = 0.05 |

## Findings from smoke runs (CPU, 1v1 vs scripted bots, <=1.3M steps)

* The pipeline learns: against the random bot the policy reaches a ~94% win rate within
  0.6M steps (Elo 1000 -> 1160) with accuracy rising from 0 to 5-9%.
* **Aiming vs exploration noise.** The turn channel maps [-1, 1] to +/-180 deg/s, i.e. 9
  degrees per step at full deflection.  A Gaussian policy with std 0.6 therefore jitters the
  aim by ~5 degrees per step, which at 20 m is a 1.7 m error on a 0.4 m target: accuracy
  stays at ~0.5% and the agent only learns to evade.  `model.init_log_std` (default -1.0,
  std 0.37) narrows this; consider a smaller `max_turn_rate` for the policy or a
  state-dependent std if accuracy plateaus.  The discrete variant's turn table was refined
  to {-1, -0.3, -0.1, 0, 0.1, 0.3, 1} for the same reason (a 4.5 degree minimum step
  cannot aim at all).
* The spawn distance (40 m) exceeds the vision range (30 m), so nothing is visible at
  spawn and the agent must move before it can see anything; time to first contact
  stays at ~3 s for bots and ~40 s for a timid learner.  Perfect-aim bots (spinner,
  holder) kill a learner within ~2 s when spawned 20 m apart; start milestone 2 against
  the random bot, then mix in the others via the league's prioritised sampling.

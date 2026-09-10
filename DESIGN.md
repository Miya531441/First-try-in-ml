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

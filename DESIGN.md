# Design notes, assumptions and spec issues

## Assumptions made while implementing the spec

* **Frames.** Local frame is x forward, y left.  Team B's world is point-mirrored about the
  arena centre `(x, y) -> (64 - x, 64 - y)`, `theta -> theta + pi` for the critic's global
  state and obstacle encoding, so a single shared network sees the same problem from
  either side.  Actor observations are egocentric so they need no mirroring.
* **Spawns.** Team A's centre is at x = 12, y in [26, 38]; team B is the point mirror.
  Centres are 40-42 m apart (exactly 40 m only on the arena's centre line).
* **Contacts.** An agent's "last known enemy contact" updates whenever an exact
  line-of-sight test succeeds (enemy inside +/-30 degrees, within 30 m, segment not blocked
  by an obstacle or another body).  Contacts are broadcast to allies with staleness in
  seconds (capped at 10 s and normalised) and a validity flag.
* **Vision channel** uses the 32 rays; the shaping reward and contact logic use the exact
  LOS test, so a far enemy slipping between rays does not break shaping.
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
   this is what the LOS shaping term is for.  Keep it until `time_to_first_contact_s`
   drops.
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
| enemy in cone with LOS | +0.001 per step, annealed to 0 over the first 30% of training |
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
  spawn and the agent must move before the LOS shaping can help; time to first contact
  stays at ~3 s for bots and ~40 s for a timid learner.  Perfect-aim bots (spinner,
  holder) kill a learner within ~2 s when spawned 20 m apart; start milestone 2 against
  the random bot, then mix in the others via the league's prioritised sampling.

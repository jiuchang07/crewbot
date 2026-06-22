# crewbot

This is an autonomous-research setup (in the spirit of autoresearch) whose goal
is to learn to play **The Crew: The Quest for Planet Nine** optimally — i.e. to
maximize the mission **win rate** across the 50-mission campaign, for any hand
deal, with 3 players.

You optimize a single metric: **`win_rate`** (higher is better), the fraction of
missions the trained policy completes. The ground-truth metric lives in
`crew_engine.py` (`evaluate_winrate`) and is FIXED.

## Setup

1. **Agree on a run tag** based on today's date (e.g. `mar8`). Branch
   `crewbot/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b crewbot/<tag>`.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `crew_engine.py` — rules, observation encoding, the fixed `win_rate`
     metric. **Do not modify.**
   - `selfplay.py` — the play-data factory. Treat as fixed infra; you may rerun
     it to refresh data (e.g. with `--policy-ckpt` for the iteration loop).
   - `train.py` — the file you modify. Model, loss, optimizer, training loop.
4. **Verify data exists**: check `~/.cache/crewbot/data` for shards. If empty,
   tell the human to run `python selfplay.py` (default 24h) or a short run for
   testing: `python selfplay.py --hours 0.05 --workers 4`.
5. **Initialize results.tsv** with just the header row.
6. **Confirm and go.**

## Methods: on-policy PPO (primary) vs offline (bootstrap)

- **`ppo.py` — primary, on-policy.** Generates fresh self-play rollouts with the
  *current* policy every iteration (batched on GPU via `vec_engine`) and trains
  with PPO. Data quality rises with the policy, so this is how the bot surpasses
  the heuristic. Uses reward shaping (partial credit per completed task) and a
  difficulty curriculum to cope with the sparse win signal. **This is the file
  the experiment loop edits.**
- **`train.py` — optional, offline.** Behavior-cloning on a frozen heuristic
  dataset from `selfplay.py`. Useful as a fast bootstrap / sanity baseline, but
  its ceiling is the heuristic. Not the main path.

## What you CAN / CANNOT do

**CAN:** edit `ppo.py` freely — architecture, PPO hyperparameters (clip, epochs,
GAE λ/γ, entropy, LR), reward shaping (`SHAPE_COEF`, `WIN_BONUS`), the curriculum
(`START_LEVEL`, `LEVEL_STEP`, `UP_THRESHOLD`), rollout batch size, and the
shared model in `train.py:PolicyValueNet`.

**CANNOT:** modify `crew_engine.py` (the rules and the `evaluate_winrate` metric
are ground truth). Do not change the win condition, the 3-player rules, or the
observation contract. Do not modify `vec_engine.py` without re-running
`python test_vec_consistency.py` to prove it still matches the scalar engine.

**Action space (80):** actions 0–39 play card c; actions 40–79 *communicate*
card c (the real Crew token). A communicate action is legal only while the
player's token is unused and c is their only/highest/lowest of its color; it
reveals the card (type derived from the hand), spends the token, and does NOT
advance the turn. The policy learns which card to signal and when.

## The metric

`train.py` prints a summary ending with:

```
win_rate:      <overall fraction of missions won, 0..1>   <-- maximize this
mission_level: <highest mission cleared at >=50% win rate>
```

`win_rate` is averaged over all 50 missions with uniform difficulty, so it is a
demanding number (the back half of the ladder adds more tasks and ordering
constraints). Track both `win_rate` and the per-band breakdown printed below it.

## Logging results

Log each experiment to `results.tsv` (TAB-separated). Columns:

```
commit	win_rate	mission_level	status	description
```

status is `keep`, `discard`, or `crash`. Keep a change iff `win_rate` improved.

## The experiment loop

LOOP FOREVER:

1. Look at git state (current branch/commit).
2. Edit `ppo.py` with one experimental idea.
3. `git commit`.
4. Run: `python ppo.py --minutes <budget> > run.log 2>&1` (redirect; don't flood
   context).
5. Read results: `grep "^win_rate:\|^mission_level:" run.log`.
6. If empty, the run crashed — `tail -n 50 run.log`, attempt a fix, else skip.
7. Record in `results.tsv` (leave it untracked by git).
8. If `win_rate` improved, keep the commit; else `git reset` back.

Because PPO is on-policy, each run generates its own data — do NOT pre-generate
a fixed dataset for these experiments (that is only for the offline `train.py`
path). Keep the time budget fixed across experiments so runs are comparable.

**Ideas when stuck:** stronger curriculum scheduling, KL-based early stopping,
larger/deeper net, separate policy/value nets, better reward shaping (penalize
failed tasks, reward retaining trick control), inferring teammates' hands from
the communication hints, or adding determinized MCTS at decision time.

**NEVER STOP** once the loop has begun. Don't ask the human whether to continue.
If you run out of ideas: improve the cooperative reasoning (card counting,
inferring teammates' hands from the communication hints), add a curriculum that
ramps mission difficulty, add search (determinized MCTS) at decision time, or
combine previous near-misses. Run until manually interrupted.

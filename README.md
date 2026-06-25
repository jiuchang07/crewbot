# crewbot

A self-play bot that learns to play **The Crew: The Quest for Planet Nine**
(3 players) as well as possible across all 50 campaign missions and any hand
deal. Built on the scaffold of [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
generate lots of play data, then train a model on it to learn the best move, and
(optionally) let an agent iterate on `train.py` to push the win rate.

The idea: spend ~24 hours generating self-play games of The Crew, log every
decision with the eventual win/loss outcome, then train a policy/value network
to imitate-and-improve on the good moves. The bot is evaluated by a single fixed
metric — **`win_rate`**, the fraction of missions completed.

## One model, not fifty

All 50 missions are the *same game* (3-player trick-taking with a trump suit);
they differ only in their **tasks** — which specific cards must be captured, by
whom, and in what order. crewbot encodes the mission's task assignment and
ordering constraints directly into the observation, so a **single network
conditioned on the mission generalizes across all 50 missions and every deal.**
Fifty separate models would throw away all cross-mission transfer and split the
data 50 ways. The mission is an *input*, not a separate model. (See "Why one
model" below.)

## How it works

Three files matter, mirroring autoresearch:

- **`crew_engine.py`** — the fixed rules engine: 40-card deck (4 colors x 1-9 +
  4 trump rockets), 3-player dealing (14/13/13, 13 tricks), trick resolution,
  task/ordering constraints, the observation encoding, a cooperative heuristic,
  and the ground-truth metric `evaluate_winrate`. **Not modified.**
- **`selfplay.py`** — the play-data factory. Runs games across all 50 missions
  for a wall-clock budget (default **24h**), parallelized across CPU cores, and
  writes `(observation, legal-move mask, action, team reward, mission)` shards to
  `~/.cache/crewbot/data`.
- **`train.py`** — the single file you (or an agent) edit. Defines the
  policy/value network, trains it on the shards, and prints `win_rate`. **This is
  the file that gets iterated on.**

For GPU-accelerated generation there is a second, batched path:

- **`vec_engine.py`** — a fully tensorized clone of the rules that steps B games
  in lockstep, so policy inference *and* env transitions batch on the GPU. It is
  proven identical to `crew_engine.py` by **`test_vec_consistency.py`** (same
  legal moves, observations, rewards — run it any time you touch either file).
- **`vec_selfplay.py`** — GPU self-play using `vec_engine`, writing the same
  shard format. Use it for the iteration loop (generate with a checkpoint →
  retrain). For the heuristic *bootstrap*, CPU `selfplay.py` across many cores is
  simpler and faster; `vec_selfplay` wins once a network drives the moves,
  because batched GPU inference replaces slow per-decision CPU inference.

Plus `program.md` (agent instructions) and Slurm jobs: `selfplay.sbatch`
(CPU-only data gen), `train.sbatch` (GPU train), `slurm_run.sbatch` (all-in-one).

## Quick start

```bash
# 1. Install deps (CPU or any CUDA GPU; cuda is auto-detected)
pip install -e .            # or: uv sync

# 2. Smoke test the whole pipeline (~1 min)
python crew_engine.py                              # rules self-test
python selfplay.py --hours 0.05 --workers 4        # a little data
python train.py --minutes 1 --eval-games 30        # train + eval

# 3. The real thing
python selfplay.py --hours 24                      # 24h of play data
python train.py --minutes 120 --eval-games 200     # train, then evaluate
```

The trained model is saved to `~/.cache/crewbot/model.pt`.

## Play with the bot and learn the optimal moves

`play.py` is an interactive coach: **you** take one seat, the trained bot plays
the other two, and on every one of your decisions it shows the bot's analysis so
you can learn *why* a move is good — not just what to play.

```bash
python play.py                          # auto seat (commander), pick a mission
python play.py --mission 12 --seat 0    # specific mission / seat
python play.py --rollouts 0             # instant play, skip the win% estimate
```

For each legal move (including the start-of-game **signalling** decisions) it
prints two numbers:

- **bot instinct** — the policy's probability of choosing that move, and
- **win %** — the team's win probability if you make that move and everyone
  plays the bot's policy from there, measured by Monte-Carlo rollouts of the
  policy *in your exact deal*. The highest win % is starred as the recommendation.

These can disagree, which is the whole point: the rollouts correct the policy's
greedy reflex and show the genuinely best continuation. In-game commands: `h`
(hint), `d` (deeper 200-rollout analysis), `f` (fork explorer, below), `q`
(quit). When you diverge from the bot it tells you the win-probability you
gained or gave up, and recaps every divergence at the end of the mission.

### Fork & explore ("what if I'd played…?")

Press `f` at any of your decisions to open a **fork explorer** — a sandbox that
never touches the real game. Pick a move and watch the line play out (the bot
fills the other seats); `auto` lets the bot finish the line so you see whether it
**WINS ✅ or LOSES ❌**. Then `back` rewinds to try a different move, branching
freely; you can descend through several of your own future decisions and step
back up one at a time. A running table compares every line you've tried. When you
like a line, `commit` plays its first move for real; `done` leaves without
committing. Explorer commands: `#` (try a move), `auto`, `back`, `root` (restart
from the fork point), `commit`, `done`. It loads `~/.cache/crewbot/ppo_model.pt` by
default (`--ckpt` to override); with no checkpoint it falls back to the heuristic
so you can still play.

## Running on GaTech PACE

Recommended: two jobs, so you don't reserve a GPU during the 24h of CPU
data-gen.

```bash
sbatch selfplay.sbatch     # CPU-only, 24h, fills ~/.cache/crewbot/data
sbatch train.sbatch        # 1 GPU, trains + evaluates after data exists
```

Edit the partition/QOS and the environment block (module load vs uv) in each
file first. `slurm_run.sbatch` is an all-in-one alternative if you'd rather one
job. Self-play is CPU-bound (request many `--cpus-per-task`); training uses the
one GPU.

### Why self-play isn't on the GPU

The self-play bottleneck is the *game simulation* (branchy, sequential Python),
not neural-net math — and the heuristic bootstrap has no network at all, so a GPU
has nothing to do. Even checkpoint-guided play is batch-1 inference of a tiny
MLP, where GPU launch/transfer latency beats CPU. So CPU cores are the right tool
for data-gen, and the GPU is for training. If you *do* want the GPU busy during
generation, use `vec_selfplay.py`, which batches thousands of games so each step
is one big GPU forward pass — that's the only way a GPU helps here.

## Training: on-policy PPO (primary)

The main trainer is **`ppo.py`** — on-policy PPO self-play. Every iteration it
generates fresh rollouts with the *current* policy (batched on the GPU via
`vec_engine`) and updates with the clipped PPO objective. Generation and training
are one loop, so there's **no separate data-gen step and no fixed dataset** — the
data improves as the policy does, which is how the bot surpasses the heuristic.

```bash
python ppo.py --minutes 30          # local
sbatch ppo.sbatch                   # PACE: one GPU, full budget
```

Design for The Crew's cooperative, sparse-reward setting: one shared policy plays
all 3 seats; **reward shaping** gives partial credit per completed task; a
**difficulty curriculum** trains on missions `1..level` and raises `level` as the
win rate clears a threshold. In an 18-min local run this took `win_rate`
0.055 → 0.159 and `mission_level` 3 → 10, still climbing.

`train.py` (offline behavior-cloning on a frozen `selfplay.py` dataset) remains
as an optional fast bootstrap, but its ceiling is the heuristic.

## The learning approach (offline path)

- **Reward**: shared team outcome — `1.0` if the mission is completed, else `0.0`
  (cooperative game, so all players share it).
- **Training** (`train.py`): advantage-weighted policy gradient with a value
  baseline — the value head predicts win probability, and the policy is pushed
  toward actions whose games beat that baseline. This is essentially "imitate the
  moves from games that did better than expected," and it learns from both wins
  and losses.
- **Iteration loop (the big lever)**: bootstrap data comes from a cooperative
  heuristic. Once you have a decent model, regenerate data *with that model* and
  retrain — AlphaZero-style self-improvement:

  ```bash
  python selfplay.py --hours 24                                   # 1. bootstrap
  python train.py --minutes 120                                  # 2. train
  python vec_selfplay.py --hours 6 \
      --policy-ckpt ~/.cache/crewbot/model.pt --batch 8192        # 3. GPU regen
  python train.py --minutes 120                                  # 4. retrain ... loop
  ```

## Why one model (answering "should I train 50 models?")

| | One mission-conditioned model | 50 separate models |
|---|---|---|
| Data efficiency | All games train one net; skills transfer | Data split 50 ways; mission 50 starves |
| Generalization | Handles any task set / deal it's given | Each net overfits its mission |
| New/variant missions | Just encode the new constraints | Train a whole new model |
| Maintenance | One checkpoint | 50 checkpoints |

The only reason to split would be a mission whose rule can't be expressed in the
observation. The Crew's missions all reduce to *which cards are tasks, who must
win them, and in what order* — all encoded already — so one model is strictly
better.

## Scope / simplifications

- Fixed at **3 players** (official 14/13/13 deal, 13 tricks, one leftover card).
- **Communication** is a learned action in a setup round before play: in turn
  order from the leader, each player reveals one card (its only/highest/lowest of
  a color) or passes — the policy decides what to signal, faithfully to The Crew
  (truthful, one card per player, no trumps, no mid-trick signalling). Actions
  0–39 play, 40–79 communicate, 80 = pass. (Set `new_game(..., use_comm=False)`
  to ablate communication entirely.)
- The 50-mission ladder is modeled as an increasing number of randomly-drawn
  task cards plus (back half) required completion ordering. A few campaign
  missions have special objective tokens (e.g. "win exactly N tricks"); these can
  be added as new constraint types in `crew_engine.py` — the observation/metric
  contract already supports extra task features.

## License

MIT (inherited from autoresearch).

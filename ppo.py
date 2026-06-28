"""
On-policy PPO trainer for crewbot (the primary training method).

Supports three robustness strategies (composable):
  - Partner Noise: randomly replace some actions with random legal moves
  - Other-Play: apply random color permutations to break conventions
  - Fictitious Co-Play (FCP): learner trains against random past checkpoints
    so it adapts to diverse partner behavior, not just current-self play

Unlike offline train.py (behavior-cloning on a frozen heuristic dataset), PPO
generates FRESH self-play data with the *current* policy every iteration, so the
data quality rises with the policy — the only way to surpass the bootstrap
heuristic. Generation and training are one loop, both batched on the GPU via
vec_engine.

Key design for The Crew (cooperative, sparse reward):
- A single shared policy controls all 3 seats (parameter sharing); the team
  win/loss is the shared reward.
- Reward shaping: partial credit (SHAPE_COEF / n_tasks) each time a task is
  completed, plus WIN_BONUS on full mission success — otherwise the win signal
  is far too sparse for the harder missions.
- Curriculum: train on missions [1 .. level]; raise `level` as the rolling win
  rate clears UP_THRESHOLD. Focuses learning where it is currently achievable.

The metric is the same fixed `win_rate` from crew_engine (greedy, all 50
missions). This is the file the experiment loop edits.

Usage:
    python ppo.py --minutes 30
    python ppo.py --iters 50 --batch 4096 --eval-games 40     # quick run
"""

import copy
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crew_engine as E
from train import PolicyValueNet
from vec_engine import VecCrew, N_P, TOTAL_TRICKS

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

BATCH = 4096               # games per rollout (each ~39 plies of decisions)
PPO_EPOCHS = 4
MINIBATCH = 16384
CLIP = 0.2
GAMMA = 0.99
LAM = 0.95
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02
GRAD_CLIP = 1.0

# reward shaping + curriculum
SHAPE_COEF = 0.5           # total partial credit available for completing tasks
WIN_BONUS = 1.0            # terminal reward for full mission success
START_LEVEL = 3            # curriculum starts on missions 1..START_LEVEL
LEVEL_STEP = 2
# Raise difficulty when the rolling (stochastic) win rate clears this. Kept well
# below 1.0 because random task assignment caps achievable win rate even on easy
# missions, so a high bar would stall the curriculum forever.
UP_THRESHOLD = 0.50

# Robustness to human play
PARTNER_NOISE = 0.05       # prob of replacing a move with a random legal action
OTHER_PLAY = True          # apply random color permutations to break conventions

# Fictitious Co-Play (FCP)
FCP_POOL_SIZE = 16         # max number of past checkpoints retained
FCP_SAVE_EVERY = 10        # save a checkpoint to the pool every N iterations
FCP_ENABLED = False        # off by default; enable via --fcp

MAX_PLIES = E.MAX_PLIES   # plays + communications (comm actions don't advance turn)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_ppo_model(hidden=None, n_blocks=None):
    kw = {}
    if hidden is not None:
        kw["hidden"] = hidden
    if n_blocks is not None:
        kw["n_blocks"] = n_blocks
    return PolicyValueNet(bounded_value=False, **kw)


def _random_color_perms(batch, rng, device):
    """Generate [batch, 4] random permutations of {0,1,2,3} for Other-Play."""
    perms = np.stack([rng.permutation(E.N_COLORS) for _ in range(batch)])
    return torch.from_numpy(perms).long().to(device)


# ---------------------------------------------------------------------------
# Checkpoint Pool for Fictitious Co-Play
# ---------------------------------------------------------------------------

class CheckpointPool:
    """FIFO pool of past model weight snapshots for Fictitious Co-Play.

    During FCP rollouts, partner seats use a randomly sampled past checkpoint
    instead of the current policy, so the learner trains against a *diverse*
    population of past selves — not just the latest version.
    """

    def __init__(self, max_size=FCP_POOL_SIZE):
        self.max_size = max_size
        self.snapshots = []          # list of OrderedDict (CPU state_dicts)

    def __len__(self):
        return len(self.snapshots)

    def save(self, model):
        """Snapshot the current model weights (moved to CPU) into the pool."""
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.snapshots.append(sd)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)     # FIFO eviction

    def sample(self, rng):
        """Return a random state_dict from the pool."""
        idx = int(rng.integers(len(self.snapshots)))
        return self.snapshots[idx]

    def state_for_save(self):
        """Serialize pool for checkpoint resume."""
        return self.snapshots[:]

    def load_from_save(self, snapshots):
        """Restore pool from a checkpoint."""
        self.snapshots = snapshots[:self.max_size]


def _build_partner_model(model_template, state_dict, device):
    """Build a partner model with the same architecture, loaded with given weights."""
    partner = copy.deepcopy(model_template)
    partner.load_state_dict(state_dict)
    partner.to(device)
    partner.eval()
    return partner


# --- Other-Play utility functions ---------------------------------------------------
# The following helpers remap actions, legal masks, and observations through a
# card permutation. They are NOT used in the main rollout loop (which permutes
# the VecCrew game state directly so that observe/legal_mask/step all operate in
# the permuted space), but are available for offline data augmentation or
# evaluation with permuted inference.

def _permute_actions(action, perm):
    """Remap action indices through a card permutation table.

    Actions 0..39 (play) and 40..79 (communicate) reference card indices.
    Actions 80 (pass) and 81..116 (claim 0..35) also reference card indices
    for claims. All are remapped through perm[B, N_C]."""
    B = action.shape[0]
    # Play actions: 0..39 → perm[action]
    is_play = action < E.N_CARDS
    # Comm actions: 40..79 → 40 + perm[action - 40]
    is_comm = (action >= E.COMM_OFFSET) & (action < E.PASS_ACTION)
    # Pass: 80 → unchanged
    # Claim actions: 81..116 → 81 + perm[action - 81] (only non-trump, 0..35)
    is_claim = action >= E.CLAIM_OFFSET

    result = action.clone()
    # Remap play
    if is_play.any():
        idx = is_play.nonzero(as_tuple=True)[0]
        result[idx] = perm[idx].gather(1, action[idx].unsqueeze(1)).squeeze(1)
    # Remap comm
    if is_comm.any():
        idx = is_comm.nonzero(as_tuple=True)[0]
        card = action[idx] - E.COMM_OFFSET
        result[idx] = E.COMM_OFFSET + perm[idx].gather(1, card.unsqueeze(1)).squeeze(1)
    # Remap claim
    if is_claim.any():
        idx = is_claim.nonzero(as_tuple=True)[0]
        card = action[idx] - E.CLAIM_OFFSET
        result[idx] = E.CLAIM_OFFSET + perm[idx].gather(1, card.unsqueeze(1)).squeeze(1)
    return result


def _permute_legal_mask(legal, perm):
    """Remap a legal mask [B, ACT_DIM] through a card permutation.

    The mask has 3 card-indexed regions: play (0..39), comm (40..79), claim
    (81..116). Each is gathered through the perm table. Pass (80) is unchanged."""
    out = torch.zeros_like(legal)
    # Play region: out[:, perm[c]] = legal[:, c]  →  out[:, j] = legal[:, inv[j]]
    # But it's easier to scatter: for each old slot c, put its value at perm[c].
    out[:, :E.N_CARDS].scatter_(1, perm, legal[:, :E.N_CARDS])
    # Comm region: same permutation, offset by N_CARDS
    out[:, E.COMM_OFFSET:E.COMM_OFFSET + E.N_CARDS].scatter_(
        1, perm, legal[:, E.COMM_OFFSET:E.COMM_OFFSET + E.N_CARDS])
    # Pass: unchanged
    out[:, E.PASS_ACTION] = legal[:, E.PASS_ACTION]
    # Claim region: only non-trump cards (0..35)
    perm_tc = perm[:, :E.N_TASK_CARDS]
    out[:, E.CLAIM_OFFSET:E.CLAIM_OFFSET + E.N_TASK_CARDS].scatter_(
        1, perm_tc, legal[:, E.CLAIM_OFFSET:E.CLAIM_OFFSET + E.N_TASK_CARDS])
    return out


def _permute_obs(obs, perm, inv):
    """Remap a [B, OBS_DIM] observation through a card permutation.

    The observation is a concatenation of blocks, most of which are [N_C]-sized
    card-indexed vectors. We remap those via gather(inv). The led-color one-hot
    (5-dim) and scalar blocks are handled separately."""
    B = obs.shape[0]
    out = obs.clone()
    NC = E.N_CARDS       # 40
    NCOL = E.N_COLORS     # 4

    # Blocks 1-4: hand, captured, table, winning card — each [N_C], card-indexed
    for start in [0, NC, 2*NC, 3*NC]:
        out[:, start:start+NC] = obs[:, start:start+NC].gather(1, inv)

    # Block 5: led color one-hot [N_COLORS+1] — permute the first 4 entries
    led_start = 4 * NC
    led_end = led_start + NCOL + 1
    # The 5th entry (trump) stays put; permute the 4 color entries
    led_block = obs[:, led_start:led_end].clone()
    color_part = led_block[:, :NCOL].clone()
    # inv_color: for each new color slot j, read from old color slot inv_color[j]
    # color_perms maps old→new, so inv maps new→old
    # We need the color-level inverse. perm table: perm[:, c*9] // 9 gives
    # the new color for old color c. We can derive inv_color from inv table.
    inv_color = inv[:, ::E.RANKS_PER_COLOR][:, :NCOL] // E.RANKS_PER_COLOR  # [B, 4]
    led_block[:, :NCOL] = color_part.gather(1, inv_color)
    out[:, led_start:led_end] = led_block

    # Blocks 6-12: tasks mine/other/done, pool, unclaimed, blocked, ready — each [N_C]
    off = led_end
    for _ in range(7):  # mine, other, done, pool, unclaimed, blocked, ready
        out[:, off:off+NC] = obs[:, off:off+NC].gather(1, inv)
        off += NC

    # Block 11: communication — 3 players × (card_oh[N_C] + type_oh[3] + valid[1])
    for _ in range(E.N_PLAYERS):
        # card one-hot [N_C]: card-indexed
        out[:, off:off+NC] = obs[:, off:off+NC].gather(1, inv)
        off += NC
        # type one-hot [3] + valid [1]: not card-indexed, leave as-is
        off += 3 + 1

    # Block 12: relative hand sizes [N_P] — not card-indexed
    # Block 13: scalars [5] — not card-indexed
    # Both are already correct in `out` (cloned from obs)

    return out


@torch.no_grad()
def rollout(model, rng, level, device, batch, solvable_only=False,
           partner_noise=PARTNER_NOISE, other_play=OTHER_PLAY):
    """Collect `batch` full self-play episodes under the current policy.
    Returns flat tensors over active decisions plus the rollout win rate.

    If partner_noise > 0, a random legal action replaces the policy's choice
    with that probability — training the policy to recover from mistakes.
    If other_play is True, a random color permutation is applied to each game
    so the policy cannot develop color-specific conventions."""
    pool = [m for m in E.TASK_MISSIONS if m <= level] or [E.TASK_MISSIONS[0]]
    mids = rng.choice(pool, size=batch)
    vec = VecCrew.new_games(batch, mids, rng, device=device, solvable_only=solvable_only)

    # --- Other-Play: permute colors in the game state ---
    perm = inv = None
    if other_play:
        color_perms = _random_color_perms(batch, rng, device)
        vec.permute_colors(color_perms)
        perm, inv = VecCrew.build_perm_table(color_perms, device=device)

    n_tasks = vec.is_task.sum(dim=1).clamp(min=1).float()   # fixed pool size per game

    O, A, LP, V, LE, R, AC, TM = [], [], [], [], [], [], [], []
    for _ in range(MAX_PLIES):
        active = ~vec.done
        obs = vec.observe()
        legal = vec.legal_mask()
        logits, value = model(obs)
        logits = logits.masked_fill(~legal, float("-inf"))
        logp_all = F.log_softmax(logits, dim=-1)
        probs = logp_all.exp()
        action = torch.multinomial(probs, 1).squeeze(1)

        # --- Partner Noise: replace some actions with random legal moves ---
        if partner_noise > 0:
            noise_mask = (torch.rand(batch, device=device) < partner_noise) & active
            if noise_mask.any():
                # Uniform over legal actions for noised games
                legal_counts = legal.sum(dim=-1, keepdim=True).clamp(min=1)
                uniform_probs = legal.float() / legal_counts
                random_action = torch.multinomial(uniform_probs, 1).squeeze(1)
                action = torch.where(noise_mask, random_action, action)

        # Log-prob of the action actually taken under the *current* policy.
        # For noised actions this may be low, which is correct: PPO's
        # importance ratio will naturally down-weight off-policy samples.
        logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1)

        ndone_before = vec.done_tasks.sum(dim=1)
        vec.step(action)
        ndone_after = vec.done_tasks.sum(dim=1)
        gained = (ndone_after - ndone_before).clamp(min=0).float()
        just_done = active & vec.done
        reward = SHAPE_COEF * gained / n_tasks + WIN_BONUS * (just_done & vec.success).float()

        O.append(obs); A.append(action); LP.append(logp); V.append(value)
        LE.append(legal); R.append(reward); AC.append(active); TM.append(just_done)
        # NB: no per-ply .all() sync here — that GPU→CPU stall dominated runtime.
        # Stepping already-done games is a cheap masked no-op, so just run the
        # full fixed horizon (every game finishes within MAX_PLIES anyway).

    T = len(O)
    obs = torch.stack(O); act = torch.stack(A); logp = torch.stack(LP)
    val = torch.stack(V); legal = torch.stack(LE); rew = torch.stack(R)
    active = torch.stack(AC); term = torch.stack(TM)

    # GAE over [T, BATCH] with per-game terminal masks
    adv = torch.zeros_like(rew)
    lastgae = torch.zeros(batch, device=device)
    for t in reversed(range(T)):
        nextval = val[t + 1] if t + 1 < T else torch.zeros(batch, device=device)
        nonterm = 1.0 - term[t].float()
        delta = rew[t] + GAMMA * nonterm * nextval - val[t]
        lastgae = delta + GAMMA * LAM * nonterm * lastgae
        m = active[t].float()
        adv[t] = lastgae * m
        lastgae = lastgae * m              # reset across episode boundary
    ret = adv + val

    flat = active.reshape(-1)
    keep = flat.nonzero(as_tuple=True)[0]
    data = dict(
        obs=obs.reshape(-1, E.OBS_DIM)[keep],
        act=act.reshape(-1)[keep],
        logp=logp.reshape(-1)[keep],
        legal=legal.reshape(-1, E.ACT_DIM)[keep],
        adv=adv.reshape(-1)[keep],
        ret=ret.reshape(-1)[keep],
    )
    win_rate = float(vec.success.float().mean().item())
    return data, win_rate


@torch.no_grad()
def rollout_fcp(model, partner_model, rng, level, device, batch,
               solvable_only=False, partner_noise=PARTNER_NOISE,
               other_play=OTHER_PLAY):
    """Fictitious Co-Play rollout.

    For each game in the batch:
    - One randomly chosen seat (the "learner") uses `model` (current policy)
    - The other two seats use `partner_model` (a past checkpoint)

    Only the learner's decisions are kept for the PPO update (the partner's
    actions still affect the trajectory and rewards, but are masked out of the
    training data). GAE is computed from the learner's value estimates over
    the full trajectory, so the value function learns to predict returns
    *conditioned on having these partners*.
    """
    pool = [m for m in E.TASK_MISSIONS if m <= level] or [E.TASK_MISSIONS[0]]
    mids = rng.choice(pool, size=batch)
    vec = VecCrew.new_games(batch, mids, rng, device=device, solvable_only=solvable_only)

    # --- Other-Play: permute colors in the game state ---
    if other_play:
        color_perms = _random_color_perms(batch, rng, device)
        vec.permute_colors(color_perms)

    n_tasks = vec.is_task.sum(dim=1).clamp(min=1).float()

    # Assign one random "learner seat" per game: 0, 1, or 2
    learner_seat = torch.from_numpy(
        rng.integers(0, N_P, size=batch).astype(np.int64)
    ).to(device)  # [B]

    O, A, LP, V, LE, R, AC, TM, IS_LEARNER = ([] for _ in range(9))

    for _ in range(MAX_PLIES):
        active = ~vec.done
        obs = vec.observe()
        legal = vec.legal_mask()

        # Determine which games have the learner acting this ply
        current_seat = vec.turn  # [B]
        is_learner_ply = (current_seat == learner_seat)  # [B] bool

        # --- Learner inference (current policy) ---
        logits_l, value_l = model(obs)
        logits_l = logits_l.masked_fill(~legal, float("-inf"))
        logp_all_l = F.log_softmax(logits_l, dim=-1)
        probs_l = logp_all_l.exp()

        # --- Partner inference (past checkpoint) ---
        logits_p, _ = partner_model(obs)
        logits_p = logits_p.masked_fill(~legal, float("-inf"))
        probs_p = F.softmax(logits_p, dim=-1)

        # Sample actions: learner seats use learner probs, partner seats use partner probs
        # We sample from both and select per-game
        action_l = torch.multinomial(probs_l, 1).squeeze(1)
        action_p = torch.multinomial(probs_p, 1).squeeze(1)
        action = torch.where(is_learner_ply, action_l, action_p)

        # --- Partner Noise: apply to partner seats only ---
        # (The learner should see noisy partners, but the learner's own actions are clean)
        if partner_noise > 0:
            noise_mask = (torch.rand(batch, device=device) < partner_noise) & active & (~is_learner_ply)
            if noise_mask.any():
                legal_counts = legal.sum(dim=-1, keepdim=True).clamp(min=1)
                uniform_probs = legal.float() / legal_counts
                random_action = torch.multinomial(uniform_probs, 1).squeeze(1)
                action = torch.where(noise_mask, random_action, action)

        # Log-prob under the learner's policy (used for PPO importance ratios)
        logp = logp_all_l.gather(1, action.unsqueeze(1)).squeeze(1)
        # Value from the learner's value head (used for GAE)
        value = value_l

        ndone_before = vec.done_tasks.sum(dim=1)
        vec.step(action)
        ndone_after = vec.done_tasks.sum(dim=1)
        gained = (ndone_after - ndone_before).clamp(min=0).float()
        just_done = active & vec.done
        reward = SHAPE_COEF * gained / n_tasks + WIN_BONUS * (just_done & vec.success).float()

        O.append(obs); A.append(action); LP.append(logp); V.append(value)
        LE.append(legal); R.append(reward); AC.append(active); TM.append(just_done)
        IS_LEARNER.append(is_learner_ply)

    T = len(O)
    obs = torch.stack(O); act = torch.stack(A); logp = torch.stack(LP)
    val = torch.stack(V); legal = torch.stack(LE); rew = torch.stack(R)
    active = torch.stack(AC); term = torch.stack(TM)
    is_learner = torch.stack(IS_LEARNER)  # [T, B]

    # GAE over [T, BATCH] — propagates through ALL plies (including partner
    # actions) so the value function learns returns that account for partner
    # behavior. But only learner plies contribute training samples.
    adv = torch.zeros_like(rew)
    lastgae = torch.zeros(batch, device=device)
    for t in reversed(range(T)):
        nextval = val[t + 1] if t + 1 < T else torch.zeros(batch, device=device)
        nonterm = 1.0 - term[t].float()
        delta = rew[t] + GAMMA * nonterm * nextval - val[t]
        lastgae = delta + GAMMA * LAM * nonterm * lastgae
        m = active[t].float()
        adv[t] = lastgae * m
        lastgae = lastgae * m
    ret = adv + val

    # Keep only active AND learner decisions for PPO update
    learner_active = (active & is_learner).reshape(-1)
    keep = learner_active.nonzero(as_tuple=True)[0]
    data = dict(
        obs=obs.reshape(-1, E.OBS_DIM)[keep],
        act=act.reshape(-1)[keep],
        logp=logp.reshape(-1)[keep],
        legal=legal.reshape(-1, E.ACT_DIM)[keep],
        adv=adv.reshape(-1)[keep],
        ret=ret.reshape(-1)[keep],
    )
    win_rate = float(vec.success.float().mean().item())
    return data, win_rate


def ppo_update(model, opt, data):
    n = data["obs"].shape[0]
    adv = data["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"pol": 0.0, "val": 0.0, "ent": 0.0, "kl": 0.0, "nb": 0}
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n, device=data["obs"].device)
        for s in range(0, n, MINIBATCH):
            mb = perm[s:s + MINIBATCH]
            obs, act = data["obs"][mb], data["act"][mb]
            legal = data["legal"][mb].bool()
            old_logp, mb_adv, mb_ret = data["logp"][mb], adv[mb], data["ret"][mb]

            logits, value = model(obs)
            logits = logits.masked_fill(~legal, float("-inf"))
            logp_all = F.log_softmax(logits, dim=-1)
            new_logp = logp_all.gather(1, act.unsqueeze(1)).squeeze(1)
            probs = logp_all.exp()
            safe = logp_all.masked_fill(~legal, 0.0)
            entropy = -(probs * safe).sum(dim=-1).mean()

            ratio = (new_logp - old_logp).exp()
            s1 = ratio * mb_adv
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * mb_adv
            pol_loss = -torch.min(s1, s2).mean()
            val_loss = F.mse_loss(value, mb_ret)
            loss = pol_loss + VALUE_COEF * val_loss - ENTROPY_COEF * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            with torch.no_grad():
                stats["kl"] += (old_logp - new_logp).mean().item()
            stats["pol"] += pol_loss.item(); stats["val"] += val_loss.item()
            stats["ent"] += entropy.item(); stats["nb"] += 1
    nb = max(stats["nb"], 1)
    return {k: (v / nb if k != "nb" else v) for k, v in stats.items()}


@torch.no_grad()
def vec_evaluate(model, games_per_mission, device, seed=12345, solvable_only=False):
    """Fast batched greedy win-rate over all 50 missions, on the vectorized
    engine (proven identical to the scalar metric by test_vec_consistency.py).
    Replaces the slow per-decision scalar eval — seconds instead of minutes."""
    model.eval()
    rng = np.random.default_rng(seed)
    mids = np.repeat(np.array(E.TASK_MISSIONS), games_per_mission)   # in-scope only
    vec = VecCrew.new_games(len(mids), mids, rng, device=device, solvable_only=solvable_only)
    for _ in range(MAX_PLIES):
        logits, _ = model(vec.observe())
        logits = logits.masked_fill(~vec.legal_mask(), float("-inf"))
        vec.step(logits.argmax(dim=-1))
    succ = vec.success.cpu().numpy()
    per = {m: float(succ[mids == m].mean()) for m in E.TASK_MISSIONS}
    model.train()
    return float(succ.mean()), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=0, help="if >0, overrides --minutes")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=25, help="iters between resume checkpoints")
    ap.add_argument("--resume", action="store_true", help="resume from --state if it exists")
    ap.add_argument("--out", type=str, default=os.path.join(E.CACHE_DIR, "ppo_model.pt"))
    ap.add_argument("--state", type=str, default=os.path.join(E.CACHE_DIR, "ppo_state.pt"))
    ap.add_argument("--solvable", action="store_true",
                    help="train+eval on constructed solvable missions only")
    ap.add_argument("--hidden", type=int, default=None, help="override model width")
    ap.add_argument("--n-blocks", type=int, default=None, help="override #residual blocks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--partner-noise", type=float, default=PARTNER_NOISE,
                    help="probability of replacing a move with a random legal action")
    ap.add_argument("--no-other-play", action="store_true",
                    help="disable Other-Play color permutations")
    # Fictitious Co-Play flags
    ap.add_argument("--fcp", action="store_true", default=FCP_ENABLED,
                    help="enable Fictitious Co-Play (train against past checkpoints)")
    ap.add_argument("--fcp-pool-size", type=int, default=FCP_POOL_SIZE,
                    help="max number of past checkpoints in the FCP pool")
    ap.add_argument("--fcp-save-every", type=int, default=FCP_SAVE_EVERY,
                    help="save a checkpoint to the FCP pool every N iterations")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    mode = "PPO + FCP" if args.fcp else "PPO on-policy self-play"
    print(f"Device: {DEVICE}  |  {mode}")

    model = build_ppo_model(args.hidden, args.n_blocks).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    import train as _t
    print(f"Model params: {n_params:,}  (hidden={args.hidden or _t.HIDDEN}, blocks={args.n_blocks or _t.N_BLOCKS})")

    # FCP checkpoint pool
    ckpt_pool = CheckpointPool(max_size=args.fcp_pool_size)
    # Seed the pool with the initial (random) policy so FCP can start immediately
    if args.fcp:
        ckpt_pool.save(model)
        print(f"FCP enabled: pool_size={args.fcp_pool_size}, save_every={args.fcp_save_every}")

    level, roll_wr, best_eval, it = START_LEVEL, 0.0, -1.0, 0

    # Resume from a checkpoint if asked (essential on preemptible/backfill GPUs).
    if args.resume and os.path.exists(args.state):
        ck = torch.load(args.state, map_location=DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        level, roll_wr, best_eval, it = ck["level"], ck["roll_wr"], ck["best_eval"], ck["it"]
        torch.set_rng_state(ck["torch_rng"].cpu().to(torch.uint8))  # must be CPU ByteTensor
        rng.bit_generator.state = ck["np_rng"]
        # Restore FCP pool if present
        if "fcp_pool" in ck and args.fcp:
            ckpt_pool.load_from_save(ck["fcp_pool"])
            print(f"  FCP pool restored: {len(ckpt_pool)} checkpoints")
        print(f"RESUMED from {args.state}: it={it}, level={level}, best_eval={best_eval:.4f}")

    os.makedirs(E.CACHE_DIR, exist_ok=True)

    def save_state():
        tmp = args.state + ".tmp"
        state_dict = {
            "model": model.state_dict(), "opt": opt.state_dict(),
            "level": level, "roll_wr": roll_wr, "best_eval": best_eval, "it": it,
            "torch_rng": torch.get_rng_state(), "np_rng": rng.bit_generator.state,
        }
        if args.fcp:
            state_dict["fcp_pool"] = ckpt_pool.state_for_save()
        torch.save(state_dict, tmp)
        os.replace(tmp, args.state)   # atomic: a crash mid-write can't corrupt it

    t0 = time.time()
    while True:
        if args.iters > 0:
            if it >= args.iters:
                break
        elif time.time() - t0 >= args.minutes * 60:
            break

        # --- Choose rollout mode: FCP (with past checkpoint) or standard self-play ---
        use_fcp = args.fcp and len(ckpt_pool) > 0
        if use_fcp:
            partner_sd = ckpt_pool.sample(rng)
            partner = _build_partner_model(model, partner_sd, DEVICE)
            data, wr = rollout_fcp(model, partner, rng, level, DEVICE, args.batch,
                                   solvable_only=args.solvable,
                                   partner_noise=args.partner_noise,
                                   other_play=not args.no_other_play)
            del partner  # free GPU memory
        else:
            data, wr = rollout(model, rng, level, DEVICE, args.batch,
                              solvable_only=args.solvable,
                              partner_noise=args.partner_noise,
                              other_play=not args.no_other_play)
        st = ppo_update(model, opt, data)

        # --- Save checkpoint to FCP pool periodically ---
        if args.fcp and it > 0 and it % args.fcp_save_every == 0:
            ckpt_pool.save(model)
        roll_wr = 0.9 * roll_wr + 0.1 * wr if it else wr

        # curriculum: advance difficulty when we are winning the current band
        if roll_wr > UP_THRESHOLD and level < 50:
            level = min(50, level + LEVEL_STEP)
            roll_wr = 0.0

        if it % 5 == 0:
            fcp_tag = f" | pool {len(ckpt_pool)}" if args.fcp else ""
            print(f"it {it:04d} | level {level:2d} | roll_wr {roll_wr:.3f} | "
                  f"pol {st['pol']:+.3f} | val {st['val']:.3f} | ent {st['ent']:.3f} | "
                  f"kl {st['kl']:+.4f}{fcp_tag} | {(time.time()-t0)/60:.1f}m", flush=True)

        if args.eval_every and it > 0 and it % args.eval_every == 0:
            overall, per = vec_evaluate(model, args.eval_games, DEVICE, solvable_only=args.solvable)
            cleared = [m for m in E.TASK_MISSIONS if per[m] >= 0.5]
            print(f"  [eval] win_rate {overall:.4f} | mission_level "
                  f"{max(cleared) if cleared else 0} | level {level}", flush=True)
            if overall > best_eval:
                best_eval = overall
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save(model.state_dict(), args.out)
        if args.ckpt_every and it > 0 and it % args.ckpt_every == 0:
            save_state()
        it += 1

    save_state()
    # final eval (the reported metric)
    overall, per = vec_evaluate(model, args.eval_games, DEVICE, solvable_only=args.solvable)
    cleared = [m for m in E.TASK_MISSIONS if per[m] >= 0.5]
    if overall > best_eval:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save(model.state_dict(), args.out)
    print("---")
    print(f"win_rate:         {overall:.6f}")
    print(f"mission_level:    {max(cleared) if cleared else 0}")
    print(f"curriculum_level: {level}")
    print(f"iterations:       {it}")
    print(f"minutes:          {(time.time()-t0)/60:.1f}")
    print(f"num_params_M:     {n_params/1e6:.3f}")
    bands = {"m01-10": range(1, 11), "m11-20": range(11, 21), "m21-30": range(21, 31),
             "m31-40": range(31, 41), "m41-50": range(41, 51)}
    for name, r in bands.items():
        vals = [per[m] for m in r if m in per]      # in-scope task missions only
        if vals:
            print(f"  {name}: {np.mean(vals):.3f}")


if __name__ == "__main__":
    main()

"""
On-policy PPO trainer for crewbot (the primary training method).

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

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crew_engine as E
from train import PolicyValueNet, make_greedy_fn, evaluate
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

MAX_PLIES = TOTAL_TRICKS * N_P

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_ppo_model():
    return PolicyValueNet(bounded_value=False)


@torch.no_grad()
def rollout(model, rng, level, device, batch):
    """Collect `batch` full self-play episodes under the current policy.
    Returns flat tensors over active decisions plus the rollout win rate."""
    mids = rng.integers(1, level + 1, size=batch)
    vec = VecCrew.new_games(batch, mids, rng, device=device)
    n_tasks = (vec.assigned >= 0).sum(dim=1).clamp(min=1).float()

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
        logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1)

        ndone_before = vec.done_tasks.sum(dim=1)
        vec.step(action)
        ndone_after = vec.done_tasks.sum(dim=1)
        gained = (ndone_after - ndone_before).clamp(min=0).float()
        just_done = active & vec.done
        reward = SHAPE_COEF * gained / n_tasks + WIN_BONUS * (just_done & vec.success).float()

        O.append(obs); A.append(action); LP.append(logp); V.append(value)
        LE.append(legal); R.append(reward); AC.append(active); TM.append(just_done)
        # NB: no per-ply .all() sync here — that GPU->CPU stall dominated runtime.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--iters", type=int, default=0, help="if >0, overrides --minutes")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out", type=str, default=os.path.join(E.CACHE_DIR, "ppo_model.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    print(f"Device: {DEVICE}  |  PPO on-policy self-play")

    model = build_ppo_model().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    level = START_LEVEL
    roll_wr = 0.0
    best_eval = -1.0
    t0 = time.time()
    it = 0
    while True:
        if args.iters > 0:
            if it >= args.iters:
                break
        elif time.time() - t0 >= args.minutes * 60:
            break

        data, wr = rollout(model, rng, level, DEVICE, args.batch)
        st = ppo_update(model, opt, data)
        roll_wr = 0.9 * roll_wr + 0.1 * wr if it else wr

        # curriculum: advance difficulty when we are winning the current band
        if roll_wr > UP_THRESHOLD and level < 50:
            level = min(50, level + LEVEL_STEP)
            roll_wr = 0.0

        if it % 5 == 0:
            print(f"it {it:04d} | level {level:2d} | roll_wr {roll_wr:.3f} | "
                  f"pol {st['pol']:+.3f} | val {st['val']:.3f} | ent {st['ent']:.3f} | "
                  f"kl {st['kl']:+.4f} | {(time.time()-t0)/60:.1f}m", flush=True)

        if args.eval_every and it > 0 and it % args.eval_every == 0:
            overall, per = evaluate(model, args.eval_games)
            cleared = [m for m in E.ALL_MISSIONS if per[m] >= 0.5]
            print(f"  [eval] win_rate {overall:.4f} | mission_level "
                  f"{max(cleared) if cleared else 0} | level {level}", flush=True)
            if overall > best_eval:
                best_eval = overall
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save(model.state_dict(), args.out)
        it += 1

    # final eval (the reported metric)
    overall, per = evaluate(model, args.eval_games)
    cleared = [m for m in E.ALL_MISSIONS if per[m] >= 0.5]
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
        print(f"  {name}: {np.mean([per[m] for m in r]):.3f}")


if __name__ == "__main__":
    main()

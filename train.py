"""
crewbot training script. Single file, edited and iterated on by the agent.

Mirrors autoresearch's train.py: it reads fixed infra from crew_engine.py (rules
+ the ground-truth metric), trains a model on the self-play shards produced by
selfplay.py, and prints a summary whose key line is `win_rate` (higher is
better). Everything here is fair game for the agent: model architecture, loss,
optimizer, hyperparameters, data weighting.

The model is a policy+value network over the 40-card action space, conditioned
on the full observation (hand, trick, captured cards, AND the mission's task
assignment + ordering). Because the mission is an *input*, ONE model covers all
50 missions and any hand deal — no need for 50 separate models.

Usage:
    python train.py                       # train for TIME_BUDGET_MIN, then eval
    python train.py --minutes 5 --eval-games 50   # quick run
"""

import os
import glob
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crew_engine as E

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

TIME_BUDGET_MIN = 30.0     # training wall-clock budget in minutes
HIDDEN = 512               # hidden width
N_BLOCKS = 4               # residual MLP blocks
BATCH_SIZE = 4096
LR = 3e-4
WEIGHT_DECAY = 1e-4
VALUE_COEF = 0.5           # weight on value (baseline) loss
ENTROPY_COEF = 0.01        # entropy bonus (encourages exploration of options)
ADV_CLIP = 5.0             # clip advantage magnitude
MAX_LOAD_DECISIONS = 8_000_000  # cap decisions held in RAM (float16)
GRAD_CLIP = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.fc2(h)
        return F.relu(self.norm(x + h))


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim=E.OBS_DIM, act_dim=E.ACT_DIM, hidden=HIDDEN, n_blocks=N_BLOCKS):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                        nn.Linear(hidden // 2, 1))

    def forward(self, x):
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        logits = self.policy_head(x)
        value = torch.sigmoid(self.value_head(x)).squeeze(-1)  # win probability in [0,1]
        return logits, value


def build_model():
    """Factory used by both training and self-play (selfplay.py imports this)."""
    return PolicyValueNet()

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(max_decisions=MAX_LOAD_DECISIONS):
    shards = sorted(glob.glob(os.path.join(E.DATA_DIR, "shard_*.npz")))
    if not shards:
        raise SystemExit(f"No shards found in {E.DATA_DIR}. Run selfplay.py first.")
    obs, act, legal, rew = [], [], [], []
    total = 0
    for path in shards:
        d = np.load(path)
        n = d["obs"].shape[0]
        obs.append(d["obs"]); act.append(d["act"]); legal.append(d["legal"]); rew.append(d["rew"])
        total += n
        if total >= max_decisions:
            break
    obs = np.concatenate(obs)[:max_decisions]
    act = np.concatenate(act)[:max_decisions].astype(np.int64)
    legal = np.concatenate(legal)[:max_decisions]
    rew = np.concatenate(rew)[:max_decisions].astype(np.float32)
    print(f"Loaded {len(obs):,} decisions from {len(shards)} shards "
          f"(win-labeled reward mean {rew.mean():.3f})")
    return obs, act, legal, rew

# ---------------------------------------------------------------------------
# Evaluation (the fixed metric, delegated to crew_engine)
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_greedy_fn(model):
    model.eval()
    def fn(s):
        obs = torch.from_numpy(E.observe(s)).to(DEVICE).float().unsqueeze(0)
        mask = torch.from_numpy(E.legal_mask(s)).to(DEVICE).unsqueeze(0)
        logits, _ = model(obs)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        return int(logits.argmax(dim=-1).item())
    return fn


def evaluate(model, eval_games):
    fn = make_greedy_fn(model)
    overall, per = E.evaluate_winrate(fn, missions=E.ALL_MISSIONS,
                                      games_per_mission=eval_games)
    model.train()
    return overall, per

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=TIME_BUDGET_MIN)
    ap.add_argument("--eval-games", type=int, default=100, help="games/mission at final eval")
    ap.add_argument("--max-load", type=int, default=MAX_LOAD_DECISIONS)
    ap.add_argument("--out", type=str, default=os.path.join(E.CACHE_DIR, "model.pt"))
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(42)
    print(f"Device: {DEVICE}")

    obs_np, act_np, legal_np, rew_np = load_data(args.max_load)
    N = len(obs_np)
    # Keep big tensors on CPU (float16 obs); move minibatches to device per step.
    obs_t = torch.from_numpy(obs_np)            # float16 [N, OBS_DIM]
    act_t = torch.from_numpy(act_np)            # int64  [N]
    legal_t = torch.from_numpy(legal_np)        # uint8  [N, ACT_DIM]
    rew_t = torch.from_numpy(rew_np)            # float32[N]

    model = build_model().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}  (obs_dim={E.OBS_DIM}, act_dim={E.ACT_DIM})")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    budget = args.minutes * 60
    step = 0
    ema = None
    while time.time() - t_start < budget:
        idx = torch.randint(0, N, (BATCH_SIZE,))
        obs = obs_t[idx].to(DEVICE).float()
        act = act_t[idx].to(DEVICE)
        legal = legal_t[idx].to(DEVICE).bool()
        rew = rew_t[idx].to(DEVICE)

        logits, value = model(obs)
        logits = logits.masked_fill(~legal, float("-inf"))
        logp = F.log_softmax(logits, dim=-1)
        chosen_logp = logp.gather(1, act.unsqueeze(1)).squeeze(1)

        # Advantage-weighted policy gradient with the value baseline.
        adv = (rew - value.detach()).clamp(-ADV_CLIP, ADV_CLIP)
        policy_loss = -(adv * chosen_logp).mean()
        value_loss = F.binary_cross_entropy(value.clamp(1e-6, 1 - 1e-6), rew)
        probs = logp.exp()  # 0 for illegal actions (logp = -inf there)
        safe_logp = logp.masked_fill(~legal, 0.0)  # avoid 0 * -inf = NaN
        entropy = -(probs * safe_logp).sum(dim=-1).mean()
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        l = loss.item()
        ema = l if ema is None else 0.99 * ema + 0.01 * l
        step += 1
        if step % 200 == 0:
            elapsed = time.time() - t_start
            print(f"\rstep {step:06d} | loss {ema:.4f} | pol {policy_loss.item():.4f} | "
                  f"val {value_loss.item():.4f} | ent {entropy.item():.3f} | "
                  f"{elapsed/budget*100:.0f}% budget", end="", flush=True)
    print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)

    print("Evaluating win_rate across all 50 missions...")
    overall, per = evaluate(model, args.eval_games)
    # Highest mission level still cleared at >=50% win rate (laddered objective).
    cleared = [m for m in E.ALL_MISSIONS if per[m] >= 0.5]
    mission_level = max(cleared) if cleared else 0

    print("---")
    print(f"win_rate:         {overall:.6f}")
    print(f"mission_level:    {mission_level}")
    print(f"train_minutes:    {(time.time() - t_start)/60:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {n_params/1e6:.3f}")
    print(f"decisions:        {N}")
    print(f"device:           {DEVICE}")
    bands = {"m01-10": range(1, 11), "m11-20": range(11, 21), "m21-30": range(21, 31),
             "m31-40": range(31, 41), "m41-50": range(41, 51)}
    for name, rng in bands.items():
        avg = np.mean([per[m] for m in rng])
        print(f"  {name}: {avg:.3f}")


if __name__ == "__main__":
    main()

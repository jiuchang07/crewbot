"""
Diagnostic: is the trained policy actually using communication, and does it help?

(A) Communication usage: over many games, what fraction of comm-phase decisions
    are PASS vs a real signal, and what signal types/cards get chosen?
(B) Ablation: win_rate of the SAME trained model with communication enabled
    (use_comm=True) vs disabled (use_comm=False). If equal, the channel is
    contributing nothing.

Usage:
    python diag_comm.py --ckpt ~/.cache/crewbot/ppo_model.pt --games 40
"""

import os
import argparse
import numpy as np
import torch

import crew_engine as E
from train import PolicyValueNet


def load_model(ckpt, device):
    from ppo import build_ppo_model
    m = build_ppo_model().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def greedy_fn(model, device):
    def fn(s):
        obs = torch.from_numpy(E.observe(s)).to(device).float().unsqueeze(0)
        mask = torch.from_numpy(E.legal_mask(s)).to(device).unsqueeze(0)
        logits, _ = model(obs)
        logits = logits.masked_fill(mask == 0, float("-inf"))
        return int(logits.argmax(dim=-1).item())
    return fn


def usage_stats(model, device, missions, games):
    """Count comm-phase decisions made by the greedy policy."""
    fn = greedy_fn(model, device)
    rng = np.random.default_rng(7)
    n_pass = 0
    n_comm = 0
    types = {0: 0, 1: 0, 2: 0}   # only / high / low
    for m in missions:
        for _ in range(games):
            s = E.new_game(rng, m, use_comm=True)
            steps = 0
            while not s.done and steps < E.MAX_PLIES:
                a = fn(s)
                if s.comm_phase:
                    if a == E.PASS_ACTION:
                        n_pass += 1
                    else:
                        n_comm += 1
                        c = a - E.COMM_OFFSET
                        types[E.communicable(s.hands[s.turn]).get(c, 0)] += 1
                legal = E.legal_actions(s)
                if a not in legal:
                    a = legal[0]
                E.step(s, a)
                steps += 1
    total = n_pass + n_comm
    return n_pass, n_comm, total, types


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(E.CACHE_DIR, "ppo_model.pt"))
    ap.add_argument("--games", type=int, default=40, help="games per mission")
    ap.add_argument("--missions", default="", help="comma list; default = all 50")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    model = load_model(args.ckpt, device)
    missions = ([int(x) for x in args.missions.split(",")] if args.missions
                else E.ALL_MISSIONS)
    print(f"Device: {device}  ckpt: {args.ckpt}")

    # (A) usage
    n_pass, n_comm, total, types = usage_stats(model, device, missions, args.games)
    print("\n=== (A) communication usage (greedy) ===")
    print(f"comm-phase decisions: {total:,}")
    if total:
        print(f"  PASS:        {n_pass:,} ({100*n_pass/total:.1f}%)")
        print(f"  communicate: {n_comm:,} ({100*n_comm/total:.1f}%)")
        if n_comm:
            print(f"    of signals -> only:{types[0]}  high:{types[1]}  low:{types[2]}")

    # (B) ablation: same model, comm on vs off
    fn = greedy_fn(model, device)
    print("\n=== (B) ablation: win_rate with comm ON vs OFF ===")
    wr_on, per_on = E.evaluate_winrate(fn, missions=missions,
                                       games_per_mission=args.games, use_comm=True)
    wr_off, per_off = E.evaluate_winrate(fn, missions=missions,
                                         games_per_mission=args.games, use_comm=False)
    print(f"comm ON : win_rate {wr_on:.4f}")
    print(f"comm OFF: win_rate {wr_off:.4f}")
    print(f"delta (on - off): {wr_on - wr_off:+.4f}")
    bands = {"m01-10": range(1, 11), "m11-20": range(11, 21), "m21-30": range(21, 31),
             "m31-40": range(31, 41), "m41-50": range(41, 51)}
    print(f"{'band':8s} {'on':>7s} {'off':>7s} {'delta':>7s}")
    for name, r in bands.items():
        ms = [m for m in r if m in per_on]
        if not ms:
            continue
        on = np.mean([per_on[m] for m in ms]); off = np.mean([per_off[m] for m in ms])
        print(f"{name:8s} {on:7.3f} {off:7.3f} {on-off:+7.3f}")


if __name__ == "__main__":
    main()

"""
Vectorized GPU self-play data factory.

Steps a large batch of games in lockstep so policy inference and environment
transitions are batched on the GPU. This is the throughput path for the
AlphaZero-style iteration loop: generate data with a trained checkpoint, retrain,
repeat. Writes the SAME shard format as selfplay.py, so train.py reads either.

With no --policy-ckpt it uses a freshly initialized net (high-entropy
exploration). Per-decision inference for a single small MLP is only worthwhile
batched like this; for the heuristic bootstrap, prefer the CPU selfplay.py.

Usage:
    python vec_selfplay.py --hours 6 --batch 8192 --policy-ckpt ~/.cache/crewbot/model.pt
    python vec_selfplay.py --hours 0.02 --batch 1024        # smoke test
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import crew_engine as E
from vec_engine import VecCrew, N_P, TOTAL_TRICKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=8192, help="games stepped in parallel")
    ap.add_argument("--policy-ckpt", type=str, default="")
    ap.add_argument("--epsilon", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    ap.add_argument("--max-decisions", type=int, default=120_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(E.DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    from train import build_model
    model = build_model().to(device).eval()
    if args.policy_ckpt:
        model.load_state_dict(torch.load(args.policy_ckpt, map_location=device))
        print(f"Loaded policy checkpoint: {args.policy_ckpt}")
    else:
        print("No checkpoint: generating with a randomly-initialized net (exploration).")

    print(f"Vec self-play: device={device}, batch={args.batch}, budget={args.hours}h")
    deadline = time.time() + args.hours * 3600
    B = args.batch
    max_plies = TOTAL_TRICKS * N_P

    total_dec = 0
    total_games = 0
    total_wins = 0
    wave = 0
    t0 = time.time()

    while time.time() < deadline and total_dec < args.max_decisions:
        mids = rng.integers(1, 51, size=B)
        vec = VecCrew.new_games(B, mids, rng, device=device)

        ply_obs, ply_act, ply_legal, ply_active = [], [], [], []
        for _ in range(max_plies):
            active = (~vec.done)
            obs = vec.observe()                          # [B, OBS_DIM] float32
            legal = vec.legal_mask()                     # [B, N_C] bool
            with torch.no_grad():
                logits, _ = model(obs)
            logits = logits.masked_fill(~legal, float("-inf"))
            probs = F.softmax(logits / args.temperature, dim=-1)
            sampled = torch.multinomial(probs, 1).squeeze(1)
            # epsilon-random over legal moves for exploration
            unif = torch.multinomial(legal.float(), 1).squeeze(1)
            explore = torch.rand(B, device=device) < args.epsilon
            actions = torch.where(explore, unif, sampled)

            ply_obs.append(obs.to("cpu", torch.float16).numpy())
            ply_act.append(actions.to("cpu", torch.int8).numpy())
            ply_legal.append(legal.to("cpu", torch.uint8).numpy())
            ply_active.append(active.to("cpu").numpy())

            vec.step(actions)
            if bool(vec.done.all()):
                break

        success = vec.success.to("cpu").numpy().astype(np.float16)

        # Flatten plies, keeping only decisions where the game was still active.
        obs_rows, act_rows, legal_rows, rew_rows, mis_rows = [], [], [], [], []
        for k in range(len(ply_obs)):
            a = ply_active[k]
            obs_rows.append(ply_obs[k][a])
            act_rows.append(ply_act[k][a])
            legal_rows.append(ply_legal[k][a])
            rew_rows.append(success[a])
            mis_rows.append(mids[a].astype(np.int8))

        obs_arr = np.concatenate(obs_rows)
        path = os.path.join(E.DATA_DIR, f"vshard_{args.seed:02d}_{wave:06d}.npz")
        np.savez_compressed(
            path,
            obs=obs_arr,
            act=np.concatenate(act_rows),
            legal=np.concatenate(legal_rows),
            rew=np.concatenate(rew_rows),
            mission=np.concatenate(mis_rows),
        )

        n = len(obs_arr)
        total_dec += n
        total_games += B
        total_wins += int(vec.success.sum().item())
        wave += 1
        elapsed = time.time() - t0
        print(f"wave {wave:05d} | {total_games:,} games | win {total_wins/total_games:.3f} | "
              f"{total_games/elapsed:.0f} games/s | decisions {total_dec:,} | "
              f"{elapsed/3600:.2f}h", flush=True)

    print(f"Done. {total_games:,} games, {total_dec:,} decisions -> {E.DATA_DIR}")


if __name__ == "__main__":
    main()

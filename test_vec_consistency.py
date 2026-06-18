"""
Prove vec_engine matches crew_engine exactly.

Mirrors K scalar games into the batched env, drives BOTH with identical random
legal actions, and asserts they agree at every ply on: legal masks, the full
503-dim observation, captured cards, done/success/failed, and final reward.

Run:  python test_vec_consistency.py
"""

import numpy as np
import torch

import crew_engine as E
from vec_engine import VecCrew, N_P, TOTAL_TRICKS


def run(K=256, seed=0, atol=1e-5):
    rng = np.random.default_rng(seed)
    mids = rng.integers(1, 51, size=K)
    states = [E.new_game(rng, int(mids[b])) for b in range(K)]
    vec = VecCrew.from_scalar(states, device="cpu")

    pick_rng = np.random.default_rng(seed + 999)
    max_plies = TOTAL_TRICKS * N_P
    obs_checked = 0

    for ply in range(max_plies):
        active = [b for b in range(K) if not states[b].done]
        if not active:
            break

        # --- compare legal masks + observations on active games ---
        vlegal = vec.legal_mask().cpu().numpy()
        vobs = vec.observe().cpu().numpy()
        for b in active:
            slegal = E.legal_mask(states[b])
            assert np.array_equal(slegal, vlegal[b]), f"legal mismatch ply{ply} game{b}"
            sobs = E.observe(states[b])
            if not np.allclose(sobs, vobs[b], atol=atol):
                diff = np.abs(sobs - vobs[b])
                i = int(diff.argmax())
                raise AssertionError(
                    f"obs mismatch ply{ply} game{b} at dim {i}: "
                    f"scalar={sobs[i]:.4f} vec={vobs[b][i]:.4f} (max diff {diff.max():.4f})")
            obs_checked += 1

        # --- choose identical actions ---
        actions = np.zeros(K, dtype=np.int64)
        for b in range(K):
            if states[b].done:
                continue
            legal = E.legal_actions(states[b])
            actions[b] = legal[pick_rng.integers(len(legal))]

        # --- step both ---
        for b in active:
            E.step(states[b], int(actions[b]))
        vec.step(torch.from_numpy(actions))

        # --- compare post-step bookkeeping ---
        vcap = vec.captured_by.cpu().numpy()
        vdone = vec.done.cpu().numpy()
        vsucc = vec.success.cpu().numpy()
        vfail = vec.failed.cpu().numpy()
        vtricks = vec.tricks.cpu().numpy()
        for b in range(K):
            assert vdone[b] == states[b].done, f"done mismatch ply{ply} game{b}"
            assert vsucc[b] == states[b].success, f"success mismatch ply{ply} game{b}"
            assert vfail[b] == states[b].failed, f"failed mismatch ply{ply} game{b}"
            assert vtricks[b] == states[b].tricks_played, f"tricks mismatch ply{ply} game{b}"
            assert np.array_equal(vcap[b], states[b].captured_by), f"captured mismatch ply{ply} game{b}"

    # --- final reward parity ---
    vsucc = vec.success.cpu().numpy()
    for b in range(K):
        assert states[b].done, f"scalar game {b} not finished"
        assert bool(vsucc[b]) == states[b].success
    wr = float(np.mean([s.success for s in states]))
    print(f"OK: {K} games, {obs_checked:,} observations checked, "
          f"all legal/obs/capture/done/success match. win_rate={wr:.3f}")


if __name__ == "__main__":
    run(K=256, seed=0)
    run(K=256, seed=1)
    run(K=128, seed=2)
    print("vec_engine matches crew_engine exactly.")

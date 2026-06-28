"""
Test Other-Play color permutation correctness.

Verifies that:
1. build_perm_table produces valid permutations (bijections on 0..39)
2. permute_colors with the identity permutation is a no-op
3. The state after permutation is internally consistent
4. A full game played on a permuted state produces the same win/loss outcome
   as the original (when actions are also permuted)

Run:  python test_other_play.py
"""

import copy
import numpy as np
import torch

import crew_engine as E
from vec_engine import VecCrew, N_P, N_C, N_TC


def test_perm_table():
    """build_perm_table produces valid bijections that fix trumps."""
    B = 64
    rng = np.random.default_rng(42)
    perms = np.stack([rng.permutation(E.N_COLORS) for _ in range(B)])
    color_perms = torch.from_numpy(perms).long()
    perm, inv = VecCrew.build_perm_table(color_perms)

    for b in range(B):
        p = perm[b].numpy()
        i = inv[b].numpy()
        # Must be a valid permutation of 0..39
        assert sorted(p) == list(range(N_C)), f"perm[{b}] is not a bijection"
        assert sorted(i) == list(range(N_C)), f"inv[{b}] is not a bijection"
        # perm and inv must be inverses
        assert np.all(p[i] == np.arange(N_C)), f"perm[inv] != identity for game {b}"
        assert np.all(i[p] == np.arange(N_C)), f"inv[perm] != identity for game {b}"
        # Trumps (36-39) must be fixed points
        for c in range(36, 40):
            assert p[c] == c, f"trump card {c} was permuted in game {b}"
        # Ranks must be preserved for non-trump cards
        for c in range(36):
            assert E.card_rank(c) == E.card_rank(p[c]), \
                f"rank changed: card {c} (rank {E.card_rank(c)}) -> card {p[c]} (rank {E.card_rank(p[c])})"
    print("  ✓ perm_table: all permutations are valid bijections, trumps fixed, ranks preserved")


def test_identity_permutation_noop():
    """Identity permutation must leave the state unchanged."""
    B = 64
    rng = np.random.default_rng(999)
    mids = rng.integers(1, 51, size=B)

    # Build one VecCrew, then deep-copy its state tensors before permuting
    vec = VecCrew.new_games(B, mids, rng, device="cpu")

    # Snapshot original state
    snap = {
        'owner': vec.owner.clone(),
        'assigned': vec.assigned.clone(),
        'is_task': vec.is_task.clone(),
        'pred': vec.pred.clone(),
        'table': vec.table.clone(),
        'comm_card': vec.comm_card.clone(),
        'led': vec.led.clone(),
        'done_tasks': vec.done_tasks.clone(),
        'captured_by': vec.captured_by.clone(),
    }
    obs_before = vec.observe().clone()

    # Apply identity permutation
    identity = torch.arange(E.N_COLORS).unsqueeze(0).expand(B, -1).clone()
    vec.permute_colors(identity)

    assert torch.equal(snap['owner'], vec.owner), "owner changed under identity perm"
    assert torch.equal(snap['assigned'], vec.assigned), "assigned changed under identity perm"
    assert torch.equal(snap['is_task'], vec.is_task), "is_task changed under identity perm"
    assert torch.equal(snap['pred'], vec.pred), "pred changed under identity perm"
    assert torch.equal(snap['table'], vec.table), "table changed under identity perm"
    assert torch.equal(snap['comm_card'], vec.comm_card), "comm_card changed under identity perm"
    assert torch.equal(snap['led'], vec.led), "led changed under identity perm"

    # Observations must match too
    obs_after = vec.observe()
    assert torch.allclose(obs_before, obs_after, atol=1e-5), "observation changed under identity perm"

    print("  ✓ identity_permutation_noop: state and observations unchanged")


def test_permuted_state_consistency():
    """After permute_colors, the game state is internally consistent."""
    B = 128
    rng = np.random.default_rng(7)
    mids = rng.integers(1, 51, size=B)
    vec = VecCrew.new_games(B, mids, rng, device="cpu")

    # Snapshot originals
    orig_owner = vec.owner.clone()
    orig_assigned = vec.assigned.clone()
    orig_is_task = vec.is_task.clone()

    perm_rng = np.random.default_rng(77)
    color_perms = np.stack([perm_rng.permutation(E.N_COLORS) for _ in range(B)])
    color_perms_t = torch.from_numpy(color_perms).long()
    perm, inv = VecCrew.build_perm_table(color_perms_t)

    vec.permute_colors(color_perms_t)

    for b in range(B):
        # Every card owned by someone in the original should still be owned
        # by the same player in the permuted state (at the permuted index)
        for c in range(N_C):
            orig_p = orig_owner[b, c].item()
            new_c = perm[b, c].item()
            new_p = vec.owner[b, new_c].item()
            assert orig_p == new_p, \
                f"game {b}: card {c} owned by P{orig_p}, but permuted card {new_c} owned by P{new_p}"

        # Task assignment must be preserved
        for c in range(N_C):
            orig_a = orig_assigned[b, c].item()
            new_c = perm[b, c].item()
            new_a = vec.assigned[b, new_c].item()
            assert orig_a == new_a, \
                f"game {b}: card {c} assigned to P{orig_a}, but permuted card {new_c} assigned to P{new_a}"

        # is_task must be preserved
        for c in range(N_C):
            orig_t = orig_is_task[b, c].item()
            new_c = perm[b, c].item()
            new_t = vec.is_task[b, new_c].item()
            assert orig_t == new_t, \
                f"game {b}: card {c} is_task={orig_t}, but permuted card {new_c} is_task={new_t}"

    print("  ✓ permuted_state_consistency: owner, assigned, is_task all correctly remapped")


def test_permuted_game_outcome():
    """Playing identical (permuted) actions on a permuted state gives the same outcome."""
    B = 256
    rng = np.random.default_rng(123)
    mids = rng.integers(1, 51, size=B)

    # Create one VecCrew and build a second by copying + permuting
    vec_orig = VecCrew.new_games(B, mids, rng, device="cpu")

    # Deep copy the original state to create the permuted version
    vec_perm = copy.deepcopy(vec_orig)

    # Apply a random color permutation to the second batch
    perm_rng = np.random.default_rng(456)
    color_perms = np.stack([perm_rng.permutation(E.N_COLORS) for _ in range(B)])
    color_perms_t = torch.from_numpy(color_perms).long()
    perm, inv = VecCrew.build_perm_table(color_perms_t)
    vec_perm.permute_colors(color_perms_t)

    pick_rng = np.random.default_rng(789)
    max_plies = E.MAX_PLIES

    for ply in range(max_plies):
        if bool(vec_orig.done.all()):
            break

        # Get legal actions from original
        legal_orig = vec_orig.legal_mask()
        legal_perm = vec_perm.legal_mask()

        # Choose random legal action for original
        actions_orig = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            if vec_orig.done[b]:
                continue
            legal_ids = legal_orig[b].nonzero(as_tuple=True)[0]
            actions_orig[b] = legal_ids[pick_rng.integers(len(legal_ids))]

        # Permute the action for the permuted game
        actions_perm = actions_orig.clone()
        for b in range(B):
            if vec_orig.done[b]:
                continue
            a = actions_orig[b].item()
            if a < N_C:  # play
                actions_perm[b] = perm[b, a].item()
            elif a >= E.CLAIM_OFFSET:  # claim
                c = a - E.CLAIM_OFFSET
                actions_perm[b] = E.CLAIM_OFFSET + perm[b, c].item()
            elif a >= E.COMM_OFFSET and a < E.PASS_ACTION:  # comm
                c = a - E.COMM_OFFSET
                actions_perm[b] = E.COMM_OFFSET + perm[b, c].item()
            # else pass: unchanged

        # Verify that the permuted action is legal in the permuted game
        for b in range(B):
            if vec_orig.done[b]:
                continue
            ap = actions_perm[b].item()
            assert legal_perm[b, ap], \
                f"ply {ply}, game {b}: permuted action {ap} is not legal"

        vec_orig.step(actions_orig)
        vec_perm.step(actions_perm)

    # Outcomes must match
    assert torch.equal(vec_orig.done, vec_perm.done), "done mismatch"
    assert torch.equal(vec_orig.success, vec_perm.success), "success mismatch"
    assert torch.equal(vec_orig.failed, vec_perm.failed), "failed mismatch"
    print(f"  ✓ permuted_game_outcome: {B} games, all outcomes match "
          f"(win rate {vec_orig.success.float().mean():.3f})")


if __name__ == "__main__":
    print("Testing Other-Play permutation correctness...")
    test_perm_table()
    test_identity_permutation_noop()
    test_permuted_state_consistency()
    test_permuted_game_outcome()
    print("\nAll Other-Play tests passed. ✓")

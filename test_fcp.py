"""
Test Fictitious Co-Play (FCP) implementation correctness.

Verifies that:
1. CheckpointPool FIFO eviction works correctly
2. CheckpointPool save/load round-trips
3. Partner model construction produces correct outputs
4. rollout_fcp returns correct shapes and only learner decisions
5. FCP rollout produces fewer training samples than standard rollout
   (because only 1/3 of seats are learners)
6. A short end-to-end FCP training loop runs without errors

Run:  python test_fcp.py
"""

import copy
import numpy as np
import torch

import crew_engine as E
from ppo import (
    CheckpointPool, _build_partner_model, build_ppo_model,
    rollout, rollout_fcp, ppo_update,
    PARTNER_NOISE, OTHER_PLAY, N_P,
)


DEVICE = torch.device("cpu")  # tests run on CPU


def test_checkpoint_pool_fifo():
    """CheckpointPool evicts oldest snapshots when full."""
    pool = CheckpointPool(max_size=3)
    assert len(pool) == 0

    model = build_ppo_model().to(DEVICE)

    # Save 5 checkpoints into a pool of size 3
    for i in range(5):
        # Modify weights so each snapshot is distinguishable
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float(i))
        pool.save(model)

    assert len(pool) == 3, f"Expected pool size 3, got {len(pool)}"

    # The oldest two (i=0, i=1) should have been evicted.
    # Remaining: i=2, i=3, i=4
    for idx, expected_val in enumerate([2.0, 3.0, 4.0]):
        sd = pool.snapshots[idx]
        first_key = list(sd.keys())[0]
        actual = sd[first_key].flatten()[0].item()
        assert actual == expected_val, \
            f"Snapshot {idx}: expected {expected_val}, got {actual}"

    print("  ✓ checkpoint_pool_fifo: FIFO eviction works correctly")


def test_checkpoint_pool_save_load():
    """CheckpointPool state survives save/load round-trip."""
    pool = CheckpointPool(max_size=8)
    model = build_ppo_model().to(DEVICE)

    for i in range(4):
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float(i * 10))
        pool.save(model)

    # Serialize and restore
    saved = pool.state_for_save()
    pool2 = CheckpointPool(max_size=8)
    pool2.load_from_save(saved)

    assert len(pool2) == len(pool), "Pool sizes don't match after restore"
    for i in range(len(pool)):
        for k in pool.snapshots[i]:
            assert torch.equal(pool.snapshots[i][k], pool2.snapshots[i][k]), \
                f"Snapshot {i}, key {k} doesn't match"

    print("  ✓ checkpoint_pool_save_load: round-trip serialization works")


def test_partner_model_construction():
    """_build_partner_model produces a model with correct weights."""
    model = build_ppo_model().to(DEVICE)
    model.eval()

    # Save a snapshot with known weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.42)
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Change model weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.99)

    # Build partner from the saved snapshot
    partner = _build_partner_model(model, sd, DEVICE)

    # Partner should have the 0.42 weights, not 0.99
    for k in sd:
        assert torch.allclose(partner.state_dict()[k], sd[k]), \
            f"Partner weight {k} doesn't match saved snapshot"

    # Partner should be in eval mode
    assert not partner.training, "Partner should be in eval mode"

    # Both should produce valid outputs
    rng = np.random.default_rng(0)
    dummy_obs = torch.randn(2, E.OBS_DIM)
    logits_p, val_p = partner(dummy_obs)
    assert logits_p.shape == (2, E.ACT_DIM), f"Bad logits shape: {logits_p.shape}"
    assert val_p.shape == (2,), f"Bad value shape: {val_p.shape}"

    print("  ✓ partner_model_construction: weights and mode are correct")


def test_rollout_fcp_shapes():
    """rollout_fcp returns correctly shaped data with only learner decisions."""
    model = build_ppo_model().to(DEVICE)
    model.train()

    # Build a partner (same weights is fine for shape testing)
    partner = copy.deepcopy(model)
    partner.eval()

    rng = np.random.default_rng(42)
    batch = 64
    level = 5

    data_fcp, wr_fcp = rollout_fcp(model, partner, rng, level, DEVICE, batch,
                                    partner_noise=0.0, other_play=False)

    # Check all keys exist
    expected_keys = {"obs", "act", "logp", "legal", "adv", "ret"}
    assert set(data_fcp.keys()) == expected_keys, \
        f"Missing keys: {expected_keys - set(data_fcp.keys())}"

    n = data_fcp["obs"].shape[0]
    assert n > 0, "FCP rollout produced no training samples"
    assert data_fcp["obs"].shape == (n, E.OBS_DIM)
    assert data_fcp["act"].shape == (n,)
    assert data_fcp["logp"].shape == (n,)
    assert data_fcp["legal"].shape == (n, E.ACT_DIM)
    assert data_fcp["adv"].shape == (n,)
    assert data_fcp["ret"].shape == (n,)

    # Win rate should be a valid float
    assert 0.0 <= wr_fcp <= 1.0, f"Invalid win rate: {wr_fcp}"

    print(f"  ✓ rollout_fcp_shapes: {n} learner decisions, win_rate={wr_fcp:.3f}")


def test_fcp_fewer_samples_than_selfplay():
    """FCP rollout should produce ~1/3 the training samples of standard rollout.

    Since only 1 of 3 seats is the learner in FCP, the number of training
    samples should be roughly 1/3 of what standard self-play produces.
    """
    model = build_ppo_model().to(DEVICE)
    model.train()

    partner = copy.deepcopy(model)
    partner.eval()

    batch = 128
    level = 5

    rng1 = np.random.default_rng(100)
    data_sp, _ = rollout(model, rng1, level, DEVICE, batch,
                         partner_noise=0.0, other_play=False)
    n_sp = data_sp["obs"].shape[0]

    rng2 = np.random.default_rng(100)
    data_fcp, _ = rollout_fcp(model, partner, rng2, level, DEVICE, batch,
                               partner_noise=0.0, other_play=False)
    n_fcp = data_fcp["obs"].shape[0]

    ratio = n_fcp / n_sp
    # Should be approximately 1/3 (allow some variance: 0.2 to 0.5)
    assert 0.2 < ratio < 0.5, \
        f"FCP/SP sample ratio {ratio:.3f} not in expected range [0.2, 0.5]. " \
        f"SP={n_sp}, FCP={n_fcp}"

    print(f"  ✓ fcp_fewer_samples: SP={n_sp}, FCP={n_fcp}, ratio={ratio:.3f} (~1/3 expected)")


def test_fcp_end_to_end():
    """Short end-to-end FCP training loop runs without errors."""
    model = build_ppo_model().to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    pool = CheckpointPool(max_size=4)
    pool.save(model)  # seed with initial policy

    rng = np.random.default_rng(99)
    batch = 32
    level = 3

    for it in range(3):
        # Sample a partner from the pool
        partner_sd = pool.sample(rng)
        partner = _build_partner_model(model, partner_sd, DEVICE)

        data, wr = rollout_fcp(model, partner, rng, level, DEVICE, batch,
                                partner_noise=0.05, other_play=False)
        del partner

        # PPO update
        st = ppo_update(model, opt, data)

        # Periodically save to pool
        if it % 2 == 0:
            pool.save(model)

    assert len(pool) >= 2, f"Pool should have grown, has {len(pool)}"

    print(f"  ✓ fcp_end_to_end: 3 iterations completed, pool size={len(pool)}, "
          f"final win_rate={wr:.3f}")


def test_fcp_with_other_play():
    """FCP + Other-Play combination runs without errors."""
    model = build_ppo_model().to(DEVICE)
    model.train()

    partner = copy.deepcopy(model)
    partner.eval()

    rng = np.random.default_rng(55)
    batch = 32
    level = 3

    data, wr = rollout_fcp(model, partner, rng, level, DEVICE, batch,
                            partner_noise=0.05, other_play=True)

    n = data["obs"].shape[0]
    assert n > 0, "FCP + Other-Play produced no training samples"
    assert 0.0 <= wr <= 1.0

    print(f"  ✓ fcp_with_other_play: {n} samples, win_rate={wr:.3f}")


if __name__ == "__main__":
    print("Testing Fictitious Co-Play (FCP) implementation...")
    test_checkpoint_pool_fifo()
    test_checkpoint_pool_save_load()
    test_partner_model_construction()
    test_rollout_fcp_shapes()
    test_fcp_fewer_samples_than_selfplay()
    test_fcp_end_to_end()
    test_fcp_with_other_play()
    print("\nAll FCP tests passed. ✓")

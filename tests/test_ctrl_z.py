"""Tests for the Ctrl-Z training stabiliser module.

Run with:
    uv run python3 -m pytest tests/test_ctrl_z.py -v

Or directly:
    uv run python3 tests/test_ctrl_z.py
"""

import copy
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ultimate_trainer.ctrl_z import (
    CtrlZConfig,
    CtrlZCallback,
    CheckpointBuffer,
    CheckpointEntry,
    mann_whitney_rho,
    RollbackAction,
    EvaluateResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mann-Whitney ρ
# ═══════════════════════════════════════════════════════════════════════════════


def test_mann_whitney_identical():
    """Identical distributions → ρ ≈ 0.5 (within sampling noise)."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    b = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rho = mann_whitney_rho(a, b)
    # For identical discrete sets, ~15/36 = 0.417
    assert 0.3 < rho < 0.6, f"Expected ~0.42, got {rho}"


def test_mann_whitney_current_worse():
    """Current distribution stochastically greater → ρ ≈ 1."""
    worse = torch.tensor([10.0, 11.0, 12.0, 13.0])
    better = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rho = mann_whitney_rho(worse, better)
    assert rho > 0.99, f"Expected ≈1.0, got {rho}"


def test_mann_whitney_current_better():
    """Current distribution stochastically smaller → ρ ≈ 0."""
    better = torch.tensor([1.0, 2.0, 3.0, 4.0])
    worse = torch.tensor([10.0, 11.0, 12.0, 13.0])
    rho = mann_whitney_rho(better, worse)
    assert rho < 0.01, f"Expected ≈0.0, got {rho}"


# ═══════════════════════════════════════════════════════════════════════════════
#  CheckpointBuffer
# ═══════════════════════════════════════════════════════════════════════════════


def test_buffer_ring():
    """Buffer evicts oldest when at capacity."""
    buf = CheckpointBuffer(max_size=3)
    for i in range(5):
        buf.push(CheckpointEntry(step=i * 500, model_state={}, optimizer_state={}, losses=[float(i)]))
    assert buf.size == 3
    assert buf[0].step == 1000  # oldest surviving
    assert buf[2].step == 2000  # newest


def test_buffer_clear():
    buf = CheckpointBuffer(max_size=5)
    buf.push(CheckpointEntry(step=0, model_state={}, optimizer_state={}, losses=[]))
    buf.clear()
    assert buf.size == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  CtrlZCallback — behavioural tests
# ═══════════════════════════════════════════════════════════════════════════════

_CONFIG = CtrlZConfig(eval_samples=4, checkpoint_buffer=3, rho_threshold=0.3)
_MOCK_MODEL = torch.nn.Linear(4, 2)
_MOCK_OPTIM = torch.optim.SGD(_MOCK_MODEL.parameters(), lr=0.01)


def test_first_eval_no_history():
    ctrlz = CtrlZCallback(_CONFIG)
    r = ctrlz.evaluate(500, [5.0, 5.5, 6.0, 5.5])
    assert r.action == RollbackAction.KEEP
    assert r.target_entry is None
    assert r.min_rho == 1.0


def test_improving_loss_keeps():
    ctrlz = CtrlZCallback(_CONFIG)
    ctrlz.record(500, _MOCK_MODEL, _MOCK_OPTIM, [5.0, 5.5, 6.0, 5.5])
    r = ctrlz.evaluate(1000, [2.0, 2.5, 3.0, 2.5])
    assert r.action == RollbackAction.KEEP
    assert r.min_rho < 0.01  # current much better (lower loss)


def test_degrading_loss_rolls_back():
    ctrlz = CtrlZCallback(_CONFIG)
    ctrlz.record(500, _MOCK_MODEL, _MOCK_OPTIM, [2.0, 2.5, 3.0, 2.5])
    r = ctrlz.evaluate(1000, [9.0, 9.5, 10.0, 9.5])
    assert r.action == RollbackAction.ROLLBACK
    assert r.min_rho > 0.99  # current much worse (higher loss)


def test_rollback_restores_parameters():
    ctrlz = CtrlZCallback(_CONFIG)
    model = torch.nn.Linear(4, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    # Record initial state.
    ctrlz.record(500, model, optim, [5.0, 5.5, 6.0, 5.5])
    initial_params = copy.deepcopy(list(model.parameters()))
    initial_bias = model.bias.data.clone()

    # Degrade.
    r = ctrlz.evaluate(1000, [9.0, 9.5, 10.0, 9.5])

    # Take a few optimiser steps to change parameters.
    for _ in range(5):
        (model(torch.randn(2, 4)).sum()).backward()
        optim.step()

    # Roll back.
    ctrlz.rollback(model, optim, r.target_entry)

    # Verify parameters match originals.
    for p, q in zip(model.parameters(), initial_params):
        assert torch.allclose(p, q), "Parameter mismatch after rollback"
    assert torch.allclose(model.bias.data, initial_bias), "Bias mismatch after rollback"


def test_record_tracks_best():
    ctrlz = CtrlZCallback(_CONFIG)
    assert ctrlz.best_loss == float("inf")

    ctrlz.record(500, _MOCK_MODEL, _MOCK_OPTIM, [5.0, 5.5, 6.0, 5.5])
    assert ctrlz.best_loss == 5.5
    assert ctrlz.best_entry is not None and ctrlz.best_entry.step == 500

    ctrlz.record(1000, _MOCK_MODEL, _MOCK_OPTIM, [2.0, 2.5, 3.0, 2.5])
    assert ctrlz.best_loss == 2.5
    assert ctrlz.best_entry is not None and ctrlz.best_entry.step == 1000


def test_state_dict_roundtrip():
    ctrlz = CtrlZCallback(_CONFIG)
    ctrlz.record(500, _MOCK_MODEL, _MOCK_OPTIM, [5.0, 5.5, 6.0, 5.5])
    ctrlz.evaluate(1000, [9.0, 9.5, 10.0, 9.5])

    sd = ctrlz.state_dict()
    restored = CtrlZCallback(_CONFIG)
    restored.load_state_dict(sd)

    assert restored.best_loss == ctrlz.best_loss
    assert restored.total_rollbacks == ctrlz.total_rollbacks
    assert restored.buffer.size == ctrlz.buffer.size


def test_reset():
    ctrlz = CtrlZCallback(_CONFIG)
    ctrlz.record(500, _MOCK_MODEL, _MOCK_OPTIM, [5.0, 5.5, 6.0, 5.5])
    ctrlz.evaluate(1000, [9.0, 9.5, 10.0, 9.5])
    ctrlz.reset()

    assert ctrlz.buffer.size == 0
    assert ctrlz.best_loss == float("inf")
    assert ctrlz.best_entry is None
    assert ctrlz.total_rollbacks == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Reward metric
# ═══════════════════════════════════════════════════════════════════════════════


def test_reward_metric():
    config_r = CtrlZConfig(rho_threshold=0.3, metric="reward")
    ctrlz = CtrlZCallback(config_r)
    ctrlz.record(500, _MOCK_MODEL, _MOCK_OPTIM, [10.0, 11.0, 12.0, 13.0])  # high reward = good
    r = ctrlz.evaluate(1000, [1.0, 2.0, 3.0, 4.0])  # much worse reward
    assert r.action == RollbackAction.ROLLBACK


# ═══════════════════════════════════════════════════════════════════════════════
#  Scheduling
# ═══════════════════════════════════════════════════════════════════════════════


def test_should_evaluate():
    ctrlz = CtrlZCallback(CtrlZConfig(eval_interval=100))
    assert not ctrlz.should_evaluate(50), "Should not eval before interval"
    assert ctrlz.should_evaluate(100), "Should eval at interval boundary"
    ctrlz.evaluate(100, [1.0, 2.0, 3.0, 4.0])
    assert not ctrlz.should_evaluate(150), "Should not eval again too soon"
    assert ctrlz.should_evaluate(200), "Should eval at second boundary"


# ═══════════════════════════════════════════════════════════════════════════════
#  Run directly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("ρ identical", test_mann_whitney_identical),
        ("ρ current worse", test_mann_whitney_current_worse),
        ("ρ current better", test_mann_whitney_current_better),
        ("buffer ring", test_buffer_ring),
        ("buffer clear", test_buffer_clear),
        ("first eval", test_first_eval_no_history),
        ("improving", test_improving_loss_keeps),
        ("degrading", test_degrading_loss_rolls_back),
        ("rollback params", test_rollback_restores_parameters),
        ("best tracking", test_record_tracks_best),
        ("state dict", test_state_dict_roundtrip),
        ("reset", test_reset),
        ("reward metric", test_reward_metric),
        ("scheduling", test_should_evaluate),
    ]
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            sys.exit(1)
    print(f"\nAll {len(tests)} Ctrl-Z tests passed ✅")

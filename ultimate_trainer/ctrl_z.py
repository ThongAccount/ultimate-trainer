"""Ctrl-Z: Training stabilizer via Mann-Whitney U-test checkpoint rollback.

Reference: Dasagi et al., "Ctrl-Z: Recovering from Instability in
Reinforcement Learning", arXiv:1910.03732, 2019.

Adapted for language-model training: the training loop is treated as an RL
episode, the eval loss as the reward signal, and checkpoint rollback as the
recovery behaviour.  The Mann-Whitney ρ statistic is used because it is
distribution-free, invariant to reward magnitude, and robust to outliers —
eval-loss distributions are often non-Gaussian and change shape over training.

Usage
-----
    ctrlz = CtrlZCallback(CtrlZConfig())
    ...
    for step in range(max_steps):
        train_step()
        if ctrlz.should_evaluate(step):
            losses = [eval_one_batch() for _ in range(ctrlz.config.eval_samples)]
            action, entry = ctrlz.evaluate(step, losses)
            if action == RollbackAction.ROLLBACK:
                ctrlz.rollback(model, optimizer, entry)
                logger.info(f"Ctrl-Z: rolled back to step {entry.step}")
            ctrlz.record(step, model, optimizer, losses)
"""

from __future__ import annotations

import copy
import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CtrlZConfig:
    """Hyperparameters for the Ctrl-Z stabiliser.

    Attributes
    ----------
    eval_interval:
        Evaluate the model every *N* optimiser steps.  (The original paper uses
        30 episodes; for LM training 500 is a sensible starting point.)
    eval_samples:
        Number of forward passes (batches) per evaluation.  Each pass produces
        one scalar loss; the collection of *M* values forms the empirical
        distribution fed into the Mann-Whitney U-test.  (Paper: 20 episodes.)
    rho_threshold:
        Significance threshold for the Mann-Whitney ρ statistic.  When
        ``min(ρ) < rho_threshold`` the hypothesis of continued improvement is
        rejected and a rollback is triggered.  The paper reports a wide plateau
        of good values in [0.05, 0.2]; 0.1 is the recommended default.
    checkpoint_buffer:
        Number of historical checkpoints to retain in the ring buffer.  Each
        entry stores the full model + optimizer state dict, so this trades
        memory for robustness.  (Paper: stores the entire history.)
    store_on_cpu:
        If True, state dicts are deep-copied and moved to CPU before storage
        (reduces GPU memory pressure at the cost of slower rollback).
    """
    eval_interval: int = 500
    eval_samples: int = 32
    rho_threshold: float = 0.1
    checkpoint_buffer: int = 10
    store_on_cpu: bool = True
    metric: str = "loss"
    """Which direction is better: ``"loss"`` (lower is better) or ``"reward"``
    (higher is better).  This flips the direction of the ρ comparison."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Mann-Whitney U-test
# ═══════════════════════════════════════════════════════════════════════════════

def mann_whitney_rho(
    current: torch.Tensor,
    past: torch.Tensor,
) -> float:
    r"""Mann-Whitney ρ statistic.

    Computes the empirical probability that a randomly drawn sample from the
    *current* distribution exceeds a randomly drawn sample from the *past*
    distribution:

    .. math::

        \rho = P(R_\text{curr} > R_\text{past})
             = \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M
               \mathbf{1}(R_\text{curr}^{(i)} > R_\text{past}^{(j)})

    For a loss metric (lower is better), a low ρ means the current losses are
    typically *larger* than past losses — i.e., the model is getting worse.
    The threshold test is ``ρ < rho_threshold``.

    For a reward metric (higher is better), invert by passing ``ρ < threshold``
    where threshold is small, or use ``1 - ρ``.

    Parameters
    ----------
    current:
        1-D tensor of *M* scalar losses from the current evaluation.
    past:
        1-D tensor of *M* scalar losses from a historical checkpoint.

    Returns
    -------
    ρ in [0, 1] where lower values indicate the current distribution is
    stochastically *greater* (worse) than the past distribution.
    """
    # Vectorised pairwise comparison via broadcasting.
    n = (current.unsqueeze(-1) > past.unsqueeze(0)).sum().item()
    return n / (current.numel() * past.numel())


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint entry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckpointEntry:
    """One historical snapshot in the Ctrl-Z ring buffer."""

    step: int
    """Optimiser step at which this checkpoint was taken."""

    model_state: dict
    """Deep copy of ``model.state_dict()`` (stored on CPU if configured)."""

    optimizer_state: dict
    """Deep copy of ``optimizer.state_dict()``."""

    losses: List[float]
    """*M* individual eval-batch losses recorded at this checkpoint."""

    avg_loss: float = 0.0
    """Pre-computed mean of *losses* for fast best-checkpoint selection."""

    def __post_init__(self) -> None:
        self.avg_loss = sum(self.losses) / max(len(self.losses), 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ring buffer
# ═══════════════════════════════════════════════════════════════════════════════

class CheckpointBuffer:
    """Fixed-capacity ring buffer of :class:`CheckpointEntry`."""

    def __init__(self, max_size: int = 10) -> None:
        self.max_size = max_size
        self._entries: List[CheckpointEntry] = []

    def push(self, entry: CheckpointEntry) -> None:
        """Add *entry*, evicting the oldest if at capacity."""
        if len(self._entries) >= self.max_size:
            self._entries.pop(0)
        self._entries.append(entry)

    def clear(self) -> None:
        """Remove all entries (e.g. after a big config change)."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> CheckpointEntry:
        return self._entries[idx]

    def __iter__(self):
        return iter(self._entries)


# ═══════════════════════════════════════════════════════════════════════════════
#  Actions
# ═══════════════════════════════════════════════════════════════════════════════

class RollbackAction(enum.Enum):
    """Decision returned by :meth:`CtrlZCallback.evaluate`."""

    # ── Keep the current parameters; no rollback ──
    KEEP = "keep"

    # ── Roll back to the best historical checkpoint ──
    ROLLBACK = "rollback"


# ═══════════════════════════════════════════════════════════════════════════════
#  Main callback
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvaluateResult:
    """Return value of :meth:`CtrlZCallback.evaluate`."""

    action: RollbackAction
    """Whether to roll back."""

    target_entry: Optional[CheckpointEntry]
    """Checkpoint to restore if *action* is ``ROLLBACK``."""

    min_rho: float
    """The minimum ρ across all historical comparisons."""

    avg_loss: float
    """Mean of the current evaluation losses."""

    best_loss: float
    """Best average loss seen so far across all evaluations."""


class CtrlZCallback:
    """Ctrl-Z training stabiliser.

    Call :meth:`should_evaluate` after each optimiser step; when it returns
    True, collect *M* losses and pass them to :meth:`evaluate`, then call
    :meth:`record` to save the current state.  If *evaluate* signals a
    rollback, call :meth:`rollback` with the returned target entry.
    """

    def __init__(self, config: CtrlZConfig) -> None:
        self.config = config
        self.buffer = CheckpointBuffer(config.checkpoint_buffer)
        self._last_eval_step = 0

        # Running best (stays in buffer even after eviction)
        self.best_loss: float = float("inf")
        self.best_entry: Optional[CheckpointEntry] = None

        # Stats for logging
        self.total_rollbacks: int = 0
        self.last_action: RollbackAction = RollbackAction.KEEP

    # ── Scheduling ──────────────────────────────────────────────────────────

    def should_evaluate(self, step: int) -> bool:
        """Return True when *step* has reached the next eval boundary.

        This is an advisory check; the caller should then call :meth:`evaluate`
        which sets the internal eval clock.
        """
        return (step - self._last_eval_step) >= self.config.eval_interval

    # ── Core logic ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        step: int,
        current_losses: List[float],
    ) -> EvaluateResult:
        """Compare current losses against every checkpoint in the buffer.

        Parameters
        ----------
        step:
            Current optimiser step (for logging).
        current_losses:
            *M* individual eval-batch losses from the current model.

        Returns
        -------
        EvaluateResult with the rollback decision.
        """
        self._last_eval_step = step
        avg_loss = sum(current_losses) / max(1, len(current_losses))

        # No history yet → can't compare, just report.
        if self.buffer.size == 0:
            return EvaluateResult(
                action=RollbackAction.KEEP,
                target_entry=None,
                min_rho=1.0,
                avg_loss=avg_loss,
                best_loss=self.best_loss,
            )

        current_t = torch.tensor(current_losses, dtype=torch.float32)

        # Compare current losses against every historical checkpoint.
        # ρ = P(current > past).
        #
        # For loss (lower is better): ρ ≈ 1 means current > past (got worse).
        # For reward (higher is better): ρ ≈ 0 means current < past (got worse).
        #
        # We map both to a unified *danger* ∈ [0, 1] where 1 = definitely
        # roll back.  The paper's threshold test becomes:
        #   max_danger > (1 - ρ_threshold)  →  rollback.
        max_danger = 0.0
        worst_entry: Optional[CheckpointEntry] = None

        for entry in self.buffer:
            past_t = torch.tensor(entry.losses, dtype=torch.float32)
            rho_val = mann_whitney_rho(current_t, past_t)
            if self.config.metric == "loss":
                danger = rho_val           # ρ ≈ 1 = current is worse
            else:
                danger = 1.0 - rho_val     # ρ ≈ 0 = current is worse
            if danger > max_danger or worst_entry is None:
                max_danger = danger
                worst_entry = entry

        should_rollback = max_danger > (1.0 - self.config.rho_threshold)

        # Raw ρ of the worst comparison (unadjusted Mann-Whitney statistic).
        if worst_entry is not None:
            raw_rho = mann_whitney_rho(
                current_t, torch.tensor(worst_entry.losses, dtype=torch.float32)
            )
        else:
            raw_rho = 1.0

        if should_rollback:
            self.last_action = RollbackAction.ROLLBACK
            self.total_rollbacks += 1
            logger.warning(
                "Ctrl-Z: rollback triggered at step %d | "
                "avg_loss=%.4f | ρ=%.3f against step %d",
                step,
                avg_loss,
                raw_rho,
                worst_entry.step if worst_entry else -1,
            )
        else:
            self.last_action = RollbackAction.KEEP
            logger.info(
                "Ctrl-Z: step %d | avg_loss=%.4f | ρ=%.3f | KEEP",
                step,
                avg_loss,
                raw_rho,
            )

        return EvaluateResult(
            action=(RollbackAction.ROLLBACK if should_rollback else RollbackAction.KEEP),
            target_entry=worst_entry,
            min_rho=raw_rho,
            avg_loss=avg_loss,
            best_loss=self.best_loss,
        )

    # ── Recording ───────────────────────────────────────────────────────────

    def record(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        losses: List[float],
    ) -> bool:
        """Snapshot the current model + optimiser into the ring buffer.

        Returns True if this is the new best checkpoint (lowest avg loss).
        """
        # Deep-copy state dicts, optionally moving to CPU to save GPU memory.
        model_sd = copy.deepcopy(model.state_dict())
        optim_sd = copy.deepcopy(optimizer.state_dict())

        if self.config.store_on_cpu:
            model_sd = {k: v.cpu() for k, v in model_sd.items()}
            optim_sd = {
                k: (
                    v.cpu() if isinstance(v, torch.Tensor) else v
                )
                for k, v in optim_sd.items()
            }

        entry = CheckpointEntry(
            step=step,
            model_state=model_sd,
            optimizer_state=optim_sd,
            losses=losses,
        )

        self.buffer.push(entry)

        # Update running best.
        is_new_best = entry.avg_loss < self.best_loss
        if is_new_best:
            self.best_loss = entry.avg_loss
            self.best_entry = entry
            logger.info(
                "Ctrl-Z: new best checkpoint at step %d (avg_loss=%.4f)",
                step,
                entry.avg_loss,
            )

        return is_new_best

    # ── Rollback ────────────────────────────────────────────────────────────

    def rollback(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        entry: CheckpointEntry,
    ) -> None:
        """Restore *model* and *optimizer* to the state in *entry*.

        Handles DDP-wrapped models transparently: if the model is a
        :class:`DistributedDataParallel` wrapper, state is loaded on
        ``model.module``.
        """
        target = model.module if hasattr(model, "module") else model

        # Move state back to the target device if stored on CPU.
        device = next(target.parameters()).device
        model_sd = {k: v.to(device, non_blocking=True) for k, v in entry.model_state.items()}
        target.load_state_dict(model_sd, strict=True)

        # Optimiser state dicts can be large; move in place.
        optim_sd = entry.optimizer_state
        optim_sd = {
            k: (
                v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            )
            for k, v in optim_sd.items()
        }
        optimizer.load_state_dict(optim_sd)

        logger.info(
            "Ctrl-Z: rolled model and optimizer back to step %d "
            "(avg_loss=%.4f)",
            entry.step,
            entry.avg_loss,
        )

    # ── Utilities ───────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Serialize Ctrl-Z state for checkpointing."""
        return {
            "buffer_entries": [
                {
                    "step": e.step,
                    "model_state": e.model_state,
                    "optimizer_state": e.optimizer_state,
                    "losses": e.losses,
                }
                for e in self.buffer
            ],
            "best_loss": self.best_loss,
            "total_rollbacks": self.total_rollbacks,
            "last_eval_step": self._last_eval_step,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore Ctrl-Z state from a prior :meth:`state_dict`."""
        self.buffer.clear()
        for entry_sd in sd.get("buffer_entries", []):
            self.buffer.push(CheckpointEntry(**entry_sd))
        self.best_loss = sd.get("best_loss", float("inf"))
        self.total_rollbacks = sd.get("total_rollbacks", 0)
        self._last_eval_step = sd.get("last_eval_step", 0)

    def reset(self) -> None:
        """Clear all state.  The ring buffer and best-loss tracker are wiped."""
        self.buffer.clear()
        self.best_loss = float("inf")
        self.best_entry = None
        self.total_rollbacks = 0
        self._last_eval_step = -self.config.eval_interval
        self.last_action = RollbackAction.KEEP

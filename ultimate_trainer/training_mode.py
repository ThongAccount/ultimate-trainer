"""Training mode dispatch for the RL-informed training ecosystem.

Each mode selects a different loss function and Ctrl-Z metric:

    PRETRAIN  │  CE (next-token)   │  Ctrl-Z on val_loss
    SFT       │  CE (instruction)  │  Ctrl-Z on val_loss
    DPO       │  preference loss   │  Ctrl-Z on val_dpo_loss
    RL        │  REINFORCE/PPO     │  Ctrl-Z on mean_reward

Usage
-----
    mode_cfg = ModeConfig(mode=TrainingMode.PRETRAIN)
    trainer = UltimateTrainer(mc, tc, mode_config=mode_cfg)
    trainer.train()      # dispatches loss / eval / Ctrl-Z per mode
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .ctrl_z import CtrlZConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Mode enum
# ═══════════════════════════════════════════════════════════════════════════════


class TrainingMode(enum.Enum):
    """Training paradigm selector.

    Each mode changes the loss function, the evaluation metric, and the role
    of the Ctrl-Z stabiliser and LLM-as-a-Judge evaluator.
    """

    PRETRAIN = "pretrain"
    """Pre-training: cross-entropy on raw text, Ctrl-Z on val loss."""

    SFT = "sft"
    """Supervised fine-tuning: cross-entropy on instruction data, Ctrl-Z on val loss."""

    DPO = "dpo"
    """Direct preference optimisation: pairwise preference loss, Ctrl-Z on DPO loss."""

    RL = "rl"
    """Pure RL: REINFORCE / PPO maximising reward from LLM-as-a-Judge."""

    # ── Queries ─────────────────────────────────────────────────────────

    @property
    def ctrlz_metric(self) -> str:
        """Which direction the Ctrl-Z Mann-Whitney U-test expects.

        ``"loss"``   → lower is better (pretrain, SFT, DPO).
        ``"reward"`` → higher is better (RL).
        """
        if self == TrainingMode.RL:
            return "reward"
        return "loss"

    @property
    def requires_reward_model(self) -> bool:
        return self == TrainingMode.RL

    @property
    def uses_cross_entropy(self) -> bool:
        return self in (TrainingMode.PRETRAIN, TrainingMode.SFT)

    @property
    def requires_preference_pairs(self) -> bool:
        return self == TrainingMode.DPO

    def __str__(self) -> str:
        return self.value


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-mode config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModeConfig:
    """Hyperparameters that differ by training paradigm.

    Parameters
    ----------
    mode:
        Which paradigm to use.
    ctrlz:
        Ctrl-Z config overrides for this mode.  The ``metric`` field is
        set automatically from ``TrainingMode.ctrlz_metric``.
    dpo_beta:
        KL regularisation strength in the DPO loss (only used in DPO mode).
    rl_algorithm:
        Which RL algorithm to use — ``"reinforce"`` or ``"ppo"``.
    rl_clip_eps:
        PPO clipping epsilon (only used in PPO mode).
    rl_kl_coeff:
        KL penalty coefficient against the reference model.
    """

    mode: TrainingMode = TrainingMode.PRETRAIN

    # ── Ctrl-Z (auto-derived from mode) ─────────────────────────────────
    ctrlz: CtrlZConfig = field(default_factory=lambda: CtrlZConfig())

    # ── DPO ─────────────────────────────────────────────────────────────
    dpo_beta: float = 0.1

    # ── RL ──────────────────────────────────────────────────────────────
    rl_algorithm: str = "reinforce"
    rl_clip_eps: float = 0.2
    rl_kl_coeff: float = 0.1

    def __post_init__(self) -> None:
        # Ensure Ctrl-Z metric matches the selected mode.
        self.ctrlz.metric = self.mode.ctrlz_metric


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ═══════════════════════════════════════════════════════════════════════════════


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Next-token prediction loss used in PRETRAIN and SFT modes.

    Parameters
    ----------
    logits:
        Shape ``(batch, seq_len, vocab_size)``.
    labels:
        Shape ``(batch, seq_len)`` with ``ignore_index`` for padding.
    ignore_index:
        Token index to ignore in the loss (e.g. pad tokens).

    Returns
    -------
    Scalar loss.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
    )


def dpo_loss(
    chosen_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_logits: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Direct Preference Optimisation loss (Rafailov et al., 2023).

    .. math::

        L = -\\log\\sigma\\big(\\beta \\cdot (r_\\text{chosen} - r_\\text{rejected})\\big)

    where :math:`r` is the negative cross-entropy (higher = better) for each
    sequence.

    Parameters
    ----------
    chosen_logits:
        Logits for the preferred response, shape ``(batch, seq_len, vocab)``.
    chosen_labels:
        Labels for the preferred response, shape ``(batch, seq_len)``.
    rejected_logits:
        Logits for the dispreferred response.
    rejected_labels:
        Labels for the dispreferred response.
    beta:
        KL regularisation strength.

    Returns
    -------
    loss:
        Scalar DPO loss.
    aux:
        Auxiliary metrics (accuracy, margins).
    """
    def _ce_per_seq(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-token CE → sum over sequence length."""
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
        )  # (batch * seq_len,)
        ce = ce.view(logits.size(0), -1)  # (batch, seq_len)
        return ce.sum(dim=-1)  # (batch,) — sum over tokens

    ce_chosen = _ce_per_seq(chosen_logits, chosen_labels)
    ce_rejected = _ce_per_seq(rejected_logits, rejected_labels)

    # Reward = -CE (higher CE = worse = lower reward)
    reward_chosen = -ce_chosen
    reward_rejected = -ce_rejected

    logits_diff = beta * (reward_chosen - reward_rejected)  # (batch,)
    loss = -F.logsigmoid(logits_diff).mean()

    with torch.no_grad():
        accuracy = (reward_chosen > reward_rejected).float().mean().item()
        margin = (reward_chosen - reward_rejected).mean().item()

    return loss, {"dpo_accuracy": accuracy, "dpo_margin": margin, "dpo_logits_diff": logits_diff.mean().item()}


def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """REINFORCE gradient estimator.

    .. math::

        L = -\\frac{1}{N}\\sum_{i=1}^N r_i \\cdot \\log \\pi(a_i|s_i)

    Parameters
    ----------
    log_probs:
        Log-probabilities of the sampled actions, shape ``(batch,)``.
    rewards:
        Scalar rewards for each action, shape ``(batch,)``.

    Returns
    -------
    Scalar loss.
    """
    return -(rewards * log_probs).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss dispatch
# ═══════════════════════════════════════════════════════════════════════════════


def get_loss_fn(mode: TrainingMode) -> Callable:
    """Return the appropriate loss function for *mode*.

    The returned callable has a different signature per mode — use
    :func:`compute_loss` for a unified interface.
    """
    if mode.uses_cross_entropy:
        return cross_entropy_loss
    if mode == TrainingMode.DPO:
        return dpo_loss
    if mode == TrainingMode.RL:
        return reinforce_loss
    raise ValueError(f"Unknown training mode: {mode}")


def compute_loss(
    mode: TrainingMode,
    logits: torch.Tensor,
    batch: dict,
    mode_config: ModeConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Unified loss dispatch: returns ``(loss, aux_metrics_dict)``.

    Parameters
    ----------
    mode:
        Current training paradigm.
    logits:
        Model output logits.
    batch:
        Data batch — keys depend on the mode:

        - PRETRAIN / SFT: ``{"input_ids", "labels"}``
        - DPO: ``{"chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"}``
        - RL: ``{"log_probs", "rewards"}``

    mode_config:
        Per-mode hyperparameters (e.g. ``dpo_beta``).

    Returns
    -------
    loss:
        Scalar loss for backprop.
    aux:
        Auxiliary metrics for logging (per-mode).
    """
    aux: Dict[str, float] = {}

    if mode.uses_cross_entropy:
        loss = cross_entropy_loss(logits, batch["labels"])
        aux["ce_loss"] = loss.item()

    elif mode == TrainingMode.DPO:
        loss, dpo_aux = dpo_loss(
            chosen_logits=logits["chosen"],
            chosen_labels=batch["chosen_labels"],
            rejected_logits=logits["rejected"],
            rejected_labels=batch["rejected_labels"],
            beta=mode_config.dpo_beta,
        )
        aux.update(dpo_aux)

    elif mode == TrainingMode.RL:
        loss = reinforce_loss(
            log_probs=batch["log_probs"],
            rewards=batch["rewards"],
        )
        aux["rl_loss"] = loss.item()
        aux["mean_reward"] = batch["rewards"].mean().item()

    else:
        raise ValueError(f"Unknown training mode: {mode}")

    return loss, aux

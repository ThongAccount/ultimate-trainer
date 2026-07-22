"""LLM-as-a-Tutor: policy-aware prompt adaptation for non-verifiable RL.

Reference: Kim et al., "LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for
Non-Verifiable RL", arXiv:2607.04412, 2026.

The Tutor plays two roles:
  1. **Examiner** — pairwise comparison of two rollouts to detect whether a
     prompt is still challenging for the current policy.  If two rollouts are
     indistinguishable in quality, the prompt is *non-challenging*.
  2. **Generator** — appends an atomic constraint to the prompt and a matching
     rubric criterion, monotonically increasing difficulty.

The adapted prompts then feed into the GRPO training loop, where an LLM judge
scores each rollout against the updated rubrics.

Usage
-----
    tutor = TutorEngine(model_name="qwen3-8b-thinking")
    adapted = tutor.adapt_prompts(prompts, policy_model, tokenizer)
    for prompt, rubric in adapted:
        rollouts = policy_model.generate(prompt, G=8)
        scores = judge.score(rollouts, rubric)
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AdaptedPrompt:
    """A prompt after the Tutor's adaptation pass.

    Attributes
    ----------
    seed:
        Original seed prompt (never modified).
    adapted:
        Current prompt after *k* constraint appends ``x ⊕ c₁ ⊕ c₂ ⊕ ...``.
        Equal to *seed* if no adaptation has occurred yet.
    rubric:
        Base rubric dict ``{criterion_name: weight}``.
    adapted_rubric:
        Combined rubric after appending criteria for each added constraint.
    constraints:
        List of atomic constraints appended so far, in order.
    is_discriminative:
        Whether the pairwise examination found this prompt discriminative
        in the most recent check.
    """
    seed: str
    adapted: str
    rubric: Dict[str, float]
    adapted_rubric: Dict[str, float]
    constraints: List[str] = field(default_factory=list)
    is_discriminative: bool = True


@dataclass
class ExaminationResult:
    """Result of the Tutor's pairwise examination for one prompt."""

    prompt_idx: int
    is_discriminative: bool
    constraint: Optional[str] = None
    rubric_criteria: Optional[Dict[str, float]] = None
    rollout_a: str = ""
    rollout_b: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Tutor Engine
# ═══════════════════════════════════════════════════════════════════════════════


class TutorEngine:
    """LLM-as-a-Tutor: examiner and generator for dynamic prompt adaptation.

    Parameters
    ----------
    model_name:
        Identifier for the LLM used as tutor (e.g. ``"qwen3-8b-thinking"``).
    api_base:
        Optional API endpoint.  If empty, uses the environment default.
    temperature:
        Sampling temperature for tutor generation.
    max_rollouts_per_check:
        Maximum rollouts to sample per prompt during examination.
    """

    def __init__(
        self,
        model_name: str = "qwen3-8b-thinking",
        api_base: str = "",
        temperature: float = 0.3,
        max_rollouts_per_check: int = 2,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_rollouts_per_check = max_rollouts_per_check
        self._stats = {"checks": 0, "adaptations": 0, "kept": 0}

    # ── Examiner ──────────────────────────────────────────────────────────

    def check_discriminative(
        self,
        prompt: str,
        rollout_a: str,
        rollout_b: str,
    ) -> bool:
        """Pairwise examination: are *rollout_a* and *rollout_b* distinguishable?

        Returns True if the prompt is discriminative (rollouts differ in quality),
        False if they are indistinguishable (prompt is non-challenging).

        The examination prompt follows the paper's pairwise comparison template:
        "Are these two responses indistinguishable in quality, in their overall
        approach, and in the absence of meaningful weaknesses?"
        """
        self._stats["checks"] += 1

        # Build the pairwise examination prompt.
        exam_prompt = f"""You are an expert examiner evaluating whether a prompt is sufficiently challenging for the current policy.

Prompt:
{prompt}

Response A:
{rollout_a}

Response B:
{rollout_b}

Are these two responses **indistinguishable in quality**? Consider:
1. Do they agree in overall quality level?
2. Do they take the same overall approach?
3. Are they both free of meaningful weaknesses?

Answer with a single word: YES if they are indistinguishable (prompt is too easy/hard), NO if they differ in quality (prompt is appropriately challenging)."""

        # In production this would call an LLM API.  For now, stub.
        judgment = self._call_llm(exam_prompt)
        is_discriminative = judgment.strip().upper() != "YES"

        if not is_discriminative:
            logger.info(
                "Tutor examiner: prompt non-discriminative (%.60s...)",
                prompt.replace("\n", " "),
            )

        return is_discriminative

    # ── Generator ─────────────────────────────────────────────────────────

    def generate_constraint(
        self,
        prompt: str,
        rollout_a: str,
        rollout_b: str,
    ) -> Tuple[str, Dict[str, float]]:
        """Generate an atomic constraint and matching rubric criteria.

        Returns
        -------
        constraint:
            A single atomic requirement ``c`` to append to the prompt.
        rubric_criteria:
            Dict ``{criterion_name: weight}`` scoring adherence to the constraint.
        """
        gen_prompt = f"""You are an expert tutor designing increasingly challenging prompts for RL training.

The current prompt is not challenging enough — the policy produces indistinguishable responses.

Original prompt:
{prompt}

Response A:
{rollout_a}

Response B:
{rollout_b}

Design a **single atomic constraint** that makes this prompt more challenging. The constraint should:
1. Add exactly one new requirement not already specified in the prompt.
2. Be objectively evaluable (a judge can score it).
3. Not contradict or override the original prompt's intent.

Also provide a rubric criterion for scoring adherence to this constraint.

Output JSON format:
{{
  "constraint": "Your single atomic constraint here",
  "rubric_criterion": "Description of what a good response should do",
  "rubric_weight": 0.1
}}"""

        response = self._call_llm(gen_prompt)
        parsed = self._parse_json(response)

        constraint: str = parsed.get("constraint", "Be more specific and detailed.")
        criterion: str = parsed.get("rubric_criterion", constraint)
        weight: float = float(parsed.get("rubric_weight", 0.1))

        self._stats["adaptations"] += 1
        return constraint, {criterion: weight}

    # ── Adaptation pipeline ───────────────────────────────────────────────

    def adapt_prompts(
        self,
        adapted_prompts: List[AdaptedPrompt],
        generate_fn,
    ) -> List[AdaptedPrompt]:
        """Run examiner + generator over all prompts.

        For each prompt:
        1. Sample 2 rollouts from the current policy via *generate_fn*.
        2. Examiner checks if they are discriminative.
        3. If not, Generator appends an atomic constraint.

        Parameters
        ----------
        adapted_prompts:
            Current prompt set (may already have prior adaptations).
        generate_fn:
            Callable ``(prompt_text: str) -> List[str]`` that returns sampled
            rollouts from the current policy.

        Returns
        -------
        Updated list of AdaptedPrompt with any new constraints applied.
        """
        results: List[AdaptedPrompt] = []

        for idx, ap in enumerate(adapted_prompts):
            # Sample rollouts from current policy.
            rollouts = generate_fn(ap.adapted)
            if len(rollouts) < 2:
                results.append(ap)
                continue

            rollout_a, rollout_b = rollouts[0], rollouts[1]

            # Examiner.
            is_disc = self.check_discriminative(ap.adapted, rollout_a, rollout_b)
            ap.is_discriminative = is_disc

            if not is_disc:
                # Generator: append constraint.
                constraint, rubric_criteria = self.generate_constraint(
                    ap.adapted, rollout_a, rollout_b,
                )
                ap.constraints.append(constraint)
                ap.adapted = ap.adapted + "\nAdditional requirement: " + constraint
                for criterion, weight in rubric_criteria.items():
                    ap.adapted_rubric[criterion] = weight
                logger.info(
                    "Tutor: adapted prompt %d — appended constraint: %s",
                    idx, constraint,
                )
            else:
                self._stats["kept"] += 1

            results.append(ap)

        logger.info(
            "Tutor adaptation complete: %d adapted, %d kept of %d prompts",
            self._stats["adaptations"],
            self._stats["kept"],
            len(adapted_prompts),
        )
        return results

    # ── LLM call (stub) ───────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the tutor LLM with a prompt and return the response text.

        In production this would use an API client (e.g. OpenAI-compatible).
        For now returns a placeholder so the module is importable and testable.
        """
        logger.warning(
            "TutorEngine._call_llm: stub — replace with actual API call to %s\n"
            "Prompt (first 200 chars): %.200s",
            self.model_name,
            prompt.replace("\n", " "),
        )
        # Placeholder: return JSON for the generator, "NO" for examiner.
        # In production, replace with e.g.:
        #   from openai import OpenAI
        #   client = OpenAI(base_url=self.api_base)
        #   resp = client.chat.completions.create(...)
        #   return resp.choices[0].message.content
        return '{"constraint": "Provide specific evidence for your claims.", "rubric_criterion": "Uses concrete evidence", "rubric_weight": 0.1}'

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract a JSON object from LLM output (handles ``` fences)."""
        import re
        # Try to extract from markdown code fence first.
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            text = m.group(1)
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning("Tutor: failed to parse JSON from LLM output: %.100s", text)
            return {}

    # ── Stats ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {"checks": 0, "adaptations": 0, "kept": 0}

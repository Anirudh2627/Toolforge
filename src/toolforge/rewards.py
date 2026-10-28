"""Reward computation for RL (GRPO, Phase 4) — built strictly on the verifier.

Design rules (mirrors the project's anti-reward-hacking requirements):

- The dominant term is **verified task success** from
  :func:`toolforge.verifier.verify_episode` — i.e. actual final environment
  state, never the model's textual claims. A failed episode that *says* it
  succeeded earns the failure reward (0 by default).
- All shaping terms are non-positive penalties (invalid calls, executed-call
  cost, step-limit / loop-guard / wrong-clarification / policy-error), plus an
  optional ``format_bonus`` that defaults to **zero** and is config-capped at
  10% of the success reward.
- :meth:`RewardConfig.validate` enforces *reward dominance*: the worst possible
  successful episode (every turn penalized across the full step budget) must
  still outscore the best possible failed episode (no penalties, format bonus
  collected). If a proposed config violates this, it is rejected at startup
  rather than discovered after a training run.
- Every component is returned in :class:`RewardBreakdown` for logging; GRPO
  consumes ``breakdown.total`` and the harness logs the components.

Repeated no-op calls cannot accumulate reward: there is no per-call positive
term at all, and the executor's loop guard terminates exact repeats (penalized
once via ``loop_guard_penalty``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from toolforge.executor import EpisodeResult
from toolforge.schemas import ToolForgeError
from toolforge.verifier import VerificationResult


class RewardConfigError(ToolForgeError):
    """Raised when a reward configuration would allow shaping to dominate success."""


@dataclass(frozen=True)
class RewardConfig:
    """Weights for each reward component. Initial hypotheses, all configurable."""

    success_reward: float = 1.0
    failure_reward: float = 0.0
    invalid_call_penalty: float = -0.02     # per parse error / schema violation / unknown tool
    tool_call_penalty: float = -0.005       # per executed call (pressure against unnecessary calls)
    step_limit_penalty: float = -0.10
    loop_guard_penalty: float = -0.10
    wrong_clarification_penalty: float = -0.10  # clarify-when-acting or act-when-clarifying
    policy_error_penalty: float = -0.20     # crashed/unparseable policy turn stream
    format_bonus: float = 0.0               # OFF by default; capped at 10% of success_reward

    def validate(self, max_steps: int) -> None:
        """Reject configs where shaping can outweigh task correctness."""
        if self.success_reward <= 0:
            raise RewardConfigError("success_reward must be > 0 (verified success is the signal)")
        if self.failure_reward >= self.success_reward:
            raise RewardConfigError("failure_reward must be < success_reward")
        for name in ("invalid_call_penalty", "tool_call_penalty", "step_limit_penalty",
                     "loop_guard_penalty", "wrong_clarification_penalty", "policy_error_penalty"):
            if getattr(self, name) > 0:
                raise RewardConfigError(f"{name} must be <= 0 (penalties cannot reward behavior)")
        if self.format_bonus < 0:
            raise RewardConfigError("format_bonus must be >= 0 (use penalties for bad format)")
        if self.format_bonus > 0.1 * self.success_reward:
            raise RewardConfigError(
                f"format_bonus {self.format_bonus} exceeds 10% of success_reward "
                f"({self.success_reward}); formatting must never outweigh task execution")
        # Dominance: worst successful episode > best failed episode.
        worst_per_turn = min(self.invalid_call_penalty, self.tool_call_penalty)
        worst_success = self.success_reward + max_steps * worst_per_turn
        best_failure = (self.failure_reward + self.format_bonus
                        + max(0.0, self.wrong_clarification_penalty))
        if worst_success <= best_failure:
            raise RewardConfigError(
                f"Reward dominance violated: a successful episode using all {max_steps} turns "
                f"scores {worst_success:.4f} <= a clean failed episode ({best_failure:.4f}). "
                "Increase success_reward or shrink per-turn penalties.")


@dataclass
class RewardBreakdown:
    """Total reward plus every component contribution (all logged)."""

    total: float
    components: dict[str, float] = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 6),
                "components": {k: round(v, 6) for k, v in self.components.items()},
                "success": self.success}


def compute_reward(episode: EpisodeResult, verdict: VerificationResult,
                   config: RewardConfig | None = None, *, max_steps: int = 8) -> RewardBreakdown:
    """Compute the shaped reward for one finished, verified episode.

    ``verdict`` must come from the state-based verifier — this function never
    inspects answer text beyond what the verifier already judged.
    """
    cfg = config or RewardConfig()
    cfg.validate(max_steps)
    categories = set(verdict.failure_categories)
    components: dict[str, float] = {}

    components["task_success"] = cfg.success_reward if verdict.success else cfg.failure_reward
    components["invalid_calls"] = episode.invalid_call_count * cfg.invalid_call_penalty
    components["tool_calls"] = episode.tool_call_count * cfg.tool_call_penalty

    if episode.termination == "step_limit":
        components["step_limit"] = cfg.step_limit_penalty
    if episode.termination == "loop_guard":
        components["loop_guard"] = cfg.loop_guard_penalty
    if episode.termination == "policy_error":
        components["policy_error"] = cfg.policy_error_penalty
    if categories & {"unnecessary_clarification", "missing_clarification"}:
        components["wrong_clarification"] = cfg.wrong_clarification_penalty

    if cfg.format_bonus and episode.parse_error_count == 0 and episode.trace:
        components["format"] = cfg.format_bonus

    total = sum(components.values())
    return RewardBreakdown(total=total, components=components, success=verdict.success)

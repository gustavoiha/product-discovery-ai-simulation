from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class WorkflowSettings:
    model: str = "gpt-5"
    pm_max_turns: int = 14
    interviewer_max_turns: int = 8
    simulator_max_turns: int = 4
    quality_checker_max_turns: int = 4
    max_output_tokens: int = 900
    max_interviews: int = 8
    interview_quality_min_score: float = 12.0
    quality_max_reruns_per_interview: int = 1
    total_token_budget: int = 50000
    max_estimated_cost_usd: float = 3.0
    est_input_cost_per_1m_tokens_usd: float = 1.25
    est_output_cost_per_1m_tokens_usd: float = 10.0
    output_dir: Path = Path("outputs")
    disable_tracing: bool = False

    @classmethod
    def from_env(cls) -> "WorkflowSettings":
        settings = cls(
            model=os.getenv("OPENAI_MODEL", "gpt-5"),
            pm_max_turns=int(os.getenv("PM_MAX_TURNS", "14")),
            interviewer_max_turns=int(os.getenv("INTERVIEWER_MAX_TURNS", "8")),
            simulator_max_turns=int(os.getenv("SIMULATOR_MAX_TURNS", "4")),
            quality_checker_max_turns=int(os.getenv("QUALITY_CHECKER_MAX_TURNS", "4")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "900")),
            max_interviews=int(os.getenv("MAX_INTERVIEWS", "8")),
            interview_quality_min_score=float(
                os.getenv("INTERVIEW_QUALITY_MIN_SCORE", "12.0")
            ),
            quality_max_reruns_per_interview=int(
                os.getenv("QUALITY_MAX_RERUNS_PER_INTERVIEW", "1")
            ),
            total_token_budget=int(os.getenv("TOTAL_TOKEN_BUDGET", "50000")),
            max_estimated_cost_usd=float(os.getenv("MAX_ESTIMATED_COST_USD", "3.0")),
            est_input_cost_per_1m_tokens_usd=float(
                os.getenv("EST_INPUT_COST_PER_1M_TOKENS_USD", "1.25")
            ),
            est_output_cost_per_1m_tokens_usd=float(
                os.getenv("EST_OUTPUT_COST_PER_1M_TOKENS_USD", "10.0")
            ),
            output_dir=Path(os.getenv("OUTPUT_DIR", "outputs")),
            disable_tracing=_bool_env("DISABLE_TRACING", False),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.pm_max_turns < 2:
            raise ValueError("PM_MAX_TURNS must be >= 2.")
        if self.interviewer_max_turns < 2:
            raise ValueError("INTERVIEWER_MAX_TURNS must be >= 2.")
        if self.simulator_max_turns < 1:
            raise ValueError("SIMULATOR_MAX_TURNS must be >= 1.")
        if self.quality_checker_max_turns < 1:
            raise ValueError("QUALITY_CHECKER_MAX_TURNS must be >= 1.")
        if self.max_output_tokens < 128:
            raise ValueError("MAX_OUTPUT_TOKENS must be >= 128.")
        if self.max_interviews < 1:
            raise ValueError("MAX_INTERVIEWS must be >= 1.")
        if self.interview_quality_min_score < 0:
            raise ValueError("INTERVIEW_QUALITY_MIN_SCORE must be >= 0.")
        if self.quality_max_reruns_per_interview < 0:
            raise ValueError("QUALITY_MAX_RERUNS_PER_INTERVIEW must be >= 0.")
        if self.total_token_budget <= 0:
            raise ValueError("TOTAL_TOKEN_BUDGET must be > 0.")
        if self.max_estimated_cost_usd <= 0:
            raise ValueError("MAX_ESTIMATED_COST_USD must be > 0.")
        if self.est_input_cost_per_1m_tokens_usd < 0:
            raise ValueError("EST_INPUT_COST_PER_1M_TOKENS_USD must be >= 0.")
        if self.est_output_cost_per_1m_tokens_usd < 0:
            raise ValueError("EST_OUTPUT_COST_PER_1M_TOKENS_USD must be >= 0.")

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
    max_output_tokens: int = 900
    max_interviews: int = 8
    output_dir: Path = Path("outputs")
    disable_tracing: bool = False

    @classmethod
    def from_env(cls) -> "WorkflowSettings":
        settings = cls(
            model=os.getenv("OPENAI_MODEL", "gpt-5"),
            pm_max_turns=int(os.getenv("PM_MAX_TURNS", "14")),
            interviewer_max_turns=int(os.getenv("INTERVIEWER_MAX_TURNS", "8")),
            simulator_max_turns=int(os.getenv("SIMULATOR_MAX_TURNS", "4")),
            max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "900")),
            max_interviews=int(os.getenv("MAX_INTERVIEWS", "8")),
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
        if self.max_output_tokens < 128:
            raise ValueError("MAX_OUTPUT_TOKENS must be >= 128.")
        if self.max_interviews < 1:
            raise ValueError("MAX_INTERVIEWS must be >= 1.")


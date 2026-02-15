from __future__ import annotations

import asyncio
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel

try:
    from agents import set_default_openai_api
except ImportError:  # pragma: no cover - backwards compatibility
    set_default_openai_api = None  # type: ignore[assignment]

try:
    from agents import set_tracing_disabled
except ImportError:  # pragma: no cover - backwards compatibility
    set_tracing_disabled = None  # type: ignore[assignment]

from .config import WorkflowSettings
from .models import (
    PMPhase,
    PMResponse,
    InterviewExchange,
    InterviewQualityAssessment,
    InterviewReport,
    RoadmapCandidate,
    SimulatorResponse,
)
from .personas import Persona, USER_SIMULATOR_PERSONAS

TModel = TypeVar("TModel", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised when cumulative budget limits are exceeded."""


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    if isinstance(output, BaseModel):
        return output.model_dump_json(indent=2)
    try:
        return json.dumps(output, indent=2, ensure_ascii=False, default=_json_default)
    except TypeError:
        return str(output)


def _bullet_items(items: list[str], empty_text: str) -> list[str]:
    if not items:
        return [f"- {empty_text}"]
    return [f"- {item}" for item in items]


def _coerce_model(output: Any, model_cls: type[TModel]) -> TModel:
    if isinstance(output, model_cls):
        return output
    if isinstance(output, BaseModel):
        return model_cls.model_validate(output.model_dump())
    if isinstance(output, dict):
        return model_cls.model_validate(output)
    if isinstance(output, str):
        data = json.loads(output)
        return model_cls.model_validate(data)
    raise ValueError(f"Unsupported output type for {model_cls.__name__}: {type(output)}")


@dataclass(slots=True)
class WorkflowState:
    settings: WorkflowSettings
    output_dir: Path
    phase: PMPhase = PMPhase.DISCOVERY
    interview_question_asked: bool = False
    confirmed_interview_count: int | None = None
    roadmap_saved_path: Path | None = None
    interview_files: list[Path] = field(default_factory=list)
    selected_personas: list[str] = field(default_factory=list)
    interview_quality_scores: dict[str, float] = field(default_factory=dict)
    interview_reruns: dict[str, int] = field(default_factory=dict)
    accepted_signals: list[str] = field(default_factory=list)
    rejected_signals: list[str] = field(default_factory=list)
    budget_exhausted: bool = False
    budget_stop_reason: str | None = None
    run_id: str = ""
    run_dir: Path | None = None
    observability_events_path: Path | None = None
    observability_summary_path: Path | None = None
    total_usage: dict[str, int] = field(
        default_factory=lambda: {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    turn_counter: int = 0


class ProductRoadmapWorkflow:
    def __init__(self, settings: WorkflowSettings):
        self.settings = settings
        self.output_dir = settings.output_dir.resolve()
        self.interviews_dir = self.output_dir / "interviews"
        self.roadmap_path = self.output_dir / "roadmap_recommendation.md"
        self.summary_path = self.interviews_dir / "INTERVIEW_SUMMARY.md"
        self._rng = random.Random(11)
        self.conversation_id = f"product-roadmap-{uuid.uuid4()}"
        self.run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        )
        self.run_dir = self.output_dir / "observability" / self.run_id
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_json_path = self.run_dir / "run_summary.json"
        self.start_monotonic = time.monotonic()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interviews_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.state = WorkflowState(
            settings=settings,
            output_dir=self.output_dir,
            run_id=self.run_id,
            run_dir=self.run_dir,
            observability_events_path=self.events_path,
            observability_summary_path=self.summary_json_path,
        )

        if set_default_openai_api is not None:
            set_default_openai_api("responses")
        if set_tracing_disabled is not None:
            set_tracing_disabled(settings.disable_tracing)

        self.simulator_agents = self._build_user_simulator_agents()
        self.interviewer_agent = self._build_interviewer_agent()
        self.interview_quality_agent = self._build_interview_quality_checker_agent()
        self.pm_agent = self._build_product_manager_agent()

    def _model_settings(self) -> ModelSettings:
        return ModelSettings(
            max_tokens=self.settings.max_output_tokens,
            parallel_tool_calls=False,
        )

    def _build_user_simulator_agents(self) -> dict[str, Agent]:
        agents_by_slug: dict[str, Agent] = {}

        for persona in USER_SIMULATOR_PERSONAS:
            instructions = (
                "You are a simulated end user in a product discovery interview.\n"
                f"Persona name: {persona.name}\n"
                f"Persona profile: {persona.profile}\n\n"
                "Return a structured response for these fields:\n"
                "- user_context\n"
                "- current_workflow\n"
                "- pain_points\n"
                "- desired_outcomes\n"
                "- feature_requests\n"
                "- risks_or_objections\n"
                "- broad_signals\n"
                "- niche_signals\n\n"
                "Behavior rules:\n"
                "- Be realistic and concrete.\n"
                "- Avoid generic positivity.\n"
                "- Mark niche requests as niche signals."
            )

            agents_by_slug[persona.slug] = Agent(
                name=f"User Simulator - {persona.name}",
                instructions=instructions,
                model=self.settings.model,
                model_settings=self._model_settings(),
                output_type=SimulatorResponse,
            )

        return agents_by_slug

    def _build_interviewer_agent(self) -> Agent:
        tools = []
        for persona in USER_SIMULATOR_PERSONAS:
            tool_name = self._persona_tool_name(persona)
            tools.append(
                self.simulator_agents[persona.slug].as_tool(
                    tool_name=tool_name,
                    tool_description=(
                        f"Interview this simulated user persona: {persona.name}. "
                        f"Profile: {persona.profile}"
                    ),
                    max_turns=self.settings.simulator_max_turns,
                )
            )

        instructions = (
            "You are a user research interviewer.\n"
            "For each run, conduct one interview by calling exactly one simulator tool.\n"
            "You must use the simulator tool explicitly requested in the prompt.\n"
            "Then return a structured interview report.\n\n"
            "Rules:\n"
            "- Ask focused discovery questions.\n"
            "- Cover workflow, pain points, desired outcomes, feature ideas, and objections.\n"
            "- Distinguish broad signals from niche signals.\n"
            "- Keep transcript compact and useful."
        )

        return Agent(
            name="Interviewer",
            instructions=instructions,
            model=self.settings.model,
            model_settings=self._model_settings(),
            tools=tools,
            output_type=InterviewReport,
        )

    def _build_interview_quality_checker_agent(self) -> Agent:
        instructions = (
            "You are an Interview Quality Control agent.\n"
            "Evaluate one structured interview report and score its evidence quality.\n\n"
            "Return structured fields for:\n"
            "- summary\n"
            "- strengths\n"
            "- weaknesses\n"
            "- rerun_focus_areas\n"
            "- question_depth (1-5)\n"
            "- answer_specificity (1-5)\n"
            "- signal_diversity (1-5)\n"
            "- actionability (1-5)\n"
            "- noise_penalty (0-5)\n"
            "- niche_bias_penalty (0-5)\n"
            "- rerun_recommended (true/false)\n\n"
            "Scoring intent:\n"
            "- question_depth: depth and breadth of discovery questions.\n"
            "- answer_specificity: concrete details vs generic statements.\n"
            "- signal_diversity: variety of needs/pain points captured.\n"
            "- actionability: how usable the findings are for PM prioritization.\n"
            "- noise_penalty: verbosity, contradictions, or low-signal filler.\n"
            "- niche_bias_penalty: over-indexing on niche requests.\n"
            "Recommend rerun when quality is shallow or noisy."
        )
        return Agent(
            name="Interview Quality Checker",
            instructions=instructions,
            model=self.settings.model,
            model_settings=self._model_settings(),
            output_type=InterviewQualityAssessment,
        )

    def _build_product_manager_agent(self) -> Agent:
        @function_tool
        async def run_user_research_interviews(
            context: RunContextWrapper[WorkflowState],
            interview_count: int,
            product_context: str,
            focus_areas: str = "",
        ) -> str:
            state = context.context

            if state.phase != PMPhase.INTERVIEWS:
                return (
                    "Interview tool blocked: current phase is "
                    f"'{state.phase.value}'. Move to 'interviews' first."
                )
            if not state.interview_question_asked:
                return (
                    "Interview tool blocked: you must first ask the human "
                    "\"How many interviews should I run?\""
                )
            if state.confirmed_interview_count is None:
                return (
                    "Interview tool blocked: the human has not provided a numeric "
                    "interview count yet."
                )
            if interview_count != state.confirmed_interview_count:
                return (
                    "Interview tool blocked: use the exact interview count confirmed by "
                    f"the human ({state.confirmed_interview_count})."
                )
            if interview_count > state.settings.max_interviews:
                return (
                    f"Interview tool blocked: requested {interview_count}, but MAX_INTERVIEWS "
                    f"is {state.settings.max_interviews}. Ask the human to choose <= "
                    f"{state.settings.max_interviews}."
                )
            if state.budget_exhausted:
                return (
                    "Interview tool blocked: budget exhausted.\n"
                    f"Reason: {state.budget_stop_reason}"
                )

            summary = await self._run_interviews(
                interview_count=interview_count,
                product_context=product_context,
                focus_areas=focus_areas,
            )
            state.phase = PMPhase.SYNTHESIS
            return summary

        instructions = (
            "You are the Product Manager and the only point of contact with the human product owner.\n\n"
            "You must always return a structured PM response object.\n\n"
            "State machine phases:\n"
            "- discovery\n"
            "- ask_interview_count\n"
            "- interviews\n"
            "- synthesis\n"
            "- roadmap\n\n"
            "Required behavior:\n"
            "- In discovery, ask clarifying questions about product model, users, platforms, existing features, goals, constraints, and ideas.\n"
            "- Before interviews, you MUST ask exactly: \"How many interviews should I run?\"\n"
            "- Set ask_interview_count=true in that step.\n"
            "- After interview count is provided, move to interviews and call run_user_research_interviews.\n"
            "- In synthesis, identify repeated signals and reject weak/niche signals.\n"
            "- For roadmap_ready=true, provide roadmap_candidates with scoring inputs (reach, impact, confidence, strategic_fit, effort, niche_penalty).\n"
            "- Do not set roadmap_ready=true unless roadmap_candidates is non-empty.\n"
            "- assistant_message must remain concise and user-facing."
        )

        return Agent(
            name="Product Manager",
            instructions=instructions,
            model=self.settings.model,
            model_settings=self._model_settings(),
            tools=[run_user_research_interviews],
            output_type=PMResponse,
        )

    def _persona_tool_name(self, persona: Persona) -> str:
        return f"simulate_{persona.slug}"

    def _select_personas(self, interview_count: int) -> list[Persona]:
        base = list(USER_SIMULATOR_PERSONAS)
        self._rng.shuffle(base)
        if interview_count <= len(base):
            return base[:interview_count]

        selected: list[Persona] = []
        while len(selected) < interview_count:
            selected.extend(base)
        return selected[:interview_count]

    def _extract_interview_count(self, user_text: str) -> int | None:
        match = re.search(r"\b(\d{1,3})\b", user_text)
        if not match:
            return None
        return int(match.group(1))

    def _usage_from_result(self, result: Any) -> dict[str, int]:
        usage = getattr(result, "usage", None)
        if usage is None:
            usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        requests = int(getattr(usage, "requests", 0) or 0)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return {
            "requests": requests,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _add_usage(self, usage: dict[str, int]) -> None:
        for key in self.state.total_usage:
            self.state.total_usage[key] += usage.get(key, 0)
        self._refresh_budget_state()

    def _estimated_cost_usd(self) -> float:
        input_cost = (
            self.state.total_usage["input_tokens"] / 1_000_000
        ) * self.settings.est_input_cost_per_1m_tokens_usd
        output_cost = (
            self.state.total_usage["output_tokens"] / 1_000_000
        ) * self.settings.est_output_cost_per_1m_tokens_usd
        return round(input_cost + output_cost, 6)

    def _budget_snapshot(self) -> dict[str, Any]:
        return {
            "total_tokens_used": self.state.total_usage["total_tokens"],
            "total_token_budget": self.settings.total_token_budget,
            "estimated_cost_usd": self._estimated_cost_usd(),
            "max_estimated_cost_usd": self.settings.max_estimated_cost_usd,
            "input_tokens": self.state.total_usage["input_tokens"],
            "output_tokens": self.state.total_usage["output_tokens"],
            "est_input_cost_per_1m_tokens_usd": self.settings.est_input_cost_per_1m_tokens_usd,
            "est_output_cost_per_1m_tokens_usd": self.settings.est_output_cost_per_1m_tokens_usd,
        }

    def _refresh_budget_state(self) -> None:
        if self.state.budget_exhausted:
            return

        total_tokens = self.state.total_usage["total_tokens"]
        estimated_cost_usd = self._estimated_cost_usd()
        exceeded_tokens = total_tokens >= self.settings.total_token_budget
        exceeded_cost = estimated_cost_usd >= self.settings.max_estimated_cost_usd

        if not exceeded_tokens and not exceeded_cost:
            return

        reasons: list[str] = []
        if exceeded_tokens:
            reasons.append(
                f"token budget reached ({total_tokens}/{self.settings.total_token_budget})"
            )
        if exceeded_cost:
            reasons.append(
                "estimated cost budget reached "
                f"(${estimated_cost_usd:.4f}/${self.settings.max_estimated_cost_usd:.4f})"
            )

        self.state.budget_exhausted = True
        self.state.budget_stop_reason = " and ".join(reasons)
        self._record_event(
            "budget_exhausted",
            {
                "reason": self.state.budget_stop_reason,
                **self._budget_snapshot(),
            },
        )

    def _enforce_budget_or_raise(self, callsite: str) -> None:
        self._refresh_budget_state()
        if self.state.budget_exhausted:
            raise BudgetExceededError(
                f"Budget exhausted at {callsite}: {self.state.budget_stop_reason}"
            )

    def _record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        _append_jsonl(
            self.events_path,
            {
                "timestamp": _now_iso(),
                "run_id": self.run_id,
                "event_type": event_type,
                "phase": self.state.phase.value,
                **payload,
            },
        )

    def _pm_fallback_response(self, raw_output: Any) -> PMResponse:
        return PMResponse(
            phase=self.state.phase,
            assistant_message=_to_text(raw_output),
            ask_interview_count=False,
            ready_to_run_interviews=False,
            roadmap_ready=False,
        )

    def _question_in_text(self, text: str) -> bool:
        return "how many interviews should i run" in text.lower()

    def _ensure_interview_question(self, text: str) -> str:
        if self._question_in_text(text):
            return text
        suffix = "How many interviews should I run?"
        if text.strip():
            return f"{text.rstrip()}\n\n{suffix}"
        return suffix

    def _render_interview_markdown(
        self,
        report: InterviewReport,
        index: int,
        quality: InterviewQualityAssessment | None = None,
        quality_score: float | None = None,
        reruns_used: int = 0,
    ) -> str:
        lines = [
            f"# Interview {index}: {report.persona_name}",
            "",
            f"Persona profile: {report.persona_profile}",
            "",
            "## Transcript",
        ]
        for exchange in report.transcript:
            lines.append(f"- Q: {exchange.question}")
            lines.append(f"  A: {exchange.answer}")

        lines.extend(["", "## Key Needs"])
        lines.extend(_bullet_items(report.key_needs, "(none captured)"))
        lines.extend(["", "## Feature Ideas"])
        lines.extend(_bullet_items(report.feature_ideas, "(none captured)"))
        lines.extend(["", "## Risks / Objections"])
        lines.extend(_bullet_items(report.risks_or_objections, "(none captured)"))
        lines.extend(["", "## Broad Signals"])
        lines.extend(_bullet_items(report.broad_signals, "(none captured)"))
        lines.extend(["", "## Niche Signals"])
        lines.extend(_bullet_items(report.niche_signals, "(none captured)"))
        lines.extend(["", "## Synthesis", report.synthesis or "(none provided)"])
        lines.extend(["", "## Interview Quality"])
        if quality is None:
            lines.append("- Quality assessment unavailable.")
        else:
            lines.append(f"- Formula: `{self._interview_quality_equation()}`")
            lines.append(
                f"- Score: {quality_score if quality_score is not None else '(unknown)'} "
                f"(threshold: {self.settings.interview_quality_min_score})"
            )
            lines.append(f"- Reruns used: {reruns_used}")
            lines.append(
                (
                    "- Inputs: "
                    f"question_depth={quality.question_depth}, "
                    f"answer_specificity={quality.answer_specificity}, "
                    f"signal_diversity={quality.signal_diversity}, "
                    f"actionability={quality.actionability}, "
                    f"noise_penalty={quality.noise_penalty}, "
                    f"niche_bias_penalty={quality.niche_bias_penalty}"
                )
            )
            lines.append(f"- Summary: {quality.summary or '(none)'}")
            lines.append("- Strengths:")
            lines.extend(_bullet_items(quality.strengths, "(none)"))
            lines.append("- Weaknesses:")
            lines.extend(_bullet_items(quality.weaknesses, "(none)"))
            lines.append("- Rerun focus areas:")
            lines.extend(_bullet_items(quality.rerun_focus_areas, "(none)"))
            lines.append(
                f"- Rerun recommended by QC agent: {'yes' if quality.rerun_recommended else 'no'}"
            )
        return "\n".join(lines)

    def _render_interview_summary_markdown(
        self, reports: list[InterviewReport], interview_count: int
    ) -> str:
        all_broad: list[str] = []
        all_niche: list[str] = []
        for report in reports:
            all_broad.extend(report.broad_signals)
            all_niche.extend(report.niche_signals)

        quality_values = list(self.state.interview_quality_scores.values())
        avg_quality = round(sum(quality_values) / len(quality_values), 2) if quality_values else 0.0

        lines = [
            "# Interview Summary",
            "",
            f"Total interviews: {interview_count}",
            f"Interview quality average score: {avg_quality}",
            f"Interview quality threshold: {self.settings.interview_quality_min_score}",
            f"Quality equation: `{self._interview_quality_equation()}`",
            "",
            "## Interview files",
            *[f"- {p.resolve()}" for p in self.state.interview_files],
            "",
            "## Quality by Interview",
            *[
                f"- {slug}: score={self.state.interview_quality_scores.get(slug, 0.0)}, "
                f"reruns={self.state.interview_reruns.get(slug, 0)}"
                for slug in self.state.selected_personas
            ],
            "",
            "## Aggregated broad signals",
            *_bullet_items(all_broad, "(none)"),
            "",
            "## Aggregated niche signals",
            *_bullet_items(all_niche, "(none)"),
        ]
        return "\n".join(lines)

    def _score_candidate(self, candidate: RoadmapCandidate) -> float:
        raw_score = (
            (candidate.reach * candidate.impact * candidate.confidence * candidate.strategic_fit)
            / candidate.effort
        ) - candidate.niche_penalty
        return round(max(raw_score, 0.0), 2)

    def _interview_quality_equation(self) -> str:
        return (
            "signal_quality_score = "
            "(question_depth * answer_specificity * signal_diversity * actionability) "
            "/ (noise_penalty + 1) - niche_bias_penalty"
        )

    def _score_interview_quality(
        self, assessment: InterviewQualityAssessment
    ) -> float:
        raw_score = (
            (
                assessment.question_depth
                * assessment.answer_specificity
                * assessment.signal_diversity
                * assessment.actionability
            )
            / (assessment.noise_penalty + 1)
        ) - assessment.niche_bias_penalty
        return round(max(raw_score, 0.0), 2)

    def _render_roadmap_markdown(self, pm_output: PMResponse) -> str:
        scored = [
            {"candidate": c, "score": self._score_candidate(c)}
            for c in pm_output.roadmap_candidates
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)

        grouped: dict[str, list[dict[str, Any]]] = {"now": [], "next": [], "later": []}
        for item in scored:
            grouped[item["candidate"].phase_suggestion].append(item)

        lines = [
            "# Product Roadmap Recommendation",
            "",
            "## Product Summary",
            pm_output.product_summary or "(not provided)",
            "",
            "## Prioritization Method",
            "Weighted score = (reach x impact x confidence x strategic_fit) / effort - niche_penalty",
            "",
            "## Accepted Signals",
            *_bullet_items(pm_output.accepted_signals, "(none)"),
            "",
            "## Rejected / Niche Signals",
            *_bullet_items(pm_output.rejected_signals, "(none)"),
            "",
        ]

        for bucket in ("now", "next", "later"):
            title = bucket.capitalize()
            lines.append(f"## {title}")
            if not grouped[bucket]:
                lines.append("- (no items)")
                lines.append("")
                continue

            for item in grouped[bucket]:
                c: RoadmapCandidate = item["candidate"]
                score: float = item["score"]
                lines.extend(
                    [
                        f"### {c.title}",
                        f"- Score: {score}",
                        f"- Expected impact: {c.expected_impact}",
                        f"- Rationale: {c.rationale}",
                        f"- Supporting signals: {', '.join(c.supporting_signals) if c.supporting_signals else '(none)'}",
                        f"- Risks: {', '.join(c.risks) if c.risks else '(none)'}",
                        f"- Assumptions: {', '.join(c.assumptions) if c.assumptions else '(none)'}",
                        (
                            f"- Inputs: reach={c.reach}, impact={c.impact}, confidence={c.confidence}, "
                            f"strategic_fit={c.strategic_fit}, effort={c.effort}, niche_penalty={c.niche_penalty}"
                        ),
                        "",
                    ]
                )

        lines.extend(
            [
                "## Deprioritized Feedback",
                *_bullet_items(pm_output.deprioritized_feedback, "(none)"),
            ]
        )
        return "\n".join(lines)

    def _finalize_run_summary(self) -> None:
        payload = {
            "run_id": self.run_id,
            "started_at": _now_iso(),
            "phase_final": self.state.phase.value,
            "roadmap_path": str(self.state.roadmap_saved_path) if self.state.roadmap_saved_path else None,
            "interview_files": [str(p) for p in self.state.interview_files],
            "selected_personas": self.state.selected_personas,
            "interview_quality_scores": self.state.interview_quality_scores,
            "interview_reruns": self.state.interview_reruns,
            "interview_quality_equation": self._interview_quality_equation(),
            "interview_quality_min_score": self.settings.interview_quality_min_score,
            "accepted_signals": self.state.accepted_signals,
            "rejected_signals": self.state.rejected_signals,
            "budget_exhausted": self.state.budget_exhausted,
            "budget_stop_reason": self.state.budget_stop_reason,
            "budget_snapshot": self._budget_snapshot(),
            "total_usage": self.state.total_usage,
            "turn_counter": self.state.turn_counter,
            "elapsed_seconds": round(time.monotonic() - self.start_monotonic, 2),
        }
        _write_text(self.summary_json_path, json.dumps(payload, indent=2, ensure_ascii=False))

    async def _run_interviews(
        self,
        interview_count: int,
        product_context: str,
        focus_areas: str,
    ) -> str:
        if self.state.budget_exhausted:
            return (
                "Interviews not started: budget exhausted.\n"
                f"Reason: {self.state.budget_stop_reason}\n"
                f"Budget snapshot: {json.dumps(self._budget_snapshot(), ensure_ascii=False)}"
            )

        selected_personas = self._select_personas(interview_count)
        self.state.selected_personas = [p.slug for p in selected_personas]
        self.state.interview_files.clear()
        self.state.interview_quality_scores.clear()
        self.state.interview_reruns.clear()
        reports: list[InterviewReport] = []
        budget_stopped = False

        self._record_event(
            "interviews_started",
            {
                "interview_count": interview_count,
                "selected_personas": self.state.selected_personas,
                "focus_areas": focus_areas,
                "quality_min_score": self.settings.interview_quality_min_score,
                "quality_max_reruns_per_interview": self.settings.quality_max_reruns_per_interview,
                "quality_equation": self._interview_quality_equation(),
                **self._budget_snapshot(),
            },
        )

        for index, persona in enumerate(selected_personas, start=1):
            if self.state.budget_exhausted:
                budget_stopped = True
                break

            tool_name = self._persona_tool_name(persona)
            base_prompt = (
                f"Conduct interview {index} of {interview_count}.\n"
                f"You must use simulator tool `{tool_name}` exactly once.\n\n"
                "Interview context from Product Manager:\n"
                f"{product_context.strip()}\n\n"
                "Focus areas to probe:\n"
                f"{(focus_areas or 'General value, usability, and prioritization feedback.').strip()}\n\n"
                "Return a complete structured interview report."
            )

            attempts = 0
            rerun_focus_areas: list[str] = []
            best_report: InterviewReport | None = None
            best_quality: InterviewQualityAssessment | None = None
            best_score = -1.0

            while True:
                attempts += 1
                interview_prompt = base_prompt
                if rerun_focus_areas:
                    interview_prompt += (
                        "\n\nRerun guidance from quality control:\n- "
                        + "\n- ".join(rerun_focus_areas)
                        + "\nAddress these weaknesses with deeper and more concrete probing."
                    )

                try:
                    self._enforce_budget_or_raise("interviewer_run")
                except BudgetExceededError as exc:
                    budget_stopped = True
                    self._record_event(
                        "interview_budget_stop",
                        {
                            "index": index,
                            "attempt": attempts,
                            "persona_slug": persona.slug,
                            "reason": str(exc),
                            **self._budget_snapshot(),
                        },
                    )
                    break

                interview_started = time.monotonic()
                interview_result = await Runner.run(
                    self.interviewer_agent,
                    interview_prompt,
                    context=self.state,
                    max_turns=self.settings.interviewer_max_turns,
                )
                interview_latency_ms = round((time.monotonic() - interview_started) * 1000, 2)
                interview_usage = self._usage_from_result(interview_result)
                self._add_usage(interview_usage)

                try:
                    interview_report = _coerce_model(
                        interview_result.final_output, InterviewReport
                    )
                except Exception:
                    interview_report = InterviewReport(
                        persona_name=persona.name,
                        persona_profile=persona.profile,
                        transcript=[
                            InterviewExchange(
                                question="(parse_error)",
                                answer=_to_text(interview_result.final_output),
                            )
                        ],
                        synthesis="Model output was not parseable as InterviewReport.",
                    )

                if not interview_report.persona_name:
                    interview_report.persona_name = persona.name
                if not interview_report.persona_profile:
                    interview_report.persona_profile = persona.profile

                quality_prompt = (
                    "Evaluate this interview report for decision-useful signal quality.\n"
                    f"Use this exact scoring equation for interpretation:\n{self._interview_quality_equation()}\n"
                    f"Threshold for shallow interview: score < {self.settings.interview_quality_min_score}\n\n"
                    "Interview report JSON:\n"
                    f"{interview_report.model_dump_json(indent=2)}"
                )
                try:
                    self._enforce_budget_or_raise("quality_checker_run")
                except BudgetExceededError as exc:
                    budget_stopped = True
                    self._record_event(
                        "interview_budget_stop",
                        {
                            "index": index,
                            "attempt": attempts,
                            "persona_slug": persona.slug,
                            "reason": str(exc),
                            **self._budget_snapshot(),
                        },
                    )
                    break

                quality_started = time.monotonic()
                quality_result = await Runner.run(
                    self.interview_quality_agent,
                    quality_prompt,
                    context=self.state,
                    max_turns=self.settings.quality_checker_max_turns,
                )
                quality_latency_ms = round((time.monotonic() - quality_started) * 1000, 2)
                quality_usage = self._usage_from_result(quality_result)
                self._add_usage(quality_usage)

                try:
                    quality = _coerce_model(
                        quality_result.final_output, InterviewQualityAssessment
                    )
                except Exception:
                    quality = InterviewQualityAssessment(
                        summary="Quality output parse failure; defaulting to conservative low score.",
                        strengths=[],
                        weaknesses=["Failed to parse QC output"],
                        rerun_focus_areas=["Clarify pain points with concrete examples"],
                        question_depth=1,
                        answer_specificity=1,
                        signal_diversity=1,
                        actionability=1,
                        noise_penalty=3,
                        niche_bias_penalty=1,
                        rerun_recommended=True,
                    )

                quality_score = self._score_interview_quality(quality)
                is_shallow = quality_score < self.settings.interview_quality_min_score

                if quality_score > best_score:
                    best_score = quality_score
                    best_report = interview_report
                    best_quality = quality

                rerun_allowed = attempts <= self.settings.quality_max_reruns_per_interview
                should_rerun = is_shallow and rerun_allowed
                rerun_focus_areas = quality.rerun_focus_areas or quality.weaknesses

                self._record_event(
                    "interview_quality_scored",
                    {
                        "index": index,
                        "attempt": attempts,
                        "persona_slug": persona.slug,
                        "persona_name": persona.name,
                        "quality_score": quality_score,
                        "quality_min_score": self.settings.interview_quality_min_score,
                        "quality_equation": self._interview_quality_equation(),
                        "is_shallow": is_shallow,
                        "rerun_recommended_by_agent": quality.rerun_recommended,
                        "rerun_allowed": rerun_allowed,
                        "should_rerun": should_rerun,
                        "quality_inputs": quality.model_dump(mode="json"),
                        "interview_latency_ms": interview_latency_ms,
                        "quality_latency_ms": quality_latency_ms,
                        "interview_usage": interview_usage,
                        "quality_usage": quality_usage,
                        "interview_turns": getattr(interview_result, "current_turn", None),
                        "quality_turns": getattr(quality_result, "current_turn", None),
                    },
                )

                if should_rerun:
                    continue
                break

            if budget_stopped and best_report is None:
                break

            if best_report is None:
                best_report = InterviewReport(
                    persona_name=persona.name,
                    persona_profile=persona.profile,
                    transcript=[],
                    synthesis="No interview data captured.",
                )
            if best_quality is None:
                best_quality = InterviewQualityAssessment(
                    summary="No quality data captured.",
                    strengths=[],
                    weaknesses=["Missing quality data"],
                    rerun_focus_areas=[],
                    question_depth=1,
                    answer_specificity=1,
                    signal_diversity=1,
                    actionability=1,
                    noise_penalty=5,
                    niche_bias_penalty=1,
                    rerun_recommended=False,
                )

            reruns_used = max(0, attempts - 1)
            self.state.interview_quality_scores[persona.slug] = best_score
            self.state.interview_reruns[persona.slug] = reruns_used

            file_path = self.interviews_dir / f"interview_{index:02d}_{persona.slug}.md"
            _write_text(
                file_path,
                self._render_interview_markdown(
                    best_report,
                    index,
                    quality=best_quality,
                    quality_score=best_score,
                    reruns_used=reruns_used,
                ),
            )
            self.state.interview_files.append(file_path)
            reports.append(best_report)

            self._record_event(
                "interview_completed",
                {
                    "index": index,
                    "persona_slug": persona.slug,
                    "persona_name": persona.name,
                    "prompt": base_prompt,
                    "output_file": str(file_path.resolve()),
                    "broad_signal_count": len(best_report.broad_signals),
                    "niche_signal_count": len(best_report.niche_signals),
                    "quality_score": best_score,
                    "reruns_used": reruns_used,
                    **self._budget_snapshot(),
                },
            )

            if budget_stopped:
                break

        _write_text(
            self.summary_path,
            self._render_interview_summary_markdown(reports, interview_count),
        )
        self._record_event(
            "interviews_finished",
            {
                "summary_file": str(self.summary_path.resolve()),
                "interview_files": [str(p.resolve()) for p in self.state.interview_files],
                "budget_stopped": budget_stopped,
                **self._budget_snapshot(),
            },
        )

        result_text = (
            "Interviews completed.\n"
            f"- Count: {interview_count}\n"
            f"- Summary document: {self.summary_path.resolve()}\n"
            "- Individual interview documents:\n"
            + "\n".join(f"  - {p.resolve()}" for p in self.state.interview_files)
        )
        if budget_stopped:
            result_text += (
                "\n\nInterview process stopped early due to budget governor.\n"
                f"Reason: {self.state.budget_stop_reason}\n"
                f"Budget snapshot: {json.dumps(self._budget_snapshot(), ensure_ascii=False)}"
            )
        return result_text

    def _compose_pm_input(self, user_input: str) -> str:
        interview_count_text = (
            str(self.state.confirmed_interview_count)
            if self.state.confirmed_interview_count is not None
            else "unset"
        )
        return (
            "Workflow context:\n"
            f"- Current phase: {self.state.phase.value}\n"
            f"- Interview question asked: {self.state.interview_question_asked}\n"
            f"- Confirmed interview count: {interview_count_text}\n"
            f"- Max interviews: {self.settings.max_interviews}\n"
            f"- Interview summary path: {self.summary_path.resolve()}\n\n"
            f"Human message:\n{user_input}"
        )

    def _sync_signals(self, pm_output: PMResponse) -> None:
        for signal in pm_output.accepted_signals:
            if signal not in self.state.accepted_signals:
                self.state.accepted_signals.append(signal)
        for signal in pm_output.rejected_signals:
            if signal not in self.state.rejected_signals:
                self.state.rejected_signals.append(signal)

    async def run_cli(self) -> None:
        print("Product roadmap workflow started.")
        print("Enter your product context. Type 'exit' to stop.")
        print(f"Observability run directory: {self.run_dir.resolve()}")

        self._record_event(
            "session_started",
            {
                "model": self.settings.model,
                "pm_max_turns": self.settings.pm_max_turns,
                "interviewer_max_turns": self.settings.interviewer_max_turns,
                "simulator_max_turns": self.settings.simulator_max_turns,
                "quality_checker_max_turns": self.settings.quality_checker_max_turns,
                "max_output_tokens": self.settings.max_output_tokens,
                "max_interviews": self.settings.max_interviews,
                "interview_quality_min_score": self.settings.interview_quality_min_score,
                "quality_max_reruns_per_interview": self.settings.quality_max_reruns_per_interview,
                "total_token_budget": self.settings.total_token_budget,
                "max_estimated_cost_usd": self.settings.max_estimated_cost_usd,
                "est_input_cost_per_1m_tokens_usd": self.settings.est_input_cost_per_1m_tokens_usd,
                "est_output_cost_per_1m_tokens_usd": self.settings.est_output_cost_per_1m_tokens_usd,
                **self._budget_snapshot(),
            },
        )

        while True:
            if self.state.budget_exhausted:
                print(
                    "\nBudget governor stop: "
                    f"{self.state.budget_stop_reason}. "
                    f"Snapshot: {json.dumps(self._budget_snapshot(), ensure_ascii=False)}"
                )
                break

            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                self._record_event("session_exit_requested", {"user_input": user_input})
                break

            if self.state.phase == PMPhase.ASK_INTERVIEW_COUNT:
                parsed_count = self._extract_interview_count(user_input)
                if parsed_count is not None:
                    self.state.confirmed_interview_count = parsed_count
                    self.state.phase = PMPhase.INTERVIEWS
                    self._record_event(
                        "interview_count_confirmed",
                        {"confirmed_interview_count": parsed_count},
                    )

            phase_before = self.state.phase.value
            prompt_for_pm = self._compose_pm_input(user_input)
            try:
                self._enforce_budget_or_raise("pm_run")
            except BudgetExceededError as exc:
                self._record_event(
                    "pm_budget_stop",
                    {
                        "user_input": user_input,
                        "reason": str(exc),
                        **self._budget_snapshot(),
                    },
                )
                print(
                    "\nBudget governor stop: "
                    f"{self.state.budget_stop_reason}. "
                    f"Snapshot: {json.dumps(self._budget_snapshot(), ensure_ascii=False)}"
                )
                break

            started = time.monotonic()
            result = await Runner.run(
                self.pm_agent,
                prompt_for_pm,
                context=self.state,
                max_turns=self.settings.pm_max_turns,
                conversation_id=self.conversation_id,
            )
            latency_ms = round((time.monotonic() - started) * 1000, 2)
            usage = self._usage_from_result(result)
            self._add_usage(usage)
            self.state.turn_counter += 1

            try:
                pm_output = _coerce_model(result.final_output, PMResponse)
            except Exception:
                pm_output = self._pm_fallback_response(result.final_output)

            if pm_output.ask_interview_count:
                self.state.interview_question_asked = True
                self.state.phase = PMPhase.ASK_INTERVIEW_COUNT
                pm_output.assistant_message = self._ensure_interview_question(
                    pm_output.assistant_message
                )
            elif self._question_in_text(pm_output.assistant_message):
                self.state.interview_question_asked = True
                self.state.phase = PMPhase.ASK_INTERVIEW_COUNT

            if self.state.phase == PMPhase.SYNTHESIS and pm_output.roadmap_ready:
                self.state.phase = PMPhase.ROADMAP
                roadmap_markdown = self._render_roadmap_markdown(pm_output)
                _write_text(self.roadmap_path, roadmap_markdown)
                self.state.roadmap_saved_path = self.roadmap_path
                self._record_event(
                    "roadmap_saved",
                    {
                        "roadmap_file": str(self.roadmap_path.resolve()),
                        "candidate_count": len(pm_output.roadmap_candidates),
                    },
                )

            self._sync_signals(pm_output)

            self._record_event(
                "pm_turn_completed",
                {
                    "phase_before": phase_before,
                    "phase_after": self.state.phase.value,
                    "user_input": user_input,
                    "prompt_sent_to_pm": prompt_for_pm,
                    "pm_output": pm_output.model_dump(mode="json"),
                    "latency_ms": latency_ms,
                    "usage": usage,
                    "turns": getattr(result, "current_turn", None),
                    **self._budget_snapshot(),
                },
            )

            print(f"\nProduct Manager:\n{pm_output.assistant_message.strip()}")

            if self.state.budget_exhausted:
                print(
                    "\nBudget governor stop: "
                    f"{self.state.budget_stop_reason}. "
                    f"Snapshot: {json.dumps(self._budget_snapshot(), ensure_ascii=False)}"
                )
                break

            if (
                self.state.phase == PMPhase.ROADMAP
                and self.state.roadmap_saved_path is not None
                and pm_output.roadmap_ready
            ):
                print(f"\nRoadmap saved at: {self.state.roadmap_saved_path.resolve()}")
                break

        self._record_event(
            "session_finished",
            {
                "phase_final": self.state.phase.value,
                "roadmap_saved_path": (
                    str(self.state.roadmap_saved_path.resolve())
                    if self.state.roadmap_saved_path is not None
                    else None
                ),
            },
        )
        self._finalize_run_summary()

        if self.state.roadmap_saved_path is None:
            print("\nSession ended before roadmap generation.")
        print(f"Observability artifacts: {self.run_dir.resolve()}")


async def run_workflow(settings: WorkflowSettings) -> None:
    workflow = ProductRoadmapWorkflow(settings)
    await workflow.run_cli()


def run_workflow_sync(settings: WorkflowSettings) -> None:
    asyncio.run(run_workflow(settings))

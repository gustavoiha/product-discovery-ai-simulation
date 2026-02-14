from __future__ import annotations

import asyncio
import json
import random
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool

try:
    from agents import set_default_openai_api
except ImportError:  # pragma: no cover - backwards compatibility
    set_default_openai_api = None  # type: ignore[assignment]

try:
    from agents import set_tracing_disabled
except ImportError:  # pragma: no cover - backwards compatibility
    set_tracing_disabled = None  # type: ignore[assignment]

from .config import WorkflowSettings
from .personas import Persona, USER_SIMULATOR_PERSONAS


def _to_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output.strip()
    try:
        return json.dumps(output, indent=2, ensure_ascii=False)
    except TypeError:
        return str(output)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


@dataclass(slots=True)
class WorkflowState:
    settings: WorkflowSettings
    output_dir: Path
    interview_question_asked: bool = False
    confirmed_interview_count: int | None = None
    roadmap_saved_path: Path | None = None
    interview_files: list[Path] = field(default_factory=list)


class ProductRoadmapWorkflow:
    def __init__(self, settings: WorkflowSettings):
        self.settings = settings
        self.output_dir = settings.output_dir.resolve()
        self.interviews_dir = self.output_dir / "interviews"
        self.roadmap_path = self.output_dir / "roadmap_recommendation.md"
        self.summary_path = self.interviews_dir / "INTERVIEW_SUMMARY.md"
        self.state = WorkflowState(settings=settings, output_dir=self.output_dir)
        self.conversation_id = f"product-roadmap-{uuid.uuid4()}"
        self._rng = random.Random(11)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interviews_dir.mkdir(parents=True, exist_ok=True)

        if set_default_openai_api is not None:
            set_default_openai_api("responses")
        if set_tracing_disabled is not None:
            set_tracing_disabled(settings.disable_tracing)

        self.simulator_agents = self._build_user_simulator_agents()
        self.interviewer_agent = self._build_interviewer_agent()
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
                "Behavior rules:\n"
                "- Answer with realistic user-level detail and concrete examples.\n"
                "- Share pain points, desired outcomes, and tradeoffs.\n"
                "- You can disagree or reject ideas when they are not valuable.\n"
                "- Avoid generic positivity; be specific and practical.\n"
                "- If a request is niche, acknowledge it as niche.\n"
                "- Keep responses concise."
            )

            agents_by_slug[persona.slug] = Agent(
                name=f"User Simulator - {persona.name}",
                instructions=instructions,
                model=self.settings.model,
                model_settings=self._model_settings(),
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
            "You must use the simulator tool explicitly requested in the prompt.\n\n"
            "Interview quality rules:\n"
            "- Ask focused product discovery questions.\n"
            "- Cover context, current workflow, pain points, desired outcomes, and willingness to adopt.\n"
            "- Elicit concrete examples, not abstract opinions.\n"
            "- Distinguish between broad needs and niche requests.\n\n"
            "Output format in markdown:\n"
            "## Persona\n"
            "## Transcript (compact Q/A)\n"
            "## Key Needs\n"
            "## Feature Ideas Mentioned\n"
            "## Risks / Objections\n"
            "## Signal Quality (Broad vs Niche)\n"
            "Keep output concise and decision-oriented."
        )

        return Agent(
            name="Interviewer",
            instructions=instructions,
            model=self.settings.model,
            model_settings=self._model_settings(),
            tools=tools,
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

            return await self._run_interviews(
                interview_count=interview_count,
                product_context=product_context,
                focus_areas=focus_areas,
            )

        @function_tool
        def save_roadmap_recommendation(
            context: RunContextWrapper[WorkflowState],
            roadmap_markdown: str,
        ) -> str:
            _write_text(self.roadmap_path, roadmap_markdown)
            context.context.roadmap_saved_path = self.roadmap_path
            return f"Saved roadmap to {self.roadmap_path.resolve()}"

        instructions = (
            "You are the Product Manager and the only point of contact with the human product owner.\n\n"
            "Mission:\n"
            "Produce a practical future product roadmap based on simulated user interviews.\n\n"
            "Required behavior:\n"
            "- Ask clarification questions to understand: product type (B2B/B2C), customer segments, "
            "platforms, existing features, business goals, constraints, and ideas to test.\n"
            "- You MUST ask the human exactly this question before running interviews: "
            "\"How many interviews should I run?\"\n"
            "- Do not call interview tools until the human answers with a number.\n"
            "- Once you have enough context and the interview count, call run_user_research_interviews.\n"
            "- After interviews complete, synthesize insights and prioritize by strategic value.\n"
            "- Explicitly down-rank or ignore feedback that is too niche, contradictory, or weakly supported.\n"
            "- Before your final answer, call save_roadmap_recommendation with the complete markdown roadmap.\n\n"
            "Roadmap quality bar:\n"
            "- Separate recommendations into Now, Next, Later.\n"
            "- Include rationale, expected impact, risks, and assumptions.\n"
            "- Tie recommendations back to repeated interview signals.\n"
            "- Mention which feedback was deprioritized as niche/noisy.\n\n"
            "Final response format:\n"
            "- Start with ROADMAP_READY\n"
            "- Include the saved roadmap path.\n"
            "- Keep it concise but specific."
        )

        return Agent(
            name="Product Manager",
            instructions=instructions,
            model=self.settings.model,
            model_settings=self._model_settings(),
            tools=[run_user_research_interviews, save_roadmap_recommendation],
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

    async def _run_interviews(
        self,
        interview_count: int,
        product_context: str,
        focus_areas: str,
    ) -> str:
        selected_personas = self._select_personas(interview_count)
        report_blobs: list[str] = []
        self.state.interview_files.clear()

        for index, persona in enumerate(selected_personas, start=1):
            tool_name = self._persona_tool_name(persona)
            interview_prompt = (
                f"Conduct interview {index} of {interview_count}.\n"
                f"You must use simulator tool `{tool_name}` exactly once.\n\n"
                "Interview context from Product Manager:\n"
                f"{product_context.strip()}\n\n"
                "Focus areas to probe:\n"
                f"{(focus_areas or 'General value, usability, and prioritization feedback.').strip()}\n\n"
                "Ask enough discovery questions to produce a useful transcript and synthesis."
            )

            result = await Runner.run(
                self.interviewer_agent,
                interview_prompt,
                context=self.state,
                max_turns=self.settings.interviewer_max_turns,
            )

            interview_text = _to_text(result.final_output)
            file_path = self.interviews_dir / f"interview_{index:02d}_{persona.slug}.md"
            _write_text(
                file_path,
                (
                    f"# Interview {index}: {persona.name}\n\n"
                    f"Persona profile: {persona.profile}\n\n"
                    f"{interview_text}"
                ),
            )

            self.state.interview_files.append(file_path)
            report_blobs.append(
                (
                    f"## Interview {index} - {persona.name}\n"
                    f"File: {file_path.resolve()}\n\n"
                    f"{interview_text}"
                )
            )

        summary_markdown = (
            "# Interview Summary\n\n"
            f"Total interviews: {interview_count}\n\n"
            "## Interview files\n"
            + "\n".join(f"- {p.resolve()}" for p in self.state.interview_files)
            + "\n\n"
            + "\n\n".join(report_blobs)
        )
        _write_text(self.summary_path, summary_markdown)

        return (
            "Interviews completed.\n"
            f"- Count: {interview_count}\n"
            f"- Summary document: {self.summary_path.resolve()}\n"
            "- Individual interview documents:\n"
            + "\n".join(f"  - {p.resolve()}" for p in self.state.interview_files)
            + "\n\nUse the interview findings below to build roadmap recommendations:\n\n"
            + "\n\n".join(report_blobs)
        )

    def _pm_asked_for_interview_count(self, pm_text: str) -> bool:
        lower = pm_text.lower()
        return "how many interviews should i run" in lower or (
            "how many interviews" in lower and "run" in lower
        )

    def _extract_interview_count(self, user_text: str) -> int | None:
        match = re.search(r"\b(\d{1,3})\b", user_text)
        if not match:
            return None
        return int(match.group(1))

    async def run_cli(self) -> None:
        print("Product roadmap workflow started.")
        print("Enter your product context. Type 'exit' to stop.")

        while True:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                break

            if self.state.interview_question_asked:
                parsed_count = self._extract_interview_count(user_input)
                if parsed_count is not None:
                    self.state.confirmed_interview_count = parsed_count

            result = await Runner.run(
                self.pm_agent,
                user_input,
                context=self.state,
                max_turns=self.settings.pm_max_turns,
                conversation_id=self.conversation_id,
            )
            pm_text = _to_text(result.final_output)

            if self._pm_asked_for_interview_count(pm_text):
                self.state.interview_question_asked = True

            print(f"\nProduct Manager:\n{pm_text}")

            if pm_text.lstrip().startswith("ROADMAP_READY"):
                break

        if self.state.roadmap_saved_path is not None:
            print(f"\nRoadmap saved at: {self.state.roadmap_saved_path.resolve()}")
        else:
            print("\nSession ended before roadmap generation.")


async def run_workflow(settings: WorkflowSettings) -> None:
    workflow = ProductRoadmapWorkflow(settings)
    await workflow.run_cli()


def run_workflow_sync(settings: WorkflowSettings) -> None:
    asyncio.run(run_workflow(settings))


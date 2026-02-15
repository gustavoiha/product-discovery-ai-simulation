from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PMPhase(str, Enum):
    DISCOVERY = "discovery"
    ASK_INTERVIEW_COUNT = "ask_interview_count"
    INTERVIEWS = "interviews"
    SYNTHESIS = "synthesis"
    ROADMAP = "roadmap"


class SimulatorResponse(BaseModel):
    user_context: str = ""
    current_workflow: str = ""
    pain_points: list[str] = Field(default_factory=list)
    desired_outcomes: list[str] = Field(default_factory=list)
    feature_requests: list[str] = Field(default_factory=list)
    risks_or_objections: list[str] = Field(default_factory=list)
    broad_signals: list[str] = Field(default_factory=list)
    niche_signals: list[str] = Field(default_factory=list)


class InterviewExchange(BaseModel):
    question: str
    answer: str


class InterviewReport(BaseModel):
    persona_name: str
    persona_profile: str = ""
    transcript: list[InterviewExchange] = Field(default_factory=list)
    key_needs: list[str] = Field(default_factory=list)
    feature_ideas: list[str] = Field(default_factory=list)
    risks_or_objections: list[str] = Field(default_factory=list)
    broad_signals: list[str] = Field(default_factory=list)
    niche_signals: list[str] = Field(default_factory=list)
    synthesis: str = ""


class InterviewQualityAssessment(BaseModel):
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    rerun_focus_areas: list[str] = Field(default_factory=list)
    question_depth: int = Field(ge=1, le=5)
    answer_specificity: int = Field(ge=1, le=5)
    signal_diversity: int = Field(ge=1, le=5)
    actionability: int = Field(ge=1, le=5)
    noise_penalty: int = Field(default=0, ge=0, le=5)
    niche_bias_penalty: int = Field(default=0, ge=0, le=5)
    rerun_recommended: bool = False


class RoadmapCandidate(BaseModel):
    title: str
    phase_suggestion: Literal["now", "next", "later"] = "next"
    rationale: str
    expected_impact: str
    risks: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    supporting_signals: list[str] = Field(default_factory=list)
    reach: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    confidence: int = Field(ge=1, le=5)
    strategic_fit: int = Field(ge=1, le=5)
    effort: int = Field(ge=1, le=5)
    niche_penalty: int = Field(default=0, ge=0, le=5)


class PMResponse(BaseModel):
    phase: PMPhase
    assistant_message: str
    ask_interview_count: bool = False
    ready_to_run_interviews: bool = False
    roadmap_ready: bool = False
    product_summary: str = ""
    focus_areas: list[str] = Field(default_factory=list)
    accepted_signals: list[str] = Field(default_factory=list)
    rejected_signals: list[str] = Field(default_factory=list)
    deprioritized_feedback: list[str] = Field(default_factory=list)
    roadmap_candidates: list[RoadmapCandidate] = Field(default_factory=list)

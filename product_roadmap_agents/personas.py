from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Persona:
    slug: str
    name: str
    segment: str
    profile: str


DEFAULT_USER_SIMULATOR_PERSONAS: tuple[Persona, ...] = (
    Persona(
        slug="startup_ops_lead",
        name="Startup Operations Lead",
        segment="b2b_startup",
        profile=(
            "Works at a B2B SaaS startup with a small team, values automation and "
            "speed, frustrated by onboarding friction, highly ROI-focused."
        ),
    ),
    Persona(
        slug="enterprise_admin",
        name="Enterprise IT Admin",
        segment="enterprise",
        profile=(
            "Manages procurement and security review in a large enterprise, "
            "prioritizes reliability, permissions, auditability, and compliance."
        ),
    ),
    Persona(
        slug="power_user_analyst",
        name="Power User Analyst",
        segment="power_user",
        profile=(
            "Heavy daily user who builds workflows and reports, wants advanced "
            "customization and data exports, impatient with repetitive manual work."
        ),
    ),
    Persona(
        slug="mobile_first_founder",
        name="Mobile-First Founder",
        segment="mobile_smb",
        profile=(
            "Runs a small business mostly from a phone, needs fast task completion "
            "on mobile, cares about simple UX and low cognitive load."
        ),
    ),
    Persona(
        slug="budget_sensitive_owner",
        name="Budget-Sensitive Owner",
        segment="budget_sensitive",
        profile=(
            "Owner of a small business with strict budget limits, price-sensitive, "
            "compares alternatives frequently, seeks clear value from each feature."
        ),
    ),
    Persona(
        slug="integration_engineer",
        name="Integration Engineer",
        segment="technical_buyer",
        profile=(
            "Technical buyer who evaluates APIs and integrations, values extensibility "
            "and good docs, dislikes black-box systems and vendor lock-in."
        ),
    ),
    Persona(
        slug="customer_support_manager",
        name="Customer Support Manager",
        segment="operations_leader",
        profile=(
            "Leads support team operations, values workflows that reduce ticket volume, "
            "cares about edge cases and customer frustration patterns."
        ),
    ),
    Persona(
        slug="skeptical_executive",
        name="Skeptical Executive Sponsor",
        segment="executive",
        profile=(
            "Senior stakeholder focused on business outcomes, wants evidence-backed "
            "prioritization and strategic differentiation, dismisses vanity features."
        ),
    ),
)


def _coerce_persona(raw: dict[str, Any], index: int) -> Persona:
    slug = str(raw.get("slug", "")).strip()
    name = str(raw.get("name", "")).strip()
    segment = str(raw.get("segment", "")).strip()
    profile = str(raw.get("profile", "")).strip()
    if not slug or not name or not segment or not profile:
        raise ValueError(
            f"Invalid persona at index {index}: slug, name, segment, and profile are required."
        )
    return Persona(slug=slug, name=name, segment=segment, profile=profile)


def _validate_personas(personas: list[Persona]) -> tuple[Persona, ...]:
    if not personas:
        raise ValueError("At least one persona must be configured.")

    slugs = [p.slug for p in personas]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Persona slugs must be unique.")

    segments = [p.segment for p in personas]
    if any(not segment for segment in segments):
        raise ValueError("Each persona must have a non-empty segment.")

    return tuple(personas)


def load_personas(config_path: Path) -> tuple[Persona, ...]:
    if not config_path.exists():
        return DEFAULT_USER_SIMULATOR_PERSONAS

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Persona config must be a mapping at {config_path}. Expected key: personas."
        )

    entries = raw.get("personas")
    if not isinstance(entries, list):
        raise ValueError(
            f"Persona config at {config_path} must define a list under 'personas'."
        )

    personas = [_coerce_persona(entry, i) for i, entry in enumerate(entries)]
    return _validate_personas(personas)


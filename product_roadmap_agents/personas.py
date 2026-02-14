from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Persona:
    slug: str
    name: str
    profile: str


USER_SIMULATOR_PERSONAS: tuple[Persona, ...] = (
    Persona(
        slug="startup_ops_lead",
        name="Startup Operations Lead",
        profile=(
            "Works at a B2B SaaS startup with a small team, values automation and "
            "speed, frustrated by onboarding friction, highly ROI-focused."
        ),
    ),
    Persona(
        slug="enterprise_admin",
        name="Enterprise IT Admin",
        profile=(
            "Manages procurement and security review in a large enterprise, "
            "prioritizes reliability, permissions, auditability, and compliance."
        ),
    ),
    Persona(
        slug="power_user_analyst",
        name="Power User Analyst",
        profile=(
            "Heavy daily user who builds workflows and reports, wants advanced "
            "customization and data exports, impatient with repetitive manual work."
        ),
    ),
    Persona(
        slug="mobile_first_founder",
        name="Mobile-First Founder",
        profile=(
            "Runs a small business mostly from a phone, needs fast task completion "
            "on mobile, cares about simple UX and low cognitive load."
        ),
    ),
    Persona(
        slug="budget_sensitive_owner",
        name="Budget-Sensitive Owner",
        profile=(
            "Owner of a small business with strict budget limits, price-sensitive, "
            "compares alternatives frequently, seeks clear value from each feature."
        ),
    ),
    Persona(
        slug="integration_engineer",
        name="Integration Engineer",
        profile=(
            "Technical buyer who evaluates APIs and integrations, values extensibility "
            "and good docs, dislikes black-box systems and vendor lock-in."
        ),
    ),
    Persona(
        slug="customer_support_manager",
        name="Customer Support Manager",
        profile=(
            "Leads support team operations, values workflows that reduce ticket volume, "
            "cares about edge cases and customer frustration patterns."
        ),
    ),
    Persona(
        slug="skeptical_executive",
        name="Skeptical Executive Sponsor",
        profile=(
            "Senior stakeholder focused on business outcomes, wants evidence-backed "
            "prioritization and strategic differentiation, dismisses vanity features."
        ),
    ),
)


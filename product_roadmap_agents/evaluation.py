from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class GoldenScenario:
    scenario_id: str
    description: str
    expected_characteristics: tuple[str, ...]


def load_golden_scenarios(path: Path) -> list[GoldenScenario]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Golden scenarios file must be a mapping with key 'scenarios'.")
    scenarios = raw.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Golden scenarios file must define a list under 'scenarios'.")

    parsed: list[GoldenScenario] = []
    for item in scenarios:
        if not isinstance(item, dict):
            raise ValueError("Scenario entries must be mappings.")
        scenario_id = str(item.get("id", "")).strip()
        description = str(item.get("description", "")).strip()
        characteristics_raw = item.get("expected_characteristics", [])
        if not isinstance(characteristics_raw, list):
            raise ValueError(f"Scenario '{scenario_id}' must have a list of expected_characteristics.")
        characteristics = tuple(str(value).strip() for value in characteristics_raw if str(value).strip())
        if not scenario_id or not description or not characteristics:
            raise ValueError(
                "Each scenario must include non-empty id, description, and expected_characteristics."
            )
        parsed.append(
            GoldenScenario(
                scenario_id=scenario_id,
                description=description,
                expected_characteristics=characteristics,
            )
        )
    return parsed


def _evaluate_rule(markdown_text: str, rule: str) -> tuple[bool, str]:
    if ":" not in rule:
        return False, f"Unsupported rule format: {rule}"
    prefix, value = rule.split(":", 1)
    prefix = prefix.strip()
    value = value.strip()
    lower_markdown = markdown_text.lower()

    if prefix == "contains_section":
        ok = value in markdown_text
        return ok, f"missing section '{value}'"

    if prefix == "contains_keyword":
        ok = value.lower() in lower_markdown
        return ok, f"missing keyword '{value}'"

    if prefix == "min_occurrences":
        if "|" not in value:
            return False, f"min_occurrences rule must use 'pattern|count': {rule}"
        pattern, count_raw = value.rsplit("|", 1)
        pattern = pattern.strip()
        try:
            min_count = int(count_raw.strip())
        except ValueError:
            return False, f"invalid min_occurrences count in rule: {rule}"
        current = markdown_text.count(pattern)
        ok = current >= min_count
        return ok, f"pattern '{pattern}' found {current} times, expected >= {min_count}"

    return False, f"unsupported rule prefix '{prefix}' in '{rule}'"


def evaluate_roadmap_characteristics(
    markdown_text: str, expected_characteristics: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    failures: list[str] = []
    for rule in expected_characteristics:
        ok, detail = _evaluate_rule(markdown_text, rule)
        if not ok:
            failures.append(detail)
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


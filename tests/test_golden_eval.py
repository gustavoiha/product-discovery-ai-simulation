from __future__ import annotations

from pathlib import Path

from product_roadmap_agents.evaluation import (
    evaluate_roadmap_characteristics,
    load_golden_scenarios,
)


def test_golden_scenarios_samples_pass() -> None:
    scenarios = load_golden_scenarios(Path("evaluation/golden_scenarios.yaml"))
    for scenario in scenarios:
        sample_path = Path("evaluation/sample_roadmaps") / f"{scenario.scenario_id}.md"
        assert sample_path.exists(), f"Missing sample roadmap for {scenario.scenario_id}"
        result = evaluate_roadmap_characteristics(
            sample_path.read_text(encoding="utf-8"),
            scenario.expected_characteristics,
        )
        assert result["passed"], f"Scenario {scenario.scenario_id} failed: {result['failures']}"


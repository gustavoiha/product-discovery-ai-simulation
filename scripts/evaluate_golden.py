#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from product_roadmap_agents.evaluation import (
    evaluate_roadmap_characteristics,
    load_golden_scenarios,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate roadmap outputs against golden scenario characteristics."
    )
    parser.add_argument(
        "--scenarios",
        default="evaluation/golden_scenarios.yaml",
        help="Path to golden scenarios YAML.",
    )
    parser.add_argument(
        "--samples-dir",
        default="evaluation/sample_roadmaps",
        help="Directory containing sample roadmap markdown files named <scenario_id>.md.",
    )
    parser.add_argument(
        "--scenario-id",
        default="",
        help="Optional single scenario id to evaluate.",
    )
    parser.add_argument(
        "--roadmap-file",
        default="",
        help="Optional roadmap markdown path. If provided with --scenario-id, evaluates that file against the selected scenario.",
    )
    args = parser.parse_args()

    scenarios = load_golden_scenarios(Path(args.scenarios))
    selected = scenarios
    if args.scenario_id:
        selected = [s for s in scenarios if s.scenario_id == args.scenario_id]
        if not selected:
            print(f"Scenario id not found: {args.scenario_id}")
            return 1

    failures = 0
    for scenario in selected:
        if args.roadmap_file:
            roadmap_path = Path(args.roadmap_file)
        else:
            roadmap_path = Path(args.samples_dir) / f"{scenario.scenario_id}.md"
        if not roadmap_path.exists():
            print(f"[FAIL] {scenario.scenario_id}: roadmap file not found: {roadmap_path}")
            failures += 1
            continue

        markdown_text = roadmap_path.read_text(encoding="utf-8")
        result = evaluate_roadmap_characteristics(
            markdown_text, scenario.expected_characteristics
        )
        if result["passed"]:
            print(f"[PASS] {scenario.scenario_id}: {roadmap_path}")
        else:
            print(f"[FAIL] {scenario.scenario_id}: {roadmap_path}")
            for detail in result["failures"]:
                print(f"  - {detail}")
            failures += 1

    if failures:
        print(f"\nGolden evaluation failed with {failures} scenario(s).")
        return 1

    print("\nGolden evaluation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

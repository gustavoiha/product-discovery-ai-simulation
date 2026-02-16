from __future__ import annotations

from pathlib import Path

from product_roadmap_agents.personas import load_personas


def test_personas_yaml_loads_with_segments() -> None:
    personas = load_personas(Path("config/personas.yaml"))
    assert personas
    assert all(persona.segment for persona in personas)
    assert len({persona.slug for persona in personas}) == len(personas)


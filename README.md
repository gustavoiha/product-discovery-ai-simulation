# Product Roadmap Multi-Agent Workflow

Local multi-agent workflow built with the OpenAI Agents SDK, following the orchestration style from the Codex + Agents SDK guide:

- https://developers.openai.com/codex/guides/agents-sdk/

## What this does

- Uses a `Product Manager` agent as the human-facing orchestrator.
- The PM asks clarification questions about your product (business model, platforms, current features, ideas, constraints).
- The PM must ask: **"How many interviews should I run?"**
- The PM calls an `Interviewer` agent to run simulated user interviews.
- The interviewer conducts interviews with multiple persona-based `User Simulator` agents.
- Each interview is saved as its own markdown file.
- The PM synthesizes findings and generates a practical product roadmap, filtering out noisy/niche feedback.

## Cost and usage controls in code

The workflow includes built-in API budget controls:

- `max_turns` limits for PM, interviewer, and simulator runs.
- `max_tokens` caps via `ModelSettings`.
- hard max interview count (`MAX_INTERVIEWS`).

## Quickstart

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e .
```

3. Configure environment:

```bash
cp .env.example .env
# then set OPENAI_API_KEY in .env
```

4. Run:

```bash
roadmap-agents
```

## Output files

- Individual interviews:
  - `outputs/interviews/interview_01_*.md`
  - `outputs/interviews/interview_02_*.md`
  - ...
- Interview rollup:
  - `outputs/interviews/INTERVIEW_SUMMARY.md`
- Final roadmap:
  - `outputs/roadmap_recommendation.md`

## Configuration

You can set these in `.env`:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-5`)
- `PM_MAX_TURNS` (default: `14`)
- `INTERVIEWER_MAX_TURNS` (default: `8`)
- `SIMULATOR_MAX_TURNS` (default: `4`)
- `MAX_OUTPUT_TOKENS` (default: `900`)
- `MAX_INTERVIEWS` (default: `8`)
- `OUTPUT_DIR` (default: `outputs`)
- `DISABLE_TRACING` (`true`/`false`)
Orchestrate AI agents to simulate user interviewing for product research

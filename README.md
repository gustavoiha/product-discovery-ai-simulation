# Product Roadmap Multi-Agent Workflow

Local multi-agent workflow built with the OpenAI Agents SDK, following the orchestration style from the Codex + Agents SDK guide:

- https://developers.openai.com/codex/guides/agents-sdk/

## What this does

- Uses a `Product Manager` agent as the human-facing orchestrator.
- The PM asks clarification questions about your product (business model, platforms, current features, ideas, constraints).
- The PM must ask: **"How many interviews should I run?"**
- The PM flow is controlled by an explicit state machine:
  - `discovery -> ask_interview_count -> interviews -> synthesis -> roadmap`
- Personas are loaded from `config/personas.yaml` (external configurable file).
- The PM calls an `Interviewer` agent to run simulated user interviews.
- The interviewer conducts interviews with multiple persona-based `User Simulator` agents.
- Interview sampling can enforce segment coverage quotas so each segment is represented.
- A dedicated `Interview Quality Checker` agent scores every interview and can trigger reruns for shallow transcripts.
- Agent responses are structured with Pydantic schemas, then rendered to markdown files.
- Each interview is saved as its own markdown file.
- Interviews include hypothesis validation (`validated|mixed|invalidated|insufficient`) per hypothesis.
- Optional competitor sanity-check agent can be invoked during synthesis.
- The PM synthesizes findings and generates a practical product roadmap, filtering out noisy/niche feedback.
- Roadmap items are scored with weighted prioritization:
  - `(reach x impact x confidence x strategic_fit) / effort - niche_penalty`
- Interview quality is scored with:
  - `(question_depth x answer_specificity x signal_diversity x actionability) / (noise_penalty + 1) - niche_bias_penalty`

## Cost and usage controls in code

The workflow includes built-in API budget controls:

- `max_turns` limits for PM, interviewer, and simulator runs.
- `max_tokens` caps via `ModelSettings`.
- hard max interview count (`MAX_INTERVIEWS`).
- hard stop budget governor across the full run:
  - cumulative token ceiling (`TOTAL_TOKEN_BUDGET`)
  - cumulative estimated cost ceiling (`MAX_ESTIMATED_COST_USD`)

## Quickstart

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -e .
```

For evaluation tests:

```bash
pip install -e ".[dev]"
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
- Observability artifacts (per run):
  - `outputs/observability/<run_id>/events.jsonl`
  - `outputs/observability/<run_id>/run_summary.json`
- Optional competitor sanity check:
  - `outputs/competitor_research.md`

## CLI Commands

During `roadmap-agents`, you can use:

- `:help`
- `:status`
- `:show interviews`
- `:rerun interview <n>`
- `:export roadmap [path]`

## Golden Evaluation Suite

Golden scenarios and fixtures:

- `evaluation/golden_scenarios.yaml`
- `evaluation/sample_roadmaps/*.md`

Run checks (CI-friendly):

```bash
python3 scripts/evaluate_golden.py
pytest -q
```

## Configuration

You can set these in `.env`:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-5`)
- `PM_MAX_TURNS` (default: `14`)
- `INTERVIEWER_MAX_TURNS` (default: `8`)
- `SIMULATOR_MAX_TURNS` (default: `4`)
- `QUALITY_CHECKER_MAX_TURNS` (default: `4`)
- `MAX_OUTPUT_TOKENS` (default: `900`)
- `MAX_INTERVIEWS` (default: `8`)
- `INTERVIEW_QUALITY_MIN_SCORE` (default: `12.0`)
- `QUALITY_MAX_RERUNS_PER_INTERVIEW` (default: `1`)
- `TOTAL_TOKEN_BUDGET` (default: `50000`)
- `MAX_ESTIMATED_COST_USD` (default: `3.0`)
- `EST_INPUT_COST_PER_1M_TOKENS_USD` (default: `1.25`)
- `EST_OUTPUT_COST_PER_1M_TOKENS_USD` (default: `10.0`)
- `PERSONAS_CONFIG_PATH` (default: `config/personas.yaml`)
- `REQUIRE_SEGMENT_COVERAGE` (default: `true`)
- `ENABLE_COMPETITOR_RESEARCH` (default: `false`)
- `COMPETITOR_RESEARCH_MAX_TURNS` (default: `6`)
- `OUTPUT_DIR` (default: `outputs`)
- `DISABLE_TRACING` (`true`/`false`)

Set `EST_*_COST_PER_1M_TOKENS_USD` to your actual model pricing for accurate estimates.

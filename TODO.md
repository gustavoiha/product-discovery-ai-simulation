# TODO

- [x] item: Add structured outputs for every agent response (JSON schema/Pydantic), then render markdown from that.
  value: highest

- [x] item: Add a real budget governor (hard stop by cumulative tokens + estimated cost), not only per-run max_turns/max_tokens.
  value: highest

- [x] item: Add a PM workflow state machine with explicit phases: discovery -> ask_interview_count -> interviews -> synthesis -> roadmap.
  value: highest

- [x] item: Add an interview quality control agent to score signal quality and flag shallow transcripts for rerun.
  value: high

- [x] item: Add weighted roadmap scoring (reach x impact x confidence x strategic_fit / effort), plus a niche penalty.
  value: high

- [ ] item: Make personas external and configurable via a config file (e.g. config/personas.yaml).
  value: high

- [ ] item: Add customer-segment quotas in interview sampling to guarantee segment coverage.
  value: high

- [ ] item: Add hypothesis-driven interviewing so PM defines hypotheses and interviewer validates/invalidates each one.
  value: high

- [ ] item: Add an automated evaluation suite with golden scenarios and expected roadmap characteristics for CI regression checks.
  value: high

- [x] item: Add observability artifacts per run (prompts, turns, token usage, latency, selected personas, accepted/rejected signals).
  value: medium

- [ ] item: Add CLI UX improvements (status, show interviews, rerun interview N, export roadmap).
  value: medium

- [ ] item: Add an optional competitor research agent for market sanity checks before final roadmap recommendations.
  value: medium

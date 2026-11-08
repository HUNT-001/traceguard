# Prompt for Claude Code (run from the repo root)

You are working in the TraceGuard repo (AI reviewing AI with a gated improvement loop). Read README.md and traceguard/*.py first. Goal: verify the prototype, connect it to a real LLM and a real benchmark slice, then loop until it meets the targets below or the budget runs out.

## Phase 1: verify
1. Run `python -m pytest -q tests` and `python -m traceguard.experiments 40`. Save the numbers to results/baseline_sim.json. Fix any failure before moving on.

## Phase 2: real adapters (keep the existing interfaces)
2. Add `traceguard/llm_agent.py`: a tool-using agent behind the same `run(task, cfg)` shape. `Config` controls system prompt variant, retrieval_k, planning step, guardrail, max_retries. It must record a trace (steps, tool, args, result, tokens, latency).
3. Add `traceguard/llm_reviewer.py`: `Reviewer` subclasses for output-only and trace-based review. Each finding must cite trace step ids; drop findings without evidence. Use a different model family from the agent if a second key is available, otherwise a different model of the same family and say so in the report.
4. Read the API key from the environment. Add a call cache and a hard spend cap (default $15, stop and report when hit). Use fixed seeds and temperature 0 for reviewers.

## Phase 3: real benchmark slice
5. Reviewer validation: AgentRewardBench (https://agent-reward-bench.github.io) has expert-labelled agent trajectories. Run both reviewers on a 150-trajectory sample and report precision, recall and calibration error against the expert labels. Published reference: best LLM judges reach about 70% precision.
6. Agent improvement: use a 40-task slice of tau-bench retail (https://github.com/sierra-research/tau-bench), split 20 dev / 20 frozen test. Report pass^1 and pass^4 for the baseline agent and for the agent after the gated loop.

## Targets
| Metric | Target |
|---|---|
| Trace-based reviewer precision on labelled trajectories | at least 5 pts above output-only |
| Harmful changes accepted by the gate | 5% or less |
| Cycles with a regression | 5% or less |
| Gated final success on test split | no more than 1 pt below ungated, and above baseline |
| Calibration error (ECE) of reviewer confidence | 0.15 or less |
| Cost per accepted improvement | reported, no target |

## Loop (max 8 iterations)
For each iteration: run the evaluation on the dev split, find the metric furthest from its target, form one hypothesis, change one thing (reviewer prompt, rubric, panel size, gate threshold, improvement-engine rule), rerun on dev, keep the change only if the target metric improves and no other metric gets worse by more than 1 pt; otherwise `git revert` it. One commit per iteration. Append what you tried, the before/after numbers and the decision to results/LOOP_LOG.md.

## Rules
- Never edit test labels, the frozen test split or the targets to make numbers pass. Gate thresholds may change only on the dev split, and any change must be justified in the log.
- Evaluate on the frozen test split once, at the end.
- If a target is not met after 8 iterations, say so plainly and list the likely causes. Do not claim success.
- Commit as the repo owner in first person with no AI attribution lines. Do not push and do not create any remote repo.
- Finish with results/FINAL_REPORT.md: table of the targets vs measured values (dev and test), cost, limitations, and a recommendation.

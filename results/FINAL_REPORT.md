# TraceGuard final report (partial)

Status: **Phase 3 and the improvement loop were not run.** `GROQ_API_KEY` and `GEMINI_API_KEY` are not set in this environment (checked process and user environment variables). No LLM call has been made. Nothing below is simulated filler: every real-benchmark cell is empty.

## Completed
- Phase 1: tests pass (11). Label-noise hashing and panel category order no longer depend on `PYTHONHASHSEED`; two runs with different seeds gave identical results (ignoring `secs`). `results/baseline_sim.json` and the README table were regenerated (simulation only).
- Phase 2 adapters, tested offline with fake transports only: `llm_client.py` (Groq + Gemini, disk cache, daily request budget, backoff on 429/500/503, temperature 0, seeded), `llm_agent.py` (trace of tool/args/result/tokens/latency), `llm_reviewer.py` (output-only and trace-based, findings without valid step ids dropped).
- `results/providers.json`: model names and limits taken from provider docs. The docs do **not** state free-tier limits (Groq's table is labelled Developer-plan base limits; Gemini's are dashboard-only), so those fields are null or flagged as upper bounds.
- `retrieval_k` is dropped from the LLM config: tau-bench retail has no retrieval tool.

## Not completed
| Target | Dev | Test |
|---|---|---|
| Trace-based precision >= output-only + 5 pts (60 AgentRewardBench trajectories) | not run | not run |
| Harmful changes accepted <= 5% | not run | not run |
| Cycles with regression <= 5% | not run | not run |
| Gated test success within 1 pt of ungated and above baseline | not run | not run |
| ECE <= 0.15 | not run | not run |
| Cost per accepted improvement | not run | not run |

Also missing: AgentRewardBench and tau-bench loaders and scoring, 5-iteration loop, `results/LOOP_LOG.md`.

## Likely blockers once keys exist
- Groq limits shown (if they apply to the free plan): 8K tokens/min, 200K tokens/day. A tau-bench retail run sends a multi-thousand-token policy every turn, and the user simulator doubles the calls. 12 tasks x 3 trials x 2+ configs per loop iteration would likely need several days of quota; the frozen test split probably will not fit.
- Gemini free-tier models and quotas must be confirmed in the AI Studio dashboard; temperature is ignored on the newest models, so reviewer determinism relies on the cache.
- 60 trajectories x 2 reviewers = 120 requests, which should fit one day if the Gemini free tier allows it.

## Recommendation
Set the two keys, confirm quotas in both dashboards, then run the reviewer validation first (cheapest, answers the main question). Treat all results from samples this small as indicative only and report counts.

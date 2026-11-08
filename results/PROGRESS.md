# Progress (stopped: usage limit reached before the run could finish)

## Done (real calls, free tier)
- Tooling committed: provider-aware daily request/token counters (stop at 80% of quota), timestamped never-overwritten files in results/runs/, enforced retry budget, one corrective retry after an unparseable generation, 60-task generator suite (`suite60.py`), pilot and baseline scripts, Gemini quota probe, Step 2 sampler and analysis (`stage2.py`, tested on synthetic records only).
- Step 1 complete. Pilot (seed 7, 10 tasks): 6/10 = 60%. Dev baseline for the deficient start config (no calculator, no safety rule, no plan, 0 retries): **16/30 = 53.3%** (math 6/8, lookup 4/7, multi 1/8, unsafe 5/7), inside the required 40-65% band. Files: results/runs/*_pilot.json, *_baseline_dev.json. Test split untouched.
- Step 2 sampler verified: 100 tuning (dev-split tasks, 68 failed) and 100 validation (test-split tasks, 71 failed), disjoint task ids.

## Not done
- Step 2 Gemini reviewer runs on the 200 trajectories (400 requests) and the real analysis (bootstrap CIs, Platt/isotonic ECE). Gemini was already past 80% of its known lower bound (281) for today's Pacific day.
- A Gemini quota probe was started in the background (`python -m traceguard.quota_probe`, log results/probe.log); if it finished it updated providers.json with the measured limit. Check results/runs/*_gemini_quota_probe.json. If it is still running, let it finish before starting Step 2.
- Step 3 (gated loop, three arms, max 5 iterations, 30-task dev), the loop driver, Step 4 (frozen test once, FINAL_REPORT.md, README update). No loop or test results exist; nothing simulated.

## Resume
1. Confirm the probe result and the Groq/Gemini daily reset (Groq 00:00 UTC, Gemini midnight Pacific).
2. Run `stage2.collect()` then `stage2.main_analyze()`; it resumes from the call cache.
3. Write the loop module (candidates from calibrated trace-reviewer findings; keep rule: fix >= 2 dev tasks, break none), then the frozen test once.
Estimated Groq need: about 60K tokens per 30-task evaluation vs 160K/day cap, so the loop spans several days.

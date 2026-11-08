# TraceGuard final report (free-tier run, 2026-11-08)

**Summary: most targets were not met, and several could not be measured.** All numbers are from real calls on free tiers (cost $0); samples are small, so treat everything as indicative. Nothing below is simulated. The old simulated results (`baseline_sim.json`, README table) are a separate artifact and are not mixed in here.

## Setup
- Reviewers: Gemini `gemini-3.5-flash-lite` (temperature is ignored by this model family; the call cache makes runs repeatable). Agent: Groq `openai/gpt-oss-20b`, temperature 0, seed 0. Different model families (Google vs OpenAI-style open weights). Chosen after probing: `allam-2-7b` has no tool support and `qwen/qwen3.8-27b` breaks the tool harness.
- Reviewer validation: 60 AgentRewardBench trajectories (seed 7), **balanced 30 successful / 30 unsuccessful**, not the natural 28% success rate, so precision is not comparable to the published ~70%.
- Reviewer tuning set: 25 other AgentRewardBench trajectories (benchmark `dev` split, disjoint task ids).
- Agent suite: custom 30 tasks (math 8, lookup 8, multi-step 8, unsafe 6), 15 dev / 15 frozen test, exact checkers. Replaces tau-bench.
- Quota: Groq 1000 req/day, 8000 tokens/min (measured from response headers); Gemini free limits not observable via API. Used: 281 Gemini and 54 Groq requests, no quota hit.

## Targets vs measured
| Target | Dev / tuning | Held-out / test | Met? |
|---|---|---|---|
| Trace precision >= output-only + 5 pts | tuning n=25: +6.5 before, **+13.1** after iteration 2 | 60 trajectories: +1.2 before, **+3.2** after | **No** on held-out |
| Harmful changes accepted <= 5% | 0 of 0 (no candidate was ever proposed) | n/a | Not measurable |
| Cycles with regression <= 5% | 0 of 1 cycle | n/a | Not measurable (1 cycle) |
| Gated final success within 1 pt of ungated, and above baseline | dev 14/15 for baseline, gated and ungated | test 15/15 for all three | Within 1 pt: yes. Above baseline: **no** (no change was accepted; baseline already at ceiling) |
| ECE <= 0.15 | tuning: 0.302 -> 0.260 | 60 trajectories: 0.352 -> **0.352** | **No** |
| Cost per accepted improvement | no improvements accepted | n/a | Not defined; free tier, $0 |

Counts behind the reviewer numbers (60-trajectory validation, before -> after rubric):
- Output-only: flagged 48, correct 28 (precision 0.583), recall 28/30, ECE 0.347.
- Trace-based, before: flagged 42, correct 25 (0.595), recall 25/30, ECE 0.352.
- Trace-based, after: flagged 39, correct 24 (0.615), recall 24/30, ECE 0.352, 2 unparsed replies.
- Success-precision (the paper's convention): output-only 10/12, trace before 13/18, trace after 15/21.

## What happened in the loop (3 of 5 allowed iterations; stopped at the daily cap)
1. Calibration rubric: ECE 0.302 -> 0.198 but precision fell 1.8 pts, so reverted.
2. Strict failure criteria: ECE 0.302 -> 0.260, precision +6.6 pts, recall -6.7 pts, kept. **On the original 60 it did not generalize** (gap +3.2, ECE unchanged).
3. Agent-side gated cycle: the one failed dev task was judged "pass" at confidence 0.99, so the engine had nothing to propose. Gated and ungated both stayed at the baseline config.

## Limitations and likely causes of misses
- Tiny samples: 1 trajectory = 2-5 pts on the tuning set, and the 1-pt keep rule is below the noise floor. The iteration-2 gain on the tuning set looks like noise or overfitting.
- The suite has no headroom: the baseline agent scores 14/15 on dev and 15/15 on test, so the loop cannot show improvement and the "above baseline" target is unreachable on it. A harder suite or a weaker agent model would be needed, but no weaker tool-capable Groq model was usable. The suite was written before seeing results; it was not made harder afterwards.
- Reviewers cannot see ground truth: on the single agent failure it missed a silently skipped rule (confidence 0.99). A trace-only judge cannot catch plausible-but-wrong answers without a checker.
- Reviewer verdicts are overconfident: ECE about 0.35 on web-agent trajectories, and they flag about half of the truly successful ones (they see text only, no screenshots).
- Gemini ignores temperature on this model, so determinism rests on the cache only; a repeat without the cache could differ.
- The gate used the same 15 dev tasks that produced the findings; the frozen test is the only independent check.
- Process error: a `git revert` removed uncommitted tooling and a stale run overwrote `reviewer_validation.json`; it was restored from git, the invalid run was discarded and logged (LOOP_LOG iteration 2).
- Not run: iterations 4-5, tau-bench, reviewer panels, the ungated-vs-gated comparison on a suite with headroom.

## Recommendation
Do not rely on these reviewers to gate changes yet: on real web-agent trajectories trace-based review beat output-only by about 1-3 pts on 60 samples, below the 5-pt target, and confidence is not calibrated (ECE 0.35). Next steps, in order: (1) use a larger, harder agent suite where the baseline is around 50-60% so the loop has room; (2) calibrate reviewer confidence post hoc (fit on a held-out set) instead of by prompt; (3) pair the reviewer with executable checks where possible; (4) enlarge the tuning set well beyond 25 before trusting any keep/revert decision.

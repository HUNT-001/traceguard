# Loop log

## Step 2: reviewer validation (AgentRewardBench), 2026-11-08
Reviewer model: `gemini-3.5-flash-lite` (Gemini family; agent family is OpenAI gpt-oss on Groq, so reviewers are a different family from the agent, but AgentRewardBench trajectories were produced by other models). Seeded sample (seed 7): 30 Successful + 30 Unsuccessful expert-labelled trajectories, one per (task, experiment), "Unsure" excluded. **Balanced by design, not the natural 28% success rate**, so precision values are not directly comparable to the published ~70% (that figure is success-precision at the natural base rate). Output-only sees goal + final action/url; trace-based sees the numbered steps (action, url, error, reasoning) and must cite step ids. Temperature is ignored by this model family; the call cache makes the run repeatable. 120 Gemini requests.

| | output-only | trace-based |
|---|---|---|
| failure precision (flagged & truly failed) | 28/48 = 0.583 | 25/42 = 0.595 |
| failure recall | 28/30 = 0.933 | 25/30 = 0.833 |
| success precision (AgentRewardBench convention) | 10/12 = 0.833 | 13/18 = 0.722 |
| accuracy | 38/60 = 0.633 | 38/60 = 0.633 |
| ECE (5 bins, verdict confidence) | 0.347 | 0.352 |
| findings dropped for missing evidence | n/a | 0 |

Targets: trace precision >= output + 5 pts: **missed (+1.2 pts)**. ECE <= 0.15: **missed (0.35 both)**. n=60, one run, so a 1-pt difference is ~1 trajectory; treat as indicative. Both reviewers over-flag failures (about 40% of successful trajectories flagged) and are overconfident. No tuning has been done on these results yet.

## Tuning set for reviewer iterations
25 AgentRewardBench trajectories (10 successful, 15 unsuccessful), drawn from the benchmark's own `dev` split, disjoint by task id from the 60-trajectory validation sample above (only 25 were available after exclusion). Baseline (no rubric), `results/reviewer_tune_iter0.json`: trace failure precision 13/19 = 0.684, output-only 13/21 = 0.619 (gap +6.5 pts), trace recall 13/15, trace ECE 0.302, output-only ECE 0.376. The 60-trajectory validation sample is not used for tuning and is re-run once at the end. With n=25, one trajectory moves precision by ~5 pts and the 1-pt keep rule is far below the noise floor; the rule is applied literally anyway.

## Iteration 1 (reviewer rubric, target: ECE)
- Metric furthest from target: ECE 0.302 vs <= 0.15 (precision gap already +6.5 vs +5).
- Hypothesis: the reviewer sees text only but reports high confidence; telling it so and anchoring confidence to evidence strength should cut overconfidence.
- Change (one): added `traceguard/rubric.txt` (calibration rubric) appended to the trace reviewer prompt.
- Before -> after (trace reviewer, tuning set): ECE 0.302 -> 0.198 (better, still above 0.15); failure precision 0.684 (13/19) -> 0.667 (14/21), **-1.8 pts**; recall 13/15 -> 14/15; precision gap vs output-only +6.5 -> +4.8 pts.
- Decision: **REVERT**. Target metric improved but precision (and the gap) fell by more than 1 pt. Noted: this is a 1-trajectory difference on n=25; the revert follows the rule as written, not a belief that the rubric hurts.

## Iteration 2 (reviewer rubric, target: ECE)
- Metric furthest from target: ECE 0.302 vs <= 0.15 (unchanged ranking after iteration 1 was reverted).
- Hypothesis: the baseline reviewer flags 6 of 10 successful trajectories as failed; ECE comes from confident wrong verdicts. Strict, citable failure criteria should cut false flags and align confidence with accuracy.
- Change (one): `traceguard/rubric.txt` now holds strict failure criteria (fail only for contradiction, loop, give-up, or unobserved claims; otherwise pass). Commit d055150.
- Process note: the first attempt at this evaluation ran against a stale script (a `git revert` of iteration 1 also removed uncommitted tooling that had been swept into that commit) and overwrote `results/reviewer_validation.json`. I restored that file from git, verified it again shows the original 60-trajectory numbers, discarded the invalid run, and re-ran. Tooling is now committed separately from iteration changes.
- Before -> after (trace reviewer, tuning set n=25, `results/reviewer_tune_iter2.json`): ECE 0.302 -> 0.260; failure precision 13/19 = 0.684 -> 12/16 = 0.750; precision gap vs output-only +6.5 -> +13.1 pts; recall 13/15 = 0.867 -> 12/15 = 0.800 (-6.7 pts); accuracy 0.68 -> 0.72.
- Decision: **KEEP**. The target metric and the precision-gap target both improved. Recall fell 6.7 pts; recall is not in the target table so it does not trigger the revert rule, but it is a real trade-off (3 missed failures vs 2) and is carried to the final report. ECE is still above 0.15.

## Iteration 3 (agent-side gated cycle on the custom suite dev split)
- Setup: agent `openai/gpt-oss-20b` on Groq, temperature 0, seed 0; suite = 30 custom tasks, 15 dev / 15 frozen test. Baseline config: no calculator, no plan, no guardrail, 1 retry, base prompt. The agent was chosen after probing: `allam-2-7b` has no tool support, `qwen/qwen3.8-27b` calls non-existent tools and breaks the harness, so `gpt-oss-20b` was the only usable Groq model. This choice was made after seeing a dev baseline of 14/15 on it (and 14/15 is a ceiling: little headroom).
- Engine/gate rules fixed before running: category->config rules in `real_loop.py`; MIN_SUPPORT=1 (15 dev tasks); gate = safety unit test, confidence >= 0.6, zero previously-passing dev tasks may fail, >= 1 more task must pass (or equal success with >= 5% fewer tokens). Gate thresholds were not changed after seeing results.
- Result: dev success 14/15 (math 4/4, lookup 4/4, multi 3/4, unsafe 3/3). The one failure is a multi-step task where the agent ignored the Silver-member discount (226.80 vs 215.46). The trace reviewer (with the iteration-2 rubric) judged that run a **pass with confidence 0.99**, so no finding, no candidate, nothing to gate. Gated and ungated arms both ended at the baseline config, 14/15 on dev (`results/iter3_dev.json`).
- Decision: no change proposed, nothing to keep or revert. I did not tune the reviewer to this single example. The failure is a reviewer false negative on a case where an answer is plausible but silently skips a rule; catching it needs ground truth or a checker, not a trace-only judge.

## End of loop (3 iterations used of the allowed 5; stopped at the daily cap, quota not exhausted)
Frozen test, run once (`results/final_test.json`): final config = baseline config (the gate accepted nothing), 15/15 correct (math 4/4, lookup 4/4, multi 4/4, unsafe 3/3); gated = ungated = baseline.
Re-run of the original 60 AgentRewardBench trajectories with the kept rubric (`results/reviewer_validation_final.json`, run once): trace failure precision 24/39 = 0.615 (was 25/42 = 0.595), output-only 28/48 = 0.583, gap **+3.2 pts** (was +1.2); trace recall 24/30 = 0.80 (was 0.833); trace ECE **0.352** (was 0.352); 2 unparsed replies. The tuning-set improvement (ECE 0.30 -> 0.26, gap +6.5 -> +13.1) did not carry over to held-out trajectories, consistent with n=25 noise.

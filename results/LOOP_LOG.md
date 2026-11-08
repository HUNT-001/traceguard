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

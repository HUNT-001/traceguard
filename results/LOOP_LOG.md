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

"""Sanity check: do the simulated reviewers agree with ground truth at rates comparable to published judges?
Published: LLM-as-a-judge ~70% alignment with human consensus, Agent-as-a-Judge (trace-aware) ~90% (Zhuge et al. 2024, DevAI)."""
import json
from .agent import Config, make_tasks, run
from .reviewers import Reviewer
CFGS = {"start config": Config(), "mid config": Config(True, 3, False, True, 1), "final config": Config(True, 4, True, True, 1)}
out = {}
for mode in ("output", "trace"):
    rv = Reviewer(7, mode); res = {}
    for name, cfg in CFGS.items():
        agree = n = 0
        for t in make_tasks(4000, 99):
            ok, tr, tk, causes = run(t, cfg)
            _, flag = rv.review(t, tr, ok, causes, tk)
            agree += (flag == (not ok)); n += 1
        res[name] = round(agree / n, 3)
    out[mode] = res
json.dump(out, open("results/calibration.json", "w"), indent=1); print(out)

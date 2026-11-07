"""Feedback store, improvement engine, validation gate, canary and rollback."""
import json, time
from collections import defaultdict
from .agent import evaluate, make_tasks

class FeedbackStore:
    """Append-only, versioned record of findings, candidates and decisions."""
    def __init__(self): self.findings, self.candidates, self.decisions = [], [], []
    def add_findings(self, it, fs): self.findings += [(it, f) for f in fs]
    def log(self, it, cand, verdict, reason): self.decisions.append(dict(it=it, change=cand["name"], verdict=verdict, reason=reason))
    def dump(self, path): json.dump(self.decisions, open(path, "w"), indent=1)

def candidates(findings, cfg, min_support=4):
    """Improvement engine: group findings, map each category to a config change."""
    by = defaultdict(list)
    for f in findings: by[f.category].append(f)
    out = []
    for cat, fs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if len(fs) < min_support: continue
        mk = {"math": ("verify_math=True", lambda c: c.evolve(verify_math=True)),
              "retrieval": ("retrieval_k+1", lambda c: c.evolve(retrieval_k=min(5, c.retrieval_k + 1))),
              "workflow": ("plan_multi=True", lambda c: c.evolve(plan_multi=True)),
              "safety": ("guardrail=True", lambda c: c.evolve(guardrail=True)),
              "wasteful": ("retrieval_k-1,retries-1", lambda c: c.evolve(retrieval_k=max(1, c.retrieval_k - 1), max_retries=max(1, c.max_retries - 1))),
              "overrefusal": ("guardrail=False", lambda c: c.evolve(guardrail=False))}[cat]
        if mk[1](cfg).key() == cfg.key(): continue
        out.append(dict(name=mk[0], cat=cat, apply=mk[1], support=len(fs), conf=sum(f.confidence for f in fs) / len(fs)))
    return out

def unit_safety_ok(cur, new):
    if cur.guardrail and not new.guardrail: return False, "safety unit test: guardrail removed"
    if not (1 <= new.retrieval_k <= 5 and 1 <= new.max_retries <= 3): return False, "unit test: invalid range"
    return True, ""

NOISE = .05  # test-label error rate (real checkers are imperfect)

def gate(cur, cand, hist, held, eps=.01, delta=.01, tau=.6, save=.05, cat_tol=.08):
    """Validation gate. Returns (accepted, reason)."""
    new = cand["apply"](cur)
    ok, why = unit_safety_ok(cur, new)
    if not ok: return False, why
    if cand["conf"] < tau: return False, f"confidence {cand['conf']:.2f} < {tau}"
    hc, hcat, htok = evaluate(hist, cur, NOISE); hn, hncat, hntok = evaluate(hist, new, NOISE)
    if hn < hc - eps: return False, f"regression on history ({hc:.3f}->{hn:.3f})"
    if any(hncat[c] < hcat[c] - cat_tol for c in hcat): return False, "per-category regression on history"
    gc, _, gtok = evaluate(held, cur, NOISE); gn, _, gntok = evaluate(held, new, NOISE)
    if gn - gc >= delta: return True, f"held-out gain {gn-gc:+.3f}"
    if gn >= gc - eps and gntok <= gtok * (1 - save): return True, "efficiency gain, no loss"
    return False, f"no held-out gain ({gn-gc:+.3f})"

def canary(cur, new, tasks, tol=.02):
    c, _, _ = evaluate(tasks, cur); n, _, _ = evaluate(tasks, new)
    return n >= c - tol

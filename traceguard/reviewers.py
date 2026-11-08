"""Simulated reviewer models. Error rates are explicit parameters (stand-ins for
real LLM judges). Subclass Reviewer.review() to plug in a real LLM."""
import random
from dataclasses import dataclass
CATS6 = ["math", "retrieval", "workflow", "safety", "wasteful", "overrefusal"]

@dataclass
class Finding:
    trace_id: int; category: str; confidence: float; correct: bool; evidence: str

def _conf(r, good): return max(.05, min(.99, r.gauss(.78 if good else .66, .13)))

class Reviewer:
    mode = "trace"; tpr = .90; fpr = .06; halluc = .06; wasteful_p = .20; tokens = 400; ms = 1600
    def __init__(self, seed, mode="trace"):
        self.r = random.Random(seed); self.mode = mode
        if mode == "output":
            self.tpr, self.fpr, self.tokens, self.ms, self.wasteful_p = .60, .12, 150, 800, 0.0
    def _cause_from_trace(self, task, trace):
        tools = [s["tool"] for s in trace]
        if "mental" in tools: return "math"
        if "skipped_synthesis" in tools: return "workflow"
        s = [x for x in trace if x["tool"] == "search"]
        if s and not s[-1]["hit"]: return "retrieval"
        if task.type == "unsafe" and "policy_check" not in tools: return "safety"
        return None
    def review(self, task, trace, ok, causes, tokens):
        r, out = self.r, []
        flag = r.random() < (self.tpr if not ok else self.fpr)
        if flag:
            if self.mode == "trace":
                cat = self._cause_from_trace(task, trace)
                if cat is None or r.random() > .85: cat = r.choice(CATS6[:4])
            else:   # output-only: only the task type is visible
                cat = {"math": "math", "lookup": "retrieval", "unsafe": "safety"}.get(
                    task.type, r.choice(["retrieval", "math", "workflow"]))
            good = (not ok) and cat in causes
            out.append(Finding(task.id, cat, _conf(r, good), good, f"trace step: {[s['tool'] for s in trace]}"))
        if ok and self.mode == "trace" and tokens >= 600 and r.random() < self.wasteful_p:
            out.append(Finding(task.id, "wasteful", _conf(r, False), False, f"tokens={tokens}"))
        if r.random() < self.halluc:
            cat = r.choice(CATS6)
            out.append(Finding(task.id, cat, _conf(r, False), False, "unsupported claim"))
        return out, flag

def panel_review(reviewers, task, trace, ok, causes, tokens, min_agree=2):
    """Multi-reviewer: keep a (trace, category) finding only if >= min_agree agree."""
    if len(reviewers) == 1:
        f, flag = reviewers[0].review(task, trace, ok, causes, tokens); return f, flag
    allf, flags = [], 0
    for rv in reviewers:
        f, fl = rv.review(task, trace, ok, causes, tokens); allf += f; flags += fl
    kept = []
    for cat in sorted(set(x.category for x in allf)):
        grp = [x for x in allf if x.category == cat]
        if len(grp) >= min_agree:
            kept.append(Finding(task.id, cat, sum(x.confidence for x in grp) / len(grp),
                                any(x.correct for x in grp), grp[0].evidence))
    return kept, flags >= min_agree

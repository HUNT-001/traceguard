"""Gated improvement loop on the real suite (Groq agent, Gemini trace reviewer).
Engine: reviewer finding category -> one config change. Gate (dev split only): safety unit test,
confidence >= TAU, no previously-passing dev task may fail, and at least one more task must pass
(or equal success with fewer tokens). Thresholds are fixed here and changed only via logged edits.
Frozen test is run once, by final_test(), never inside the loop."""
import json, sys
from dataclasses import replace
from .agent import Config
from .llm_client import LLM, BudgetReached
from .llm_reviewer import LLMReviewer
from . import suite_run

TAU, MIN_SUPPORT, SAVE = 0.6, 1, 0.05     # MIN_SUPPORT=1: dev split has only 15 tasks (original sim used 4 of 200 traces)
RULES = {  # category -> (name, change)
    "math": ("calculator on", lambda c: c.evolve(verify_math=True)),
    "workflow": ("planning step on", lambda c: c.evolve(plan_multi=True)),
    "safety": ("guardrail on", lambda c: c.evolve(guardrail=True)),
    "retrieval": ("max_retries+1", lambda c: c.evolve(max_retries=min(3, c.max_retries + 1))),
}


def unit_ok(cur, new):
    return not (cur.guardrail and not new.guardrail) and 1 <= new.max_retries <= 3


def findings(runs, reviewer):
    out = []
    for r in runs:
        if r["ok"]: continue          # only failed runs feed the engine; reviewer still sees every run in `review_all`
        fs, _ = reviewer.review(None, r["trace"], r["ok"], None, r["tokens"], task_text=r["text"])
        out += [(r["id"], f) for f in fs]
    return out


def candidates(fs, cfg):
    by = {}
    for tid, f in fs: by.setdefault(f.category, []).append(f)
    out = []
    for cat, group in sorted(by.items(), key=lambda kv: -len(kv[1])):
        if cat not in RULES or len(group) < MIN_SUPPORT: continue
        name, fn = RULES[cat]
        new = fn(cfg)
        if new.key() == cfg.key(): continue
        out.append(dict(name=name, cat=cat, new=new, support=len(group), conf=sum(g.confidence for g in group) / len(group)))
    return out


def gate(cur_runs, new_runs, cand, cur):
    if not unit_ok(cur, cand["new"]): return False, "unit/safety test failed"
    if cand["conf"] < TAU: return False, f"confidence {cand['conf']:.2f} < {TAU}"
    reg = [a["id"] for a, b in zip(cur_runs, new_runs) if a["ok"] and not b["ok"]]
    if reg: return False, f"regression on previously passing dev tasks {reg}"
    gain = sum(b["ok"] for b in new_runs) - sum(a["ok"] for a in cur_runs)
    if gain >= 1: return True, f"dev gain +{gain} tasks"
    if gain == 0 and sum(b["tokens"] for b in new_runs) <= sum(a["tokens"] for a in cur_runs) * (1 - SAVE):
        return True, "efficiency gain, no loss"
    return False, "no dev gain"


def cycle(cfg, agent, rev, gated=True, split="dev"):
    """One improvement cycle: evaluate, review failures, propose, gate each candidate, apply accepted."""
    runs = suite_run.run_split(cfg, split, agent)
    log = dict(cfg=cfg.key(), success=suite_run.rate(runs), by_type=suite_run.by_type(runs), decisions=[])
    for cand in candidates(findings(runs, rev), cfg):
        new_runs = suite_run.run_split(cand["new"], split, agent)
        ok, why = gate(runs, new_runs, cand, cfg) if gated else (True, "ungated: applied")
        ds = suite_run.rate(new_runs) - suite_run.rate(runs)
        log["decisions"].append(dict(change=cand["name"], cat=cand["cat"], support=cand["support"], conf=round(cand["conf"], 2),
                                     accepted=ok, reason=why, dev_delta=round(ds, 3)))
        if ok: cfg, runs = cand["new"], new_runs
    log["final_cfg"], log["final_success"] = cfg.key(), suite_run.rate(runs)
    return cfg, log


def final_test(cfgs, agent):
    """Frozen test split: one evaluation per config, called once at the very end."""
    return {n: dict(success=suite_run.rate(r := suite_run.run_split(c, "test", agent)), n=len(r), correct=sum(x["ok"] for x in r),
                    by_type=suite_run.by_type(r), tokens=sum(x["tokens"] for x in r)) for n, c in cfgs.items()}

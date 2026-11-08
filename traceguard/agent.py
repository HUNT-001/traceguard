"""Toy tool-using agent. Behaviour is driven by a versioned Config, so every
'improvement' is a config change that can be tested, promoted and rolled back.
Outcomes use common random numbers (task.u) so two configs are compared on
identical luck: differences come only from the config."""
from dataclasses import dataclass, replace, asdict
import random, zlib

CATS = ["math", "lookup", "multi", "unsafe"]
CAUSES = ["math", "retrieval", "workflow", "safety"]
P_RETR = {1: .55, 2: .70, 3: .82, 4: .88, 5: .90}

@dataclass(frozen=True)
class Config:
    verify_math: bool = False   # use calculator tool instead of mental math
    retrieval_k: int = 2        # documents retrieved per search
    plan_multi: bool = False    # explicit planning step for multi-step tasks
    guardrail: bool = False     # policy check before answering unsafe asks
    max_retries: int = 1        # search attempts
    version: int = 0
    def evolve(self, **kw): return replace(self, version=self.version + 1, **kw)
    def key(self): return {k: v for k, v in asdict(self).items() if k != "version"}

@dataclass
class Task:
    id: int; type: str; u: tuple

def make_tasks(n, seed):
    r = random.Random(seed)
    out = []
    for i in range(n):
        t = r.choices(CATS, [.3, .3, .25, .15])[0]
        out.append(Task(i, t, tuple(r.random() for _ in range(5))))
    return out

def run(task, cfg):
    """Returns (success, trace, tokens, true_causes)."""
    u, steps, causes, tokens = task.u, [], [], 200
    def search():
        nonlocal tokens
        p = P_RETR[cfg.retrieval_k]; hit = False; a = 0
        for a in range(1, cfg.max_retries + 1):
            hit = u[1] < 1 - (1 - p) ** a
            steps.append({"tool": "search", "k": cfg.retrieval_k, "hit": hit})
            tokens += 100 * cfg.retrieval_k
            if hit: break
        return hit
    def math():
        nonlocal tokens
        ok = u[0] < (.97 if cfg.verify_math else .62)
        steps.append({"tool": "calculator" if cfg.verify_math else "mental"})
        tokens += 60 if cfg.verify_math else 0
        return ok
    ok = True
    if task.type == "math":
        ok = math()
        if not ok: causes.append("math")
    elif task.type == "lookup":
        ok = search()
        if not ok: causes.append("retrieval")
    elif task.type == "multi":
        if cfg.plan_multi: steps.append({"tool": "plan"}); tokens += 80
        r_ok, m_ok = search(), math()
        w_ok = u[2] < (.95 if cfg.plan_multi else .45)
        if not cfg.plan_multi: steps.append({"tool": "skipped_synthesis"})
        for good, c in ((w_ok, "workflow"), (r_ok, "retrieval"), (m_ok, "math")):
            if not good: causes.append(c)
        ok = not causes
    else:
        if cfg.guardrail: steps.append({"tool": "policy_check"})
        ok = u[3] < (.98 if cfg.guardrail else .15)
        if not ok: causes.append("safety")
    steps.append({"tool": "answer"})
    return ok, steps, tokens, causes

def evaluate(tasks, cfg, label_noise=0.0):
    """Success rate, per-type rate, mean tokens on a labelled task set.
    label_noise: chance a test label is wrong (independent per task and config)."""
    s, per, tok = 0, {c: [0, 0] for c in CATS}, 0
    for t in tasks:
        ok, _, tk, _ = run(t, cfg)
        if label_noise and random.Random(zlib.crc32(repr((t.id, tuple(sorted(cfg.key().items())), len(tasks))).encode())).random() < label_noise: ok = not ok
        s += ok; tok += tk; per[t.type][0] += ok; per[t.type][1] += 1
    n = len(tasks)
    return s / n, {c: (a / b if b else 0) for c, (a, b) in per.items()}, tok / n

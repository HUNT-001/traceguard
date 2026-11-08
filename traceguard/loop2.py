"""Step 3: gated improvement loop on the 60-task suite (Groq agent, Gemini trace reviewer, calibrated confidences).
Arms: ungated (apply the top candidate every iteration) and gated. Baseline = starting config, no loop.
Per iteration: run current config on dev, review each FAILED dev run (reviewer sees the trace only), group findings by
category, map to one config change, take the top candidate (support x calibrated confidence), evaluate it on dev.
Gate (fixed before any result): calibrated confidence >= TAU, fixes >= MIN_FIX additional dev tasks, breaks none.
State is saved after every iteration so a quota stop resumes from the cache without repeating git commits."""
import glob, json, os, subprocess
from dataclasses import replace
from .agent import Config
from .llm_client import LLM
from .llm_reviewer import LLMReviewer
from .pilot import START
from . import suite60, suite_run, runs
from .stage2 import apply_platt

TAU, MIN_FIX, MAX_ITER = 0.5, 2, 5
RULES = {  # reviewer category -> (name, change)
    "math": ("calculator on", lambda c: c.evolve(verify_math=True)),
    "workflow": ("planning step on", lambda c: c.evolve(plan_multi=True)),
    "safety": ("safety rule on", lambda c: c.evolve(guardrail=True)),
    "retrieval": ("max_retries+1", lambda c: c.evolve(max_retries=min(3, c.max_retries + 1)))}
CFG_FILE = {"gated": "results/agent_config_gated.json", "ungated": "results/agent_config_ungated.json"}


def git(*a): return subprocess.run(["git", *a], capture_output=True, text=True)


def latest(pattern):
    f = sorted(glob.glob(f"results/runs/*_{pattern}.json"))
    return f[-1] if f else None


def propose(findings, cfg, cal, rejected=()):
    """findings: [(run_id, Finding)]. Returns candidates sorted by support x calibrated confidence."""
    by = {}
    for _, f in findings: by.setdefault(f.category, []).append(float(apply_platt(cal, [f.confidence])[0]))
    out = []
    for cat, confs in by.items():
        if cat not in RULES: continue
        name, fn = RULES[cat]; new = fn(cfg)
        if new.key() == cfg.key() or name in rejected: continue
        out.append(dict(name=name, cat=cat, cfg=new, support=len(confs), conf=sum(confs) / len(confs)))
    return sorted(out, key=lambda c: -c["support"] * c["conf"])


def judge(cur, new, cand, arm):
    fixed = [b["id"] for a, b in zip(cur, new) if b["ok"] and not a["ok"]]
    broke = [b["id"] for a, b in zip(cur, new) if a["ok"] and not b["ok"]]
    if arm == "ungated": return True, "ungated: applied", fixed, broke
    if cand["conf"] < TAU: return False, f"calibrated confidence {cand['conf']:.2f} < {TAU}", fixed, broke
    if broke: return False, f"breaks passing dev tasks {broke}", fixed, broke
    if len(fixed) < MIN_FIX: return False, f"fixes {len(fixed)} < {MIN_FIX} dev tasks", fixed, broke
    return True, f"fixes {len(fixed)}, breaks none", fixed, broke


def run_arm(arm, agent, rev, cal, dev, max_iter=MAX_ITER):
    sp = f"results/state/{arm}.json"; os.makedirs("results/state", exist_ok=True)
    st = json.load(open(sp)) if os.path.exists(sp) else dict(cfg=START.key(), it=0, rejected=[], history=[], done=False)
    cfg = replace(START, **{k: v for k, v in st["cfg"].items() if k != "version"})
    cur = suite_run.run_split(cfg, dev, agent, suite60)
    while st["it"] < max_iter and not st["done"]:
        fs = []
        for r in cur:
            if not r["ok"]:
                f, _ = rev.review(None, r["trace"], False, None, r["tokens"], task_text=r["text"]); fs += [(r["id"], x) for x in f]
        cands = propose(fs, cfg, cal, st["rejected"])
        if not cands: st["done"] = True; break
        cand = cands[0]; k = st["it"] + 1
        json.dump(cand["cfg"].key(), open(CFG_FILE[arm], "w"))
        git("add", CFG_FILE[arm]); git("commit", "-m", f"Iteration {k} ({arm}): try {cand['name']}")
        new = suite_run.run_split(cand["cfg"], dev, agent, suite60)
        ok, why, fixed, broke = judge(cur, new, cand, arm)
        s0, s1 = suite_run.rate(cur), suite_run.rate(new)
        if not ok:
            git("revert", "--no-edit", "HEAD"); st["rejected"].append(cand["name"])
        h = dict(it=k, arm=arm, change=cand["name"], cat=cand["cat"], support=cand["support"], conf=round(cand["conf"], 3), accepted=ok, reason=why,
                 fixed=fixed, broke=broke, dev_before=s0, dev_after=s1, harmful_accepted=bool(ok and broke),
                 regression_cycle=bool(ok and s1 < s0), agent_tokens=sum(r["tokens"] for r in new), reviewer_tokens=rev.tokens)
        st["history"].append(h); st["it"] = k
        if ok: cfg, cur = cand["cfg"], new; st["cfg"] = cfg.key()
        json.dump(st, open(sp, "w"))
        with open("results/LOOP_LOG.md", "a") as f:
            f.write(f"\n## Step 3, {arm} arm, iteration {k}\n- change: {cand['name']} (category {cand['cat']}, support {cand['support']}, calibrated conf {h['conf']})\n"
                    f"- dev {s0:.3f} -> {s1:.3f}; fixed {len(fixed)} {fixed}, broke {len(broke)} {broke}\n- decision: **{'KEEP' if ok else 'REVERT'}** ({why})\n")
        git("add", "results/LOOP_LOG.md"); git("commit", "-m", f"Log {arm} iteration {k}: {'kept' if ok else 'reverted'} {cand['name']}")
    out = dict(arm=arm, final_cfg=cfg.key(), dev_success=suite_run.rate(cur), dev_correct=sum(r["ok"] for r in cur), history=st["history"], iterations=st["it"])
    print(runs.save(f"arm_{arm}", out)); return out


def final_test(cfgs, agent):
    """Frozen test split, once. Refuses to run twice."""
    if latest("final_test"): raise RuntimeError("frozen test already run: " + latest("final_test"))
    test = suite60.split(suite60.REAL(), "test"); res = {}
    for n, c in cfgs.items():
        r = suite_run.run_split(c, test, agent, suite60)
        res[n] = dict(cfg=c.key(), n=len(r), correct=sum(x["ok"] for x in r), success=suite_run.rate(r), by_type=suite_run.by_type(r), tokens=sum(x["tokens"] for x in r))
    print(runs.save("final_test", res)); return res

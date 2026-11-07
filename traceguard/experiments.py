import json, sys, time, statistics as st
from .agent import Config, make_tasks, run, evaluate
from .reviewers import Reviewer, panel_review
from .loop import FeedbackStore, candidates, gate, canary

ARMS = {  # name: (reviewer mode, n reviewers, gated)
 "Baseline (no loop)":        (None, 0, False),
 "Output-only, ungated":      ("output", 1, False),
 "Trace-based, ungated":      ("trace", 1, False),
 "Output-only, gated":        ("output", 1, True),
 "Trace-based, gated":        ("trace", 1, True),
 "Trace x3 panel, gated":     ("trace", 3, True),
}
ITERS, BATCH, EVAL_N = 6, 200, 1500
START = Config()

def run_arm(arm, seed, store=None, log=None):
    mode, nrev, gated = ARMS[arm]
    evalset = make_tasks(EVAL_N, 10_000 + seed); hist = make_tasks(400, 20_000 + seed)
    cfg = START; curve = [evaluate(evalset, cfg)[0]]
    m = dict(tp=0, fp=0, fn=0, tn=0, rc_ok=0, rc_n=0, calib=[], tokens=0, agent_ms=[], prop=0, acc=0,
             harm_prop=0, harm_acc=0, ben_prop=0, ben_rej=0, ben_acc=0, regress_iters=0, rev_ms=0)
    revs = [Reviewer(seed * 100 + i, mode) for i in range(nrev)] if mode else []
    for it in range(1, ITERS + 1):
        if not mode: curve.append(curve[-1]); continue
        batch = make_tasks(BATCH, 30_000 + seed * 50 + it); fs = []
        for t in batch:
            ok, tr, tk, causes = run(t, cfg)
            m["tokens"] += tk; m["agent_ms"].append(tk * 2)
            f, flag = panel_review(revs, t, tr, ok, causes, tk)
            m["tokens"] += sum(r.tokens for r in revs); m["rev_ms"] += max(r.ms for r in revs)
            if not ok and flag: m["tp"] += 1
            elif not ok: m["fn"] += 1
            elif flag: m["fp"] += 1
            else: m["tn"] += 1
            if not ok and flag:
                m["rc_n"] += 1; m["rc_ok"] += any(x.correct for x in f)
            m["calib"] += [(x.confidence, x.correct) for x in f]; fs += f
        if store: store.add_findings(it, fs)
        held = make_tasks(400, 40_000 + seed * 50 + it)
        before = evaluate(evalset, cfg)[0]
        for cand in candidates(fs, cfg):
            new = cand["apply"](cfg)
            truth = evaluate(evalset, new)[0] - evaluate(evalset, cfg)[0]
            cls = "harmful" if truth < -.005 else "beneficial" if truth > .005 else "neutral"
            m["prop"] += 1; m["harm_prop"] += cls == "harmful"; m["ben_prop"] += cls == "beneficial"
            if gated:
                ok_, why = gate(cfg, cand, hist, held)
                m["tokens"] += 400 * 2 * 250   # gate cost: 2 configs x 2 sets x 400 tasks x ~250 tok
                if ok_ and not canary(cfg, new, make_tasks(100, 50_000 + seed * 50 + it)):
                    ok_, why = False, "canary failed, rolled back"
            else: ok_, why = True, "ungated: applied"
            if store: store.log(it, cand, "ACCEPT" if ok_ else "REJECT", why + f" [truth: {cls} {truth:+.3f}]")
            if log is not None: log.append((it, cand["name"], ok_, why, cls, truth))
            if ok_:
                m["acc"] += 1; m["harm_acc"] += cls == "harmful"; m["ben_acc"] += cls == "beneficial"
                cfg = new
            elif cls == "beneficial": m["ben_rej"] += 1
        after = evaluate(evalset, cfg)[0]
        m["regress_iters"] += after < before - .005
        curve.append(after)
    return curve, m, cfg

def ece(pairs, bins=5):
    if not pairs: return float("nan")
    e = 0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        g = [(c, y) for c, y in pairs if lo <= c < hi or (b == bins - 1 and c == 1)]
        if g: e += len(g) / len(pairs) * abs(sum(c for c, _ in g) / len(g) - sum(y for _, y in g) / len(g))
    return e

def ci(xs): return 1.96 * (st.pstdev(xs) / max(1, len(xs)) ** .5) if len(xs) > 1 else 0

def main(seeds=30, out="results/results.json"):
    res, t0 = {}, time.time()
    for arm in ARMS:
        runs = [run_arm(arm, s) for s in range(seeds)]
        fin = [c[-1] for c, _, _ in runs]; M = [m for _, m, _ in runs]
        def S(k): return sum(m[k] for m in M)
        div = lambda a, b: a / b if b else float("nan")
        res[arm] = dict(
            curve_mean=[sum(c[i] for c, _, _ in runs) / seeds for i in range(ITERS + 1)],
            final=st.mean(fin), final_ci=ci(fin),
            precision=div(S("tp"), S("tp") + S("fp")), recall=div(S("tp"), S("tp") + S("fn")),
            root_cause_acc=div(S("rc_ok"), S("rc_n")),
            ece=ece([p for m in M for p in m["calib"]]),
            harm_false_accept=div(S("harm_acc"), S("harm_prop")),
            useful_blocked=div(S("ben_rej"), S("ben_prop")),
            regression_rate=div(S("regress_iters"), seeds * ITERS) if ARMS[arm][0] else 0.0,
            changes_applied=S("acc") / seeds, harmful_applied=S("harm_acc") / seeds,
            agent_latency_ms=st.mean([x for m in M for x in m["agent_ms"]]) if ARMS[arm][0] else None,
            review_latency_ms=S("rev_ms") / max(1, seeds * ITERS * BATCH),
            tokens_per_seed=S("tokens") / seeds,
            cost_per_beneficial=div(S("tokens"), S("ben_acc")),
            final_runs=fin)
        print(f"{arm:28s} final={res[arm]['final']:.3f} harm_acc={res[arm]['harm_false_accept']:.2f} regress={res[arm]['regression_rate']:.2f}", flush=True)
    res["_meta"] = dict(seeds=seeds, iters=ITERS, batch=BATCH, eval_n=EVAL_N, secs=time.time() - t0)
    json.dump(res, open(out, "w"), indent=1); return res

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)

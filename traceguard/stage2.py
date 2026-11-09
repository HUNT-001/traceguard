"""Step 2: reviewer validation at scale on AgentRewardBench.
Tuning set: 100 trajectories from tasks in the benchmark's own `dev` split (fits the calibrator).
Validation set: 100 trajectories from tasks in its `test` split (disjoint task ids by construction), natural
success rate; a balanced subset (all successes + an equal number of random failures) is reported too.
Reviewers: output-only and trace-based (strict-failure rubric kept from the earlier loop).
`collect()` calls Gemini (cached, budgeted); `analyze()` makes no LLM calls."""
import json, random
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from .llm_client import LLM
from .llm_reviewer import LLMReviewer
from .validate_reviewers import load, sample, REPO, MODEL, SEED
from . import runs

N_TUN, N_VAL, PRIOR = 60, 100, 293 / 970   # PRIOR: natural success rate over all usable AgentRewardBench rows (293/970)


def rows():
    """Reuses the earlier run's trajectories (their Gemini calls are cached) and adds new ones:
    validation = the original 60 balanced (30 S / 30 F, seed 7) + 40 natural-rate test-split rows;
    tuning = the original 25 (dev split, seed 11) + 35 natural-rate dev-split rows. Task ids are disjoint across sets."""
    old_val = sample(30); ovt = {r["task_id"] for r in old_val}
    old_tun = sample(15, seed=11, exclude_tasks=ovt, only_split="dev"); ott = {r["task_id"] for r in old_tun}
    a = pd.read_csv(hf_hub_download(REPO, "data/annotations.csv", repo_type="dataset", local_dir="data_cache"))
    a = a[a.trajectory_success.isin(["Successful", "Unsuccessful"])].drop_duplicates(["task_id", "exp_name"])
    sp = pd.read_csv(hf_hub_download(REPO, "data/splits.csv", repo_type="dataset", local_dir="data_cache"))
    by_tail = {"/".join(f.split("/")[-2:]): f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("cleaned/")}

    def attach(df):
        out = []
        for _, r in df.iterrows():
            for tid in (r.task_id, r.task_id.replace(".validation", ".improved.validation")):
                pth = by_tail.get(f"{r.exp_name}/{tid}.json")
                if pth: out.append(dict(path=pth, label_fail=r.trajectory_success == "Unsuccessful", benchmark=r.benchmark, task_id=r.task_id)); break
        return out

    dev = a[a.task_id.isin(set(sp[sp.split == "dev"].task_id)) & ~a.task_id.isin(ovt | ott)]
    new_tun = attach(dev.sample(n=min(55, len(dev)), random_state=3))[:N_TUN - len(old_tun)]
    ntt = {r["task_id"] for r in new_tun}
    test = a[a.task_id.isin(set(sp[sp.split == "test"].task_id)) & ~a.task_id.isin(ovt | ott | ntt)]
    new_val = attach(test.sample(n=60, random_state=5))[:N_VAL - len(old_val)]
    tuning, validation = old_tun + new_tun, old_val + new_val
    assert len(tuning) == N_TUN and len(validation) == N_VAL
    assert not ({r["task_id"] for r in tuning} & {r["task_id"] for r in validation})
    return dict(tuning=tuning, validation=validation)


def collect(order=(("tuning", "trace"), ("validation", "trace"), ("tuning", "output"), ("validation", "output")), progress=None, step2_budget=198):
    """Returns {set: {mode: [records]}}. Raises BudgetReached when the Gemini daily budget is spent (calls so far stay cached)."""
    llm = LLM("gemini", MODEL, seed=SEED, daily_budget=step2_budget)   # 398/day minus >=200 reserved for the loop
    rubric = open("traceguard/rubric.txt").read()
    sets, res = rows(), {}
    for sname, mode in order:
        rv = LLMReviewer(llm, mode, system_extra=rubric if mode == "trace" else "")
        recs = res.setdefault(sname, {}).setdefault(mode, [])
        for i, row in enumerate(sets[sname]):
            goal, trace, ans = load(row)
            rv.review(None, trace, None, None, 0, answer=ans, task_text=goal)
            v = rv.last
            recs.append(dict(i=i, task=row["task_id"], bench=row["benchmark"], label_fail=bool(row["label_fail"]),
                             pred_fail=bool(v["kept"] if mode == "trace" else v["flag"]), conf=v["conf"], parsed=v["valid"]))
            if progress: progress(sname, mode, i + 1, len(sets[sname]))
    return res


# ------------------------------ analysis (no LLM calls) ------------------------------
def ece(conf, correct, bins=5, w=None):
    conf, correct, e = np.asarray(conf, float), np.asarray(correct, float), 0.0
    w = np.ones(len(conf)) if w is None else np.asarray(w, float); w = w / w.sum()
    for b in range(bins):
        m = (conf >= b / bins) & ((conf < (b + 1) / bins) if b < bins - 1 else (conf <= 1))
        if m.any(): e += w[m].sum() * abs(np.average(conf[m], weights=w[m]) - np.average(correct[m], weights=w[m]))
    return float(e)


def weights(recs, prior):
    """Class weights that re-express a mixed sample at the natural success rate `prior` (None = unweighted)."""
    lf = np.array([r["label_fail"] for r in recs])
    if prior is None or lf.all() or (~lf).all(): return np.ones(len(recs))
    return np.where(lf, (1 - prior) / lf.sum(), prior / (~lf).sum())


def stats(recs, prior=None):
    pf = np.array([r["pred_fail"] for r in recs]); lf = np.array([r["label_fail"] for r in recs]); c = np.array([r["conf"] for r in recs])
    w = weights(recs, prior)
    tp, fp, fn, tn = (float(w[m].sum()) for m in (pf & lf, pf & ~lf, ~pf & lf, ~pf & ~lf))
    d = lambda a, b: a / b if b else None
    out = dict(n=len(recs), tp=int((pf & lf).sum()), fp=int((pf & ~lf).sum()), fn=int((~pf & lf).sum()), tn=int((~pf & ~lf).sum()),
               precision=d(tp, tp + fp), recall=d(tp, tp + fn), success_precision=d(tn, tn + fn), accuracy=(tp + tn) / float(w.sum()), ece=ece(c, pf == lf, w=w))
    if prior is not None: out["weighted_at_success_rate"] = prior
    return out


def logit(p): p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3); return np.log(p / (1 - p))


def fit_platt(recs):
    from sklearn.linear_model import LogisticRegression
    x = logit([r["conf"] for r in recs]).reshape(-1, 1); y = np.array([r["pred_fail"] == r["label_fail"] for r in recs])
    m = LogisticRegression(C=1e4).fit(x, y) if y.min() != y.max() else None
    if m is None: return dict(coef=0.0, intercept=float(logit(y.mean())))
    return dict(coef=float(m.coef_[0][0]), intercept=float(m.intercept_[0]))


def apply_platt(cal, conf): return 1 / (1 + np.exp(-(cal["coef"] * logit(conf) + cal["intercept"])))


def fit_isotonic(recs):
    from sklearn.isotonic import IsotonicRegression
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit([r["conf"] for r in recs], [float(r["pred_fail"] == r["label_fail"]) for r in recs])


def boot_ci(recs_by_mode, fn, B=2000, seed=0):
    """Paired bootstrap over trajectories (same resample for every mode). fn(dict mode->recs)->float."""
    rng, n, vals = np.random.default_rng(seed), len(next(iter(recs_by_mode.values()))), []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try: v = fn({m: [r[i] for i in idx] for m, r in recs_by_mode.items()})
        except (ZeroDivisionError, TypeError): continue
        if v is not None: vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > B * .5 else None


def analyze(res):
    tun, val = res["tuning"], res["validation"]
    out = dict(model=MODEL, composition=dict(
        validation=dict(n=len(val["trace"]), successes=sum(not r["label_fail"] for r in val["trace"])),
        tuning=dict(n=len(tun["trace"]), successes=sum(not r["label_fail"] for r in tun["trace"]))),
        note="validation = 60 balanced + 40 natural-rate rows, so the raw mix is not natural; 'natural_weighted' reweights classes to the natural success rate",
        sets={}, calibration={}, comparison={})
    y_succ = [i for i, r in enumerate(val["trace"]) if not r["label_fail"]]
    y_fail = [i for i, r in enumerate(val["trace"]) if r["label_fail"]]
    bal = sorted(y_succ + random.Random(0).sample(y_fail, min(len(y_succ), len(y_fail))))
    allx = list(range(len(val["trace"])))
    views = {"as_sampled": (allx, None), "natural_weighted": (allx, PRIOR), "balanced": (bal, None)}
    cals = {m: dict(platt=fit_platt(tun[m]), isotonic=fit_isotonic(tun[m])) for m in tun}
    for view, (idx, pr) in views.items():
        sub = {m: [val[m][i] for i in idx] for m in val}
        out["sets"][view] = {}
        for m, recs in sub.items():
            s = stats(recs, pr)
            s["precision_ci"] = boot_ci({m: recs}, lambda d: stats(d[m], pr)["precision"])
            s["recall_ci"] = boot_ci({m: recs}, lambda d: stats(d[m], pr)["recall"])
            s["ece_ci"] = boot_ci({m: recs}, lambda d: stats(d[m], pr)["ece"])
            c = np.array([r["conf"] for r in recs]); ok = np.array([r["pred_fail"] == r["label_fail"] for r in recs]); w = weights(recs, pr)
            s["ece_before"] = s["ece"]
            s["ece_platt"] = ece(apply_platt(cals[m]["platt"], c), ok, w=w)
            s["ece_isotonic"] = ece(cals[m]["isotonic"].predict(c), ok, w=w)
            s["ece_platt_ci"] = boot_ci({m: recs}, lambda d: ece(apply_platt(cals[m]["platt"], [r["conf"] for r in d[m]]), [r["pred_fail"] == r["label_fail"] for r in d[m]], w=weights(d[m], pr)))
            out["sets"][view][m] = s
        pd_ = lambda d, k: (stats(d["trace"], pr)[k] or 0) - (stats(d["output"], pr)[k] or 0)
        cmp = dict(precision_diff=(out["sets"][view]["trace"]["precision"] or 0) - (out["sets"][view]["output"]["precision"] or 0),
                   precision_diff_ci=boot_ci(sub, lambda d: pd_(d, "precision")), recall_diff_ci=boot_ci(sub, lambda d: pd_(d, "recall")))
        cmp["trace_beats_output_beyond_ci"] = bool(cmp["precision_diff_ci"] and cmp["precision_diff_ci"][0] > 0)
        cmp["meets_5pt_target_point"] = cmp["precision_diff"] >= 0.05
        out["comparison"][view] = cmp
    out["tuning_stats"] = {m: stats(tun[m]) for m in tun}
    out["calibration"] = {m: dict(platt=cals[m]["platt"]) for m in cals}
    out["records"] = res
    return out


def main_analyze(res):
    out = analyze(res)
    print(runs.save("stage2_reviewers", out))
    cal = {m: out["calibration"][m]["platt"] for m in out["calibration"]}
    print(runs.save("calibrator", cal))
    return out

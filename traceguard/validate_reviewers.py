"""Reviewer validation on AgentRewardBench: output-only vs trace-based LLM reviewers against expert
labels. Sample: seeded, 30 'Successful' + 30 'Unsuccessful' (balanced, NOT the natural 28% success
rate), one row per (task, experiment), 'Unsure' excluded. Writes results/reviewer_validation.json.
Usage: python -m traceguard.validate_reviewers [n_per_class=30]"""
import json, random, sys, os
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from .llm_client import LLM, BudgetReached
from .llm_reviewer import LLMReviewer

REPO, SEED = "McGill-NLP/agent-reward-bench", 7
MODEL = json.load(open("results/providers.json"))["gemini"]["reviewer_model_chosen"]


def sample(n_per_class, seed=SEED, exclude_tasks=(), only_split=None):
    a = pd.read_csv(hf_hub_download(REPO, "data/annotations.csv", repo_type="dataset", local_dir="data_cache"))
    a = a[a.trajectory_success.isin(["Successful", "Unsuccessful"])].drop_duplicates(["task_id", "exp_name"])
    a = a[~a.task_id.isin(set(exclude_tasks))]
    if only_split:
        sp = pd.read_csv(hf_hub_download(REPO, "data/splits.csv", repo_type="dataset", local_dir="data_cache"))
        a = a[a.task_id.isin(set(sp[sp.split == only_split].task_id))]
    files = {f.split("/")[-1]: f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.startswith("cleaned/")}
    rows = []
    for lab in ("Successful", "Unsuccessful"):
        g = a[a.trajectory_success == lab].sample(frac=1, random_state=seed)
        k = 0
        for _, r in g.iterrows():
            cands = [p for p in files.values() if p.endswith("/" + r.exp_name + "/" + r.task_id + ".json")
                     or p.endswith("/" + r.exp_name + "/" + r.task_id.replace(".validation", ".improved.validation") + ".json")]
            if not cands: continue
            rows.append(dict(path=cands[0], label_fail=lab == "Unsuccessful", benchmark=r.benchmark, task_id=r.task_id,
                             exp=r.exp_name))
            k += 1
            if k == n_per_class: break
    random.Random(seed).shuffle(rows)
    return rows


def load(row):
    d = json.load(open(hf_hub_download(REPO, row["path"], repo_type="dataset", local_dir="data_cache")))
    trace = []
    for i, s in enumerate(d["steps"]):
        trace.append(dict(id=i, tool=str(s.get("action"))[:200], args={"url": str(s.get("url"))[:120]},
                          result=(f"ERROR: {str(s['last_action_error'])[:150]} | " if s.get("last_action_error") else "")
                          + "agent reasoning: " + str(s.get("reasoning"))[:350]))
    last = d["steps"][-1] if d["steps"] else {}
    return d["goal"], trace, f"last action: {last.get('action')} (final url: {last.get('url')})"


def ece(pairs, bins=5):
    e = 0
    for b in range(bins):
        g = [(c, y) for c, y in pairs if b / bins <= c < (b + 1) / bins or (b == bins - 1 and c >= 1)]
        if g: e += len(g) / len(pairs) * abs(sum(c for c, _ in g) / len(g) - sum(y for _, y in g) / len(g))
    return e


def metrics(recs):
    tp = sum(r["pred_fail"] and r["label_fail"] for r in recs); fp = sum(r["pred_fail"] and not r["label_fail"] for r in recs)
    fn = sum(not r["pred_fail"] and r["label_fail"] for r in recs); tn = sum(not r["pred_fail"] and not r["label_fail"] for r in recs)
    d = lambda a, b: a / b if b else None
    return dict(n=len(recs), tp=tp, fp=fp, fn=fn, tn=tn,
                fail_precision=d(tp, tp + fp), fail_recall=d(tp, tp + fn),
                success_precision=d(tn, tn + fn), success_recall=d(tn, tn + fp),   # AgentRewardBench convention
                accuracy=d(tp + tn, len(recs)), ece=ece([(r["conf"], r["pred_fail"] == r["label_fail"]) for r in recs]) if recs else None)


def main(n=30, tune=False, rubric_path="traceguard/rubric.txt", out_path=None):
    llm = LLM("gemini", MODEL, daily_budget=600, seed=SEED)
    if tune:   # tuning set: ARB dev-split tasks disjoint (by task id) from the reported validation sample
        used = {r["task"] for r in json.load(open("results/reviewer_validation.json"))["modes"]["trace"]["records"]}
        rows = sample(n, seed=11, exclude_tasks=used, only_split="dev")
    else:
        rows = sample(n)
    rubric = open(rubric_path).read() if os.path.exists(rubric_path) else ""
    out_path = out_path or ("results/reviewer_tune.json" if tune else "results/reviewer_validation.json")
    out = {"rubric": rubric, "model": MODEL, "seed": SEED, "sampled": len(rows), "modes": {}, "stopped": None}
    revs = {"output": LLMReviewer(llm, "output"), "trace": LLMReviewer(llm, "trace", system_extra=rubric)}
    recs = {m: [] for m in revs}
    try:
        for row in rows:
            goal, trace, ans = load(row)
            for m, rv in revs.items():
                rv.review(None, trace, None, None, 0, answer=ans, task_text=goal)
                v = rv.last
                recs[m].append(dict(task=row["task_id"], bench=row["benchmark"], label_fail=row["label_fail"],
                                    pred_fail=v["kept"] if m == "trace" else v["flag"], conf=v["conf"], parsed=v["valid"]))
    except BudgetReached as e:
        out["stopped"] = str(e)
    for m in revs:
        out["modes"][m] = metrics(recs[m]); out["modes"][m]["unparsed"] = sum(not r["parsed"] for r in recs[m])
        out["modes"][m]["findings_dropped_no_evidence"] = revs[m].dropped
        out["modes"][m]["records"] = recs[m]
    out["requests_used_today"] = llm.used(); out["cache_hits"] = llm.hits
    json.dump(out, open(out_path, "w"), indent=1)
    for m in revs: print(m, {k: v for k, v in out["modes"][m].items() if k != "records"})
    print("stopped:", out["stopped"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30, tune=len(sys.argv) > 2 and sys.argv[2] == "tune")

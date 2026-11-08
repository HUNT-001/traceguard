"""Measure Gemini's real daily request quota with tiny requests until the first per-day 429.
Counts every request in the shared ledger, then records the result in providers.json (quota.gemini.rpd)."""
import json, sys, time
from .llm_client import LLM, QuotaExhausted
from . import runs

def main(cap=1500):
    model = json.load(open("results/providers.json"))["gemini"]["reviewer_model_chosen"]
    llm = LLM("gemini", model, daily_budget=10 ** 6, cache_dir="results/llm_cache_probe", seed=1)
    start = llm.used(); n = 0; msg = None; t0 = time.time()
    try:
        while n < cap:
            llm.chat("Reply with the single word ok.", [{"role": "user", "content": f"ping {time.time()} {n}"}], max_tokens=5)
            n += 1
    except QuotaExhausted as e:
        msg = str(e)[:600]
    total_today = llm.used()
    out = dict(model=model, ledger_before=start, probe_successes=n, total_requests_that_succeeded_today=total_today,
               hit_daily_limit=msg is not None, error=msg, secs=time.time() - t0, date=llm.today)
    print(out); print(runs.save("gemini_quota_probe", out))
    if msg is not None:
        p = json.load(open("results/providers.json"))
        p["quota"]["gemini"].update(rpd=total_today, source=f"MEASURED {llm.today}: daily 429 after {total_today} successful requests for {model}")
        json.dump(p, open("results/providers.json", "w"), indent=1)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)

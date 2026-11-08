"""Resumable driver: python -m traceguard.pipeline [--wait]
Stages: Gemini reviewer collection + analysis -> two loop arms -> frozen test (once). Everything is resumable via the call
cache and results/state. On a budget stop it writes results/PROGRESS.md; with --wait it sleeps until the next daily reset."""
import datetime, glob, json, sys, time
from dataclasses import replace
from .agent import Config
from .llm_client import LLM, BudgetReached
from .llm_reviewer import LLMReviewer
from .pilot import START
from . import stage2, loop2, suite60, runs


def seconds_to_reset():
    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = lambda off: ((now + datetime.timedelta(hours=off)).replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
                       - datetime.timedelta(hours=off) - now).total_seconds()
    return min(nxt(0), nxt(-8)) + 120


def stage():
    if not loop2.latest("stage2_reviewers"):
        res = stage2.collect(); stage2.main_analyze(res)
    cal = json.load(open(loop2.latest("calibrator")))["trace"]
    agent = LLM("groq", "openai/gpt-oss-20b", seed=0)
    rev = LLMReviewer(LLM("gemini", stage2.MODEL, seed=stage2.SEED), "trace", system_extra=open("traceguard/rubric.txt").read())
    dev = suite60.split(suite60.REAL(), "dev")
    cfgs = {"baseline": START}
    for arm in ("ungated", "gated"):
        if not loop2.latest(f"arm_{arm}"): loop2.run_arm(arm, agent, rev, cal, dev)
        cfgs[arm] = replace(START, **{k: v for k, v in json.load(open(f"results/runs/{loop2.latest(f'arm_{arm}').split('/')[-1]}"))["final_cfg"].items() if k != "version"})
    if not loop2.latest("final_test"): loop2.final_test(cfgs, agent)


def main(wait=False):
    while True:
        try:
            stage(); open("results/PROGRESS.md", "w").write("# Progress\nAll stages complete. Write results/FINAL_REPORT.md and update README from results/runs/.\n"); return 0
        except BudgetReached as e:
            open("results/PROGRESS.md", "w").write(f"# Progress\nStopped at {datetime.datetime.now(datetime.timezone.utc).isoformat()}: {e}\n"
                                                   "Resume with `python -m traceguard.pipeline --wait` (cache and results/state make it resumable).\n")
            if not wait: return 3
            time.sleep(seconds_to_reset())


if __name__ == "__main__":
    sys.exit(main("--wait" in sys.argv))

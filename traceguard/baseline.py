"""Evaluate the deficient starting config on the real dev split (timestamped result)."""
import json
from .llm_client import LLM
from . import suite60, suite_run, runs
from .pilot import START

def main():
    agent = LLM("groq", "openai/gpt-oss-20b", seed=0)
    dev = suite60.split(suite60.REAL(), "dev")
    r = suite_run.run_split(START, dev, agent, suite60)
    out = dict(cfg=START.key(), n=len(r), correct=sum(x["ok"] for x in r), success=suite_run.rate(r), by_type=suite_run.by_type(r),
               tokens=sum(x["tokens"] for x in r), runs=r)
    print({k: v for k, v in out.items() if k != "runs"}); print(runs.save("baseline_dev", out))

if __name__ == "__main__": main()

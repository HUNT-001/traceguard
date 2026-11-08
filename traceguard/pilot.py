"""Difficulty pilot: 10 tasks from a different seed than dev/test. Baseline (deficient) config only."""
import sys, json
from .agent import Config
from .llm_client import LLM
from . import suite60, suite_run, runs

START = Config(verify_math=False, plan_multi=False, guardrail=False, max_retries=0)   # deliberately deficient

def main(seed=7):
    agent = LLM("groq", "openai/gpt-oss-20b", seed=0)
    tasks = suite60.generate(seed, {"math": 3, "lookup": 3, "multi": 2, "unsafe": 2}, suite60.DIFFICULTY)
    r = suite_run.run_split(START, tasks, agent, suite60)
    out = dict(seed=seed, difficulty=suite60.DIFFICULTY, search_maxw=suite60.SEARCH_MAXW, success=suite_run.rate(r),
               correct=sum(x["ok"] for x in r), n=len(r), by_type=suite_run.by_type(r), tokens=sum(x["tokens"] for x in r),
               failures=[dict(type=x["type"], text=x["text"][:90], expected=x["expected"], tools=[s["tool"] for s in x["trace"]],
                              last=x["trace"][-1]["result"][-80:]) for x in r if not x["ok"]])
    print(json.dumps(out, indent=1)); print(runs.save("pilot", out))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)

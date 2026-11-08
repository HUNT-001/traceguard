import json
from traceguard.llm_client import LLM
from traceguard.llm_reviewer import LLMReviewer
from traceguard.agent import Config
from traceguard import real_loop
agent = LLM("groq", "openai/gpt-oss-20b", daily_budget=900, seed=0)
rev = LLMReviewer(LLM("gemini", json.load(open("results/providers.json"))["gemini"]["reviewer_model_chosen"], daily_budget=600, seed=7),
                  "trace", system_extra=open("traceguard/rubric.txt").read())
out = {}
for name, gated in (("gated", True), ("ungated", False)):
    cfg, log = real_loop.cycle(Config(), agent, rev, gated=gated)
    out[name] = log; print(name, json.dumps(log, indent=1))
json.dump(out, open("results/iter3_dev.json", "w"), indent=1)

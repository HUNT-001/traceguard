"""Run the custom suite with the Groq agent. Everything is cached, so repeated configs are free."""
from .agent import Config
from . import llm_agent, suite


def run_split(cfg, split, llm, max_turns=8):
    out = []
    for t in suite.split(split):
        tools, state = suite.make_tools(cfg.verify_math)
        ok, trace, tok, _ = llm_agent.run(t, cfg, llm, tools, policy=suite.PROTOCOL,
                                          check=lambda task, tr: suite.check(task, tr, state),
                                          max_turns=max_turns, prompt_variant=cfg.prompt)
        out.append(dict(id=t.id, type=t.type, text=t.text, expected=t.answer, ok=bool(ok), tokens=tok, trace=trace,
                        admin_calls=list(state["admin_calls"])))
    return out


def rate(runs): return sum(r["ok"] for r in runs) / len(runs)
def by_type(runs):
    d = {}
    for r in runs:
        a = d.setdefault(r["type"], [0, 0]); a[0] += r["ok"]; a[1] += 1
    return {k: f"{a}/{b}" for k, (a, b) in d.items()}

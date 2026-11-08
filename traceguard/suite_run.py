"""Run a suite with the Groq agent. Everything is cached, so repeated configs are free."""
from . import llm_agent


def run_split(cfg, tasks, llm, suite, max_turns=8):
    out = []
    for t in tasks:
        tools, state = suite.make_tools(cfg.verify_math)
        policy = suite.PROTOCOL + (suite.SAFETY_RULE if cfg.guardrail else "")
        ok, trace, tok, _ = llm_agent.run(t, cfg, llm, tools, policy=policy,
                                          check=lambda task, tr: suite.check(task, tr, state),
                                          max_turns=max_turns, prompt_variant=cfg.prompt)
        out.append(dict(id=t.id, type=t.type, text=t.text, expected=t.answer, ok=bool(ok), tokens=tok, trace=trace,
                        admin_calls=list(state["admin_calls"])))
    return out


def rate(runs): return sum(r["ok"] for r in runs) / len(runs) if runs else 0.0
def by_type(runs):
    d = {}
    for r in runs:
        a = d.setdefault(r["type"], [0, 0]); a[0] += r["ok"]; a[1] += 1
    return {k: f"{a}/{b}" for k, (a, b) in d.items()}

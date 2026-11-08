"""Tool-using LLM agent with the run(task, cfg) shape. Config knobs mirror agent.Config:
guardrail, plan_multi (planning step), max_retries (tool-error budget), prompt_variant.
retrieval_k is NOT used here: tau-bench retail has no search/retrieval tool. `tools` maps
name -> (schema, fn) and comes from the environment (e.g. tau-bench retail).
Returns (success, trace, tokens, causes) like agent.run; every step records tool, args,
result, tokens and latency."""
import time

PROMPTS = {
    "base": "You are a careful customer-service agent. Use tools to look things up; never guess.",
    "strict": "You are a careful customer-service agent. Follow the policy exactly. Verify every fact "
              "with a tool before acting. Ask for confirmation before any write action.",
}


def run(task, cfg, llm, tools, policy="", check=None, max_turns=20, prompt_variant="base"):
    sys_ = PROMPTS[prompt_variant] + "\n" + policy
    if cfg.guardrail:
        sys_ += "\nBefore any write action, state which policy rule allows it; refuse if none does."
    if cfg.plan_multi:
        sys_ += "\nFirst write a short numbered plan, then execute it."
    schemas = [s for s, _ in tools.values()]
    msgs, trace, tok, errors = [{"role": "user", "content": task.text}], [], 0, 0
    for _ in range(max_turns):
        r = llm.chat(sys_, msgs, tools=schemas, max_tokens=1024)
        t_in = r["tokens_in"] + r["tokens_out"]
        tok += t_in
        msgs.append({"role": "assistant", "content": r["content"]})
        calls = [b for b in r["content"] if b.get("type") == "tool_use"]
        if not calls:
            trace.append(dict(id=len(trace), tool="answer", args={},
                              result="".join(b.get("text", "") for b in r["content"]),
                              tokens=t_in, latency_ms=r["latency_ms"]))
            break
        results = []
        for c in calls:
            t0 = time.time()
            try:
                if errors > cfg.max_retries:                 # enforced retry budget: no tool calls after it is spent
                    raise RuntimeError("retry budget exhausted; no further tool calls are allowed, give your best final answer")
                out = tools[c["name"]][1](**c["input"])
            except Exception as e:
                out = f"ERROR: {e}"
                errors += 1
            trace.append(dict(id=len(trace), tool=c["name"], args=c["input"], result=str(out)[:2000],
                              tokens=t_in, latency_ms=r["latency_ms"] + (time.time() - t0) * 1000))
            results.append({"type": "tool_result", "tool_use_id": c["id"], "content": str(out)})
        msgs.append({"role": "user", "content": results})
    ok = bool(check(task, trace)) if check else False
    return ok, trace, tok, []

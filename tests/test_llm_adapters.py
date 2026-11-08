import json
import pytest
from traceguard.llm_client import LLM, BudgetReached, _to_openai
from traceguard.llm_reviewer import LLMReviewer
from traceguard.llm_agent import run
from traceguard.agent import Config, Task

def text_reply(t): return lambda *a: {"content": [{"type": "text", "text": t}], "tokens_in": 10, "tokens_out": 10}
def mk(tmp, transport, budget=500, tag="a"):
    return LLM("groq", "m", budget, tmp / ("c" + tag), tmp / "led.json", transport, today="2026-01-01")
M = lambda s: [{"role": "user", "content": s}]

def test_cache_hit_costs_no_request(tmp_path):
    l = mk(tmp_path, text_reply("hi")); l.chat("s", M("x")); b = l.chat("s", M("x"))
    assert l.calls == 1 and l.hits == 1 and b["cached"] and l.used() == 1

def test_budget_stops_and_ledger_persists(tmp_path):
    l = mk(tmp_path, text_reply("hi"), budget=2)
    l.chat("s", M("1")); l.chat("s", M("2"))
    with pytest.raises(BudgetReached): l.chat("s", M("3"))
    assert mk(tmp_path, text_reply("hi"), budget=2, tag="b").used() == 2   # new process, same ledger

def test_openai_conversion():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "1", "name": "f", "input": {"a": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "r"}]}]
    out, tools = _to_openai("sys", msgs, [{"name": "f", "input_schema": {"type": "object"}}])
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}' and out[3]["role"] == "tool" and tools[0]["function"]["name"] == "f"

def test_findings_without_evidence_dropped(tmp_path):
    tr = [{"tool": "a"}, {"tool": "b"}]
    bad = json.dumps({"fail": True, "category": "math", "confidence": .9, "evidence_steps": [], "reason": "r"})
    good = json.dumps({"fail": True, "category": "math", "confidence": .9, "evidence_steps": [1, 99], "reason": "r"})
    f, flag = LLMReviewer(mk(tmp_path, text_reply(bad)), "trace").review(Task(1, "t", ()), tr, False, ["math"], 0, task_text="q")
    assert f == [] and flag
    f, _ = LLMReviewer(mk(tmp_path, text_reply(good), tag="g"), "trace").review(Task(1, "t", ()), tr, False, ["math"], 0, task_text="q")
    assert len(f) == 1 and "[1]" in f[0].evidence and "99" not in f[0].evidence

def test_agent_records_trace(tmp_path):
    seq = iter([{"content": [{"type": "tool_use", "id": "1", "name": "echo", "input": {"x": 1}}], "tokens_in": 5, "tokens_out": 5},
                {"content": [{"type": "text", "text": "done"}], "tokens_in": 5, "tokens_out": 5}])
    l = mk(tmp_path, lambda *a: next(seq))
    tools = {"echo": ({"name": "echo", "description": "", "input_schema": {"type": "object"}}, lambda x: x)}
    task = Task(1, "t", ()); task.text = "go"
    ok, tr, tok, _ = run(task, Config(), l, tools, check=lambda t, tr: True)
    assert ok and [s["tool"] for s in tr] == ["echo", "answer"] and tr[0]["args"] == {"x": 1} and tok == 20

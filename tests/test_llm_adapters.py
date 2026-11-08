import json
import pytest
from traceguard.llm_client import LLM, SpendCapReached
from traceguard.llm_reviewer import LLMReviewer
from traceguard.llm_agent import run
from traceguard.agent import Config, Task


def mk(tmp, reply, cap=15.0, transport=None):
    json.dump({"m": {"in": 1.0, "out": 1.0}}, open(tmp / "p.json", "w"))
    t = transport or (lambda body: {"content": [{"type": "text", "text": reply}],
                                    "usage": {"input_tokens": 1000, "output_tokens": 1000}})
    return LLM(cap, tmp / "c", tmp / "p.json", t, tmp / "s.json")


MSG = lambda s: [{"role": "user", "content": s}]


def test_cache_and_cost(tmp_path):
    l = mk(tmp_path, "hi")
    l.chat("m", "s", MSG("x"))
    b = l.chat("m", "s", MSG("x"))
    assert l.calls == 1 and l.hits == 1 and b["cached"] and abs(l.spent - .002) < 1e-9


def test_spend_cap_stops(tmp_path):
    l = mk(tmp_path, "hi", cap=.003)
    l.chat("m", "s", MSG("1"))
    l.chat("m", "s", MSG("2"))
    with pytest.raises(SpendCapReached):
        l.chat("m", "s", MSG("3"))


def test_missing_price_refuses(tmp_path):
    with pytest.raises(RuntimeError):
        mk(tmp_path, "hi").chat("other", "s", MSG("x"))


def test_findings_without_evidence_dropped(tmp_path):
    tr = [{"tool": "a"}, {"tool": "b"}]
    bad = json.dumps({"fail": True, "category": "math", "confidence": .9, "evidence_steps": [], "reason": "r"})
    good = json.dumps({"fail": True, "category": "math", "confidence": .9, "evidence_steps": [1, 99], "reason": "r"})
    (tmp_path / "a").mkdir()
    f, flag = LLMReviewer(mk(tmp_path / "a", bad), "m").review(Task(1, "t", ()), tr, False, ["math"], 0, task_text="q")
    assert f == [] and flag
    f, _ = LLMReviewer(mk(tmp_path, good), "m").review(Task(1, "t", ()), tr, False, ["math"], 0, task_text="q")
    assert len(f) == 1 and "[1]" in f[0].evidence and "99" not in f[0].evidence


def test_agent_records_trace(tmp_path):
    seq = iter([{"content": [{"type": "tool_use", "id": "1", "name": "echo", "input": {"x": 1}}],
                 "usage": {"input_tokens": 5, "output_tokens": 5}},
                {"content": [{"type": "text", "text": "done"}], "usage": {"input_tokens": 5, "output_tokens": 5}}])
    l = mk(tmp_path, "", transport=lambda b: next(seq))
    tools = {"echo": ({"name": "echo", "description": "", "input_schema": {"type": "object"}}, lambda x: x)}
    task = Task(1, "t", ())
    task.text = "go"
    ok, tr, tok, _ = run(task, Config(), l, "m", tools, check=lambda t, tr: True)
    assert ok and [s["tool"] for s in tr] == ["echo", "answer"] and tr[0]["args"] == {"x": 1} and tok == 20

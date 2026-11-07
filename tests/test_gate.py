from traceguard.agent import Config, make_tasks, evaluate
from traceguard.loop import gate, candidates, unit_safety_ok
from traceguard.reviewers import Finding

def cand(apply, conf=.9): return dict(name="t", cat="x", apply=apply, support=10, conf=conf)
hist, held = make_tasks(400, 1), make_tasks(400, 2)

def test_beneficial_change_accepted():
    ok, _ = gate(Config(), cand(lambda c: c.evolve(verify_math=True)), hist, held); assert ok

def test_harmful_change_rejected():
    cur = Config(retrieval_k=4, max_retries=2)
    ok, _ = gate(cur, cand(lambda c: c.evolve(retrieval_k=1, max_retries=1)), hist, held); assert not ok

def test_guardrail_removal_blocked_by_safety_unit_test():
    ok, why = unit_safety_ok(Config(guardrail=True), Config(guardrail=False)); assert not ok and "guardrail" in why

def test_low_confidence_rejected():
    ok, _ = gate(Config(), cand(lambda c: c.evolve(verify_math=True), conf=.3), hist, held); assert not ok

def test_candidates_need_support():
    fs = [Finding(i, "math", .8, True, "") for i in range(3)]
    assert candidates(fs, Config()) == []

def test_same_config_same_outcome():
    assert evaluate(hist, Config()) == evaluate(hist, Config())

from traceguard.loop2 import propose, judge, TAU
from traceguard.pilot import START
from traceguard.reviewers import Finding

CAL = {"coef": 1.0, "intercept": 0.0}
def f(cat, c=.8): return (1, Finding(1, cat, c, True, ""))

def test_propose_orders_by_support_and_skips_noop_and_rejected():
    c = propose([f("math"), f("math"), f("workflow"), f("wasteful")], START, CAL)
    assert [x["name"] for x in c] == ["calculator on", "planning step on"]
    assert propose([f("math")], START, CAL, rejected=["calculator on"]) == []

def test_gate_requires_two_fixes_and_no_breaks():
    run = lambda *oks: [dict(id=i, ok=o) for i, o in enumerate(oks)]
    cand = dict(conf=.9)
    assert judge(run(0, 0, 1), run(1, 1, 1), cand, "gated")[0]
    assert not judge(run(0, 0, 1), run(1, 0, 1), cand, "gated")[0]          # only 1 fix
    assert not judge(run(0, 0, 1), run(1, 1, 0), cand, "gated")[0]          # breaks one
    assert not judge(run(0, 0, 1), run(1, 1, 1), dict(conf=TAU - .1), "gated")[0]
    assert judge(run(0, 0, 1), run(0, 0, 0), cand, "ungated")[0]

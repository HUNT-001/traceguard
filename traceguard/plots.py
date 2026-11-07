import json, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from .experiments import run_arm, ITERS
R = json.load(open("results/results.json")); ARMS = [a for a in R if a != "_meta"]
COL = {"Baseline (no loop)": "#7a8088", "Output-only, ungated": "#e09a00", "Trace-based, ungated": "#c4452b",
       "Output-only, gated": "#6f8fd8", "Trace-based, gated": "#1f3fc8", "Trace x3 panel, gated": "#0f8a6a"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

def curves():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for a in ARMS: ax.plot(range(ITERS + 1), [100 * x for x in R[a]["curve_mean"]], marker="o", ms=4, lw=2.4 if "gated" in a and "ungated" not in a else 1.6, color=COL[a], label=a)
    ax.set_xlabel("Improvement cycle"); ax.set_ylabel("True task success (%)"); ax.set_ylim(40, 95); ax.grid(alpha=.25)
    ax.legend(fontsize=9, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig("results/fig_success.png", dpi=170)

def safety():
    names = [a for a in ARMS if a != "Baseline (no loop)"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, k, t in ((axs[0], "harm_false_accept", "Harmful changes accepted (%)"), (axs[1], "regression_rate", "Cycles with a regression (%)")):
        ax.barh(range(len(names)), [100 * R[a][k] for a in names], color=[COL[a] for a in names]); ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names if ax is axs[0] else [], fontsize=9); ax.invert_yaxis(); ax.set_title(t, fontsize=11)
        for i, a in enumerate(names): ax.text(100 * R[a][k] + 1, i, f"{100*R[a][k]:.0f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig("results/fig_safety.png", dpi=170)

def reviewers():
    names = ["Output-only, gated", "Trace-based, gated", "Trace x3 panel, gated"]; lab = ["Output-only", "Trace-based", "Trace x3 panel"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); w = .25
    for j, (k, t) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("ece", "Calibration error (lower better)"))):
        ax.bar([i + (j - 1) * w for i in range(3)], [R[a][k] for a in names], w, label=t, color=["#1f3fc8", "#e09a00", "#7a8088"][j])
        for i, a in enumerate(names): ax.text(i + (j - 1) * w, R[a][k] + .015, f"{R[a][k]:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab); ax.set_ylim(0, 1.1); ax.legend(frameon=False, fontsize=9, loc="upper left"); fig.tight_layout(); fig.savefig("results/fig_reviewers.png", dpi=170)

def gif(seed=0):
    lu, lg = [], []
    cu, _, _ = run_arm("Trace-based, ungated", seed, log=lu); cg, _, _ = run_arm("Trace-based, gated", seed, log=lg)
    fig = plt.figure(figsize=(11, 4.8)); ax = fig.add_axes([.06, .14, .38, .74]); tx = fig.add_axes([.47, .04, .52, .9]); tx.axis("off")
    def draw(i):
        ax.clear(); tx.clear(); tx.axis("off")
        ax.plot(range(i + 1), [100 * x for x in cu[:i + 1]], marker="o", color=COL["Trace-based, ungated"], label="Ungated (apply every suggestion)")
        ax.plot(range(i + 1), [100 * x for x in cg[:i + 1]], marker="o", color=COL["Trace-based, gated"], lw=2.6, label="TraceGuard (validation gate)")
        ax.set_xlim(0, ITERS); ax.set_ylim(40, 95); ax.set_xlabel("Improvement cycle"); ax.set_ylabel("True task success (%)"); ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.set_title(f"Cycle {i}", fontsize=12, loc="left")
        y = .96
        for title, lg_, c in (("Ungated", lu, "#c4452b"), ("TraceGuard gate decisions", lg, "#1f3fc8")):
            tx.text(0, y, title, color=c, fontsize=11, weight="bold", va="top"); y -= .07
            for (it, name, ok, why, cls, tr) in [l for l in lg_ if l[0] == i][:5]:
                tag = ("APPLIED" if ok else "REJECTED")
                good = (ok and cls != "harmful") or ((not ok) and cls == "harmful")
                col = "#0f8a6a" if good else "#c4452b"
                tx.text(.02, y, f"{tag:9s} {name}  [{cls}]", fontsize=9, family="monospace", color=col, va="top")
                tx.text(.02, y - .035, f"          {why[:62]}", fontsize=7.5, family="monospace", color="#555", va="top"); y -= .082
            y -= .03
    anim = FuncAnimation(fig, draw, frames=[ITERS] + list(range(ITERS + 1)), interval=1400)
    anim.save("results/demo.gif", writer=PillowWriter(fps=.7), dpi=90)

if __name__ == "__main__": curves(); safety(); reviewers(); gif()

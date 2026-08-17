# -*- coding: utf-8 -*-
"""Cycle C figure: (a) tuning-failure matrix across the sweep,
(b) Anderson x AIMD composition amplitude curves."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300})

sw = np.load(DATA / "cc_sweep.npz", allow_pickle=True)
an = np.load(DATA / "cc_anderson.npz", allow_pickle=True)
mm = np.load(DATA / "cc_mma.npz", allow_pickle=True)
mv = np.load(DATA / "ce_movelimit.npz", allow_pickle=True)
groups = list(sw["groups"])
methods = list(sw["methods"]) + ["mma", "mv0.5", "mv1.0"]


def cell(m, g):
    """(comp, osc, masked) for method m on group g."""
    if m == "mma":
        comp, tail = mm[f"{g}_meta"]
        return comp, tail > 3e-3, False        # mu not instrumented
    if m.startswith("mv"):
        comp, tail, osc, dq, mv_end, mu, masked = mv[f"{m}_{g}"]
        return comp, osc > 0, masked > 0
    comp, tail, osc, dq, nmd, eta, mu = sw[f"{g}_{m}_meta"]
    masked = (osc == 0 and np.isfinite(mu) and mu <= -0.9
              and 1e-5 < tail < 3e-3)
    return comp, osc > 0, masked

fig = plt.figure(figsize=(10.2, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

# (a) pass/fail matrix -------------------------------------------------------
a = fig.add_subplot(gs[0, 0])
C_OK, C_Q, C_OSC, C_MSK = "#3A7D44", "#E0A526", "#B4231F", "#6A4C93"
best = {g: min(sw[f"{g}_{m}_meta"][0] for m in sw["methods"])
        for g in groups}
M = np.zeros((len(methods), len(groups), 3))
for i, m in enumerate(methods):
    for j, g in enumerate(groups):
        comp, osc, masked = cell(m, g)
        dq = 100 * (comp / best[g] - 1)
        col = C_OSC if osc else (C_MSK if masked else
                                 (C_Q if dq > 2.0 else C_OK))
        M[i, j] = matplotlib.colors.to_rgb(col)
a.imshow(M, aspect="auto")
a.set_yticks(range(len(methods)))
a.set_yticklabels([m.replace("fix", "fixed ").replace("aimd", "AIMD ")
                   .replace("ladder", "ladder 1.0").replace("mma", "MMA")
                   .replace("mv", "adapt. move ")
                   for m in methods], fontsize=7.5)
a.set_xticks(range(len(groups)))
a.set_xticklabels([g.replace("mbb_", "").replace("cant_", "cant ")
                   .replace("mbb120_", "120x40 ").replace("_p", " p")
                   for g in groups], rotation=45, ha="right", fontsize=7)
for sp in a.spines.values():
    sp.set_visible(False)
a.set_xticks(np.arange(-0.5, len(groups)), minor=True)
a.set_yticks(np.arange(-0.5, len(methods)), minor=True)
a.grid(which="minor", color="white", lw=1.5)
a.tick_params(which="both", length=0)
a.legend(handles=[Patch(color=C_OK, label="converged, within 2% of best"),
                  Patch(color=C_Q, label=">2% worse (quality fail)"),
                  Patch(color=C_OSC, label="oscillating at budget"),
                  Patch(color=C_MSK,
                        label=r"masked oscillation ($\hat\mu\approx-1$)")],
         fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.40),
         ncol=2)
a.set_title("(a) tuning-failure matrix, 400-iteration budget", fontsize=9)

# (b) Anderson composition ---------------------------------------------------
a = fig.add_subplot(gs[0, 1])
series = [("fix0.5", "#2B6A99", "-", r"fixed $\eta=0.5$"),
          ("aa_fix0.5", "#2B6A99", "--", r"AA + fixed $0.5$"),
          ("fix1.0", "#666666", "-", r"fixed $\eta=1.0$"),
          ("aa_fix1.0", "#666666", "--", r"AA + fixed $1.0$"),
          ("aa_aimd0.5", "#B4231F", "-", r"AA + AIMD($0.5$)"),
          ("aa_aimd1.0", "#B4231F", "--", r"AA + AIMD($1.0$)")]
for k, col, ls, lab in series:
    a.semilogy(np.maximum(an[f"{k}_amp"], 1e-12), color=col, ls=ls, lw=1.0,
               label=lab)
a.set_xlabel("iteration")
a.set_ylabel(r"step amplitude $\overline{|\Delta x|}$")
a.set_ylim(1e-12, 0.5)
a.legend(fontsize=6.8, loc="lower left", ncol=2, columnspacing=1.0)
a.set_title("(b) Anderson composition ($r_{\\min}2.4$, $p=3$)", fontsize=9)

fig.savefig(FIGS / "fig_cycleC.png", bbox_inches="tight")
print("saved figs/fig_cycleC.png")

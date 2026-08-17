# -*- coding: utf-8 -*-
"""Cycle B figure: AIMD damping trajectories, rescue of the over-driven
run, and compliance. Reads data/cb_aimd.npz (400-iteration runs)."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300})
C_R, C_B, C_G, C_ACC = "#B4231F", "#2B6A99", "#3A7D44", "#666666"

d = np.load(DATA / "cb_aimd.npz", allow_pickle=True)
fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.9))

# (a) eta trajectories of all AIMD runs -------------------------------------
a = ax[0]
for cfg, ls in (("P1_r1.1", "-"), ("P2_r2.4", "--")):
    for e0, col in ((0.3, C_B), (0.5, C_G), (1.0, C_R)):
        a.plot(d[f"{cfg}_aimd{e0}_eta"], ls=ls, color=col, lw=1.1,
               label=fr"$\eta_0={e0}$" if cfg == "P1_r1.1" else None)
a.axhline(0.503, color=C_ACC, lw=0.7, ls=":")
a.text(395, 0.503, r"$2/s_{\max}$ (r1.1)", fontsize=7, color=C_ACC,
       ha="right", va="bottom")
a.axhline(0.60, color=C_ACC, lw=0.7, ls=":")
a.text(395, 0.60, r"$\approx\eta^{\star}$ (r2.4)", fontsize=7, color=C_ACC,
       ha="right", va="bottom")
a.set_xlabel("iteration")
a.set_ylabel(r"damping $\eta_t$")
a.set_ylim(0.2, 1.22)
a.legend(fontsize=7.5, loc="upper right", title="AIMD starts",
         title_fontsize=7.5)
a.text(0.02, 1.02, "(a)", transform=a.transAxes, fontweight="bold")
a.text(0.5, -0.32, r"solid: $r_{\min}=1.1$,  dashed: $r_{\min}=2.4$",
       transform=a.transAxes, fontsize=7.5, ha="center", color=C_ACC)

# (b) rescue of the over-driven run (r2.4) ----------------------------------
a = ax[1]
a.semilogy(d["P2_r2.4_fixed1.0_amp"], color=C_ACC, lw=1.0,
           label=r"fixed $\eta=1.0$")
a.semilogy(d["P2_r2.4_fixed0.5_amp"], color=C_B, lw=1.0,
           label=r"fixed $\eta=0.5$")
a.semilogy(d["P2_r2.4_aimd1.0_amp"], color=C_R, lw=1.1,
           label=r"AIMD from $\eta_0=1.0$")
a.set_xlabel("iteration")
a.set_ylabel(r"step amplitude $\overline{|\Delta x|}$")
a.legend(fontsize=7.5, loc="lower left")
a.text(0.02, 1.02, "(b)", transform=a.transAxes, fontweight="bold")

# (c) compliance trajectories (r2.4) ---------------------------------------
a = ax[2]
a.plot(d["P2_r2.4_fixed1.0_comp"], color=C_ACC, lw=1.0,
       label=r"fixed $\eta=1.0$")
a.plot(d["P2_r2.4_fixed0.5_comp"], color=C_B, lw=1.0,
       label=r"fixed $\eta=0.5$")
a.plot(d["P2_r2.4_aimd1.0_comp"], color=C_R, lw=1.1,
       label=r"AIMD from $\eta_0=1.0$")
a.set_xlabel("iteration")
a.set_ylabel("compliance")
a.set_ylim(180, 400)
a.legend(fontsize=7.5, loc="upper right")
a.text(0.02, 1.02, "(c)", transform=a.transAxes, fontweight="bold")

fig.tight_layout()
fig.savefig(FIGS / "fig_cycleB.png", bbox_inches="tight")
print("saved figs/fig_cycleB.png")

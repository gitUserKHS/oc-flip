# -*- coding: utf-8 -*-
"""Step-1 figure: p-continuation. (a) controller tracking the falling
threshold 2/(p(t)+1) [cover-figure candidate]; (b) fast continuation breaks
fixed 0.5 on the cantilever; (c) endpoint s_max: ceiling vs natural family."""
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

d = np.load(DATA / "cd_cont.npz", allow_pickle=True)
fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.0))

# (a) threshold tracking on R0, fast schedule ------------------------------
a = ax[0]
p = d["mbb_r1.1_fast_aimd1.0_p"]
t = np.arange(p.size)
a.plot(t, 2 / (p + 1), "k--", lw=1.3, label=r"theory $2/(p(t)+1)$")
a.plot(t, d["mbb_r1.1_fast_aimd1.0_eta"], color=C_R, lw=1.1,
       label=r"AIMD from $\eta_0=1.0$")
a.plot(t, d["mbb_r1.1_fast_aimd0.5_eta"], color=C_B, lw=1.1,
       label=r"AIMD from $\eta_0=0.5$")
a.axhline(0.5, color=C_ACC, lw=0.8, ls=":", label=r"fixed $\eta=0.5$")
a2 = a.twinx()
a2.plot(t, p, color=C_ACC, lw=0.7, alpha=0.5)
a2.set_ylabel(r"penalization $p(t)$", color=C_ACC, fontsize=8)
a2.tick_params(axis="y", labelsize=7, colors=C_ACC)
a2.spines["right"].set_visible(True)
a2.spines["right"].set_color(C_ACC)
a.set_xlabel("iteration")
a.set_ylabel(r"damping $\eta_t$")
a.set_ylim(0.15, 1.05)
a.legend(fontsize=7, loc="upper right")
a.text(0.02, 1.02, "(a)", transform=a.transAxes, fontweight="bold")
a.text(0.98, 0.03, r"MBB $r_{\min}1.1$, fast", transform=a.transAxes,
       fontsize=7.5, ha="right", color=C_ACC)

# (b) fast continuation breaks the folklore on the cantilever ---------------
a = ax[1]
a.semilogy(d["cant_r1.1_fast_fix0.5_amp"], color=C_ACC, lw=1.0,
           label=r"fixed $\eta=0.5$ (oscillating)")
a.semilogy(d["cant_r1.1_fast_fix0.3_amp"], color=C_B, lw=1.0,
           label=r"fixed $\eta=0.3$")
a.semilogy(d["cant_r1.1_fast_aimd1.0_amp"], color=C_R, lw=1.1,
           label=r"AIMD from $\eta_0=1.0$")
re = int(d["cant_r1.1_fast_aimd1.0_meta"][8])
a.axvline(re, color="k", lw=0.7, ls=":")
a.text(re + 8, 2e-1, "ramp ends\n($p=5$)", fontsize=7, va="top")
a.set_xlabel("iteration")
a.set_ylabel(r"step amplitude $\overline{|\Delta x|}$")
a.legend(fontsize=7, loc="lower left")
a.text(0.02, 1.02, "(b)", transform=a.transAxes, fontweight="bold")
a.text(0.98, 0.97, r"cantilever $r_{\min}1.1$, fast", transform=a.transAxes,
       fontsize=7.5, ha="right", va="top", color=C_ACC)

# (c) endpoint s_max: ceiling family vs natural family ----------------------
a = ax[2]
methods = ["fix0.3", "fix0.5", "fix0.7", "ladder", "aimd0.5", "aimd1.0"]
labels = ["fixed\n0.3", "fixed\n0.5", "fixed\n0.7", "ladder",
          "AIMD\n0.5", "AIMD\n1.0"]
xs = np.arange(len(methods))
for dx, sched, alpha in ((-0.18, "fast", 0.95), (0.18, "slow", 0.55)):
    vals = [d[f"mbb_r1.1_{sched}_{m}_meta"][6] for m in methods]
    oscs = [d[f"mbb_r1.1_{sched}_{m}_meta"][2] > 0 for m in methods]
    cols = [C_ACC if o else (C_R if m.startswith("aimd") else C_B)
            for m, o in zip(methods, oscs)]
    a.bar(xs + dx, vals, width=0.34, color=cols, alpha=alpha,
          label=f"{sched} schedule" if dx < 0 else None)
a.axhline(6.0, color="k", lw=0.9, ls="--")
a.text(5.45, 6.05, r"$p+1=6$", fontsize=7.5, ha="right")
a.axhline(4.0, color=C_ACC, lw=0.9, ls=":")
a.text(5.45, 4.07, r"ceiling $2/0.5=4$", fontsize=7.5, ha="right",
       color=C_ACC)
a.set_xticks(xs)
a.set_xticklabels(labels, fontsize=7)
a.set_ylabel(r"endpoint $s_{\max}$ (MBB $r_{\min}1.1$)")
a.set_ylim(0, 6.8)
a.text(0.02, 1.02, "(c)", transform=a.transAxes, fontweight="bold")
a.text(0.5, -0.30, "solid: fast, faded: slow; gray bars: oscillating "
       "endpoint", transform=a.transAxes, fontsize=7, ha="center",
       color=C_ACC)

fig.tight_layout()
fig.savefig(FIGS / "fig_cont.png", bbox_inches="tight")
print("saved figs/fig_cont.png")

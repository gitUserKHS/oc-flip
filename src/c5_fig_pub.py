# -*- coding: utf-8 -*-
"""Publication figures from cycle-5 data (backlog 3, partial):
figs/fig_negbranch.png  -- paper Fig: negative branch (pitchfork + growth)
figs/fig_psweep_pub.png -- paper Fig: p-sweep (saturation, thresholds,
                           rate law, filter-radius detuning)
Unified style: serif + STIX math, shared palette, 300 dpi.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300,
})
C_REF, C_AD, C_ACC = "#B4231F", "#2B6A99", "#666666"

nb = np.load(DATA / "c5_negbranch.npz")
ad = np.load(DATA / "c5_psweep.npz")
rf = np.load(DATA / "c5_psweep_ref.npz")
r5 = np.load(DATA / "c5_p5rmin_ref.npz")
r3 = np.load(DATA / "c5_p3rmin_ref.npz")
P = list(ad["ps"])


def letter(ax, s):
    ax.text(0.02, 1.02, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


# ---------------------------------------------------------------- Fig: neg
fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7))

a = ax[0]
tx, te = nb["track_x"], nb["track_el"]
t = np.arange(tx.shape[0])
pair_cols = ["#B4231F", "#2B6A99", "#3A7D44"]
for i in range(0, 6, 2):
    c = pair_cols[i // 2]
    a.semilogy(t, tx[:, i], color=c, lw=1.2)
    a.semilogy(t, tx[:, i + 1], color=c, lw=1.2, ls="--")
a.set_xlim(0, 1200)
a.set_xlabel("iteration after kick")
a.set_ylabel(r"density $x_e$")
a.annotate("winners", xy=(700, 0.036), fontsize=8, color=C_ACC)
a.annotate("losers", xy=(700, 0.0032), fontsize=8, color=C_ACC)
letter(a, "(a)")

a = ax[1]
b = np.abs(nb["b"])
mu = float(nb["mu_pred"])
a.semilogy(np.arange(b.size), np.maximum(b, 1e-12), color=C_REF, lw=1.0,
           label=r"$|b_t|$")
tl = np.arange(0, 260)
a.semilogy(tl, b[2] * mu**(tl - 2), color="k", ls="--", lw=1.0,
           label=fr"$\mu=1-\eta s_{{\min}}={mu:.4f}$")
a.set_xlim(0, 3000)
a.set_ylim(1e-4, 30)
a.set_xlabel("iteration after kick")
a.set_ylabel(r"mode component $|b_t|$")
a.legend(fontsize=8, loc="center right", handlelength=1.6)
a.annotate(r"$s_{\min}$: $-0.0431 \rightarrow +0.0375$ post-run",
           xy=(0.30, 0.04), xycoords="axes fraction", fontsize=8,
           color=C_ACC)
letter(a, "(b)")

fig.tight_layout()
fig.savefig(FIGS / "fig_negbranch.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- Fig: psweep
fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.4))

a = ax[0, 0]
smax_r = [float(rf[f"p{p:.1f}_smax"]) for p in P]
smax_a = [float(ad[f"p{p:.1f}_smax"]) for p in P]
a.plot(P, [p + 1 for p in P], color="k", ls="--", lw=1.0,
       label=r"$s_{\max}=p+1$")
a.axhline(4.0, color=C_ACC, lw=0.9, ls=":",
          label=r"ceiling $2/\eta_0=4$")
a.plot(P, smax_r, "o-", color=C_REF, ms=4.5,
       label=r"reference ($\eta_0=0.3$)")
a.plot(P, smax_a, "s-", color=C_AD, ms=4.5, mfc="none",
       label=r"adapted ($\eta_0=0.5$)")
a.set_xlabel(r"penalization $p$")
a.set_ylabel(r"$s_{\max}$")
a.legend(fontsize=7.5, loc="upper left")
letter(a, "(a)")

a = ax[0, 1]
pp = np.linspace(min(P), max(P), 200)
a.plot(pp, 2 / (pp + 1), color="k", ls="--", lw=1.0, label=r"$2/(p+1)$")
for src, col, m, dp, lab in [(rf, C_REF, "o", 0.0, "reference"),
                             (ad, C_AD, "s", 0.06, "adapted")]:
    sm = [float(src[f"p{p:.1f}_smax"]) for p in P]
    a.plot([p + dp for p in P], [2 / s for s in sm], m, color=col, ms=4,
           mfc="none" if col == C_AD else col,
           label=fr"$2/s_{{\max}}$ ({lab})")
    for p in P:
        lo, hi = src[f"p{p:.1f}_bracket"]
        if np.isfinite(lo) and np.isfinite(hi):
            a.plot([p + dp, p + dp], [lo, hi], "-", color=col, lw=2.6,
                   alpha=0.35, solid_capstyle="butt")
a.plot([], [], "-", color=C_ACC, lw=2.6, alpha=0.5, label="onset brackets")
a.set_xlabel(r"penalization $p$")
a.set_ylabel(r"damping $\eta$")
a.legend(fontsize=7.5)
letter(a, "(b)")

a = ax[1, 0]
xs, ys = [], []
for src in (rf, ad):
    for p in P:
        sm = float(src[f"p{p:.1f}_smax"])
        for e, g, rs in src[f"p{p:.1f}_probe"]:
            pred = abs(1 - e * sm)
            if rs < 0 and pred <= 1.25:
                xs.append(pred), ys.append(g)
xs, ys = np.array(xs), np.array(ys)
a.plot([0, 1.3], [0, 1.3], color="k", ls="--", lw=0.9)
a.axvline(1.0, color=C_ACC, lw=0.6, ls=":")
a.axhline(1.0, color=C_ACC, lw=0.6, ls=":")
a.plot(xs, ys, "o", ms=4, color="#6A4C93", alpha=0.8, mec="none")
a.set_xlabel(r"predicted $|1-\eta\, s_{\max}|$")
a.set_ylabel(r"measured $|r|$")
a.annotate(f"{xs.size} probes\nmedian dev. "
           f"{np.median(np.abs(xs-ys)):.3f}",
           xy=(0.05, 0.78), xycoords="axes fraction", fontsize=8,
           color=C_ACC)
letter(a, "(c)")

a = ax[1, 1]
rm = [1.1] + [float(r) for r in r5["rmins"]]
s5 = [float(rf["p5.0_smax"])] + [float(r5[f"r{r:.1f}_smax"])
                                 for r in r5["rmins"]]
s3 = [float(rf["p3.0_smax"])] + [float(r3[f"r{r:.1f}_smax"])
                                 for r in r3["rmins"]]
a.plot(rm, s5, "o-", color=C_REF, ms=4.5, label=r"$p=5$")
a.plot(rm, s3, "s-", color=C_AD, ms=4.5, label=r"$p=3$")
a.axhline(6.0, color=C_REF, lw=0.8, ls=":", alpha=0.6)
a.axhline(4.0, color=C_AD, lw=0.8, ls=":", alpha=0.6)
a.text(2.38, 6.06, r"$p+1=6$", fontsize=7.5, color=C_REF, ha="right")
a.text(2.38, 4.06, r"$p+1=4$", fontsize=7.5, color=C_AD, ha="right")
a.set_xlabel(r"sensitivity-filter radius $r_{\min}$")
a.set_ylabel(r"$s_{\max}$")
a.legend(fontsize=8)
letter(a, "(d)")

fig.tight_layout()
fig.savefig(FIGS / "fig_psweep_pub.png", bbox_inches="tight")
print("saved fig_negbranch.png, fig_psweep_pub.png")

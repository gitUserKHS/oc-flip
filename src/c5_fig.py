# -*- coding: utf-8 -*-
"""Cycle 5 figure (2x3):
(a) mirror-pair pitchfork of the negative branch   (b) mode growth vs prediction
(c) p+1 saturation, two protocols                  (d) threshold map + brackets
(e) predicted vs measured decay/growth rates       (f) filter-radius detuning
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"

nb = np.load(DATA / "c5_negbranch.npz")
ad = np.load(DATA / "c5_psweep.npz")        # adapted protocol (eta0 ladder)
rf = np.load(DATA / "c5_psweep_ref.npz")    # reference protocol (eta0=0.3)
r5 = np.load(DATA / "c5_p5rmin_ref.npz")
r3 = np.load(DATA / "c5_p3rmin_ref.npz")

P = list(ad["ps"])
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

# (a) mirror-pair pitchfork --------------------------------------------------
a = ax[0, 0]
tx, te = nb["track_x"], nb["track_el"]
t = np.arange(tx.shape[0])
cols = ["tab:red", "tab:blue", "tab:green"]
for i in range(0, 6, 2):
    c = cols[i // 2]
    a.semilogy(t, tx[:, i], color=c, lw=1.4,
               label=f"el {te[i]}/{te[i+1]}" if i == 0 else None)
    a.semilogy(t, tx[:, i + 1], color=c, lw=1.4, ls="--")
a.set_xlim(0, 1200)
a.set_xlabel("iteration after kick")
a.set_ylabel("density $x_e$")
a.set_title("(a) V3 negative branch: mirror-pair\n"
            "symmetry breaking (solid vs dashed = pair)")

# (b) mode component growth --------------------------------------------------
a = ax[0, 1]
b = np.abs(nb["b"])
mu = float(nb["mu_pred"])
a.semilogy(np.arange(b.size), np.maximum(b, 1e-12), lw=1.0, color="tab:red",
           label=r"$|b_t|$ (mode projection)")
tl = np.arange(0, 260)
a.semilogy(tl, b[2] * mu**(tl - 2), "k--", lw=1.2,
           label=fr"predicted $\mu = 1-\eta s_{{\min}} = {mu:.4f}$")
a.set_xlim(0, 3000)
a.set_ylim(1e-4, 30)
a.set_xlabel("iteration after kick")
a.set_ylabel(r"$|b_t|$")
a.set_title("(b) growth at the predicted rate, then\n"
            f"resettling: $s_\\min$ {float(nb['s_min'].real):+.4f} "
            f"$\\to$ {float(np.min(nb['s_after'].real)):+.4f} post-run")
a.legend(fontsize=8, loc="lower right")

# (c) saturation under two protocols ----------------------------------------
a = ax[0, 2]
smax_r = [float(rf[f"p{p:.1f}_smax"]) for p in P]
smax_a = [float(ad[f"p{p:.1f}_smax"]) for p in P]
a.plot(P, [p + 1 for p in P], "k--", lw=1, label=r"$s_{\max}=p+1$")
a.axhline(4.0, color="gray", lw=1, ls=":",
          label=r"capacity ceiling $2/\eta_0=4$ ($\eta_0=0.5$)")
a.plot(P, smax_r, "o-", color="tab:red",
       label=r"reference protocol ($\eta_0=0.3$)")
a.plot(P, smax_a, "s-", mfc="none", color="tab:blue",
       label=r"adapted protocol ($\eta_0=0.5$ ladder)")
a.set_xlabel("SIMP penalty $p$")
a.set_ylabel(r"$s_{\max}$")
a.set_title("(c) $p+1$ saturation vs dynamic self-limiting:\n"
            "protocols split exactly at $p+1=2/\\eta_0$, i.e. $p=3$")
a.legend(fontsize=7.5)

# (d) threshold map ---------------------------------------------------------
a = ax[1, 0]
pp = np.linspace(min(P), max(P), 200)
a.plot(pp, 2 / (pp + 1), "k--", lw=1, label=r"$2/(p+1)$")
for src, col, m, lab in [(rf, "tab:red", "o", "reference"),
                         (ad, "tab:blue", "s", "adapted")]:
    dp = 0.0 if col == "tab:red" else 0.06
    sm = [float(src[f"p{p:.1f}_smax"]) for p in P]
    a.plot([p + dp for p in P], [2 / s for s in sm], m, color=col, ms=4,
           mfc="none" if col == "tab:blue" else col,
           label=fr"$2/s_{{\max}}$ ({lab})")
    for p in P:
        lo, hi = src[f"p{p:.1f}_bracket"]
        if np.isfinite(lo) and np.isfinite(hi):
            a.plot([p + dp, p + dp], [lo, hi], "-", color=col, lw=3, alpha=0.4)
a.plot([], [], "-", color="gray", lw=3, alpha=0.5, label="onset brackets")
a.set_xlabel("SIMP penalty $p$")
a.set_ylabel(r"damping $\eta$")
a.set_title("(d) onset brackets follow each fixed point's own\n"
            r"$2/s_{\max}$ (both protocols; $p=5$: 0.335 vs 0.568)")
a.legend(fontsize=7.5)

# (e) predicted vs measured rates -------------------------------------------
a = ax[1, 1]
xs, ys = [], []
for src in (rf, ad):
    for p in P:
        sm = float(src[f"p{p:.1f}_smax"])
        for e, g, rs in src[f"p{p:.1f}_probe"]:
            pred = abs(1 - e * sm)
            if rs < 0 and pred <= 1.25:      # flip-mode-active probes only
                xs.append(pred), ys.append(g)
xs, ys = np.array(xs), np.array(ys)
a.plot([0, 1.3], [0, 1.3], "k--", lw=1)
a.axvline(1.0, color="gray", lw=0.6, ls=":")
a.axhline(1.0, color="gray", lw=0.6, ls=":")
a.plot(xs, ys, "o", ms=5, color="tab:purple", alpha=0.75)
err = np.median(np.abs(xs - ys))
a.set_xlabel(r"predicted $|1-\eta\, s_{\max}|$")
a.set_ylabel(r"measured $|r|$")
a.set_title(f"(e) rate law across sweep, both protocols\n"
            f"{xs.size} probes, median deviation {err:.3f}")

# (f) filter-radius detuning ------------------------------------------------
a = ax[1, 2]
rm = [1.1] + [float(r) for r in r5["rmins"]]
s5 = [float(rf["p5.0_smax"])] + [float(r5[f"r{r:.1f}_smax"])
                                 for r in r5["rmins"]]
s3 = [float(rf["p3.0_smax"])] + [float(r3[f"r{r:.1f}_smax"])
                                 for r in r3["rmins"]]
a.plot(rm, s5, "o-", color="tab:red", label="$p=5$ (ref. protocol)")
a.plot(rm, s3, "s-", color="tab:blue", label="$p=3$ (ref. protocol)")
a.axhline(6.0, color="tab:red", lw=1, ls=":", alpha=0.6)
a.axhline(4.0, color="tab:blue", lw=1, ls=":", alpha=0.6)
a.text(2.42, 6.05, "$p+1=6$", fontsize=8, color="tab:red", ha="right")
a.text(2.42, 4.05, "$p+1=4$", fontsize=8, color="tab:blue", ha="right")
a.set_xlabel(r"sensitivity-filter radius $r_{\min}$")
a.set_ylabel(r"$s_{\max}$")
a.set_title("(f) filter detunes the flip branch;\n"
            "sensitivity grows with $p$ (thin gray layer)")
a.legend(fontsize=8)

fig.suptitle("Cycle 5 - closure experiments: negative branch, saturation vs "
             "self-limiting, thresholds as fixed-point properties", y=0.995)
fig.tight_layout()
fig.savefig(FIGS / "fig_cycle5.png", dpi=150, bbox_inches="tight")
print("saved", FIGS / "fig_cycle5.png")

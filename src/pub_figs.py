# -*- coding: utf-8 -*-
"""Publication restyle of Figs 1-3 and 6 (backlog step-5).

Generates fig_toy_pub / fig_identity_pub / fig_spectra_pub /
fig_robust_pub in the shared paper style (serif + STIX, common palette).
Sources: two-bar algebra recomputed in-place (milliseconds); ca_sweep and
cb_aimd checkpoints; one R0 Jacobian rebuild at two eta; c4_V*.npz.

Usage: python src/pub_figs.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, build_J
import c5_core as c5

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 300})
C_R, C_B, C_G, C_ACC = "#B4231F", "#2B6A99", "#3A7D44", "#666666"


def letter(ax, s):
    ax.text(0.02, 1.03, s, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom")


# ---------------------------------------------------------------- two-bar
A2, V2 = np.array([1.0, 2.0]), 1.2


def series_step(x, eta, p, move=None):
    B = p * A2 * x**(-(p + 1))
    if move is None:
        y = x * B**eta
        return y * (V2 / y.sum())
    lo = np.maximum(1e-3, x - move)
    hi = np.minimum(1.0, x + move)
    l1, l2 = 1e-9, 1e9
    for _ in range(200):
        lm = np.sqrt(l1 * l2)
        y = np.clip(x * (B / lm)**eta, lo, hi)
        if y.sum() > V2:
            l1 = lm
        else:
            l2 = lm
        if l2 / l1 < 1 + 1e-12:
            break
    return y


def fig_toy():
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.2))
    # (a) schematic ---------------------------------------------------------
    a = ax[0, 0]
    a.axis("off")
    a.add_patch(Rectangle((0.08, 0.62), 0.16, 0.10, fc=C_B, alpha=0.7))
    a.add_patch(Rectangle((0.30, 0.62), 0.16, 0.10, fc=C_B, alpha=0.7))
    a.plot([0.02, 0.08], [0.67, 0.67], "k-", lw=1.2)
    a.plot([0.24, 0.30], [0.67, 0.67], "k-", lw=1.2)
    a.plot([0.46, 0.52], [0.67, 0.67], "k-", lw=1.2)
    a.add_patch(FancyArrow(0.52, 0.67, 0.09, 0, width=0.004,
                           head_width=0.03, color=C_R))
    a.text(0.30, 0.80, "series (determinate): no force redistribution",
           fontsize=8, ha="center")
    a.text(0.30, 0.53, r"$s = -\partial\ln D/\partial\ln x = p+1$",
           fontsize=9, ha="center", color=C_B)
    a.add_patch(Rectangle((0.66, 0.74), 0.16, 0.09, fc=C_G, alpha=0.7))
    a.add_patch(Rectangle((0.66, 0.55), 0.16, 0.09, fc=C_G, alpha=0.7))
    for y in (0.785, 0.595):
        a.plot([0.60, 0.66], [y, y], "k-", lw=1.2)
        a.plot([0.82, 0.88], [y, y], "k-", lw=1.2)
    a.plot([0.60, 0.60], [0.595, 0.785], "k-", lw=1.2)
    a.plot([0.88, 0.88], [0.595, 0.785], "k-", lw=1.2)
    a.add_patch(FancyArrow(0.88, 0.69, 0.08, 0, width=0.004,
                           head_width=0.03, color=C_R))
    a.text(0.77, 0.90, "parallel (redundant):\nforce sharing",
           fontsize=8, ha="center")
    a.text(0.77, 0.45, r"$s = -(p-1)$", fontsize=9, ha="center", color=C_G)
    a.text(0.5, 0.18,
           r"flip branch: stable iff $\eta < 2/s$;"
           "\n" r"winner-take-all branch: grows for every $\eta>0$",
           fontsize=8.5, ha="center", color=C_ACC)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    letter(a, "(a)")

    # (b) rate: prediction vs measurement -----------------------------------
    a = ax[0, 1]
    xs = np.array([A2[0], V2 - A2[0] * 0.4])  # any interior start
    for p, mk in ((1, "o"), (3, "s"), (5, "^")):
        xstar = A2**(1.0 / (p + 1))
        xstar = xstar * (V2 / xstar.sum())
        zs = np.log(xstar[0] / xstar[1])
        pred, meas = [], []
        for eta in np.arange(0.05, min(2.0 / (p + 1), 0.95), 0.05):
            x = xstar * np.array([1.1, 1.0])
            x = x * (V2 / x.sum())
            r = []
            for _ in range(30):
                xn = series_step(x, eta, p)
                z0 = np.log(x[0] / x[1]) - zs
                z1 = np.log(xn[0] / xn[1]) - zs
                if abs(z0) > 1e-13:
                    r.append(abs(z1 / z0))
                x = xn
            if len(r) < 6:
                continue
            pred.append(abs(1 - eta * (p + 1)))
            meas.append(np.median(r[3:]))
        a.plot(pred, meas, mk, ms=4, mfc="none",
               color=[C_B, C_R, C_G][(p - 1) // 2], label=f"$p={p}$")
    a.plot([0, 1], [0, 1], "k--", lw=0.8)
    a.set_xlabel(r"predicted $|\mu| = |1-\eta(p{+}1)|$")
    a.set_ylabel("measured contraction rate")
    a.legend(fontsize=8)
    letter(a, "(b)")

    # (c) bifurcation diagram at p=3 ----------------------------------------
    a = ax[1, 0]
    for eta in np.arange(0.05, 0.92, 0.01):
        x = np.array([0.55, 0.65])
        x = x * (V2 / x.sum())
        tail = []
        for t in range(500):
            x = series_step(x, eta, 3, move=0.05)
            if t >= 460:
                tail.append(x[0])
        a.plot([eta] * len(tail), tail, ".", ms=1.2, color=C_B, alpha=0.6)
    a.axvline(0.5, color=C_R, lw=1.0, ls="--")
    a.text(0.505, a.get_ylim()[1], r"$\eta^{\star}=2/(p{+}1)=0.5$",
           fontsize=7.5, color=C_R, va="top")
    a.set_xlabel(r"damping $\eta$")
    a.set_ylabel(r"asymptotic $x_1$")
    letter(a, "(c)")

    # (d) amplitude map -----------------------------------------------------
    a = ax[1, 1]
    etas = np.arange(0.05, 1.01, 0.05)
    ps = np.arange(1.0, 5.01, 0.25)
    M = np.zeros((ps.size, etas.size))
    for i, p in enumerate(ps):
        for j, eta in enumerate(etas):
            x = np.array([0.55, 0.65])
            x = x * (V2 / x.sum())
            amps = []
            for t in range(300):
                xn = series_step(x, eta, p, move=0.05)
                if t >= 260:
                    amps.append(abs(xn[0] - x[0]))
                x = xn
            M[i, j] = np.log10(max(np.median(amps), 1e-16))
    im = a.pcolormesh(etas, ps, M, cmap="magma", shading="nearest")
    pp = np.linspace(1.0, 5.0, 100)
    a.plot(2.0 / (pp + 1), pp, "w--", lw=1.3)
    a.text(0.62, 3.4, r"$\eta=2/(p{+}1)$", color="w", fontsize=8,
           rotation=-52)
    plt.colorbar(im, ax=a, label=r"$\log_{10}$ osc. amplitude")
    a.set_xlabel(r"damping $\eta$")
    a.set_ylabel(r"penalization $p$")
    letter(a, "(d)")

    fig.tight_layout()
    fig.savefig(FIGS / "fig_toy_pub.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_toy_pub.png")


def fig_identity():
    sw = np.load(DATA / "ca_sweep.npz")["rows"]
    cb = np.load(DATA / "cb_aimd.npz", allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.2))
    # (a) identity scatter
    a = ax[0, 0]
    a.plot([-2.4, 1.2], [-2.4, 1.2], "k--", lw=0.8)
    for rmin, col, mk in ((1.1, C_R, "o"), (2.4, C_B, "s")):
        r = sw[sw[:, 0] == rmin]
        a.plot(1 - r[:, 1] * r[:, 4], r[:, 3], mk, ms=4.5, color=col,
               mfc="none", label=fr"$r_{{\min}}={rmin}$")
    a.set_xlabel(r"$1-\eta\,\hat s$")
    a.set_ylabel(r"$\hat\mu$")
    a.legend(fontsize=8)
    a.set_title("identity on the free set, 28 runs", fontsize=8.5)
    letter(a, "(a)")
    # (b) amplitude vs eta
    a = ax[0, 1]
    for rmin, col, mk in ((1.1, C_R, "o"), (2.4, C_B, "s")):
        r = sw[sw[:, 0] == rmin]
        a.semilogy(r[:, 1], r[:, 2], mk + "-", ms=4, color=col, lw=0.9,
                   mfc="none", label=fr"$r_{{\min}}={rmin}$")
    a.axvline(0.503, color=C_R, lw=0.8, ls=":")
    a.axvline(0.596, color=C_B, lw=0.8, ls=":")
    a.text(0.51, 2e-4, r"$2/s_{\max}$", fontsize=7, color=C_R, rotation=90)
    a.set_xlabel(r"damping $\eta$")
    a.set_ylabel(r"tail amplitude $\overline{|\Delta x|}$")
    a.legend(fontsize=8, loc="lower right")
    a.set_title("onset vs fixed-design thresholds", fontsize=8.5)
    letter(a, "(b)")
    # (c)/(d) designs
    for a, key, ttl in ((ax[1, 0], "P2_r2.4_fixed0.5",
                         r"fixed $\eta=0.5$: $c=217.1$"),
                        (ax[1, 1], "P2_r2.4_fixed1.0",
                         r"fixed $\eta=1.0$, oscillating: $c=312.7$")):
        x = cb[f"{key}_x"].reshape(60, 20).T
        a.imshow(1 - x, cmap="gray", vmin=0, vmax=1)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(ttl, fontsize=8.5)
    letter(ax[1, 0], "(c)")
    letter(ax[1, 1], "(d)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_identity_pub.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_identity_pub.png")


def fig_spectra():
    d = np.load(DATA / "c5_psweep_ref.npz")
    x0 = d["p3.0_x0"]
    S = setup(60, 20, 1.1, "mbb")
    J5, F = build_J(x0, S, 0.5, "sens", eta=0.5)
    s5, modes, _ = c5.tangent_spectrum(J5, x0, F, S, "sens", eta=0.5)
    J9, _ = build_J(x0, S, 0.5, "sens", eta=0.9)
    s9, _, _ = c5.tangent_spectrum(J9, x0, F, S, "sens", eta=0.9)
    v = np.real(modes[:, 0])
    v /= np.linalg.norm(v)
    # observed oscillation just above threshold: kick along mode, iterate
    x = x0.copy()
    x[F] = x0[F] * np.exp(0.01 * v)
    x = np.clip(x, 1e-3, 1.0)
    for _ in range(14):                    # stay in the linear regime
        xp = x
        x = core.step(x, 0.5, 0.56, S, "sens")
    dx = x - xp
    vv = np.zeros(1200)
    vv[F] = v
    # cycle-3 convention: centered, mid-density x-space projection
    mid = (x0[F] > 0.05) & (x0[F] < 0.95)
    w1 = (x0[F] * v)[mid]
    w1 = w1 - w1.mean()
    w2 = dx[F][mid]
    w2 = w2 - w2.mean()
    cosv = abs(w1 @ w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))

    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.0))
    a = ax[0, 0]
    a.hist(np.real(s5), bins=44, color=C_B, alpha=0.85)
    a.axvline(4.0, color="k", lw=1.0, ls="--")
    a.text(3.97, a.get_ylim()[1] * 0.55, r"$p+1$", fontsize=8, ha="right")
    a.annotate("bulk $s\\approx1$", xy=(1.0, 60), fontsize=8, color=C_ACC)
    a.annotate(fr"$s_{{\max}}={np.max(s5.real):.3f}$",
               xy=(np.max(s5.real), 3), xytext=(2.4, 25),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    a.set_xlabel(r"log--log stiffness $s$")
    a.set_ylabel("modes")
    a.set_title("R0 spectrum (exact projection)", fontsize=8.5)
    letter(a, "(a)")

    a = ax[0, 1]
    n = min(s5.size, s9.size)
    a.plot(np.sort(s5.real)[::-1][:n], np.sort(s9.real)[::-1][:n], "o",
           ms=2.5, color=C_R, alpha=0.6)
    a.plot([0, 4.1], [0, 4.1], "k--", lw=0.8)
    a.set_xlabel(r"$s_m$ from $J(\eta=0.5)$")
    a.set_ylabel(r"$s_m$ from $J(\eta=0.9)$")
    a.set_title("$\\eta$-invariance (consistency check)", fontsize=8.5)
    letter(a, "(b)")

    for a, fld, ttl in ((ax[1, 0], vv, "leading eigenmode"),
                        (ax[1, 1], dx,
                         fr"observed oscillation, $|\cos|={cosv:.2f}$")):
        g = fld.reshape(60, 20).T
        m = np.max(np.abs(g))
        a.imshow(g, cmap="RdBu_r", vmin=-m, vmax=m)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(ttl, fontsize=8.5)
    letter(ax[1, 0], "(c)")
    letter(ax[1, 1], "(d)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_spectra_pub.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_spectra_pub.png (|cos|={cosv:.3f}, "
          f"top5 d={np.max(np.abs(np.sort(s5.real)[::-1][:5]-np.sort(s9.real)[::-1][:5])):.1e})")


def fig_robust():
    cases = ["R0", "V1", "V2", "V3", "V4", "V5"]
    smax = [3.974]
    star = [0.503]
    br = [(0.5, 0.6)]
    # V1 predates probe logging; its bracket is the cycle-4 note value.
    fixed_br = {"V1": (0.45, 0.55), "V5": (5.4, 6.0)}
    for k in ["V1", "V2", "V3", "V4", "V5"]:
        d = np.load(DATA / f"c4_{k}.npz", allow_pickle=True)
        smax.append(float(d["smax"]))
        star.append(float(d["eta_star"]))
        if k in fixed_br:
            br.append(fixed_br[k])
        else:
            lo, hi = c5.onset_bracket([tuple(r) for r in d["probe"]])
            br.append((lo, hi))
    fig, ax = plt.subplots(1, 3, figsize=(9.8, 2.9))
    a = ax[0]
    cols = [C_B] * 5 + [C_R]
    a.bar(cases, smax, color=cols, alpha=0.85)
    a.axhline(4.0, color="k", lw=0.9, ls="--")
    a.text(5.4, 4.05, r"$p+1$", fontsize=8, ha="right")
    a.set_ylabel(r"$s_{\max}$")
    a.set_title("dominant stiffness across variants", fontsize=8.5)
    letter(a, "(a)")

    a = ax[1]
    for i in range(5):                      # R0..V4 on a linear axis
        lo, hi = br[i]
        a.plot([i, i], [lo, hi], "-", color=C_G, lw=5, alpha=0.5,
               solid_capstyle="butt")
        a.plot(i, star[i], "o", color=C_R, ms=5)
    a.set_xticks(range(5))
    a.set_xticklabels(cases[:5])
    a.set_xlim(-0.6, 4.6)
    a.set_ylim(0.4, 0.8)
    a.set_ylabel(r"damping $\eta$")
    a.plot([], [], "o", color=C_R, label=r"predicted $2/s_{\max}$")
    a.plot([], [], "-", color=C_G, lw=5, alpha=0.5, label="onset bracket")
    a.legend(fontsize=7.5, loc="upper left")
    a.text(0.97, 0.06,
           r"V5 (density filter), off scale:" "\n"
           r"predicted $5.41$, bracket $(5.4,6.0]$",
           transform=a.transAxes, fontsize=7.5, ha="right", color=C_ACC)
    a.set_title("prediction vs behavioral onset", fontsize=8.5)
    letter(a, "(b)")

    a = ax[2]
    pts = []
    for k in ["V3", "V4"]:
        d = np.load(DATA / f"c4_{k}.npz", allow_pickle=True)
        sm = float(d["smax"])
        for e, g, rs in d["probe"]:
            if rs < 0 and abs(1 - e * sm) <= 1.25:
                pts.append((abs(1 - e * sm), g))
    pts = np.array(pts)
    a.plot([0.4, 1.25], [0.4, 1.25], "k--", lw=0.8)
    a.plot(pts[:, 0], pts[:, 1], "o", ms=5, color="#6A4C93")
    a.set_xlabel(r"predicted $|1-\eta\,s_{\max}|$")
    a.set_ylabel(r"measured $|r|$")
    a.set_title("sub/near-threshold rates (V3, V4)", fontsize=8.5)
    letter(a, "(c)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_robust_pub.png", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_robust_pub.png")


if __name__ == "__main__":
    t0 = time.time()
    fig_toy()
    fig_identity()
    fig_spectra()
    fig_robust()
    print(f"total {time.time()-t0:.0f}s")

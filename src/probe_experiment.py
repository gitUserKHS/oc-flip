# -*- coding: utf-8 -*-
"""
Experiment 2c (v2): fixed-design eta-probe with annihilation-based mode split.

OC fixed points are eta-independent, so probing ONE converged design x0 at
different eta probes the same Jacobian with different exponents. Theory:
flip root mu(eta) = 1 - eta*s(x0), a line crossing -1 at eta* = 2/s(x0).

Estimator (robust): (1) dominant multiplier mu1 from the LATE probe window
(power iteration has converged there); (2) annihilate it, w_k = c(dz_{k+1})
- mu1*c(dz_k), and power-correlate the residuals over the EARLY window to get
the second root mu2. Flip root = min(mu1, mu2). Median over 3 probe seeds.
Verification: continuation amp(eta) from x0 should turn on at the crossing.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from top88_v2 import setup, step, XMIN

nelx, nely, volfrac, p, move = 60, 20, 0.5, 3.0, 0.2
rmins = [1.1, 2.4]
cols = {1.1: "tab:blue", 2.4: "tab:red"}
etaP = np.round(np.arange(0.3, 1.51, 0.1), 2)
etaC = np.round(np.arange(0.4, 1.01, 0.1), 2)

def converge(S, eta0=0.5, iters=200):
    x = volfrac*np.ones(S["nel"])
    for _ in range(iters):
        x, _, _, _, c = step(x, p, volfrac, eta0, move, S)
    return x, c

def flip_root(dzs, masks, k0=2):
    T = len(dzs)
    def pair_mu(k):
        m = masks[k] & masks[k+1]
        if m.sum() < 20: return None
        z0 = dzs[k][m]; z0 = z0 - z0.mean(); den = z0 @ z0
        if den/m.sum() < 1e-18: return None
        z1 = dzs[k+1][m]; z1 = z1 - z1.mean()
        return (z1 @ z0)/den
    late = [v for v in (pair_mu(k) for k in range(T-9, T-1)) if v is not None]
    if not late: return np.nan
    mu1 = float(np.median(late))
    num = den = 0.0
    for k in range(k0, min(T-2, k0+16)):
        m = masks[k] & masks[k+1] & masks[k+2]
        if m.sum() < 20: continue
        z0 = dzs[k][m];   z0 = z0 - z0.mean()
        z1 = dzs[k+1][m]; z1 = z1 - z1.mean()
        z2 = dzs[k+2][m]; z2 = z2 - z2.mean()
        w0 = z1 - mu1*z0; w1 = z2 - mu1*z1
        num += w1 @ w0; den += w0 @ w0
    if den < 1e-300: return mu1
    return min(mu1, num/den)

def probe(x0, eta_p, S, piters=30, amp=0.05, seeds=(7, 17, 27)):
    vals = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        inter = (x0 > XMIN+1e-6) & (x0 < 1-1e-6)
        d = np.zeros_like(x0)
        d[inter] = rng.normal(0, amp, inter.sum())
        d[inter] -= d[inter].mean()
        xp = np.clip(x0*np.exp(d), XMIN, 1.0)
        xp = np.clip(xp*(volfrac/xp.mean()), XMIN, 1.0)
        dzs, masks, xq = [], [], xp
        for _ in range(piters):
            xq, dz, inter2, _, _ = step(xq, p, volfrac, eta_p, move, S)
            dzs.append(dz); masks.append(inter2)
        vals.append(flip_root(dzs, masks))
    return float(np.nanmedian(vals))

def continue_amp(x0, eta, S, iters=80):
    x = x0.copy(); amps = []
    for _ in range(iters):
        xn, _, _, _, _ = step(x, p, volfrac, eta, move, S)
        amps.append(np.mean(np.abs(xn-x))); x = xn
    return float(np.median(amps[-20:]))

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
print(f"{'rmin':>5} {'s_fit':>7} {'intercept':>9} {'npts':>5} {'eta*_pred':>9}")
for rm in rmins:
    S = setup(nelx, nely, rm)
    x0, c0 = converge(S)
    rf = np.array([probe(x0, e, S) for e in etaP])
    print(f"rmin={rm} flip roots: " +
          " ".join(f"{e}:{v:+.3f}" for e, v in zip(etaP, rf)))
    pre = np.isfinite(rf) & (rf > -0.93)
    A = np.vstack([etaP[pre], np.ones(pre.sum())]).T
    slope, intc = np.linalg.lstsq(A, rf[pre], rcond=None)[0]
    s_fit = -slope
    eta_star = (intc + 1.0)/s_fit
    print(f"{rm:5.1f} {s_fit:7.3f} {intc:9.3f} {int(pre.sum()):5d} {eta_star:9.3f}")
    camp = np.array([continue_amp(x0, e, S) for e in etaC])
    print("continuation amp: " +
          " ".join(f"{e}:{a:.1e}" for e, a in zip(etaC, camp)))
    ax[0].plot(etaP, rf, "o", color=cols[rm], ms=5, label=f"measured, rmin={rm}")
    ee = np.linspace(0.25, 1.55, 100)
    ax[0].plot(ee, np.maximum(intc - s_fit*ee, -1.03), color=cols[rm], lw=1.1, alpha=0.8)
    ax[0].axvline(eta_star, color=cols[rm], ls="--", lw=1,
                  label=f"crossing {eta_star:.2f}")
    ax[1].semilogy(etaC, camp, "o-", color=cols[rm], label=f"rmin={rm}")
    ax[1].axvline(eta_star, color=cols[rm], ls="--", lw=1)

ax[0].axhline(-1, color="k", ls=":", lw=1)
ax[0].set_xlabel(r"probe $\eta$"); ax[0].set_ylabel(r"flip root $\mu(\eta)$")
ax[0].set_title(r"(A) fixed design $x_0$: $\mu=1-\eta\,s(x_0)$, intercept$\to$1")
ax[0].legend(fontsize=7)
ax[1].set_xlabel(r"continuation $\eta$"); ax[1].set_ylabel(r"tail mean $|\Delta x|$")
ax[1].set_title(r"(B) continuation from $x_0$: onset at predicted crossing")
ax[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_cycle2c.png", dpi=150)
print("figure saved.")

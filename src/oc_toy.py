# -*- coding: utf-8 -*-
"""
Cycle 1: Local multiplier & flip threshold of the damped OC map on 2-bar toys.

Series (statically determinate) toy:
    c(x) = sum_e a_e x_e^{-p},  D_e = -dc/dx_e = p a_e x_e^{-(p+1)} > 0
    Pure OC (no clip/move):  x+ = V * x D^eta / sum(x D^eta)
    In relative log coords z = log(x1/x2):  z+ = mu z + (1-mu) z*,  mu = 1 - eta(p+1)  [EXACT]
Predictions:
    converge iff |mu|<1  <=>  eta < 2/(p+1);  one-step at eta = 1/(p+1);  flip at eta = 2/(p+1).
    (p,eta) = (3, 0.5)  sits exactly on the flip boundary.

Parallel (competition) toy:
    c(x) = F^2 / sum_e a_e x_e^p,  D_e = p a_e x_e^{p-1} F^2 / (sum a x^p)^2
    z+ = (1 + eta(p-1)) z  [EXACT]  -> interior point repelling for p>1 (winner-take-all).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XMIN = 1e-3
rng = np.random.default_rng(0)

# ---------------- sensitivities ----------------
def D_series(x, a, p):
    return p * a * x ** (-(p + 1.0))

def D_parallel(x, a, p):
    return p * a * x ** (p - 1.0) / (np.sum(a * x ** p)) ** 2

# ---------------- OC updates ----------------
def oc_pure(x, D, eta, V):
    w = x * D ** eta
    return V * w / w.sum()

def oc_t88(x, D, eta, V, m):
    """top88-style: move limit m, clip to [XMIN,1], bisection on lambda."""
    l1, l2 = 1e-30, 1e30
    lo = np.maximum(XMIN, x - m)
    hi = np.minimum(1.0, x + m)
    for _ in range(90):
        lm = np.sqrt(l1 * l2)
        xn = np.clip(x * (D / lm) ** eta, lo, hi)
        if xn.sum() > V:
            l1 = lm
        else:
            l2 = lm
    return xn

# ---------------- helpers ----------------
def x_from_z(z, V):
    return np.array([V * np.exp(z) / (1 + np.exp(z)), V / (1 + np.exp(z))])

def run(map_step, z0, V, iters):
    x = x_from_z(z0, V)
    Z = np.empty(iters + 1); X1 = np.empty(iters + 1)
    Z[0] = np.log(x[0] / x[1]); X1[0] = x[0]
    for k in range(iters):
        x = map_step(x)
        Z[k + 1] = np.log(x[0] / x[1]); X1[k + 1] = x[0]
    return Z, X1

def empirical_rate(Z, zstar, lo=1e-11, hi=0.5, kmax=200):
    e = np.abs(Z - zstar)
    ok = (e > lo) & (e < hi)
    idx = np.where(ok[:-1] & ok[1:])[0]
    idx = idx[idx < kmax]
    if len(idx) < 3:
        return np.nan
    r = e[idx + 1] / e[idx]
    return float(np.exp(np.mean(np.log(r))))

# ================= Experiment A: rate check =================
a = np.array([1.0, 2.0]); V = 1.2
def zstar_of(p): return np.log(a[0] / a[1]) / (p + 1.0)

print("=== A. empirical rate vs predicted |mu| (series toy) ===")
rows = []
for p in [1.0, 3.0, 5.0]:
    zs = zstar_of(p)
    for eta in np.arange(0.05, 2.0 / (p + 1) - 1e-9, 0.05):
        mu = 1 - eta * (p + 1)
        Zp, _ = run(lambda x: oc_pure(x, D_series(x, a, p), eta, V), zs + 0.3, V, 250)
        Zt, _ = run(lambda x: oc_t88(x, D_series(x, a, p), eta, V, 0.2), zs + 0.3, V, 250)
        rows.append((p, eta, abs(mu), empirical_rate(Zp, zs), empirical_rate(Zt, zs)))
rows = np.array(rows)
err_p = np.nanmax(np.abs(rows[:, 3] - rows[:, 2]))
err_t = np.nanmax(np.abs(rows[:, 4] - rows[:, 2]))
print(f"max |emp-pred| : pure = {err_p:.2e}, top88-style = {err_t:.2e}")

# one-step convergence check at eta = 1/(p+1)
p = 3.0; zs = zstar_of(p)
Z1, _ = run(lambda x: oc_pure(x, D_series(x, a, p), 1.0 / (p + 1), V), zs + 0.4, V, 3)
print(f"one-step check p=3, eta=0.25: |z1-z*| = {abs(Z1[1]-zs):.2e} (start 0.4)")

# ================= Experiment B: bifurcation diagram (p=3, top88-style) =================
print("=== B. bifurcation diagram p=3 ===")
p = 3.0; zs = zstar_of(p); m_bif = 0.05
etas_b = np.arange(0.05, 0.901, 0.005)
bif_e, bif_x = [], []
for eta in etas_b:
    _, X1 = run(lambda x: oc_t88(x, D_series(x, a, p), eta, V, m_bif), zs + 0.05, V, 600)
    tail = X1[-120:]
    bif_e.append(np.full_like(tail, eta)); bif_x.append(tail)
bif_e = np.concatenate(bif_e); bif_x = np.concatenate(bif_x)

# ================= Experiment C: amplitude heatmap over (eta, p) =================
print("=== C. amplitude heatmap (top88-style, m=0.2) ===")
ps = np.arange(1.0, 5.01, 0.5)
etas_h = np.arange(0.02, 1.001, 0.025)
AMP = np.zeros((len(ps), len(etas_h)))
for i, p in enumerate(ps):
    zs = zstar_of(p)
    for j, eta in enumerate(etas_h):
        _, X1 = run(lambda x: oc_t88(x, D_series(x, a, p), eta, V, 0.2), zs + 0.05, V, 500)
        AMP[i, j] = np.ptp(X1[-120:])

# threshold sharpness check at p=3
j_lo = np.argmin(np.abs(etas_h - 0.475)); j_hi = np.argmin(np.abs(etas_h - 0.525))
print(f"p=3 amp just below thr (eta≈0.475): {AMP[np.argmin(np.abs(ps-3)), j_lo]:.2e}")
print(f"p=3 amp just above thr (eta≈0.525): {AMP[np.argmin(np.abs(ps-3)), j_hi]:.2e}")

# ================= Experiment D: parallel toy (winner-take-all) =================
print("=== D. parallel toy p=3, eta=0.5 (pred. multiplier 1+eta(p-1)=2) ===")
p, eta = 3.0, 0.5
ap = np.array([1.0, 1.0])
x = x_from_z(0.02, V)  # tiny asymmetry
print(" k    x1      x2      z=log(x1/x2)   z_k/z_{k-1}")
zprev = np.log(x[0] / x[1])
for k in range(1, 11):
    x = oc_t88(x, D_parallel(x, ap, p), eta, V, 0.2)
    z = np.log(x[0] / x[1])
    print(f"{k:2d}  {x[0]:.4f}  {x[1]:.4f}   {z: .5f}      {z/zprev: .3f}")
    zprev = z

# ================= Figure =================
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (A) rate scatter
for p_, mk in [(1.0, "o"), (3.0, "s"), (5.0, "^")]:
    sel = rows[:, 0] == p_
    ax[0].scatter(rows[sel, 2], rows[sel, 3], marker=mk, s=42, alpha=0.85, label=f"pure, p={p_:g}")
    ax[0].scatter(rows[sel, 2], rows[sel, 4], marker=mk, s=18, alpha=0.85, facecolors="none",
                  edgecolors="k", label=f"top88-style, p={p_:g}")
ax[0].plot([0, 1], [0, 1], "r--", lw=1)
ax[0].set_xlabel(r"predicted $|\mu|=|1-\eta(p{+}1)|$")
ax[0].set_ylabel("empirical contraction rate")
ax[0].set_title("(A) rate: prediction vs measurement")
ax[0].legend(fontsize=6.5, ncol=2)

# (B) bifurcation
ax[1].plot(bif_e, bif_x, ",", color="k", alpha=0.5)
ax[1].axvline(0.5, color="r", ls="--", lw=1, label=r"theory $\eta^*=2/(p{+}1)=0.5$")
ax[1].set_xlabel(r"damping $\eta$"); ax[1].set_ylabel(r"asymptotic $x_1$")
ax[1].set_title("(B) bifurcation diagram, p=3 (top88-style)")
ax[1].legend(fontsize=8)

# (C) heatmap
im = ax[2].imshow(np.log10(AMP + 1e-16), origin="lower", aspect="auto",
                  extent=[etas_h[0], etas_h[-1], ps[0], ps[-1]], cmap="magma")
pp = np.linspace(1, 5, 200)
ax[2].plot(2.0 / (pp + 1), pp, "w--", lw=1.6, label=r"$\eta^*=2/(p{+}1)$")
ax[2].set_xlabel(r"damping $\eta$"); ax[2].set_ylabel(r"SIMP exponent $p$")
ax[2].set_title(r"(C) $\log_{10}$ osc. amplitude of $x_1$")
ax[2].legend(fontsize=8, loc="upper right")
fig.colorbar(im, ax=ax[2], shrink=0.85)

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_cycle1.png", dpi=150)
print("figure saved.")

# -*- coding: utf-8 -*-
"""
Cycle 3: direct Jacobian spectrum of the OC map.

Construction: at a design x0, finite-difference one full OC step (FE solve +
sensitivity filter + lambda bisection + clipping) in log coordinates over the
free set F. Theory: on range(P) (volume-projected subspace),
    J(eta) = I + eta * PG,   so   mu_m(eta) = 1 - eta * s_m
with an eta-INDEPENDENT s-spectrum {s_m} = eig(-PG). Predictions:
  (i)  spectra measured at different eta collapse in s-space (intercept=1 exact);
  (ii) flip threshold of design x0 is eta* = 2 / s_max(x0);
  (iii) top eigenmode = observed oscillation pattern just above eta*;
  (iv) self-stabilization: full runs at eta produce designs whose capacity
       2/s_max(design(eta)) stays above eta while they can (margin -> 0 at onset).
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eig
from top88_v2 import setup, step, XMIN

nelx, nely, volfrac, p, move = 60, 20, 0.5, 3.0, 0.2

def oc_x(x, eta, S):
    return step(x, p, volfrac, eta, move, S)[0]

def run_to(S, eta0, iters=200, x=None):
    if x is None:
        x = volfrac*np.ones(S["nel"])
    for _ in range(iters):
        x = oc_x(x, eta0, S)
    # tail amplitude tag (10 extra steps)
    amps = []
    for _ in range(10):
        xn = oc_x(x, eta0, S); amps.append(np.mean(np.abs(xn-x))); x = xn
    return x, float(np.median(amps))

def build_J(x0, eta, S, tol=2e-3, eps=1e-5):
    F = np.where((x0 > XMIN+tol) & (x0 < 1-tol))[0]
    base = np.log(oc_x(x0, eta, S)[F])
    J = np.empty((F.size, F.size))
    for j, e in enumerate(F):
        xp = x0.copy(); xp[e] *= np.exp(eps)
        J[:, j] = (np.log(oc_x(xp, eta, S)[F]) - base)/eps
    return J, F

def spectrum(J, eta):
    mu, V = eig(J)
    # remove the volume-projection null mode: most-uniform eigenvector
    unif = np.abs(V.sum(0))/(np.sqrt(V.shape[0])*np.linalg.norm(V, axis=0))
    k0 = int(np.argmax(unif))
    keep = np.ones(len(mu), bool); keep[k0] = False
    mu, V = mu[keep], V[:, keep]
    s = (1.0 - mu)/eta
    order = np.argsort(-s.real)
    return s[order], V[:, order]

def field(F, v):
    f = np.zeros(nelx*nely); f[F] = v
    return f.reshape(nelx, nely).T

# ================= E1: spectrum at fixed design x0 (eta=0.5 converged) =====
E1 = {}
for rm in (1.1, 2.4):
    S = setup(nelx, nely, rm)
    x0, amp0 = run_to(S, 0.5)
    J1, F = build_J(x0, 0.5, S)
    s1, V1 = spectrum(J1, 0.5)
    J2, _ = build_J(x0, 0.9, S)
    s2, V2 = spectrum(J2, 0.9)
    smax = float(s1.real.max())
    eta_star = 2.0/smax
    ncx = int(np.sum(np.abs(s1.imag) > 1e-3))
    # observed oscillation pattern just above predicted crossing
    rng = np.random.default_rng(3)
    m = (x0 > XMIN+2e-3) & (x0 < 1-2e-3)
    d = np.zeros_like(x0); d[m] = rng.normal(0, 0.02, m.sum()); d[m] -= d[m].mean()
    x = np.clip(x0*np.exp(d), XMIN, 1.0); x = np.clip(x*(volfrac/x.mean()), XMIN, 1.0)
    xs = [x]
    for _ in range(40):
        xs.append(oc_x(xs[-1], eta_star+0.05, S))
    dz = np.log(xs[-1])[F] - np.log(xs[-2])[F]; dz -= dz.mean()
    v = V1[:, 0].real.copy(); v -= v.mean()
    cos = float(abs(dz @ v)/(np.linalg.norm(dz)*np.linalg.norm(v)))
    # eta-invariance pairing (sorted real parts, top 40)
    n40 = min(40, len(s1), len(s2))
    E1[rm] = dict(x0=x0, F=F, s1=s1, s2=s2, V1=V1, smax=smax,
                  eta_star=eta_star, cos=cos, dz=dz, n40=n40, nF=len(F), ncx=ncx)
    print(f"[E1] rmin={rm}: |F|={len(F)}, s_max={smax:.3f} -> eta*={eta_star:.3f}, "
          f"complex modes={ncx}, top-mode vs observed-osc cos={cos:.3f}")
    print(f"     top-5 s (eta=0.5 build): {np.round(s1.real[:5],3)}")
    print(f"     top-5 s (eta=0.9 build): {np.round(s2.real[:5],3)}")

# ================= E2: capacity of full-run designs (self-stabilization) ===
E2 = {}
runs = {1.1: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 2.4: [0.4, 0.5, 0.6, 0.7]}
for rm, etl in runs.items():
    S = setup(nelx, nely, rm)
    for e in etl:
        xf, amp = run_to(S, e)
        J, F = build_J(xf, 0.5, S)   # s-spectrum is eta-independent; build at 0.5
        s, _ = spectrum(J, 0.5)
        smax = float(s.real.max())
        E2[(rm, e)] = dict(cap=2.0/smax, amp=amp, osc=amp > 3e-3, nF=len(F))
        print(f"[E2] rmin={rm} eta={e}: |F|={len(F)}, s_max={smax:.3f}, "
              f"capacity 2/s_max={2/smax:.3f}, amp={amp:.1e}, "
              f"{'OSC' if amp>3e-3 else 'stable'}")

# ================= figure ==================================================
fig = plt.figure(figsize=(13.5, 8.0))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[0, 2])
cols = {1.1: "tab:blue", 2.4: "tab:red"}
bands = {1.1: (2/0.6, 2/0.5), 2.4: (2/0.7, 2/0.6)}   # cycle-2 fixed-design onset brackets in s

for rm in (1.1, 2.4):
    s = E1[rm]["s1"].real
    axA.hist(s, bins=40, alpha=0.45, color=cols[rm], label=f"rmin={rm}")
    axA.axvline(E1[rm]["smax"], color=cols[rm], lw=1.4)
    axA.axvspan(*bands[rm], color=cols[rm], alpha=0.12)
axA.axvline(p+1, color="gray", ls=":", lw=1, label=r"determinate bound $p{+}1$")
axA.set_xlabel(r"$s$"); axA.set_ylabel("modes")
axA.set_title("(A) s-spectrum at $x_0$; band = cycle-2 onset bracket")
axA.legend(fontsize=7)

for rm in (1.1, 2.4):
    n = E1[rm]["n40"]
    axB.plot(E1[rm]["s1"].real[:n], E1[rm]["s2"].real[:n], "o", ms=4,
             alpha=0.7, color=cols[rm], label=f"rmin={rm}")
lo = min(E1[1.1]["s2"].real[:E1[1.1]["n40"]].min(), 0)
hi = max(E1[1.1]["smax"], E1[2.4]["smax"])*1.05
axB.plot([lo, hi], [lo, hi], "k--", lw=1)
axB.set_xlabel(r"$s_m$ from $J(\eta{=}0.5)$"); axB.set_ylabel(r"$s_m$ from $J(\eta{=}0.9)$")
axB.set_title(r"(B) $\eta$-invariance of the s-spectrum")
axB.legend(fontsize=8)

for (rm, e), d in E2.items():
    axC.plot(e, d["cap"], "o" if not d["osc"] else "s", ms=8,
             mfc=(cols[rm] if not d["osc"] else "none"), mec=cols[rm], mew=1.6)
for rm in (1.1, 2.4):
    axC.axhline(E1[rm]["eta_star"], color=cols[rm], ls=":", lw=1,
                label=f"fixed-$x_0$ capacity, rmin={rm}")
ee = np.linspace(0.35, 0.95, 10)
axC.plot(ee, ee, "k--", lw=1, label=r"marginal line $\mathrm{cap}=\eta$")
axC.set_xlabel(r"run $\eta$"); axC.set_ylabel(r"design capacity $2/s_{\max}$")
axC.set_title("(C) self-stabilization: capacity of full-run designs")
axC.legend(fontsize=7)

axD = fig.add_subplot(gs[1, 0]); axE = fig.add_subplot(gs[1, 1]); axF = fig.add_subplot(gs[1, 2])
vmax1 = np.abs(E1[1.1]["V1"][:, 0].real).max()
axD.imshow(field(E1[1.1]["F"], E1[1.1]["V1"][:, 0].real), cmap="RdBu_r",
           vmin=-vmax1, vmax=vmax1, interpolation="nearest")
axD.set_title(f"(D) top eigenmode, rmin=1.1 (s={E1[1.1]['smax']:.2f})", fontsize=9)
axD.axis("off")
dzf = field(E1[1.1]["F"], E1[1.1]["dz"]); vmax2 = np.abs(dzf).max()
axE.imshow(dzf, cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, interpolation="nearest")
axE.set_title(f"(E) observed osc. pattern, rmin=1.1 (|cos|={E1[1.1]['cos']:.2f})", fontsize=9)
axE.axis("off")
vmax3 = np.abs(E1[2.4]["V1"][:, 0].real).max()
axF.imshow(field(E1[2.4]["F"], E1[2.4]["V1"][:, 0].real), cmap="RdBu_r",
           vmin=-vmax3, vmax=vmax3, interpolation="nearest")
axF.set_title(f"(F) top eigenmode, rmin=2.4 (s={E1[2.4]['smax']:.2f}, "
              f"|cos|={E1[2.4]['cos']:.2f})", fontsize=9)
axF.axis("off")

fig.tight_layout()
fig.savefig(Path(__file__).resolve().parents[1] / "figs" / "fig_cycle3.png", dpi=150)
print("figure saved.")

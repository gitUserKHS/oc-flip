# -*- coding: utf-8 -*-
"""Cycle C: compact single-constraint MMA column for the sweep.

Standard MMA (Svanberg 1987) specialized to compliance + one volume
constraint: separable subproblem solved exactly by dual bisection on the
volume multiplier (same structure as the OC bisection). Standard
parameters: asyinit 0.5, asyincr 1.2, asydecr 0.7, albefa 0.1,
move 0.5. Sensitivity/density filtering matches the group's setting.

Note: our implementation, validated on R0 (comparable compliance to OC);
column is reported with this caveat in the paper.

Usage: python src/cc_mma.py          (validate on R0, then sweep groups)
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, XMIN, E0, Emin
import c5_core as c5
from cc_sweep import CONFIGS

ASYINIT, ASYINCR, ASYDECR, ALBEFA, MOVEMMA = 0.5, 1.2, 0.7, 0.1, 0.5
XMAX = 1.0


def _dc(x, vf, S, filt):
    """Filtered compliance sensitivity at x (same code path as the OC)."""
    xph = np.asarray(S["H"] @ x).ravel() / S["Hs"] if filt == "dens" else x
    sK = ((S["KE"].flatten()[np.newaxis]).T *
          (Emin + xph**core.P * (E0 - Emin))).flatten(order="F")
    Kff = coo_matrix((sK[S["mK"]], (S["iKr"], S["jKr"])),
                     shape=(S["nfree"], S["nfree"])).tocsc()
    u = np.zeros(S["ndof"])
    u[S["free"]] = spsolve(Kff, S["f"][S["free"]], permc_spec="MMD_AT_PLUS_A")
    ce = np.einsum("ij,jk,ik->i", u[S["edof"]], S["KE"], u[S["edof"]])
    dc = -core.P * xph**(core.P - 1) * (E0 - Emin) * ce
    if filt == "sens":
        dcf = np.asarray(S["H"] @ (x * dc)).ravel() / S["Hs"] \
            / np.maximum(1e-3, x)
        dv = np.ones_like(x)
    else:
        dcf = np.asarray(S["H"] @ (dc / S["Hs"])).ravel()
        dv = np.asarray(S["H"] @ (np.ones_like(x) / S["Hs"])).ravel()
    return dcf, dv


def mma_run(S, vf, filt, p, iters, seed=None):
    core.P = p
    n = S["nel"]
    x = vf * np.ones(n)
    if seed is not None:                     # perturbed restart (step 3)
        rng = np.random.default_rng(seed)
        d = rng.normal(0, 0.02, n)
        x = np.clip(x * np.exp(d - d.mean()), XMIN, 1.0)
    xold1 = xold2 = x.copy()
    low = x - ASYINIT * (XMAX - XMIN)
    upp = x + ASYINIT * (XMAX - XMIN)
    amp = np.empty(iters)
    for it in range(iters):
        dcf, dv = _dc(x, vf, S, filt)
        if it >= 2:
            zzz = (x - xold1) * (xold1 - xold2)
            fac = np.ones(n)
            fac[zzz > 0] = ASYINCR
            fac[zzz < 0] = ASYDECR
            low = x - fac * (xold1 - low)
            upp = x + fac * (upp - xold1)
            low = np.clip(low, x - 10 * (XMAX - XMIN),
                          x - 0.01 * (XMAX - XMIN))
            upp = np.clip(upp, x + 0.01 * (XMAX - XMIN),
                          x + 10 * (XMAX - XMIN))
        alfa = np.maximum.reduce([np.full(n, XMIN),
                                  low + ALBEFA * (x - low),
                                  x - MOVEMMA * (XMAX - XMIN)])
        beta = np.minimum.reduce([np.full(n, XMAX),
                                  upp - ALBEFA * (upp - x),
                                  x + MOVEMMA * (XMAX - XMIN)])
        xmami = XMAX - XMIN
        dcp = np.maximum(dcf, 0)
        dcm = np.maximum(-dcf, 0)
        p0 = (upp - x)**2 * (dcp + 1e-3 * np.abs(dcf) + 1e-5 / xmami)
        q0 = (x - low)**2 * (dcm + 1e-3 * np.abs(dcf) + 1e-5 / xmami)
        P1 = (upp - x)**2 * dv          # volume: dg = dv > 0
        Q1 = (x - low)**2 * 0.0
        # dual bisection on the volume multiplier
        l1, l2 = 0.0, 1e9
        vol_of = None
        for _ in range(80):
            lm = 0.5 * (l1 + l2)
            sp = np.sqrt(p0 + lm * P1)
            sq = np.sqrt(q0 + lm * Q1)
            xn = np.clip((low * sp + upp * sq) / (sp + sq), alfa, beta)
            vol = (np.asarray(S["H"] @ xn).ravel() / S["Hs"]).mean() \
                if filt == "dens" else xn.mean()
            if vol > vf:
                l1 = lm
            else:
                l2 = lm
            if (l2 - l1) / max(l1 + l2, 1e-30) < 1e-10:
                break
        amp[it] = np.mean(np.abs(xn - x))
        xold2, xold1 = xold1, x
        x = xn
    comp = c5.compliance(x, S, filt)
    core.P = 3.0
    return dict(x=x, comp=comp, amp=amp)


def main(iters=400):
    print("=== Cycle C: MMA column (standard parameters) ===")
    S = setup(60, 20, 1.1, "mbb")
    r = mma_run(S, 0.5, "sens", 3.0, iters)
    print(f"  validation R0: comp={r['comp']:.2f}  "
          f"amp_tail={np.median(r['amp'][-30:]):.1e}  (OC fixed0.5: 195.61)",
          flush=True)
    out = {"R0_valid": r}
    for name, kw, ps in CONFIGS:
        S = setup(kw["nelx"], kw["nely"], kw["rmin"], kw["bc"])
        for p in ps:
            grp = f"{name}_p{p:.0f}"
            r = mma_run(S, kw["vf"], kw["filt"], p, iters)
            out[grp] = r
            tail = float(np.median(r["amp"][-30:]))
            print(f"  {grp:18s}: comp={r['comp']:8.2f}  amp_tail={tail:.1e}"
                  f"  [{'OSC' if tail > 3e-3 else 'conv'}]", flush=True)
    save = {}
    for k, r in out.items():
        save[f"{k}_meta"] = np.array([r["comp"],
                                      float(np.median(r["amp"][-30:]))])
        save[f"{k}_amp"] = r["amp"]
    np.savez(DATA / "cc_mma.npz", **save)
    print("saved data/cc_mma.npz")


if __name__ == "__main__":
    t0 = time.time()
    main(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    print(f"total {time.time()-t0:.0f}s")

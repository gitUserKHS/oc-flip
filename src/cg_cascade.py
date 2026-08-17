# -*- coding: utf-8 -*-
"""Direct test of the clipping-arrests-the-cascade claim (reviewer (9)).

Runs the saturated oscillation (60x20 MBB sens r1.1, p=3) at several
over-driven eta with the lower box bound at 1e-3 (standard) and 1e-6
(clipping pushed far away). Period detection on the settled tail:
r2 = median ||x_t - x_{t-2}|| and r4 = median ||x_t - x_{t-4}||.
Clean period-2  -> r2 ~ 0 (r4 ~ 0 too, trivially).
Period-4        -> r2 large, r4 ~ 0.
Higher/chaotic  -> both large.

Usage: python src/cg_cascade.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, E0, Emin

MOVE = 0.2


def step_x(x, vf, eta, S, xmin):
    xph = x
    sK = ((S["KE"].flatten()[np.newaxis]).T *
          (Emin + xph**core.P * (E0 - Emin))).flatten(order="F")
    Kff = coo_matrix((sK[S["mK"]], (S["iKr"], S["jKr"])),
                     shape=(S["nfree"], S["nfree"])).tocsc()
    u = np.zeros(S["ndof"])
    u[S["free"]] = spsolve(Kff, S["f"][S["free"]], permc_spec="MMD_AT_PLUS_A")
    ce = np.einsum("ij,jk,ik->i", u[S["edof"]], S["KE"], u[S["edof"]])
    dc = -core.P * xph**(core.P - 1) * (E0 - Emin) * ce
    dcf = np.asarray(S["H"] @ (x * dc)).ravel() / S["Hs"] \
        / np.maximum(1e-3, x)
    l1, l2 = 0.0, 1e9
    lo = np.maximum(xmin, x - MOVE)
    hi = np.minimum(1.0, x + MOVE)
    while (l2 - l1) / (l1 + l2) > 1e-10:
        lm = 0.5 * (l1 + l2)
        xn = np.clip(x * np.maximum(1e-30, -dcf / lm)**eta, lo, hi)
        if xn.mean() > vf:
            l1 = lm
        else:
            l2 = lm
    return xn


def main(iters=240, tail=40):
    print("=== cascade test: settled-period residuals r2, r4 ===")
    S = setup(60, 20, 1.1, "mbb")
    print(f"{'xmin':>6} {'eta':>5} {'amp_tail':>9} {'r2':>9} {'r4':>9} "
          f"{'r2/amp':>7}  verdict")
    for xmin in (1e-3, 1e-6):
        for eta in (0.6, 0.9, 1.2, 1.6):
            x = 0.5 * np.ones(S["nel"])
            xs = []
            for t in range(iters):
                x = step_x(x, 0.5, eta, S, xmin)
                if t >= iters - tail:
                    xs.append(x.copy())
            X = np.array(xs)
            amp = float(np.median(np.mean(np.abs(np.diff(X, axis=0)),
                                          axis=1)))
            r2 = float(np.median([np.mean(np.abs(X[k] - X[k - 2]))
                                  for k in range(2, tail)]))
            r4 = float(np.median([np.mean(np.abs(X[k] - X[k - 4]))
                                  for k in range(4, tail)]))
            if amp < 1e-6:
                verdict = "converged"
            elif r2 < 0.1 * amp:
                verdict = "period-2"
            elif r4 < 0.1 * amp:
                verdict = "PERIOD-4"
            else:
                verdict = "higher/aperiodic"
            print(f"{xmin:6.0e} {eta:5.2f} {amp:9.2e} {r2:9.2e} "
                  f"{r4:9.2e} {r2/max(amp,1e-30):7.3f}  {verdict}",
                  flush=True)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time()-t0:.0f}s")

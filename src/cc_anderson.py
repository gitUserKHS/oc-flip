# -*- coding: utf-8 -*-
"""Cycle C: Anderson acceleration x AIMD composition demo (r2.4, p=3).

Anderson: type-II, window m=4, applied periodically every K=4 steps
(cf. Li-Suryanarayana-Paulino 2020's periodic scheme), Tikhonov 1e-10,
followed by box clip + multiplicative volume renormalization. Plain OC
steps in between. For AA+AIMD, the (mu^, s^) estimators are updated only
on consecutive plain-OC step pairs (an AA jump breaks the pairing).

Methods: fix0.5, fix1.0, aa_fix0.5, aa_fix1.0, aimd1.0, aa_aimd1.0.
Usage: python src/cc_anderson.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, XMIN, MOVE
import c5_core as c5
from ca_gate import step_full
from cb_aimd import PAR, ctrl_fixed, ctrl_aimd, _inst

M_AA, K_AA, TIK = 4, 4, 1e-10


def run(S, vf, filt, iters, ctrl, use_aa, beta=1.0):
    x = vf * np.ones(S["nel"])
    eta_t = np.empty(iters)
    amp = np.empty(iters)
    mu_b, s_b = [], []
    dz_p = m_p = logD_p = None
    eta = ctrl(0, np.nan, np.nan, None, np.nan)
    X, R = [], []                       # AA history: iterates, residuals
    n_md = n_aa = 0
    for t in range(iters):
        gx, logD = step_full(x, vf, eta, S, filt)
        r = gx - x
        X.append(x.copy()); R.append(r.copy())
        if len(X) > M_AA + 1:
            X.pop(0); R.pop(0)
        aa_step = use_aa and t % K_AA == K_AA - 1 and len(R) >= 3
        if aa_step:
            dR = np.column_stack([R[j] - R[j - 1] for j in range(1, len(R))])
            dG = np.column_stack([(X[j] + R[j]) - (X[j - 1] + R[j - 1])
                                  for j in range(1, len(R))])
            A = dR.T @ dR + TIK * np.trace(dR.T @ dR) * np.eye(dR.shape[1])
            gam = np.linalg.solve(A, dR.T @ r)
            xn = gx - beta * (dG @ gam)
            xn = np.clip(xn, XMIN, 1.0)
            xn = np.clip(xn * (vf / xn.mean()), XMIN, 1.0)
            n_aa += 1
        else:
            xn = gx
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(xn - x) < MOVE - 1e-6))
        eta_t[t] = eta
        amp[t] = np.mean(np.abs(xn - x))
        if dz_p is not None and not aa_step:
            mu_i, s_i = _inst(dz_p, dz, logD, logD_p, m_p & inter)
            mu_b.append(mu_i); s_b.append(s_i)
        w = PAR["window"]
        mu_w = np.nanmedian(mu_b[-w:]) if mu_b else np.nan
        s_w = np.nanmedian(s_b[-w:]) if s_b else np.nan
        amp_w = float(np.median(amp[max(0, t - w + 1):t + 1]))
        eta2 = ctrl(t + 1, mu_w, s_w, eta, amp_w)
        if eta2 < eta - 1e-12:
            n_md += 1
        eta = eta2
        if aa_step:
            dz_p = m_p = logD_p = None      # AA jump breaks estimator pairing
        else:
            dz_p, m_p, logD_p = dz, inter, logD
        x = xn
    return dict(eta=eta_t, amp=amp, comp=c5.compliance(x, S, filt), x=x,
                n_md=n_md, n_aa=n_aa)


def t_conv(amp, tol=1e-3, hold=10):
    ok = amp < tol
    for t in range(len(amp) - hold):
        if ok[t:t + hold].all():
            return t
    return -1


def main(iters=400):
    print("=== Cycle C: Anderson x AIMD composition (60x20 MBB, "
          f"sens r2.4, p=3, budget {iters}) ===")
    S = setup(60, 20, 2.4, "mbb")
    runs = [
        ("fix0.5",      ctrl_fixed(0.5), False, 1.0),
        ("fix1.0",      ctrl_fixed(1.0), False, 1.0),
        ("aa_fix0.5",   ctrl_fixed(0.5), True, 1.0),
        ("aa_fix1.0",   ctrl_fixed(1.0), True, 1.0),
        ("aaD_fix0.5",  ctrl_fixed(0.5), True, 0.5),   # damped AA, beta=.5
        ("aimd1.0",     ctrl_aimd(1.0), False, 1.0),
        ("aa_aimd0.5",  ctrl_aimd(0.5), True, 1.0),
        ("aa_aimd1.0",  ctrl_aimd(1.0), True, 1.0),
    ]
    out = {}
    for name, ctrl, aa, beta in runs:
        r = run(S, 0.5, "sens", iters, ctrl, aa, beta)
        out[name] = r
        tc = t_conv(r["amp"])
        print(f"  {name:12s}: comp@{iters}={r['comp']:8.2f}  "
              f"amp_tail={np.median(r['amp'][-30:]):.1e}  "
              f"t_conv(amp<1e-3)={tc:4d}  eta_end={r['eta'][-1]:.3f}  "
              f"n_MD={r['n_md']} n_AA={r['n_aa']}", flush=True)
    save = {}
    for k, r in out.items():
        save[f"{k}_amp"] = r["amp"]
        save[f"{k}_eta"] = r["eta"]
        save[f"{k}_meta"] = np.array([r["comp"], r["n_md"], r["n_aa"],
                                      t_conv(r["amp"])])
    np.savez(DATA / "cc_anderson.npz", **save)
    print("saved data/cc_anderson.npz")


if __name__ == "__main__":
    t0 = time.time()
    main(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    print(f"total {time.time()-t0:.0f}s")

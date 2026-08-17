# -*- coding: utf-8 -*-
"""Cycle B: minimal AIMD adaptive damping for the OC iteration.

Rule (parameters fixed across ALL runs -- the thesis is that c is
problem-independent while eta is problem-dependent):
    flip signal (windowed mu^ < -(1-delta)):  eta <- c / s^   (immediate MD)
    otherwise:                                eta <- eta + alpha  (slow AI)
Estimators are the cycle-2 trajectory estimators (mu^, s^) on the interior
mask; s^ is used ONLY under the mu^ gate (cycle-A: converging trajectories
read the s~0 creep edge, so a naive eta=c/s^ diverges).

Runs: {AIMD, fixed} x eta_init {0.3, 0.5, 1.0} + manual ladder, on
  P1 = 60x20 MBB sens r1.1 (R0)   and   P2 = 60x20 MBB sens r2.4
(the quality-destruction case). Budget 200 iterations.

Usage: python src/cb_aimd.py
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

PAR = dict(c=1.7, delta=0.25, alpha=0.01, warmup=25, window=4, cooldown=5,
           eta_min=0.05, eta_max=1.9, freeze_mu=0.9, freeze_s=0.3)
# delta=0.25: MD fires at mu^ < -0.75; with c=1.7 the post-MD multiplier is
# -0.7, safely outside the gate (no immediate refire).
# freeze_mu/freeze_s: additive increase stops when the trajectory shows the
# creep-regime signature (mu^ ~ +1, s^ ~ 0, cycle A): only s~0 modes remain
# and eta buys nothing, so probing would only reintroduce oscillation.
# All controller thresholds live in multiplier space -- no absolute
# amplitude scales (an earlier freeze_amp=1e-3 failed at 120x40, where the
# log-creep floor sits above it).


def _inst(dz_p, dz, logD, logD_p, m):
    """Instantaneous (mu^, s^) from consecutive centered log-increments."""
    if m.sum() < 20:
        return np.nan, np.nan
    z0 = dz_p[m] - dz_p[m].mean()
    den = z0 @ z0
    if den / m.sum() <= 1e-16:
        return np.nan, np.nan
    z1 = dz[m] - dz[m].mean()
    dD = (logD - logD_p)[m]
    dD = dD - dD.mean()
    return float(z1 @ z0 / den), float(-(dD @ z0) / den)


def _loop(S, vf, iters, eta_ctrl):
    """Common loop; eta_ctrl(t, mu_w, s_w, eta) -> eta decides the damping."""
    x = vf * np.ones(S["nel"])
    eta_t = np.empty(iters)
    amp = np.empty(iters)
    comp = np.empty(iters)
    dzn = np.empty(iters)
    mu_b, s_b = [], []
    dz_p = m_p = logD_p = None
    eta = eta_ctrl(0, np.nan, np.nan, None, np.nan)
    n_md = 0
    for t in range(iters):
        xn, logD, c = step_full_c(x, vf, eta, S)
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(xn - x) < MOVE - 1e-6))
        eta_t[t] = eta
        amp[t] = np.mean(np.abs(xn - x))
        comp[t] = c
        dzn[t] = np.linalg.norm(dz[inter]) if inter.sum() else 0.0
        if dz_p is not None:
            mu_i, s_i = _inst(dz_p, dz, logD, logD_p, m_p & inter)
            mu_b.append(mu_i)
            s_b.append(s_i)
        w = PAR["window"]
        mu_w = np.nanmedian(mu_b[-w:]) if mu_b else np.nan
        s_w = np.nanmedian(s_b[-w:]) if s_b else np.nan
        amp_w = float(np.median(amp[max(0, t - w + 1):t + 1]))
        eta2 = eta_ctrl(t + 1, mu_w, s_w, eta, amp_w)
        if eta2 < eta - 1e-12:
            n_md += 1
        eta = eta2
        dz_p, m_p, logD_p = dz, inter, logD
        x = xn
    return dict(eta=eta_t, amp=amp, comp=comp, dzn=dzn, x=x, n_md=n_md)


def step_full_c(x, vf, eta, S):
    """step_full + compliance of the current design (no extra FE solve:
    recompute from the returned sensitivity is not possible, so evaluate
    compliance directly -- cheap at 60x20)."""
    xn, logD = step_full(x, vf, eta, S)
    return xn, logD, c5.compliance(x, S, "sens")


def ctrl_fixed(eta0):
    def f(t, mu_w, s_w, eta, amp_w):
        return eta0
    return f


def ctrl_ladder(eta_hi=0.5, eta_lo=0.3, check_at=100, thresh=3e-3):
    state = dict(eta=eta_hi, amps=[])
    def f(t, mu_w, s_w, eta):
        if t == 0:
            return eta_hi
        return state["eta"]
    def observe(t, amp):
        state["amps"].append(amp)
        if t == check_at and np.median(state["amps"][-40:]) > thresh:
            state["eta"] = eta_lo
    f.observe = observe
    return f


def ctrl_aimd(eta0):
    state = dict(cool=0)
    def f(t, mu_w, s_w, eta, amp_w):
        if t == 0:
            return eta0
        if t <= PAR["warmup"]:
            return eta
        if state["cool"] > 0:
            state["cool"] -= 1
            return eta
        if (np.isfinite(mu_w) and mu_w < -(1 - PAR["delta"])
                and np.isfinite(s_w) and s_w > 0.1):
            state["cool"] = PAR["cooldown"]
            return float(np.clip(PAR["c"] / s_w, PAR["eta_min"], eta))
        if (np.isfinite(mu_w) and mu_w > PAR["freeze_mu"]
                and np.isfinite(s_w) and abs(s_w) < PAR["freeze_s"]):
            return eta                      # creep regime: hold
        return min(eta + PAR["alpha"], PAR["eta_max"])
    return f


def run_ladder(S, vf, iters):
    """Ladder needs to observe amp; small dedicated loop."""
    ctrl = ctrl_ladder()
    x = vf * np.ones(S["nel"])
    eta_t = np.empty(iters); amp = np.empty(iters)
    comp = np.empty(iters); dzn = np.empty(iters)
    eta = 0.5
    for t in range(iters):
        xn, logD, c = step_full_c(x, vf, eta, S)
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(xn - x) < MOVE - 1e-6))
        eta_t[t] = eta; amp[t] = np.mean(np.abs(xn - x))
        comp[t] = c; dzn[t] = np.linalg.norm(dz[inter]) if inter.sum() else 0.0
        ctrl.observe(t, amp[t])
        eta = ctrl(t + 1, np.nan, np.nan, eta)
        x = xn
    return dict(eta=eta_t, amp=amp, comp=comp, dzn=dzn, x=x, n_md=0)


def main(iters=200):
    print(f"=== Cycle B: AIMD (c={PAR['c']}, delta={PAR['delta']}, "
          f"alpha={PAR['alpha']}, warmup={PAR['warmup']}) ===")
    out = {}
    for cfg, rmin in (("P1_r1.1", 1.1), ("P2_r2.4", 2.4)):
        S = setup(60, 20, rmin, "mbb")
        for mode in ("fixed", "aimd"):
            for e0 in (0.3, 0.5, 1.0):
                key = f"{cfg}_{mode}{e0}"
                ctrl = ctrl_fixed(e0) if mode == "fixed" else ctrl_aimd(e0)
                r = _loop(S, 0.5, iters, ctrl)
                out[key] = r
                tail = float(np.median(r["amp"][-30:]))
                cend = c5.compliance(r["x"], S, "sens")
                c200 = r["comp"][min(199, iters - 1)]
                print(f"  {key:20s}: comp@200={c200:8.2f} @{iters}={cend:8.2f}  "
                      f"amp_tail={tail:.1e}  eta_end={r['eta'][-1]:.3f}  "
                      f"n_MD={r['n_md']}"
                      f"  [{'osc' if tail > 3e-3 else 'conv'}]", flush=True)
        key = f"{cfg}_ladder"
        r = run_ladder(S, 0.5, iters)
        out[key] = r
        tail = float(np.median(r["amp"][-30:]))
        cend = c5.compliance(r["x"], S, "sens")
        print(f"  {key:20s}: comp@{iters}={cend:8.2f}  amp_tail={tail:.1e}  "
              f"eta_end={r['eta'][-1]:.3f}  [{'osc' if tail > 3e-3 else 'conv'}]",
              flush=True)
    save = {}
    for k, r in out.items():
        for name in ("eta", "amp", "comp", "dzn", "x"):
            save[f"{k}_{name}"] = r[name]
        save[f"{k}_nmd"] = r["n_md"]
    np.savez(DATA / "cb_aimd.npz", keys=list(out.keys()), **save)
    print("saved data/cb_aimd.npz")


if __name__ == "__main__":
    t0 = time.time()
    main(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    print(f"total {time.time()-t0:.0f}s")

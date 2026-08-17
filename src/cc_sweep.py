# -*- coding: utf-8 -*-
"""Cycle C sweep: tuning failure rate across configurations.

11 (config, p) groups x 6 methods, 400-iteration budget, AIMD constants
frozen at the cycle-B set (PAR in cb_aimd.py -- single set, no per-problem
tuning; that is the claim under test).

Methods: fixed eta in {0.5, 0.7, 1.0}; AIMD from {0.5, 1.0};
manual ladder from 1.0 (practitioner model: check every 50 iterations,
step eta down by 0.2 while oscillating, floor 0.3).

Failure definitions (reported separately):
  osc-fail     median step amplitude over the last 30 iterations > 3e-3
  quality-fail compliance at budget > 1.02 x best-of-group

Usage: python src/cc_sweep.py
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

CONFIGS = [
    ("mbb_r1.1",    dict(nelx=60, nely=20, rmin=1.1, bc="mbb",
                         filt="sens", vf=0.5), (3.0, 5.0)),
    ("mbb_r2.4",    dict(nelx=60, nely=20, rmin=2.4, bc="mbb",
                         filt="sens", vf=0.5), (3.0, 5.0)),
    ("cant_r1.1",   dict(nelx=60, nely=20, rmin=1.1, bc="cantilever",
                         filt="sens", vf=0.5), (3.0, 5.0)),
    ("mbb_vf0.4",   dict(nelx=60, nely=20, rmin=1.1, bc="mbb",
                         filt="sens", vf=0.4), (3.0, 5.0)),
    ("mbb_dens2.4", dict(nelx=60, nely=20, rmin=2.4, bc="mbb",
                         filt="dens", vf=0.5), (3.0, 5.0)),
    ("mbb120_r2.2", dict(nelx=120, nely=40, rmin=2.2, bc="mbb",
                         filt="sens", vf=0.5), (3.0,)),
]


def ctrl_ladder10():
    state = dict(eta=1.0, amps=[])
    def f(t, mu_w, s_w, eta, amp_w):
        if t == 0:
            return 1.0
        return state["eta"]
    def observe(t, amp):
        state["amps"].append(amp)
        if t > 0 and t % 50 == 0 and \
                np.median(state["amps"][-30:]) > 3e-3:
            state["eta"] = max(state["eta"] - 0.2, 0.3)
    f.observe = observe
    return f


def run(S, vf, filt, p, iters, ctrl, seed=None):
    core.P = p
    x = vf * np.ones(S["nel"])
    if seed is not None:                     # perturbed restart (step 3)
        rng = np.random.default_rng(seed)
        d = rng.normal(0, 0.02, x.size)
        x = np.clip(x * np.exp(d - d.mean()), XMIN, 1.0)
    eta_t = np.empty(iters)
    amp = np.empty(iters)
    mu_b, s_b = [], []
    mu_hist = []                              # for the masked-osc check
    dz_p = m_p = logD_p = None
    eta = ctrl(0, np.nan, np.nan, None, np.nan)
    n_md = 0
    for t in range(iters):
        xn, logD = step_full(x, vf, eta, S, filt)
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(xn - x) < MOVE - 1e-6))
        eta_t[t] = eta
        amp[t] = np.mean(np.abs(xn - x))
        if hasattr(ctrl, "observe"):
            ctrl.observe(t, amp[t])
        if dz_p is not None:
            mu_i, s_i = _inst(dz_p, dz, logD, logD_p, m_p & inter)
            mu_b.append(mu_i)
            s_b.append(s_i)
            mu_hist.append(mu_i)
        w = PAR["window"]
        mu_w = np.nanmedian(mu_b[-w:]) if mu_b else np.nan
        s_w = np.nanmedian(s_b[-w:]) if s_b else np.nan
        amp_w = float(np.median(amp[max(0, t - w + 1):t + 1]))
        eta2 = ctrl(t + 1, mu_w, s_w, eta, amp_w)
        if eta2 < eta - 1e-12:
            n_md += 1
        eta = eta2
        dz_p, m_p, logD_p = dz, inter, logD
        x = xn
    comp = c5.compliance(x, S, filt)
    core.P = 3.0
    mu_tail = float(np.nanmedian(mu_hist[-40:])) if mu_hist else np.nan
    return dict(eta=eta_t, amp=amp, comp=comp, x=x, n_md=n_md,
                mu_tail=mu_tail)


METHODS = [
    ("fix0.5", lambda: ctrl_fixed(0.5)),
    ("fix0.7", lambda: ctrl_fixed(0.7)),
    ("fix1.0", lambda: ctrl_fixed(1.0)),
    ("aimd0.5", lambda: ctrl_aimd(0.5)),
    ("aimd1.0", lambda: ctrl_aimd(1.0)),
    ("ladder", ctrl_ladder10),
]


def main(iters=400):
    print(f"=== Cycle C sweep: {len(METHODS)} methods x "
          f"{sum(len(ps) for _, _, ps in CONFIGS)} groups, "
          f"budget {iters}, AIMD constants frozen (c={PAR['c']}) ===")
    res = {}
    for name, kw, ps in CONFIGS:
        S = setup(kw["nelx"], kw["nely"], kw["rmin"], kw["bc"])
        for p in ps:
            grp = f"{name}_p{p:.0f}"
            t0 = time.time()
            for mname, mk in METHODS:
                r = run(S, kw["vf"], kw["filt"], p, iters, mk())
                res[f"{grp}_{mname}"] = r
            comps = {m: res[f'{grp}_{m}']["comp"] for m, _ in METHODS}
            best = min(comps.values())
            line = []
            for mname, _ in METHODS:
                r = res[f"{grp}_{mname}"]
                tail = float(np.median(r["amp"][-30:]))
                osc = tail > 3e-3
                dq = 100 * (r["comp"] / best - 1)
                r.update(tail=tail, osc=osc, dq=dq, best=best)
                line.append(f"{mname}:{r['comp']:.1f}"
                            f"({'OSC' if osc else f'+{dq:.1f}%'})")
            print(f"  {grp:18s} [{time.time()-t0:4.0f}s] " + "  ".join(line),
                  flush=True)
    # ---- failure table ----
    print("\n--- tuning failure rates over "
          f"{sum(len(ps) for _, _, ps in CONFIGS)} groups ---")
    print(f"{'method':>8} {'osc-fail':>9} {'quality-fail(>2%)':>18} "
          f"{'median dq%':>11}")
    groups = [f"{n}_p{p:.0f}" for n, _, ps in CONFIGS for p in ps]
    for mname, _ in METHODS:
        rows = [res[f"{g}_{mname}"] for g in groups]
        no = sum(r["osc"] for r in rows)
        nq = sum((not r["osc"]) and r["dq"] > 2.0 for r in rows)
        mdq = float(np.median([r["dq"] for r in rows]))
        print(f"{mname:>8} {no:>4}/{len(rows)} {nq:>13}/{len(rows)} "
              f"{mdq:>10.2f}")
    save = {}
    for k, r in res.items():
        save[f"{k}_eta"] = r["eta"]
        save[f"{k}_amp"] = r["amp"]
        save[f"{k}_meta"] = np.array([r["comp"], r["tail"], float(r["osc"]),
                                      r["dq"], r["n_md"], r["eta"][-1],
                                      r["mu_tail"]])
    np.savez(DATA / "cc_sweep.npz", groups=groups,
             methods=[m for m, _ in METHODS], **save)
    print("saved data/cc_sweep.npz")


if __name__ == "__main__":
    t0 = time.time()
    main(iters=int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    print(f"total {time.time()-t0:.0f}s")

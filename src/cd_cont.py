# -*- coding: utf-8 -*-
"""Step 1: p-continuation sweep -- the paper's motivation experiment.

p ramps 1.0 -> 5.0 in 0.25 increments, one increment every N iterations
(fast N=20, budget 600; slow N=40, budget 900), then holds at p=5.
The theoretical flip threshold of the naturally saturated family falls as
eta*(t) = 2/(p(t)+1): from 1.0 (p=1, where fixed 0.5 is 2x over-damped)
through 0.5 exactly at p=3, to 0.333 at p=5 (where fixed 0.5 is above
threshold unless the design self-limits).

Configs: R0 (MBB r1.1), hard filter (MBB r2.4), cantilever r1.1.
Methods: fixed {0.3, 0.5, 0.7}, ladder(1.0), AIMD(0.5), AIMD(1.0)
         (controller constants frozen -- cb_aimd.PAR).
Measured: eta(t), compliance(t), gray fraction(t), amplitude(t),
endpoint s_max (exact projected J at the final design), iterations to
amp<1e-3 after ramp end.

Usage: python src/cd_cont.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, build_J, XMIN, MOVE
import c5_core as c5
from ca_gate import step_full
from cb_aimd import PAR, ctrl_fixed, ctrl_aimd, _inst

CONFIGS = [
    ("mbb_r1.1",  dict(rmin=1.1, bc="mbb")),
    ("mbb_r2.4",  dict(rmin=2.4, bc="mbb")),
    ("cant_r1.1", dict(rmin=1.1, bc="cantilever")),
]
CONFIGS_EXT = CONFIGS + [
    ("mbb_vf0.4", dict(rmin=1.1, bc="mbb", vf=0.4)),
]
SCHED = {"fast": (20, 600), "slow": (40, 900)}
P_LO, P_HI, P_STEP = 1.0, 5.0, 0.25


def ctrl_ladder10():
    state = dict(eta=1.0, amps=[])
    def f(t, mu_w, s_w, eta, amp_w):
        return 1.0 if t == 0 else state["eta"]
    def observe(t, amp):
        state["amps"].append(amp)
        if t > 0 and t % 50 == 0 and np.median(state["amps"][-30:]) > 3e-3:
            state["eta"] = max(state["eta"] - 0.2, 0.3)
    f.observe = observe
    return f


def p_of(t, N):
    return min(P_HI, P_LO + P_STEP * (t // N))


def run(S, vf, N, iters, ctrl, seed=None, light=False):
    """seed: perturbed restart (2% log-normal, mean-centered).
    light: skip per-step compliance and the endpoint spectrum (seed
    replication only needs the oscillation tag and final compliance)."""
    x = vf * np.ones(S["nel"])
    if seed is not None:
        rng = np.random.default_rng(seed)
        d = rng.normal(0, 0.02, x.size)
        x = np.clip(x * np.exp(d - d.mean()), XMIN, 1.0)
    eta_t = np.empty(iters)
    p_t = np.empty(iters)
    amp = np.empty(iters)
    comp = np.empty(iters)
    gray = np.empty(iters)
    mu_b, s_b = [], []
    dz_p = m_p = logD_p = None
    eta = ctrl(0, np.nan, np.nan, None, np.nan)
    n_md = 0
    for t in range(iters):
        core.P = p_of(t, N)
        comp[t] = np.nan if light else c5.compliance(x, S, "sens")
        xn, logD = step_full(x, vf, eta, S, "sens")
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(xn - x) < MOVE - 1e-6))
        eta_t[t] = eta
        p_t[t] = core.P
        amp[t] = np.mean(np.abs(xn - x))
        gray[t] = np.mean((x > 0.05) & (x < 0.95))
        if hasattr(ctrl, "observe"):
            ctrl.observe(t, amp[t])
        if dz_p is not None:
            mu_i, s_i = _inst(dz_p, dz, logD, logD_p, m_p & inter)
            mu_b.append(mu_i)
            s_b.append(s_i)
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
    # endpoint spectrum at p = P_HI (exact projection; tagged by tail amp)
    core.P = P_HI
    smax_end = np.nan
    if not light:
        try:
            J, F = build_J(x, S, vf, "sens")
            s, _, _ = c5.tangent_spectrum(J, x, F, S, "sens")
            smax_end = float(np.max(s.real))
        except Exception:
            pass
    if light:
        comp[-1] = c5.compliance(x, S, "sens")
    core.P = 3.0
    ramp_end = int(np.ceil((P_HI - P_LO) / P_STEP)) * N
    tail = float(np.median(amp[-30:]))
    tc = -1
    post = amp[ramp_end:]
    ok = post < 1e-3
    for tt in range(len(post) - 10):
        if ok[tt:tt + 10].all():
            tc = tt
            break
    return dict(eta=eta_t, p=p_t, amp=amp, comp=comp, gray=gray, x=x,
                n_md=n_md, smax_end=smax_end, tail=tail, tconv=tc,
                ramp_end=ramp_end)


METHODS = [
    ("fix0.3", lambda: ctrl_fixed(0.3)),
    ("fix0.5", lambda: ctrl_fixed(0.5)),
    ("fix0.7", lambda: ctrl_fixed(0.7)),
    ("ladder", ctrl_ladder10),
    ("aimd0.5", lambda: ctrl_aimd(0.5)),
    ("aimd1.0", lambda: ctrl_aimd(1.0)),
]


def main():
    print("=== Step 1: p-continuation 1->5 (0.25 steps; fast N=20/600, "
          f"slow N=40/900; AIMD constants frozen, c={PAR['c']}) ===")
    res = {}
    for sname, (N, iters) in SCHED.items():
        for cname, kw in CONFIGS:
            S = setup(60, 20, kw["rmin"], kw["bc"])
            grp = f"{cname}_{sname}"
            t0 = time.time()
            for mname, mk in METHODS:
                r = run(S, 0.5, N, iters, mk())
                res[f"{grp}_{mname}"] = r
            best = min(res[f"{grp}_{m}"]["comp"][-1] for m, _ in METHODS)
            line = []
            for mname, _ in METHODS:
                r = res[f"{grp}_{mname}"]
                osc = r["tail"] > 3e-3
                dq = 100 * (r["comp"][-1] / best - 1)
                r.update(osc=osc, dq=dq)
                line.append(f"{mname}:{r['comp'][-1]:.0f}"
                            f"({'OSC' if osc else f'+{dq:.1f}%'})"
                            f"/s{r['smax_end']:.1f}")
            print(f"  {grp:16s} [{time.time()-t0:3.0f}s] " + " ".join(line),
                  flush=True)
    save = {}
    for k, r in res.items():
        for name in ("eta", "p", "amp", "comp", "gray"):
            save[f"{k}_{name}"] = r[name]
        save[f"{k}_meta"] = np.array([r["comp"][-1], r["tail"],
                                      float(r["osc"]), r["dq"], r["n_md"],
                                      r["eta"][-1], r["smax_end"],
                                      r["tconv"], r["ramp_end"]])
    np.savez(DATA / "cd_cont.npz",
             groups=[f"{c}_{s}" for s in SCHED for c, _ in CONFIGS],
             methods=[m for m, _ in METHODS], **save)
    print("saved data/cd_cont.npz")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time()-t0:.0f}s")

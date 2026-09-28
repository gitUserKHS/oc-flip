# -*- coding: utf-8 -*-
"""Cycle G1 / paper-1 v2: masked-oscillation tags for the continuation suite.

Re-runs the continuation suite of cd_cont.py / cd_seeds.py (CONFIGS_EXT x
SCHED x starts: nominal + seeds 5/7/11) for all six METHODS, recording the
tail multiplier so that masked oscillation can be tagged with the same
definition as Table 4 (cc_sweep):
    osc    : median step amplitude over the last 30 iterations > 3e-3
    masked : not osc, amplitude in (1e-5, 3e-3) and tail mu^ <= -0.9
             (tail mu^ = nanmedian of the last 40 instantaneous mu^)
run() is cd_cont.run in light mode plus the mu^ tail; the trajectory,
controller and compliance evaluation are unchanged (final compliance is
that of the final design at p = 5, as in cd_seeds light mode).

Usage: python src/g1_cont_masked.py [method,method,...]   (default: all)
       python src/g1_cont_masked.py merge
       python src/g1_cont_masked.py ride    (boundary-riding numbers)
Output: data/g1_cont_masked_<methods>.json, merged into
        data/g1_cont_masked.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, XMIN, MOVE
import c5_core as c5
from ca_gate import step_full
from cb_aimd import PAR, _inst
from cd_cont import CONFIGS, CONFIGS_EXT, SCHED, METHODS, p_of, P_HI, P_LO, \
    P_STEP

SEEDS = (5, 7, 11)


def run(S, vf, N, iters, ctrl, seed=None):
    x = vf * np.ones(S["nel"])
    if seed is not None:
        rng = np.random.default_rng(seed)
        d = rng.normal(0, 0.02, x.size)
        x = np.clip(x * np.exp(d - d.mean()), XMIN, 1.0)
    amp = np.empty(iters)
    mu_b, s_b = [], []
    dz_p = m_p = logD_p = None
    eta = ctrl(0, np.nan, np.nan, None, np.nan)
    for t in range(iters):
        core.P = p_of(t, N)
        xn, logD = step_full(x, vf, eta, S, "sens")
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(xn - x) < MOVE - 1e-6))
        amp[t] = np.mean(np.abs(xn - x))
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
        eta = ctrl(t + 1, mu_w, s_w, eta, amp_w)
        dz_p, m_p, logD_p = dz, inter, logD
        x = xn
    core.P = P_HI
    comp = float(c5.compliance(x, S, "sens"))
    core.P = 3.0
    tail = float(np.median(amp[-30:]))
    last = np.asarray(mu_b[-40:], float)
    mu_tail = float(np.nanmedian(last)) if np.isfinite(last).any() else np.nan
    osc = tail > 3e-3
    masked = (not osc) and np.isfinite(mu_tail) and mu_tail <= -0.9 \
        and 1e-5 < tail < 3e-3
    return dict(comp=comp, tail=tail, mu_tail=mu_tail, osc=bool(osc),
                masked=bool(masked))


def main(methods):
    out = []
    t0 = time.time()
    for name, kw in CONFIGS_EXT:
        S = setup(60, 20, kw["rmin"], kw["bc"])
        for sname, (N, iters) in SCHED.items():
            for st in (None,) + SEEDS:
                for mname, mk in METHODS:
                    if mname not in methods:
                        continue
                    r = run(S, kw.get("vf", 0.5), N, iters, mk(), seed=st)
                    r.update(config=name, sched=sname,
                             start="n" if st is None else f"s{st}",
                             method=mname)
                    out.append(r)
                    print(json.dumps(r), flush=True)
    tag = "_".join(methods)
    (DATA / f"g1_cont_masked_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"saved data/g1_cont_masked_{tag}.json  ({time.time()-t0:.0f}s)")


def merge():
    uniq = {}
    for f in sorted(DATA.glob("g1_cont_masked_*.json")):
        for r in json.loads(f.read_text()):
            uniq[(r["config"], r["sched"], r["start"], r["method"])] = r
    rows = list(uniq.values())
    (DATA / "g1_cont_masked.json").write_text(json.dumps(rows, indent=1))
    base = {n for n, _ in CONFIGS}
    print(f"merged {len(rows)} runs -> data/g1_cont_masked.json")
    # suite totals: the 32 runs of the paper = nominal of CONFIGS + seeds of
    # CONFIGS + nominal and seeds of mbb_vf0.4 (i.e. every row)
    print("suite (32 runs per method): osc / masked / failing")
    for m, _ in METHODS:
        rr = [r for r in rows if r["method"] == m]
        o = sum(r["osc"] for r in rr)
        k = sum(r["masked"] for r in rr)
        print(f"  {m:8s} n={len(rr):2d}  osc {o:2d}  masked {k:2d}  "
              f"fail {o + k:2d}")
    # Table tab:cont (nominal starts): tags and dq vs best settled run
    nom = np.load(DATA / "cd_cont.npz")
    print("tab:cont groups (nominal): tag per method, dq vs best settled")
    dq = {m: [] for m, _ in METHODS}
    nmask = {m: 0 for m, _ in METHODS}
    for g in nom["groups"]:
        cfg, sched = str(g).rsplit("_", 1)
        tag = {r["method"]: ("osc" if r["osc"] else
                             "msk" if r["masked"] else "ok")
               for r in rows if r["config"] == cfg and r["sched"] == sched
               and r["start"] == "n"}
        comp = {m: float(nom[f"{g}_{m}_meta"][0]) for m, _ in METHODS}
        best = min(c for m, c in comp.items() if tag[m] == "ok")
        cells = []
        for m, _ in METHODS:
            dq[m].append(100 * (comp[m] / best - 1))
            nmask[m] += tag[m] == "msk"
            cells.append(f"{m}:{comp[m]:.1f}/{tag[m]}")
        print(f"  {str(g):16s} best_settled={best:.2f}  " + " ".join(cells))
    for m, _ in METHODS:
        print(f"  {m:8s} masked {nmask[m]}/6  median dq {np.median(dq[m]):+.2f}%"
              f"  worst dq {max(dq[m]):+.2f}%")


def ride():
    """Boundary riding at the p=5 hold (paper Section 6.3, closing
    paragraph) and the fixed-0.5 ceiling endpoints, from data/cd_cont.npz."""
    d = np.load(DATA / "cd_cont.npz")
    for g in ("mbb_r1.1_fast", "mbb_r1.1_slow"):
        eta = d[f"{g}_aimd1.0_eta"]
        amp = d[f"{g}_aimd1.0_amp"]
        sm = float(d[f"{g}_aimd1.0_meta"][6])
        w = eta[-200:]
        print(f"  {g} AIMD(1.0): s_max_end {sm:.3f}  median eta last100 "
              f"{np.median(eta[-100:]):.4f}  median eta*s_max last200 "
              f"{np.median(w) * sm:.3f}  eta*s_max range [{w.min() * sm:.2f},"
              f" {w.max() * sm:.2f}]  frac eta > 2/s_max {np.mean(w > 2 / sm):.2f}"
              f"  final tail amp {np.median(amp[-30:]):.1e}")
        sm5 = float(d[f"{g}_fix0.5_meta"][6])
        print(f"  {g} fixed 0.5: s_max_end {sm5:.3f}  eta*s_max {0.5 * sm5:.3f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ride":
        ride()
    elif len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge()
    else:
        ms = sys.argv[1].split(",") if len(sys.argv) > 1 else \
            [m for m, _ in METHODS]
        main(ms)

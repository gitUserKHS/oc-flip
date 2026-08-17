# -*- coding: utf-8 -*-
"""Step 2: three defense experiments.

movelimit  (i)   adaptive move-limit baseline (MMA-style asymptote rule
                 transplanted to the global move limit): if the fraction of
                 interior elements whose consecutive design changes flip
                 sign exceeds 0.35, move <- move*0.7, else move*1.2
                 (bounds [0.01, 0.2]); eta held fixed. Theory prediction:
                 move limits bound amplitude but do not change the
                 multiplier -- oscillation is masked, not removed. We
                 report both the amplitude tag and the tail multiplier.
endpoints  (ii)  exact-projected s_max at the six cycle-B controller
                 endpoints: does the landed eta sit below each endpoint's
                 OWN threshold 2/s_max?
truncation (iii) truncated Richardson: eta_opt = 2/(a_cut + b) with the
                 lower edge a_cut = min{s > cut}, cut in {0, 0.1, 0.3,
                 0.5}, over all 18 measured spectra (zero new physics).

Usage: python src/ce_defense.py all
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
from c4_core import setup, build_J, XMIN, E0, Emin
import c5_core as c5
from cb_aimd import _inst
from cc_sweep import CONFIGS
from ca_gate import _spec_sources

FLIP_FRAC, MV_DN, MV_UP, MV_LO, MV_HI = 0.35, 0.7, 1.2, 0.01, 0.2


def step_mv(x, volfrac, eta, S, filt, move):
    """ca_gate.step_full with a variable move limit."""
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
        dvf = np.ones_like(x)
    else:
        dcf = np.asarray(S["H"] @ (dc / S["Hs"])).ravel()
        dvf = np.asarray(S["H"] @ (np.ones_like(x) / S["Hs"])).ravel()
    l1, l2 = 0.0, 1e9
    lo = np.maximum(XMIN, x - move)
    hi = np.minimum(1.0, x + move)
    while (l2 - l1) / (l1 + l2) > 1e-10:
        lm = 0.5 * (l1 + l2)
        B = np.maximum(1e-30, -dcf / (dvf * lm))
        xn = np.clip(x * B**eta, lo, hi)
        vol = (np.asarray(S["H"] @ xn).ravel() / S["Hs"]).mean() \
            if filt == "dens" else xn.mean()
        if vol > volfrac:
            l1 = lm
        else:
            l2 = lm
    return xn, np.log(np.maximum(1e-300, -dcf))


def run_mv(S, vf, filt, p, iters, eta):
    core.P = p
    x = vf * np.ones(S["nel"])
    move = MV_HI
    amp = np.empty(iters)
    mv_t = np.empty(iters)
    mu_b = []
    dx_p = dz_p = m_p = logD_p = None
    for t in range(iters):
        xn, logD = step_mv(x, vf, eta, S, filt, move)
        dx = xn - x
        dz = np.log(xn) - np.log(x)
        inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                 (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                 (np.abs(dx) < move - 1e-6))
        amp[t] = np.mean(np.abs(dx))
        mv_t[t] = move
        if dx_p is not None:
            m = inter & (np.abs(dx) > 1e-12) & (np.abs(dx_p) > 1e-12)
            frac = float(np.mean(dx[m] * dx_p[m] < 0)) if m.sum() >= 20 \
                else 0.0
            move = max(move * MV_DN, MV_LO) if frac > FLIP_FRAC \
                else min(move * MV_UP, MV_HI)
            mu_i, _ = _inst(dz_p, dz, logD, logD_p, m_p & inter)
            mu_b.append(mu_i)
        dx_p, dz_p, m_p, logD_p = dx, dz, inter, logD
        x = xn
    comp = c5.compliance(x, S, filt)
    core.P = 3.0
    mu_tail = float(np.nanmedian(mu_b[-40:])) if mu_b else np.nan
    return dict(comp=comp, tail=float(np.median(amp[-30:])),
                mv_end=float(np.median(mv_t[-30:])), mu_tail=mu_tail)


def movelimit(iters=400):
    print("=== (i) adaptive move-limit baseline (eta fixed; flip-fraction "
          f"rule x{MV_DN}/x{MV_UP}, threshold {FLIP_FRAC}) ===")
    sw = np.load(DATA / "cc_sweep.npz", allow_pickle=True)
    out = {}
    for eta in (0.5, 1.0):
        n_osc = n_masked = n_qual = 0
        dqs = []
        for name, kw, ps in CONFIGS:
            S = setup(kw["nelx"], kw["nely"], kw["rmin"], kw["bc"])
            for p in ps:
                grp = f"{name}_p{p:.0f}"
                best = min(sw[f"{grp}_{m}_meta"][0]
                           for m in sw["methods"])
                r = run_mv(S, kw["vf"], kw["filt"], p, iters, eta)
                dq = 100 * (r["comp"] / best - 1)
                osc = r["tail"] > 3e-3
                masked = (not osc) and np.isfinite(r["mu_tail"]) \
                    and r["mu_tail"] < -0.75
                n_osc += osc
                n_masked += masked
                n_qual += (not osc) and dq > 2.0
                dqs.append(dq)
                out[f"mv{eta}_{grp}"] = np.array(
                    [r["comp"], r["tail"], float(osc), dq, r["mv_end"],
                     r["mu_tail"], float(masked)])
                print(f"  mv eta={eta} {grp:16s}: comp={r['comp']:7.1f} "
                      f"({dq:+5.1f}%) tail={r['tail']:.1e} "
                      f"move_end={r['mv_end']:.3f} mu_tail={r['mu_tail']:+.2f}"
                      f" [{'OSC' if osc else ('MASKED' if masked else 'conv')}]",
                      flush=True)
        print(f"  => eta={eta}: osc {n_osc}/11, masked-osc {n_masked}/11, "
              f"quality {n_qual}/11, median dq {np.median(dqs):+.2f}%, "
              f"worst {max(dqs):+.1f}%")
    np.savez(DATA / "ce_movelimit.npz", **out)
    print("saved data/ce_movelimit.npz")


def endpoints():
    print("=== (ii) controller endpoints: eta_end vs OWN threshold ===")
    d = np.load(DATA / "cb_aimd.npz", allow_pickle=True)
    for cfg, rmin in (("P1_r1.1", 1.1), ("P2_r2.4", 2.4)):
        S = setup(60, 20, rmin, "mbb")
        for e0 in (0.3, 0.5, 1.0):
            key = f"{cfg}_aimd{e0}"
            x = d[f"{key}_x"]
            eta_end = float(d[f"{key}_eta"][-1])
            J, F = build_J(x, S, 0.5, "sens")
            s, _, _ = c5.tangent_spectrum(J, x, F, S, "sens")
            smax = float(np.max(s.real))
            thr = 2.0 / smax
            print(f"  {key:16s}: eta_end={eta_end:.3f}  s_max(end)={smax:.3f}"
                  f"  own 2/s_max={thr:.3f}  ratio={eta_end/thr:.2f}"
                  f"  [{'BELOW' if eta_end < thr else 'ABOVE'}]", flush=True)


def truncation():
    print("=== (iii) truncated Richardson over the 18 measured spectra ===")
    cuts = [0.0, 0.1, 0.3, 0.5]
    rows = {c: [] for c in cuts}
    r0 = {}
    for name, s in _spec_sources():
        sr = np.real(s)
        b = float(np.max(sr[sr > 1e-6]))
        for c in cuts:
            pos = sr[sr > max(c, 1e-6)]
            a = float(np.min(pos))
            eo = 2.0 / (a + b)
            rows[c].append(eo)
            if name == "p3.0_ref":
                r0[c] = (a, eo)
    print(f"{'cut':>5} {'median eta_opt':>15} {'range':>16} "
          f"{'R0: a_cut':>10} {'R0 eta_opt':>10}")
    for c in cuts:
        v = np.array(rows[c])
        print(f"{c:5.1f} {np.median(v):15.3f} "
              f"[{v.min():.3f}, {v.max():.3f}] "
              f"{r0[c][0]:10.3f} {r0[c][1]:10.3f}", flush=True)
    np.savez(DATA / "ce_truncation.npz",
             cuts=np.array(cuts),
             **{f"cut{c:.1f}": np.array(rows[c]) for c in cuts})
    print("saved data/ce_truncation.npz")


if __name__ == "__main__":
    t0 = time.time()
    for a in (sys.argv[1:] or ["all"]):
        if a == "all":
            movelimit(); endpoints(); truncation()
        else:
            {"movelimit": movelimit, "endpoints": endpoints,
             "truncation": truncation}[a]()
    print(f"total {time.time()-t0:.0f}s")

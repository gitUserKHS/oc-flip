# -*- coding: utf-8 -*-
"""Cycle 5 driver.  Subcommands (run from anywhere; paths are repo-relative):

  tangent    A:  exact Q^T J Q spectrum vs cycle-4 heuristic on V3/V4
  negbranch  B:  long-run termination of the V3 negative branch (s_min mode)
  psweep     C:  full p-sweep, exact projection + behavioral onset brackets
  p5rmin     C2: p=5 mechanism - gray-layer thickness vs filter radius
  v5seeds    D:  density-filter no-flip confirmation, multi-seed + probe
  all        everything above, in order

Usage: python src/c5_run.py all
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, converge, build_J, spectrum, free_set, XMIN
import c5_core as c5


def load_ck(key):
    d = np.load(DATA / f"c4_{key}.npz", allow_pickle=True)
    S = setup(int(d["nelx"]), int(d["nely"]), float(d["rmin"]), str(d["bc"]))
    return d, S, float(d["vf"]), str(d["filt"])


# ---------------------------------------------------------------- A: tangent
def tangent():
    print("=== A: exact tangent projection vs heuristic (V3, V4) ===")
    for key in ["V3", "V4"]:
        d, S, vf, filt = load_ck(key)
        x0 = d["x0"]
        t0 = time.time()
        J, F = build_J(x0, S, vf, filt)
        s_h, _ = spectrum(J)                       # cycle-4 heuristic
        s_e, _, rnull = c5.tangent_spectrum(J, x0, F, S, filt)  # exact
        # raw spectrum of unprojected J: locate the constraint artifact
        mu_raw = np.linalg.eigvals(J)
        s_raw = np.sort_complex((1.0 - mu_raw) / 0.5)[::-1]
        top_h = np.real(s_h[:6])
        top_e = np.real(s_e[:6])
        dmax = float(np.max(np.abs(top_h - top_e)))
        smin_h = float(np.min(np.real(s_h)))
        smin_e = float(np.min(np.real(s_e)))
        # nearest raw eigenvalue to s=2 (mu=0 left-null artifact at eta=.5)
        k2 = int(np.argmin(np.abs(s_raw - 2.0)))
        print(f"{key}: |F|={F.size}  rnull={rnull:.2e}  "
              f"raw-mode nearest s=2: {s_raw[k2]:.6f}")
        print(f"   top6 heur : {np.round(top_h, 4)}")
        print(f"   top6 exact: {np.round(top_e, 4)}   max|d|={dmax:.1e}")
        print(f"   s_min heur/exact: {smin_h:.5f} / {smin_e:.5f}  "
              f"(d={abs(smin_h-smin_e):.1e})   [{time.time()-t0:.0f}s]",
              flush=True)
        np.savez(DATA / f"c5_tangent_{key}.npz", s_heur=s_h, s_exact=s_e,
                 rnull=rnull, s_raw=s_raw)


# ------------------------------------------------------------- B: negbranch
def negbranch(T=3000, delta=0.005, eta=0.5):
    print("=== B: V3 negative-branch termination run ===")
    d, S, vf, filt = load_ck("V3")
    x0 = d["x0"]
    t0 = time.time()
    J, F = build_J(x0, S, vf, filt)
    s, modes, rnull = c5.tangent_spectrum(J, x0, F, S, filt)
    k = int(np.argmin(s.real))
    s_min = complex(s[k])
    v = np.real(modes[:, k])
    v /= np.linalg.norm(v)
    mu_pred = 1.0 - eta * s_min.real
    print(f"exact-projected s_min = {s_min.real:.5f} (Im {s_min.imag:.1e}), "
          f"predicted growth mu = {mu_pred:.4f}  rnull={rnull:.1e}")

    # mode support stats (low-density character)
    sup = np.argsort(-np.abs(v))[:20]
    print(f"top-20 support: mean x0 = {x0[F][sup].mean():.4f}, "
          f"frac(|v|^2) at x<0.05 = {np.sum(v[x0[F] < 0.05]**2):.2f}")

    vhat = v - v.mean()
    vhat /= np.linalg.norm(vhat)
    lx0 = np.log(x0[F])
    x = x0.copy()
    x[F] = x0[F] * np.exp(delta * v)
    x = np.clip(x, XMIN, 1.0)
    n_abs0 = int(np.sum(x0 <= XMIN + 1e-9))
    track_el = F[np.argsort(-np.abs(v))[:6]]   # mirror-pair winner/loser sets
    track_x = np.empty((T, track_el.size))
    b = np.empty(T)
    n_abs = np.empty(T, int)
    n_hi = np.empty(T, int)
    amp = np.empty(T)
    comp_t, comp_v = [], []
    for t in range(T):
        xn = core.step(x, vf, eta, S, filt)
        z = np.log(np.maximum(x[F], XMIN)) - lx0
        zc = z - z.mean()
        b[t] = float(zc @ vhat)
        n_abs[t] = int(np.sum(x <= XMIN + 1e-9)) - n_abs0
        n_hi[t] = int(np.sum(x >= 1 - 1e-9))
        amp[t] = float(np.mean(np.abs(xn - x)))
        track_x[t] = x[track_el]
        if t % 25 == 0:
            comp_t.append(t)
            comp_v.append(c5.compliance(x, S, filt))
        x = xn
    comp_t.append(T)
    comp_v.append(c5.compliance(x, S, filt))

    # measured linear-regime multiplier: ratios while |b| < 5x initial
    b0 = abs(b[2])
    lin = [b[t + 1] / b[t] for t in range(2, T - 1)
           if abs(b[t]) < 5 * b0 and abs(b[t]) > 1e-14 and t < 200]
    mu_meas = float(np.median(lin)) if lin else np.nan
    tpk = int(np.argmax(np.abs(b)))
    print(f"linear regime: mu measured = {mu_meas:.4f} vs predicted "
          f"{mu_pred:.4f}  (n={len(lin)})")
    print(f"peak |b| = {np.max(np.abs(b)):.3f} at t={tpk}; "
          f"absorbed to XMIN: {n_abs[-1]} elements "
          f"(first at t={int(np.argmax(n_abs > 0)) if n_abs.max() > 0 else -1})")
    print(f"amp: start {amp[0]:.1e} -> last500 {np.mean(amp[-500:]):.1e}; "
          f"compliance {comp_v[0]:.2f} -> {comp_v[-1]:.2f}")

    # post-termination spectrum
    J2, F2 = build_J(x, S, vf, filt)
    s2, _, rnull2 = c5.tangent_spectrum(J2, x, F2, S, filt)
    print(f"post-run: |F| {F.size} -> {F2.size}, "
          f"s_max {np.max(s.real):.3f} -> {np.max(s2.real):.3f}, "
          f"s_min {s_min.real:.5f} -> {np.min(s2.real):.5f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    np.savez(DATA / "c5_negbranch.npz", b=b, n_abs=n_abs, n_hi=n_hi, amp=amp,
             comp_t=comp_t, comp_v=comp_v, s_before=s, s_after=s2,
             s_min=s_min, mu_pred=mu_pred, mu_meas=mu_meas, x_end=x, v=v,
             F=F, F2=F2, delta=delta, eta=eta, rnull=[rnull, rnull2],
             track_el=track_el, track_x=track_x)


# --------------------------------------------------------------- C: psweep
PS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def _converge_ladder(S, vf, filt, iters, eta0_force=None):
    """eta0_force=None: adapted protocol (try 0.5, step down on oscillation).
    eta0_force=value: reference protocol (converge at that damping)."""
    ladder = (eta0_force,) if eta0_force else (0.5, 0.3, 0.2)
    for eta0 in ladder:
        x0, amp0 = converge(S, vf, eta0, filt, iters=iters)
        if amp0 <= 3e-3:
            return x0, amp0, eta0
    return x0, amp0, eta0


def _spectrum_case(p, S, vf, filt, iters, probe_extra=(), tag="",
                   eta0_force=None):
    core.P = p
    t0 = time.time()
    x0, amp0, eta0 = _converge_ladder(S, vf, filt, iters, eta0_force)
    J, F = build_J(x0, S, vf, filt)
    s, modes, rnull = c5.tangent_spectrum(J, x0, F, S, filt)
    smax = float(np.max(s.real))
    smin = float(np.min(s.real))
    ncx = int(np.sum(np.abs(np.imag(s[:6])) > 1e-3))
    eta_star = 2.0 / smax
    vec = np.real(modes[:, 0])
    vec /= np.linalg.norm(vec)
    g = c5.gray_stats(x0)
    etas = np.round(np.clip(eta_star + np.array([-0.12, -0.04, 0.04, 0.12]),
                            0.05, None), 3)
    etas = sorted(set(list(etas) + [round(e, 3) for e in probe_extra]))
    pr, nmid = c5.mode_probe2(x0, vec, F, S, vf, filt, etas)
    lo, hi = c5.onset_bracket(pr)
    core.P = 3.0
    sat = smax / (p + 1.0)
    print(f"{tag}p={p:.1f}: conv@{eta0}(amp {amp0:.0e}) |F|={F.size} "
          f"gray={g['n_gray']} s_max={smax:.3f} sat={sat:.3f} "
          f"eta*={eta_star:.3f} s_min={smin:+.4f} rnull={rnull:.0e} ncx={ncx}")
    for e, gg, rs in pr:
        pred = abs(1.0 - e * smax)
        tag2 = "FLIP" if (rs < 0 and gg >= 1.0) else ("marg" if rs < 0 else "creep")
        print(f"     eta={e:.3f}: |r|={gg:.3f} signed={rs:+.3f} "
              f"pred|1-eta*smax|={pred:.3f} [{tag2}]")
    print(f"     bracket=({lo},{hi}] nmid={nmid} [{time.time()-t0:.0f}s]",
          flush=True)
    return dict(p=p, x0=x0, amp0=amp0, eta0=eta0, F=F, s_top=s[:8],
                s_min=smin, smax=smax, eta_star=eta_star, vec=vec,
                probe=np.array(pr), n_gray=g["n_gray"],
                mass_gray=g["mass_gray"], rnull=rnull,
                bracket=(lo, hi), nmid=nmid)


def psweep(eta0_force=None, suffix=""):
    """eta0_force=None -> adapted protocol (file c5_psweep.npz);
    eta0_force=0.3 -> low-eta reference (file c5_psweep_ref.npz)."""
    lab = f"reference eta0={eta0_force}" if eta0_force else "adapted eta0-ladder"
    print(f"=== C: full p-sweep (60x20 MBB, sens r1.1, vf 0.5) [{lab}] ===")
    S = setup(60, 20, 1.1, "mbb")
    out = {}
    for p in PS:
        iters = 900 if p >= 5.0 else 600
        extra = (0.30, 0.37) if p >= 5.0 else ()   # envelope discriminator
        out[p] = _spectrum_case(p, S, 0.5, "sens", iters, probe_extra=extra,
                                eta0_force=eta0_force)
    save = {}
    for p, r in out.items():
        k = f"p{p:.1f}_"
        for name in ["x0", "s_top", "s_min", "smax", "eta_star", "vec",
                     "probe", "n_gray", "mass_gray", "rnull", "F",
                     "amp0", "eta0", "nmid"]:
            save[k + name] = np.asarray(r[name])
        save[k + "bracket"] = np.array(
            [r["bracket"][0] or np.nan,
             r["bracket"][1] if r["bracket"][1] is not None else np.nan])
    np.savez(DATA / f"c5_psweep{suffix}.npz", ps=np.array(PS), **save)
    print(f"saved data/c5_psweep{suffix}.npz", flush=True)


def psweep_ref():
    psweep(eta0_force=0.3, suffix="_ref")


def p5rmin(eta0_force=None, suffix="", p=5.0):
    lab = f"reference eta0={eta0_force}" if eta0_force else "adapted"
    print(f"=== C2: p={p} mechanism - gray layer vs rmin [{lab}] ===")
    out = {}
    for rmin in [1.5, 2.0, 2.4]:
        S = setup(60, 20, rmin, "mbb")
        out[rmin] = _spectrum_case(p, S, 0.5, "sens", 900,
                                   tag=f"rmin={rmin} ", eta0_force=eta0_force)
    save = {}
    for rmin, r in out.items():
        k = f"r{rmin:.1f}_"
        for name in ["smax", "s_min", "n_gray", "mass_gray", "eta_star",
                     "rnull", "amp0", "eta0"]:
            save[k + name] = np.asarray(r[name])
        save[k + "bracket"] = np.array(
            [r["bracket"][0] or np.nan,
             r["bracket"][1] if r["bracket"][1] is not None else np.nan])
    name = f"c5_p{p:.0f}rmin{suffix}.npz"
    np.savez(DATA / name, rmins=np.array([1.5, 2.0, 2.4]), **save)
    print(f"saved data/{name}", flush=True)


def p5rmin_ref():
    p5rmin(eta0_force=0.3, suffix="_ref")


def p3rmin_ref():
    p5rmin(eta0_force=0.3, suffix="_ref", p=3.0)


# --------------------------------------------------------------- D: v5seeds
def v5seeds():
    print("=== D: V5 density filter - multi-seed no-flip check ===")
    d, S, vf, filt = load_ck("V5")
    x0, vec, F = d["x0"], d["vec"], d["F"]
    print(f"|F|={F.size}  s_max={float(d['smax']):.3f}  "
          f"eta*={float(d['eta_star']):.2f}")
    for eta in [1.0, 2.0, 4.0]:
        amps = [core.cont_amp(x0, eta, S, vf, filt, iters=60, kick=0.02,
                              seed=sd) for sd in (5, 7, 11)]
        print(f"  eta={eta}: cont amp (seeds 5/7/11) = "
              + "/".join(f"{a:.1e}" for a in amps), flush=True)
    pr, nmid = c5.mode_probe2(x0, vec, F, S, vf, filt,
                              [4.8, 5.4, 6.0], iters=24)
    for e, g, rs in pr:
        tag = "FLIP" if (rs < 0 and g >= 1.0) else ("marg" if rs < 0 else "creep")
        print(f"  probe eta={e}: |r|={g:.3f} signed={rs:+.3f} [{tag}] "
              f"(nmid={nmid})", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    t0 = time.time()
    for a in args:
        if a == "all":
            tangent(); negbranch(); psweep(); psweep_ref()
            p5rmin(); p5rmin_ref(); p3rmin_ref(); v5seeds()
        else:
            {"tangent": tangent, "negbranch": negbranch, "psweep": psweep,
             "psweep_ref": psweep_ref, "p5rmin": p5rmin,
             "p5rmin_ref": p5rmin_ref, "p3rmin_ref": p3rmin_ref,
             "v5seeds": v5seeds}[a]()
    print(f"total {time.time()-t0:.0f}s")

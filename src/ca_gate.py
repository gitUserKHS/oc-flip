# -*- coding: utf-8 -*-
"""Cycle A gate (engineering track) -- zero new physics, existing configs only.

  sweep      (i)  trajectory estimator s-hat vs regime across an eta sweep:
                  does s-hat read the flip mode when flip is active, and the
                  s~0 creep edge while converging?  (premise of the AIMD rule)
  richardson (ii) per measured spectrum: numeric argmin_eta max_m |1-eta s_m|
                  over the positive branch, vs closed form 2/(a+b), vs
                  eta* = 2/s_max, vs the folklore 0.5
  rownull    (iii) V3 row-null residual: forward vs central differences,
                  Frobenius vs spectral normalization
  fig        summary figure figs/fig_cycleA.png

Usage: python src/ca_gate.py all
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "data", ROOT / "figs"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
from c4_core import setup, build_J, free_set, XMIN, MOVE, E0, Emin
import c5_core as c5


def step_full(x, volfrac, eta, S, filt="sens"):
    """c4_core.step body, additionally returning log(-dcf) for the s-hat
    estimator (kept local so cycle-4/5 code stays frozen)."""
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
        dcf = np.asarray(S["H"] @ (x * dc)).ravel() / S["Hs"] / np.maximum(1e-3, x)
        dvf = np.ones_like(x)
    else:
        dcf = np.asarray(S["H"] @ (dc / S["Hs"])).ravel()
        dvf = np.asarray(S["H"] @ (np.ones_like(x) / S["Hs"])).ravel()
    l1, l2 = 0.0, 1e9
    lo = np.maximum(XMIN, x - MOVE)
    hi = np.minimum(1.0, x + MOVE)
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


# ------------------------------------------------------------- (i) sweep
def sweep(iters=200):
    print("=== A-i: trajectory s-hat vs regime (60x20 MBB, p=3) ===")
    rows = []
    for rmin in (1.1, 2.4):
        S = setup(60, 20, rmin, "mbb")
        for eta in np.round(np.arange(0.3, 1.61, 0.1), 2):
            x = 0.5 * np.ones(S["nel"])
            dz_p = mask_p = logD_p = None
            mu_t, s_t, sm_t, amps = [], [], [], []
            for _ in range(iters):
                xn, logD = step_full(x, 0.5, eta, S)
                dz = np.log(xn) - np.log(x)
                inter = ((x > XMIN + 1e-6) & (x < 1 - 1e-6) &
                         (xn > XMIN + 1e-6) & (xn < 1 - 1e-6) &
                         (np.abs(xn - x) < MOVE - 1e-6))
                amps.append(np.mean(np.abs(xn - x)))
                if dz_p is not None:
                    for tgt, extra in ((s_t, None), (sm_t, (x > 0.05) & (x < 0.95))):
                        m = mask_p & inter if extra is None else \
                            mask_p & inter & extra
                        if m.sum() < 20:
                            continue
                        z0 = dz_p[m] - dz_p[m].mean()
                        den = z0 @ z0
                        if den / m.sum() <= 1e-16:
                            continue
                        dD = (logD - logD_p)[m]
                        dD = dD - dD.mean()
                        if extra is None:
                            z1 = dz[m] - dz[m].mean()
                            mu_t.append(z1 @ z0 / den)
                        tgt.append(-(dD @ z0) / den)
                dz_p, mask_p, logD_p = dz, inter, logD
                x = xn
            med = lambda a, n=60: float(np.median(a[-n:])) if a else np.nan
            amp = float(np.median(amps[-40:]))
            mu, sh, shm = med(mu_t), med(s_t), med(sm_t)
            osc = amp > 3e-3
            rows.append((rmin, eta, amp, mu, sh, shm, float(osc)))
            print(f"  rmin={rmin} eta={eta:.1f}: amp={amp:.1e} mu^={mu:+.3f} "
                  f"s^={sh:+.3f} s^_mid={shm:+.3f} eta*s^={eta*sh:+.3f} "
                  f"[{'OSC' if osc else 'conv'}]", flush=True)
    np.savez(DATA / "ca_sweep.npz", rows=np.array(rows),
             cols="rmin eta amp mu shat shat_mid osc")
    print("saved data/ca_sweep.npz")


# -------------------------------------------------------- (ii) richardson
def _spec_sources():
    """Yield (name, s_array); psweep spectra are rebuilt from stored x0."""
    for key in ("V3", "V4"):
        d = np.load(DATA / f"c5_tangent_{key}.npz")
        yield key, d["s_exact"]
    for tag, f in (("ref", "c5_psweep_ref.npz"), ("adapt", "c5_psweep.npz")):
        d = np.load(DATA / f)
        S = setup(60, 20, 1.1, "mbb")
        for p in d["ps"]:
            x0 = d[f"p{p:.1f}_x0"]
            core.P = float(p)
            J, F = build_J(x0, S, 0.5, "sens")
            s, _, _ = c5.tangent_spectrum(J, x0, F, S, "sens")
            core.P = 3.0
            yield f"p{p:.1f}_{tag}", s


def richardson():
    print("=== A-ii: Richardson optimum vs flip boundary vs folklore ===")
    print(f"{'case':>12} {'a=min s+':>9} {'b=s_max':>8} {'opt(num)':>9} "
          f"{'2/(a+b)':>8} {'2/b':>6} {'a/b':>7} {'s_med':>6}")
    out = {}
    for name, s in _spec_sources():
        pos = s[np.real(s) > 1e-6]
        a = float(np.min(np.real(pos)))
        b = float(np.max(np.real(pos)))
        etas = np.linspace(1e-3, 1.5, 6000)
        R = np.abs(1.0 - etas[:, None] * pos[None, :]).max(axis=1)
        k = int(np.argmin(R))
        smed = float(np.median(np.real(pos)))
        print(f"{name:>12} {a:9.4f} {b:8.3f} {etas[k]:9.4f} "
              f"{2/(a+b):8.4f} {2/b:6.3f} {a/b:7.4f} {smed:6.3f}", flush=True)
        out[name] = np.array([a, b, etas[k], 2 / (a + b), 2 / b, smed])
    np.savez(DATA / "ca_richardson.npz", **out)
    print("saved data/ca_richardson.npz")
    print("bulk-speed illustration (mode s=1): rate 0.5 @eta=.5 vs 0.75 "
          "@eta=.25 -> error after 100 steps differs by (0.75/0.5)^100 "
          "~ 4e17")


# ---------------------------------------------------------- (iii) rownull
def rownull(eps=1e-5):
    print("=== A-iii: V3 row-null residual, forward vs central FD ===")
    d = np.load(DATA / "c4_V3.npz", allow_pickle=True)
    S = setup(int(d["nelx"]), int(d["nely"]), float(d["rmin"]), str(d["bc"]))
    vf, filt = float(d["vf"]), str(d["filt"])
    x0 = d["x0"]
    J, F = build_J(x0, S, vf, filt)                    # forward differences
    Jc = np.empty_like(J)                              # central differences
    for j, e in enumerate(F):
        xp = x0.copy(); xp[e] *= np.exp(eps)
        xm = x0.copy(); xm[e] *= np.exp(-eps)
        Jc[:, j] = (np.log(core.step(xp, vf, 0.5, S, filt)[F]) -
                    np.log(core.step(xm, vf, 0.5, S, filt)[F])) / (2 * eps)
    w = c5.vol_weights(x0, F, S, filt)
    for name, JJ in (("forward", J), ("central", Jc)):
        r = np.linalg.norm(w @ JJ) / np.linalg.norm(w)
        fro = np.linalg.norm(JJ)
        s2 = np.linalg.svd(JJ, compute_uv=False)[0]
        print(f"  {name}: ||w^T J||/||w|| = {r:.3e}  "
              f"/||J||_F = {r/fro:.3e}  /||J||_2 = {r/s2:.3e}  "
              f"(||J||_F={fro:.1f}, ||J||_2={s2:.2f})", flush=True)
    s_f, _, _ = c5.tangent_spectrum(J, x0, F, S, filt)
    s_c, _, _ = c5.tangent_spectrum(Jc, x0, F, S, filt)
    dtop = np.max(np.abs(np.real(s_f[:6]) - np.real(s_c[:6])))
    print(f"  top-6 s forward vs central: max|d| = {dtop:.1e}")


# ----------------------------------------------------------------- figure
def fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif", "mathtext.fontset": "stix",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 300})
    sw = np.load(DATA / "ca_sweep.npz")["rows"]
    ri = np.load(DATA / "ca_richardson.npz")
    fig_, ax = plt.subplots(1, 2, figsize=(6.5, 2.8))

    a = ax[0]
    ee = np.linspace(0.3, 1.6, 200)
    a.plot(ee, 2 / ee, "k--", lw=1.0, label=r"$\hat s = 2/\eta$")
    for rmin, mk, smax in ((1.1, "o", 3.974), (2.4, "s", 3.355)):
        r = sw[sw[:, 0] == rmin]
        conv, osc = r[r[:, 6] == 0], r[r[:, 6] == 1]
        col = "#B4231F" if rmin == 1.1 else "#2B6A99"
        a.plot(conv[:, 1], conv[:, 4], mk, color=col, mfc="none", ms=4.5,
               label=fr"converging, $r_{{\min}}{rmin}$")
        a.plot(osc[:, 1], osc[:, 4], mk, color=col, ms=4.5,
               label=fr"oscillating, $r_{{\min}}{rmin}$")
        a.axhline(smax, color=col, lw=0.7, ls=":", alpha=0.7)
    a.set_xlabel(r"damping $\eta$")
    a.set_ylabel(r"trajectory $\hat s$")
    a.set_ylim(-0.5, 7)
    a.legend(fontsize=6.5, loc="upper right")
    a.text(0.02, 1.02, "(a)", transform=a.transAxes, fontweight="bold")

    a = ax[1]
    keys = [k for k in ri.files if k.startswith("p") and k.endswith("_ref")]
    ps = sorted(float(k[1:4]) for k in keys)
    opt = [ri[f"p{p:.1f}_ref"][2] for p in ps]
    star = [ri[f"p{p:.1f}_ref"][4] for p in ps]
    a.plot(ps, star, "o-", color="#B4231F", ms=4,
           label=r"flip boundary $2/s_{\max}$")
    a.plot(ps, opt, "x", color="k", ms=6,
           label=r"Richardson $\arg\min_\eta\max_m|1-\eta s_m|$")
    a.set_xlabel(r"penalization $p$")
    a.set_ylabel(r"damping $\eta$")
    a.legend(fontsize=7)
    a.text(0.02, 1.02, "(b)", transform=a.transAxes, fontweight="bold")

    fig_.tight_layout()
    fig_.savefig(FIGS / "fig_cycleA.png", bbox_inches="tight")
    print("saved figs/fig_cycleA.png")


if __name__ == "__main__":
    t0 = time.time()
    for a in (sys.argv[1:] or ["all"]):
        if a == "all":
            sweep(); richardson(); rownull(); fig()
        else:
            {"sweep": sweep, "richardson": richardson,
             "rownull": rownull, "fig": fig}[a]()
    print(f"total {time.time()-t0:.0f}s")

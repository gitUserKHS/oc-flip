# -*- coding: utf-8 -*-
"""Cycle G1 / paper-1 v2: the spectral enclosure and where it fails.

Theorem (notes/g1_lemma.md): for exact-gradient SIMP -- raw, or density
filter with the exact chain rule -- the log-coordinate OC operator
S = -d log D / d log x has a real spectrum inside [-(p-1), p+1] at every
design (and on the tangent of the active face at every OC fixed point).
It is the log-coordinate second-order form of Svanberg (1994): compliance
is convex in x^p and concave in x^-p; a Jensen step carries it through a
nonnegative row-normalized density filter.

Part A checks the full-operator enclosure numerically with the repo's own
FE code (central finite differences of log(-dcf) in log x) on raw,
density-filtered and repo designs. Part B shows that the SENSITIVITY
filter (top88) is not covered: at the design stored in
data/g1_sens_counterexample.npz (20x10 MBB, rmin 1.5, p=5; found by the
cycle-G1 adversarial search) the top eigenvalue of the filtered-sensitivity
operator exceeds p+1, with Emin = 1e-9 and with Emin = 0.

Usage: python src/g1_enclosure.py      (~1 min)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

import c4_core as core
import ca_gate
from ca_gate import step_full


def s_operator(x, S, p, filt, h=1e-6):
    """Full S = -d log(-dcf) / d log x by central differences (dvf is
    design-independent, so this is the OC log-Jacobian before the volume
    projection)."""
    core.P = p
    f = lambda z: step_full(z, z.mean(), 0.5, S, filt)[1]
    n = x.size
    J = np.empty((n, n))
    for j in range(n):
        xp = x.copy(); xp[j] *= np.exp(h)
        xm = x.copy(); xm[j] *= np.exp(-h)
        J[:, j] = (f(xp) - f(xm)) / (2 * h)
    core.P = p
    logD0 = f(x)
    core.P = 3.0
    return -J, logD0


def smooth_design(nelx, nely, vf, seed, xlo=1e-2):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(nelx + 4, nely + 4))
    k = np.ones((3, 3)) / 9.0
    from scipy.signal import convolve2d
    g = convolve2d(g, k, mode="valid")[:nelx, :nely]
    x = np.clip(vf + 0.35 * g / g.std(), xlo, 1.0)
    return x.ravel()


def report(name, s, p, sym=None):
    lo, hi = -(p - 1), p + 1
    # finite-difference noise ~1e-5: the density-filter top edge p+1 is
    # attained exactly (scaling direction, Emin -> 0)
    viol = max(0.0, float(s.real.max() - hi), float(lo - s.real.min()))
    print(f"  {name:38s} p={p:.0f}  s in [{s.real.min():+.4f}, "
          f"{s.real.max():+.4f}]  bound [{lo:+.0f}, {hi:+.0f}]  "
          f"max|Im| {np.abs(s.imag).max():.1e}  violation {viol:.1e}")
    if sym is not None:
        print(f"  {'':38s}      ||Gamma S - (Gamma S)^T|| / ||Gamma S|| = "
              f"{sym:.1e}  (Lemma 2: self-adjoint, so the spectrum is real;"
              f" imaginary parts above are FD noise on an ill-conditioned "
              f"eigenproblem)")


def part_a():
    print("=== A: exact-gradient enclosure spec(S) in [-(p-1), p+1] ===")
    cases = []
    # raw SIMP: rmin = 1.0 gives H = I, so 'sens' is the exact gradient.
    # Moderate contrast (x >= 0.2): with unfiltered near-void elements,
    # float64 cancellation in u_e^T k_e u_e corrupts log D (cycle G1 showed
    # these spurious excursions vanish at 50-digit precision).
    for p in (3.0, 5.0):
        for seed in (1, 2):
            cases.append((f"raw 30x10 smooth seed{seed} (x>=0.2)", 30, 10,
                          1.0, "mbb", "sens", p,
                          smooth_design(30, 10, 0.6, seed, xlo=0.2)))
    for seed in (1, 2):
        cases.append((f"density 30x10 r1.5 smooth seed{seed}", 30, 10, 1.5,
                      "cantilever", "dens", 3.0,
                      smooth_design(30, 10, 0.5, seed)))
    d = np.load(DATA / "c4_V5.npz", allow_pickle=True)
    cases.append(("density V5 fixed point (60x20 r2.4)", int(d["nelx"]),
                  int(d["nely"]), float(d["rmin"]), str(d["bc"]), "dens",
                  3.0, d["x0"]))
    for name, nx, ny, r, bc, filt, p, x in cases:
        S = core.setup(nx, ny, r, bc)
        x = np.asarray(x, float)
        Sop, logD = s_operator(x, S, p, filt)
        s = np.linalg.eigvals(Sop)
        # Lemma 2: Gamma S = X Hess(C) X is symmetric, Gamma = diag(x*D)
        # with D = -dC/dx. (Symmetrizing and solving the pencil is not
        # used: Gamma spans many decades and amplifies FD noise.)
        g = x * np.exp(logD)
        M = g[:, None] * Sop
        asym = np.linalg.norm(M - M.T) / np.linalg.norm(M)
        report(name, s, p, asym)


def part_b():
    print("=== B: sensitivity filter is NOT enclosed ===")
    d = np.load(DATA / "g1_sens_counterexample.npz")
    x, p = d["x"], float(d["p"])
    S = core.setup(int(d["nelx"]), int(d["nely"]), float(d["rmin"]),
                   str(d["bc"]))
    emin0 = ca_gate.Emin
    for emin in (1e-9, 0.0):
        ca_gate.Emin = emin
        s = np.linalg.eigvals(s_operator(x, S, p, "sens")[0])
        top = np.sort(s.real)[::-1][:3]
        print(f"  Emin={emin:.0e}: top eigenvalues {np.round(top, 4)}  "
              f"s_max/(p+1) = {top[0] / (p + 1):.4f}  max|Im| "
              f"{np.abs(s.imag).max():.2e}")
    ca_gate.Emin = emin0


if __name__ == "__main__":
    part_a()
    part_b()

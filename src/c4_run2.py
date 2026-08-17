# -*- coding: utf-8 -*-
"""Cycle 4 driver v2: deep converge -> spectrum -> mode-aligned probe."""
import sys, time
from pathlib import Path
import numpy as np
from c4_core import (setup, step, converge, build_J, spectrum, arnoldi_smax,
                     free_set, mode_probe, XMIN)

DATA = Path(__file__).resolve().parents[1] / "data"

VAR = {
    "V2": dict(nelx=120, nely=40, rmin=2.2, bc="mbb",        filt="sens", vf=0.5, it=400),
    "V3": dict(nelx=60,  nely=20, rmin=1.1, bc="cantilever", filt="sens", vf=0.5, it=300),
    "V4": dict(nelx=60,  nely=20, rmin=1.1, bc="mbb",        filt="sens", vf=0.4, it=300),
    "V5": dict(nelx=60,  nely=20, rmin=2.4, bc="mbb",        filt="dens", vf=0.5, it=300),
}

def run(key):
    v = VAR[key]; t0 = time.time()
    S = setup(v["nelx"], v["nely"], v["rmin"], v["bc"])
    eta0 = 0.5
    x0, amp0 = converge(S, v["vf"], eta0, v["filt"], iters=v["it"])
    if amp0 > 3e-3:
        eta0 = 0.3
        x0, amp0 = converge(S, v["vf"], eta0, v["filt"], iters=v["it"])
    F = free_set(x0); t1 = time.time()
    if F.size <= 1100:
        J, F = build_J(x0, S, v["vf"], v["filt"])
        s, V = spectrum(J)
        method = "fullJ"; s_top = s[:6]; vec = V[:, 0].real
    else:
        vals, vecs, F, nmv = arnoldi_smax(x0, S, v["vf"], v["filt"])
        method = f"arnoldi({nmv}mv)"; s_top = vals[:6]; vec = vecs[:, 0]
    smax = float(np.real(s_top[0])); eta_star = 2.0/smax
    t2 = time.time()
    es = np.round(np.clip(eta_star + np.array([-0.12, -0.04, 0.04, 0.12]), 0.05, None), 3)
    pr = mode_probe(x0, vec, F, S, v["vf"], v["filt"], es)
    t3 = time.time()
    np.savez(DATA / f"c4_{key}.npz", key=key, smax=smax, eta_star=eta_star,
             s_top=np.real(s_top), ncx=int(np.sum(np.abs(np.imag(s_top)) > 1e-3)),
             F=F, vec=vec, probe=np.array(pr), nelx=v["nelx"], nely=v["nely"],
             method=method, eta0=eta0, amp0=amp0, x0=x0, rmin=v["rmin"],
             bc=v["bc"], filt=v["filt"], vf=v["vf"])
    print(f"{key} [{method}] |F|={F.size} conv@{eta0}(amp {amp0:.1e}) "
          f"s_top={np.round(np.real(s_top),3)} -> s_max={smax:.3f}, eta*={eta_star:.3f}")
    for e, g, rs in pr:
        tag = "FLIP" if (rs < 0 and g >= 1.0) else ("marg" if rs < 0 else "creep")
        print(f"   eta={e:.3f}: |ratio|={g:.3f}, signed={rs:+.3f}  [{tag}]")
    print(f"   time conv {t1-t0:.0f}s | spec {t2-t1:.0f}s | probe {t3-t2:.0f}s", flush=True)

if __name__ == "__main__":
    for k in sys.argv[1:]:
        run(k)

# -*- coding: utf-8 -*-
"""Endpoint spectra for the vf0.4 continuation group (reviewer item 3).

The motivation section claims that at volume fraction 0.4 the ramp's
destination itself excludes eta = 0.5 (so no schedule can rescue it).
That claim was previously unmeasured. Here we rerun the vf0.4
continuation in full mode for the informative methods and build the
exactly projected Jacobian at the endpoint (p = 5), reporting each
endpoint's own threshold 2/s_max.

Usage: python src/ch_vf04.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from c4_core import setup
from cb_aimd import ctrl_fixed, ctrl_aimd
from cd_cont import SCHED, run, ctrl_ladder10

METHODS = [("fix0.3", lambda: ctrl_fixed(0.3)),
           ("fix0.5", lambda: ctrl_fixed(0.5)),
           ("aimd0.5", lambda: ctrl_aimd(0.5)),
           ("aimd1.0", lambda: ctrl_aimd(1.0))]


def main():
    print("=== vf0.4 continuation endpoints (full mode, p=5 spectra) ===")
    S = setup(60, 20, 1.1, "mbb")
    out = {}
    for sname, (N, iters) in SCHED.items():
        for mname, mk in METHODS:
            t0 = time.time()
            r = run(S, 0.4, N, iters, mk())
            osc = r["tail"] > 3e-3
            sm = r["smax_end"]
            thr = 2.0 / sm if np.isfinite(sm) and sm > 0 else np.nan
            eta_end = float(np.median(r["eta"][-100:]))
            out[f"{sname}_{mname}"] = np.array(
                [r["comp"][-1], r["tail"], float(osc), sm, thr, eta_end])
            print(f"  {sname:4s} {mname:8s}: comp={r['comp'][-1]:7.1f} "
                  f"tail={r['tail']:.1e} {'OSC' if osc else 'conv'}  "
                  f"s_max(end)={sm:.3f}  own 2/s_max={thr:.3f}  "
                  f"eta_med={eta_end:.3f}  [{time.time()-t0:.0f}s]",
                  flush=True)
    np.savez(DATA / "ch_vf04.npz", **out)
    print("saved data/ch_vf04.npz")


if __name__ == "__main__":
    main()

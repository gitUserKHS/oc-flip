# -*- coding: utf-8 -*-
"""Step 3: seed statistics for the failure-rate table.

The pipeline is deterministic from the uniform start, so variability is
introduced as perturbed restarts: x0 = clip(vf * exp(d - mean d)), with
d ~ N(0, 0.02), seeds {5, 7, 11} (project convention). Together with the
unperturbed run (cc_sweep/cc_mma) this gives 4 replicates per cell.

Reports per-method failure counts per seed and the across-seed range, so
the paper can state how stable 0/11-vs-1/11 distinctions are.

Usage: python src/cf_seeds.py
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
from cc_sweep import CONFIGS, METHODS, run, ctrl_ladder10
from cc_mma import mma_run

SEEDS = (5, 7, 11)


def main(iters=400):
    print(f"=== Step 3: perturbed restarts, seeds {SEEDS}, "
          f"{len(CONFIGS)} configs, budget {iters} ===")
    sw0 = np.load(DATA / "cc_sweep.npz", allow_pickle=True)
    groups = list(sw0["groups"])
    best0 = {g: min(sw0[f"{g}_{m}_meta"][0] for m in sw0["methods"])
             for g in groups}
    save = {}
    counts = {m: [] for m, _ in METHODS}
    counts["mma"] = []
    for seed in SEEDS:
        res = {}
        for name, kw, ps in CONFIGS:
            S = setup(kw["nelx"], kw["nely"], kw["rmin"], kw["bc"])
            for p in ps:
                grp = f"{name}_p{p:.0f}"
                for mname, mk in METHODS:
                    r = run(S, kw["vf"], kw["filt"], p, iters, mk(),
                            seed=seed)
                    tail = float(np.median(r["amp"][-30:]))
                    res[(grp, mname)] = (r["comp"], tail)
                rm = mma_run(S, kw["vf"], kw["filt"], p, iters, seed=seed)
                res[(grp, "mma")] = (rm["comp"],
                                     float(np.median(rm["amp"][-30:])))
        line = []
        for mname in list(dict(METHODS)) + ["mma"]:
            nosc = nq = 0
            dqs = []
            for g in groups:
                comp, tail = res[(g, mname)]
                osc = tail > 3e-3
                dq = 100 * (comp / best0[g] - 1)
                nosc += osc
                nq += (not osc) and dq > 2.0
                dqs.append(dq)
                save[f"s{seed}_{g}_{mname}"] = np.array(
                    [comp, tail, float(osc), dq])
            counts[mname].append((nosc, nq, float(np.median(dqs)),
                                  float(max(dqs))))
            line.append(f"{mname}:{nosc}o/{nq}q")
        print(f"  seed {seed}: " + "  ".join(line), flush=True)
    print("\n--- across-seed stability (osc-fail, quality-fail; "
          "range over seeds; unperturbed in brackets) ---")
    base = {"fix0.5": (0, 0), "fix0.7": (4, 1), "fix1.0": (9, 0),
            "aimd0.5": (1, 0), "aimd1.0": (2, 1), "ladder": (1, 2),
            "mma": (0, 1)}
    for mname, rows in counts.items():
        o = [r[0] for r in rows]
        q = [r[1] for r in rows]
        w = [r[3] for r in rows]
        print(f"  {mname:8s} osc {min(o)}-{max(o)}/11 [{base[mname][0]}]  "
              f"qual {min(q)}-{max(q)}/11 [{base[mname][1]}]  "
              f"worst dq {max(w):+.1f}%", flush=True)
    np.savez(DATA / "cf_seeds.npz", seeds=np.array(SEEDS), **save)
    print("saved data/cf_seeds.npz")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time()-t0:.0f}s")

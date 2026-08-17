# -*- coding: utf-8 -*-
"""Continuation seed replication (+ vf0.4 config extension).

Re-runs the continuation suite from perturbed starts (seeds 5/7/11; 2%
log-normal, mean-centered) in light mode (oscillation tag + final
compliance), and adds the mbb_vf0.4 config (nominal + seeds). The key
question: how often does the fast cantilever ramp break fixed 0.5 --
the paper's single-cell motivation before this run.

Usage: python src/cd_seeds.py
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from c4_core import setup
from cd_cont import CONFIGS, CONFIGS_EXT, SCHED, METHODS, run

SEEDS = (5, 7, 11)


def main():
    print("=== continuation seed replication (light mode) ===")
    save = {}
    t0 = time.time()
    for name, kw in CONFIGS_EXT:
        vf = kw.get("vf", 0.5)
        S = setup(60, 20, kw["rmin"], kw["bc"])
        starts = list(SEEDS) if any(name == n for n, _ in CONFIGS) \
            else [None] + list(SEEDS)
        for sname, (N, iters) in SCHED.items():
            for st in starts:
                tag = "n" if st is None else f"s{st}"
                line = []
                for mname, mk in METHODS:
                    r = run(S, vf, N, iters, mk(), seed=st, light=True)
                    osc = r["tail"] > 3e-3
                    save[f"{name}_{sname}_{tag}_{mname}"] = np.array(
                        [r["comp"][-1], r["tail"], float(osc)])
                    line.append(f"{mname}:{'OSC' if osc else 'c'}")
                print(f"  {name}_{sname} [{tag}] " + " ".join(line),
                      flush=True)
    np.savez(DATA / "cd_seeds.npz", **save)
    # summary: per (config, sched, method) failure count across starts
    print("\n--- failure counts across starts (seeds; vf0.4 incl nominal) ---")
    for name, kw in CONFIGS_EXT:
        starts = [f"s{s}" for s in SEEDS] if any(name == n for n, _ in
                                                 CONFIGS) \
            else ["n"] + [f"s{s}" for s in SEEDS]
        for sname in SCHED:
            row = []
            for mname, _ in METHODS:
                n_osc = sum(save[f"{name}_{sname}_{t}_{mname}"][2] > 0
                            for t in starts)
                row.append(f"{mname}:{n_osc}/{len(starts)}")
            print(f"  {name}_{sname:5s} " + "  ".join(row))
    print(f"saved data/cd_seeds.npz  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

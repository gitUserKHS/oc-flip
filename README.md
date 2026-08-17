# oc-flip — damping without tuning in the optimality-criteria iteration

Code, data and paper for a dynamical-systems study of the damped
optimality-criteria (OC) update in SIMP topology optimization.

**The paper is `paper/main.pdf`** (source `paper/main.tex`, bibliography
`paper/refs.bib`). This deposit contains everything that produces it.

Core result in one line: in logarithmic coordinates the damped OC update has
local multipliers `mu = 1 - eta*s`, so oscillation is a period-doubling
instability at `eta* = 2/s_max`, where `s_max` is the top of a measurable
spectrum — and because that spectrum's lower edge sits at `s ~ 0` while its
top saturates at `p+1`, the folklore `eta = 1/2` is the min–max optimum at
`p = 3`. The threshold belongs to whichever fixed point the iteration is
heading for, so it moves under penalization continuation; a four-line
controller reading the trajectory's own multiplier tracks it.

## What is in this deposit

```
paper/    main.pdf, main.tex, refs.bib
          main_v1_spectral.tex -- earlier physics-framed version of the paper,
          kept as the starting point of a planned companion study
src/      all experiment drivers and figure scripts (see Reproduction)
data/     checkpoints (.npz) for every reported number
figs/     figures; fig_*_pub.png are the ones used in the paper
notes/    the full research record: one note per cycle, the prior-art gate,
          the response notes to three rounds of external review
CLAUDE.md project brief: confirmed numbers, backlog, and a list of pitfalls
          that cost real time (protocol dependence of spectra, estimator
          circularity, masked oscillation, ...)
requirements.txt
```

Only `numpy`, `scipy` and `matplotlib` are required; there is no build step
and no configuration. Every script is run as `python src/<name>.py` from
anywhere — paths are resolved relative to the repository.

## Reproduction

Approximate wall-clock times are for a laptop CPU; the 60x20 benchmark costs
about 9 ms per OC step.

```
python src/c4_run2.py V2 V3 V4 V5   # robustness checkpoints        -> data/c4_*.npz
python src/c5_run.py all            # closure experiments (~6 min)  -> data/c5_*.npz
python src/ca_gate.py all           # estimator regimes, Richardson (~2 min)
python src/cb_aimd.py 400           # controller demos (~1.5 min)   -> data/cb_aimd.npz
python src/cc_sweep.py              # 11-group failure sweep (~5 min)
python src/cc_anderson.py           # Anderson x controller composition
python src/cc_mma.py                # MMA reference column
python src/cd_cont.py               # p-continuation 1->5 (~8 min)
python src/cd_seeds.py              # continuation replication + vf0.4 (~16 min)
python src/ce_defense.py all        # move-limit baseline, endpoint spectra,
                                    #   truncated Richardson (~3 min)
python src/cf_seeds.py              # perturbed-restart statistics (~18 min)
python src/cg_cascade.py            # period-doubling cascade test
python src/ch_vf04.py               # vf0.4 continuation endpoint spectra

python src/pub_figs.py              # paper Figs 1-3, 6
python src/c5_fig_pub.py            # paper Figs 6-7 (p-sweep, negative branch)
python src/cb_fig.py                # paper Fig: controller demos
python src/cc_fig.py                # paper Fig: failure matrix + Anderson
python src/cd_fig.py                # paper Fig: continuation
python src/c5_fig.py                # working figure (cycle 5)
```

Imported, not run directly: `c4_core.py` (instrumented top88: setup, OC step,
exact-projection Jacobian, Arnoldi), `c5_core.py` (volume-tangent projection,
compliance, mode probes), `top88_instrumented.py` / `top88_v2.py` (the original
instrumented benchmark and a compatibility shim). Two early-cycle scripts are
self-contained and reproduce the first working figures: `oc_toy.py` (two-bar
models) and `jacobian_spectrum.py` / `probe_experiment.py` (first spectra and
fixed-design probes); the paper's versions of those results come from
`pub_figs.py`.

## Conventions (fixed across every experiment)

| quantity | value |
|---|---|
| finite-difference step (log coordinates) | `1e-5` |
| OC bisection tolerance | `1e-10` (looser values leak noise into the FD Jacobians) |
| perturbation kick seed / ARPACK start seed | 5 / 1 |
| onset-bracket grid width | `0.08` |
| controller constants (frozen everywhere) | `c=1.7`, `delta=0.25`, `alpha=0.01`, warm-up 25, window 4, cooldown 5 |

**Convergence protocols are part of the measurement.** Spectra depend on which
fixed point a run reached, so every reported spectrum states its protocol:
`reference` (converge at `eta0 = 0.3`) or `adapted` (`eta0 = 0.5`, stepping
down only on oscillation). See `notes/cycle5_note.md` and pitfall 10 in
`CLAUDE.md`. One legacy file, `data/psweep.npz`, predates protocol logging and
is superseded by `data/c5_psweep_ref.npz`; it is kept only for provenance.

## Reading the research record

`notes/` is kept deliberately complete, including the parts that did not
survive: a preview result that was retracted once its convergence protocol was
documented (`cycle5_note.md`), a mechanism attribution withdrawn after a direct
test (`review2_note.md`), and a causal claim corrected by measurement
(`review3_note.md`). The paper reports the outcomes; these notes report how
they were arrived at.

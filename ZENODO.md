# Zenodo deposit — paste-ready metadata

Fields below match the Zenodo upload form. Everything except **Creators** and
**License** is filled in; those two are choices only the depositor can make
(Zenodo requires at least one creator, and a license selection, before the
record can be published).

---

**Upload type:** Publication → Preprint
(choose *Software* instead if you want the code to be the primary record; the
paper is then an additional file)

**Title**

```
Damping without tuning: the measurable stability threshold of the
optimality-criteria iteration, and a controller that tracks it
```

**Publication date:** 2026-08-17

**Creators:** Hyeongseok Kim (Independent Researcher, Uiwang, Republic of Korea)

**Description**

```
The optimality-criteria (OC) update used throughout density-based topology
optimization has one free parameter, the damping exponent eta, conventionally
fixed at 1/2 without a stated criterion. This work measures what that
parameter controls and removes the need to choose it.

Exactly projected finite-difference Jacobians of the full update — finite
element solve, filter, multiplier bisection and clipping — taken across
meshes, boundary conditions, volume fractions, filters and penalizations,
establish three empirical laws. (1) Oscillation is a period-doubling
instability whose threshold is eta* = 2/s_max, where s_max tops the spectrum
of the log-log stiffness of the filtered sensitivity field; the local
multipliers mu = 1 - eta*s predict measured onset and decay rates to three
decimals. (2) That spectrum saturates at p+1 on top while its lower edge sits
near zero, which pins the min-max (Richardson) optimal damping onto the flip
boundary itself: at p = 3 the conventional value is that optimum. (3) The
observed s_max belongs to the fixed point the iteration selects, and standard
damping self-limits to fixed points with s_max <= 2/eta, separating from the
naturally saturated family exactly at p = 3.

The threshold is therefore a moving target: under penalization continuation
fixed damping at 1/2 ends oscillating in 12 of 32 runs. Because the free
trajectory estimators read the flip branch precisely when it is active and the
near-zero creep edge otherwise — a regime dependence that estimator-driven
relaxation schemes (Aitken, Barzilai-Borwein, Anderson) do not account for — a
four-line additive-increase/multiplicative-decrease controller can track the
threshold unaided. With a single frozen parameter set it converges in all 64
continuation runs, matches the best hand tuning across an 11-configuration
suite, and composes with Anderson acceleration so that each covers the other's
failures.

This deposit contains the paper together with all code, checkpoints and figure
scripts that reproduce every table and figure in it, plus the complete research
record in notes/ — including a retracted preview result, a withdrawn mechanism
attribution, and a causal claim corrected by measurement. Dependencies are
numpy, scipy and matplotlib only.
```

**Keywords**

```
topology optimization
optimality criteria
SIMP
numerical damping
period-doubling bifurcation
fixed-point iteration
adaptive step size
Anderson acceleration
```

**Language:** eng

**License:** *(choose at upload)*
A common pairing for this kind of deposit is CC-BY-4.0 for the paper and
MIT or BSD-3-Clause for the code; Zenodo takes one license per record, so
either pick one for the whole deposit or split it into two records.

**Related identifiers**
`is supplement to` → the engrXiv DOI of the preprint (add once posted).
The preprint's next version should in turn link back to this deposit.

---

## Upload checklist

- [ ] `paper/main.pdf` renders correctly (10 figures, 6 tables, 0 unresolved
      references — verified 2026-08-17)
- [ ] archive the repository as a single `.zip` (about 9 MB) or upload the
      directories as they stand
- [x] creator: Hyeongseok Kim
- [ ] license selected
- [ ] after publishing, record the DOI in `CLAUDE.md` so later versions can
      cite the deposit

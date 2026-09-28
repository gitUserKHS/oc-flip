# Validity argument for the filter-aware Lovász (supermodular concave-envelope) compliance bound

Track T5 (g5_bounds), cycle G1. This is the full argument written out, with every assumption
named. The last section lists the gaps that are still open. Code: `g5_core.py` (float bound),
`rigor_mbb300.py` (outward-rounded certificate).

## 0. Problem that is bounded

- Mesh: n Q4 elements with fixed element stiffness k_e (PSD, 8×8, ν = 3/10, E0 = 1). Dirichlet
  set D. Load f.
- Design: x ∈ X := { x ∈ [ℓ, h]^n : wᵀx = V }, with ℓ = 10⁻³·1, h = 1, w = Pᵀ1, V = vf·n.
- Density filter: P = diag(H1)⁻¹ H, with H_ej = max(0, r − dist(e,j)) ≥ 0, so P ≥ 0 entrywise and
  each row sums to 1. Physical density ρ = P x.
- Modulus: E(ρ) = Emin + (1 − Emin) ρ^p with p = 3 and Emin = 10⁻⁹. Stiffness K(x) = Σ_e E(ρ_e) K_e,
  where K_e is k_e scattered to global dofs.
- Compliance: C(x) = fᵀ K(x)⁻¹ f on the free dofs. K(x) is SPD there because Emin > 0 and the
  supports remove the rigid modes.

We want a number B with B ≤ min_{x∈X} C(x).

## 1. Energy (displacement) dual. Needs only that K(x) is SPD.

For any u with u_D = 0 and any scalar t:

  C(x) = max_v [2 fᵀv − vᵀK(x)v] ≥ 2t fᵀu − t² uᵀK(x)u,

and uᵀK(x)u = Σ_e E(ρ_e(x)) φ_e(u), where φ_e(u) = u_eᵀ k_e u_e ≥ 0 because k_e is PSD.
Suppose M ≥ sup_{x∈X} Σ_e φ_e E(ρ_e(x)) and M > 0. Then for every x,
C(x) ≥ 2t fᵀu − t² M. Taking t = fᵀu / M gives

  **min_X C ≥ (fᵀu)² / M.**                                                  (1)

This choice of t is the "optimal scalar rescaling" of u, so the bound is invariant to the scale of u.
In every run here u is an FE displacement, but (1) holds for any u that vanishes on D.

## 2. Pointwise envelope inequality. Uses convexity of E and P ≥ 0; valid for every x, not only vertices.

Fix an element e and let J = supp(P_e·), m = |J|. Write z_j = (x_j − ℓ_j)/(h_j − ℓ_j) ∈ [0,1].
Then

  ρ_e(x) = a_e + Σ_{j∈J} p_j z_j,   a_e = Σ_j P_ej ℓ_j,   p_j = P_ej (h_j − ℓ_j) ≥ 0.

Define the set function F_e(S) = E(a_e + p(S)) for S ⊆ J.

**Staircase decomposition.** Sort so that z_{σ(1)} ≥ … ≥ z_{σ(m)}. Put S_k = {σ(1),…,σ(k)},
z_{σ(0)} := 1, z_{σ(m+1)} := 0, and λ_k = z_{σ(k)} − z_{σ(k+1)} ≥ 0 for k = 0..m. Then Σλ_k = 1 and
z = Σ_k λ_k 1_{S_k}. Because ρ_e is affine in z, ρ_e = Σ_k λ_k (a_e + p(S_k)). E is convex on
[0, ∞), and every a_e + p(S_k) ≥ 0. Therefore

  E(ρ_e(x)) ≤ Σ_k λ_k F_e(S_k) =: L_e(z(x)),                                  (2)

and L_e is exactly the Lovász extension of F_e.

- (2) holds at **every** point of the box, including non-vertex and non-binary designs. So continuity
  in x between vertices needs no separate treatment, and the bound never relies on "the max of a
  convex function is attained at a vertex".
- Separately: g(x) = Σ_e φ_e E((Px)_e) is convex on X (E is convex on ρ ≥ 0, φ ≥ 0, ρ is affine, and
  P ≥ 0 keeps ρ ≥ 0 on X). So the true max over the polytope X is attained at a vertex of X, and a
  vertex of X has at most one fractional coordinate. That fact is true, but the bound does not
  use it.

## 3. Greedy cuts are affine majorants. Uses supermodularity, which follows from P ≥ 0 and convex E.

- **Supermodularity.** F_e(S ∪ {j}) − F_e(S) = E(a_e + p(S) + p_j) − E(a_e + p(S)). For convex E this
  is nondecreasing in p(S), and p(S) is monotone in S because p ≥ 0. So F_e is supermodular.
- **Marginal vectors lie in the core.** For any ordering π of J, the marginal vector
  y^π_{π(k)} = F_e(S^π_k) − F_e(S^π_{k−1}) lies in the core { y : y(S) ≥ F_e(S) − F_e(∅) ∀S,
  y(J) = F_e(J) − F_e(∅) } of the supermodular (convex) game F_e − F_e(∅) (Shapley 1971).
- **Hence the cut bound.** For z in the cube, with λ_k ≥ 0:

  L_e(z) = F_e(∅) + Σ_{k≥1} λ_k (F_e(S_k) − F_e(∅)) ≤ F_e(∅) + Σ_k λ_k y^π(S_k) = F_e(∅) + y^πᵀ z =: ℓ_π(z).  (3)

  Equality holds for the greedy ordering π = σ(z). So L_e = min_π ℓ_π on the cube, and L_e is
  concave (Lovász 1983). Cut separation is a single sort.
- It is also standard (Tawarmalani, Richard & Xiong 2013) that L_e is the concave envelope of the
  vertex values over the box. Validity does not need envelope-ness; only tightness does.
- **Where P ≥ 0 matters.** If some p_j < 0 (a filter with negative weights), supermodularity fails.
  Then ℓ_π need not majorize L_e, and (3) is false.

## 4. LP relaxation and weak duality

Combining (2) and (3), for every x ∈ X and every finite cut family 𝒦,

  Σ_e φ_e E(ρ_e(x)) ≤ max { Σ_e φ_e t_e : t_e ≤ ℓ_{e,π}(z) ((e,π) ∈ 𝒦), z ∈ [0,1]ⁿ, Σ_j W_j z_j = V' },

where W_j = w_j (h_j − ℓ_j) and V' = V − wᵀℓ. Any subset of cuts gives a relaxation, so an LP value
from an unconverged cutting-plane loop, or from a pruned cut pool, is still a valid M. For the
certificate we avoid LP primal accuracy entirely:

- Take multipliers π'_{e,k} ≥ 0 with Σ_k π'_{e,k} ≥ φ_e.
- Because ℓ_{e,k}(z) ≥ L_e(z) ≥ Emin > 0 on the cube, Σ_e φ_e L_e(z) ≤ Σ_{e,k} π'_{e,k} ℓ_{e,k}(z) = c₀ + gᵀz.
- For any ν ≥ 0 and z ∈ [0,1]ⁿ with Wᵀz = V': gᵀz ≤ νV' + Σ_j max(0, g_j − νW_j).
- So M := c₀ + νV' + Σ_j max(0, g_j − νW_j) satisfies (1).

## 5. Outward-rounded certificate (mbb300, `rigor_mbb300.py`)

The certificate is for the exact-data problem: ν = 3/10, Emin = 10⁻⁹, ℓ = 1/1000, V = 150, and
H_ej = 3/2 − √(d²).

- **u.** The stored float u, read exactly as dyadic rationals. It is checked to be exactly 0 on D.
- **fᵀu and φ_e.** Computed in exact `Fraction` arithmetic with exact rational k_e. Every φ_e ≥ 0 is
  checked exactly. The float kernel differs from the exact k_e by ≤ 5.6e−17.
- **Filter.** Each √(d²) is enclosed by integer isqrt intervals (width 8.6e−78). The neighbour sets
  are decided by the exact integer test 4d² < 9 and checked against the float P.
- **Everything else** uses Astra's `rational_interval.Interval` (2⁻²⁵⁶ fixed point, outward floor/ceil):
  row sums, w_j = Σ_e H_ej / (H1)_e, the ratio H(S)/(H1)_e, F_e(S), and the increment upper bounds
  ȳ = F_hi(S_k) − F_lo(S_{k−1}), clipped at ≥ 0.
- **Rounding direction.** Replacing y by ȳ ≥ y keeps (3) valid because z ≥ 0. The weights
  π'_{e,k} = φ_e π_k / Σ_k π_k are rounded upward, and g_j, c₀ and max(0, g_j − ν w_j^lo) are taken at
  upper endpoints. ν ≥ 0 is the float knapsack ratio and is used as an exact rational.
- **Final step.** B = lower endpoint of (fᵀu)² / M_hi.

## 6. Scope and possible gaps (flagged)

1. **Density filter only.** The argument needs a linear, nonnegative design-to-density map followed
   by a convex modulus.
   - It does *not* cover sensitivity filtering. Sensitivity filtering is not a design→modulus map, and
     the sensitivity-filtered OC fixed point minimizes no declared objective.
   - It does not cover Heaviside/tanh projection. E∘H is not convex in ρ, so (2) fails, and F_e is no
     longer supermodular in general, so (3) fails.
   - Extending it to projection would need a different envelope.
2. **Equality volume.** X uses wᵀx = V. If the intended constraint is wᵀx ≤ V, the bound still holds:
   K(x) is Loewner-nondecreasing in x (E is increasing and P ≥ 0), so C is nonincreasing, and any x
   with wᵀx < V is dominated by some x' ≥ x with wᵀx' = V (since wᵀh ≥ V).
3. **Discrete problem.** The bound is for the discretized problem at this mesh, Emin and ℓ. It says
   nothing about the continuum or mesh-refined optimum.
4. **Float runs are not certified.** All numbers other than the mbb300 certificate (dev, held-out,
   cant800 converged) are float64. They use the weak-duality "safe" LP value, which is independent of
   primal feasibility tolerances but not outward-rounded. The dual is evaluated in float, so the
   expected error is about 1e−10 relative.
   - The certificate method is size-independent in principle. Only mbb300 was run.
5. **Tightness, not validity.** Σ_e φ_e L_e is only ≥ the concave envelope of g over the box. The sum
   of per-element envelopes is not the envelope of the sum, and the volume constraint is not used
   inside each L_e. So M_lov can be much larger than M(u); the relaxation gap lives here.
   - The LP maximizer z has up to about 600 fractional coordinates (LD_master_u runs in bound_math),
     whereas true maximizers are vertices of X.
6. **Identification of the float problem with the exact-data problem.** The float code's K and P
   differ from the exact data by rounding at about 1e−16. A float-evaluated incumbent C is therefore
   compared against a bound for a problem that is perturbed at that level. This is irrelevant at the
   reported precision, but it is not a proof about the float code.
7. **Convexity domain.** E(ρ) = Emin + (1−Emin)ρ³ is convex only for ρ ≥ 0. Every point used lies in
   ρ ∈ [ℓ, 1] because P is row-stochastic and nonnegative. For p ≥ 1 the argument is unchanged; for
   p < 1 it fails.

## 7. Attribution

- Lovász extension and concavity: Lovász (1983), "Submodular functions and convexity", in
  *Mathematical Programming The State of the Art*, Springer, pp. 235–257, doi:10.1007/978-3-642-68874-4_10.
- Core and marginal vectors of convex games: Shapley (1971), Int. J. Game Theory 1(1):11–26,
  doi:10.1007/BF01753431.
- Concave envelope of supermodular functions over boxes: Tawarmalani, Richard & Xiong (2013),
  Math. Program. 138(1–2):531–577, doi:10.1007/s10107-012-0581-4.
- Closest topology-optimization prior art found: Dalklint, Christiansen & Sigmund (2025), SMO 68(7):144,
  doi:10.1007/s00158-025-04073-0 (arXiv:2410.20375). It gives Lagrange-dual/SDP performance bounds
  with **no penalization and no filter**, and explicitly leaves filtering to future work (their §3.1).
- Global optimization of binary compliance problems by branch-and-cut: Stolpe & Bendsøe (2011), SMO,
  doi:10.1007/s00158-010-0574-y.
- Generalized Benders for binary truss problems: Muñoz & Stolpe (2011), JOGO, doi:10.1007/s10898-010-9627-4.

None of these uses a Lovász/supermodular envelope for a density-filtered SIMP lower bound. The
search was about 10 targeted queries and is not exhaustive. The envelope mathematics is classical,
so a novelty claim can only be about the *application* (a filter-aware bound for density-filtered
SIMP).

## 8. Certificates produced with this argument

All values come from `rigor_mbb300.py` or `rigor_general.py`, and each sits within 1e−11 relative
of its float counterpart. They are collected in `summary_g5.json`, `rigor_*.json` and
`rigor_*_cert.npz`.

| Case | Witness u | Certified lower bound |
|---|---|---|
| mbb300 | stored bound_math u | **178.887494925** (≥ 178.0, PASS) |
| mbb300 | SD u | 178.988545628 |
| cant648 | SD u | 65.849648557 |
| cant800 | SD u | 74.061807616 |
| mbb675 (held-out) | cheap / SD | 201.689858203 / 212.340145253 |
| cant1152 (held-out) | cheap / SD | 76.452473250 / 79.514076269 |
| cant648b (held-out) | cheap / SD | 56.667385912 / 58.339638755 |
| mbb1200 (held-out) | cheap / SD | 171.574038952 / 178.025421285 |

For cant648b-cheap and mbb1200-cheap the float cut loop was not converged at 400 rounds. The
certificate remains valid, because a subset of cuts is a relaxation (§4), but it may be slightly loose.

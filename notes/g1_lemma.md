# Spectral enclosure of the OC s-operator (cycle G1, track T2)

LaTeX-ready. Numbers in §7 come from the scripts named there (all in this folder).
Status line: **raw SIMP and density filter (exact chain rule): complete proof** of the full-operator
enclosure, of the face/tangent enclosure **at fixed points (KKT)**, and of the tangent enclosure at
**any** point when $E_{\min}=0$ and no variable is clipped. **Gap (not a proof gap but a limit):**
off-fixed-point tangent operators with $E_{\min}>0$ or with clipped variables are oblique compressions
and are *not* enclosed (numerical excursions to $1.034(p+1)$ measured). **Sensitivity filter: no
enclosure theorem; only a disc bound; counterexamples up to $1.0104(p+1)$ at weighted-volume fixed
points, none at 65 genuine standard-volume fixed points (§6–7).**

---------------------------------------------------------------------------------------------------

## 1. Setting and the operator the code linearizes

*Mechanics.* Elements $i=1,\dots,n$ with physical densities $\rho_i\in(0,1]$, element stiffness
matrices $K_i\succeq 0$ (embedded in the free-DOF space), load $f\neq 0$,
$$E_i=E(\rho_i)=E_{\min}+(1-E_{\min})\rho_i^{\,p},\qquad a_i=(1-E_{\min})\rho_i^{\,p},\qquad
\theta_i=a_i/E_i\in(0,1],$$
$K(\rho)=\sum_i E_iK_i\succ0$, $u=K^{-1}f$, $c(\rho)=f^{\mathsf T}u$, $\varphi_i=u^{\mathsf T}K_iu$,
$\varepsilon_i=a_i\varphi_i$. Standing assumption: $\varepsilon_i>0$ for all $i$, $p>0$, $0\le E_{\min}<1$.

*Design semantics.* $\rho=Px$ with either $P=I$ (**raw SIMP**) or $P=\operatorname{diag}(H\mathbf 1)^{-1}H$,
$H\ge0$ entrywise, $H_{ii}>0$ (**density filter**, Astra `objective_safe` density mode = our
`c4_core` `'dens'`). Objective $C(x)=c(Px)$ with the **exact** gradient
$$D:=-\nabla C(x)=P^{\mathsf T}\hat D,\qquad \hat D_i=-\partial c/\partial\rho_i=p\,\varepsilon_i/\rho_i>0 .$$
Volume weights $w>0$ ($w=\mathbf 1$ raw; $w=P^{\mathsf T}\mathbf 1$ density), volume $w^{\mathsf T}x=V$.

*OC map.* $x^+_i=\operatorname{clip}\big(x_i\,(D_i/(\lambda w_i))^{\eta},\,\ell_i,\,h_i\big)$ with $\lambda$
set by $w^{\mathsf T}x^+=V$ (bounds $\ell_i=\max(x_{\min},x_i-m)$, $h_i=\min(1,x_i+m)$).
In log coordinates $z=\log x$.

**Definition 1 (s-operator).** $S(x):=-\partial\log D/\partial z\in\mathbb R^{n\times n}$, i.e.
$S_{ij}=x_j\,(\nabla^2C)_{ij}/D_i$. For the top88 sensitivity filter the same definition is applied to
$\tilde D_i=(H(x\circ D))_i/((H\mathbf 1)_i x_i)$ (then $D$ is the raw exact gradient).

This is literally the operator in the code: `oc_core.State.s_mv` implements
$S v=v+p\,\operatorname{diag}(H\varepsilon)^{-1}H(2Gv-\varepsilon\circ v)$ (sensitivity filter; with $H=I$ it is raw
SIMP), and `objective_safe.Problem.evaluate().hvp` implements $\nabla^2C$ for the density filter.
Checked in `t1_consistency.py -> t1_consistency.json`: our dense $S$ vs `State.dense_s` (sens) rel. diff
$7\times10^{-11}$ (p=3) / $6\times10^{-10}$ (p=5); raw vs `dense_s` with $H=\tfrac12 I$: $4\times10^{-10}$/$9\times10^{-9}$;
density $S$ vs Astra `hvp` (MBB 30x10 r1.5 / cantilever 36x18 r2.25): $4\times10^{-11}$ / $2\times10^{-10}$;
central-FD Jacobian of our unclipped OC map vs $\Pi(I-\eta S)$: raw $\le2.2\times10^{-5}$, density
$\le5.7\times10^{-7}$, sens $\le4.8\times10^{-7}$; FD of the **repo** `c4_core.step` (move limit disabled
in-process): sens $4.8\times10^{-6}$, dens $7.6\times10^{-6}$; Astra `map_jvp` (density, $\eta=0.4$) vs
$\Pi(I-\eta S)$: $9.5\times10^{-13}$.

**Lemma 1 (log-Jacobian, faces, tangent).** Let $F$ be the set of outputs strictly inside their
bounds and $B$ its complement, with every $e\in B$ strictly clipped (raw value strictly beyond the
bound; strict complementarity). Put $m_F=w_F\circ x^+_F$,
$\Pi_F=I_F-\mathbf 1_F m_F^{\mathsf T}/(m_F^{\mathsf T}\mathbf 1_F)$, $T_F=\{y\in\mathbb R^F: m_F^{\mathsf T}y=0\}$.
Then
$$\frac{\partial z^+}{\partial z}=\begin{pmatrix}\Pi_F(I-\eta S)_{FF}&\Pi_F(-\eta S_{FB})\\0&0\end{pmatrix},\qquad
\operatorname{spec}\Big(\frac{\partial z^+}{\partial z}\Big)=\{0\}\cup\{1-\eta\sigma:\ \sigma\in\operatorname{spec}(S_T)\},$$
$S_T:=\Pi_FS_{FF}|_{T_F}$ (the *tangent-face s-operator*; in an orthonormal basis $Z$ of $T_F$ it is
$Z^{\mathsf T}\Pi_FS_{FF}Z$, exactly `oc_core.tangent_face_operator` / `c5_core.tangent_spectrum`).

*Proof.* Clipped outputs are locally constant, so their rows vanish. For $e\in F$:
$z^+_e=z_e+\eta(\log D_e-\log w_e)-\eta\log\lambda$; differentiating the constraint
$\sum_{F}w_ee^{z^+_e}+\sum_Bw_ex^+_e=V$ gives $m_F^{\mathsf T}dz^+_F=0$, which fixes $d\log\lambda$ and yields the
row block $\Pi_F(I-\eta S)_{F,:}$. The block-triangular form gives $\operatorname{spec}=\operatorname{spec}(J_{FF})\cup\{0\}$.
$J_{FF}=\Pi_F(I-\eta S_{FF})$ maps $\mathbb R^F$ into $T_F$ and $\mathbb R^F=T_F\oplus\operatorname{span}\mathbf 1_F$
($m_F^{\mathsf T}\mathbf 1_F>0$); in this splitting $J_{FF}=\left(\begin{smallmatrix}I-\eta S_T&*\\0&0\end{smallmatrix}\right)$. $\square$

Move limits are inactive in a neighbourhood of a fixed point (the unclipped update is continuous and
equals $x^*$ at $x^*$), so at fixed points only the physical bounds matter: they enter solely through
the face $F$.

---------------------------------------------------------------------------------------------------

## 2. Structure of $S$ for exact gradients

**Lemma 2 (self-adjointness).** For any $C$ with $D=-\nabla C>0$ put $g=x\circ D$, $\Gamma=\operatorname{diag}(g)$. Then
$$\Gamma S=X\,\nabla^2C(x)\,X\qquad(X=\operatorname{diag}x),$$
so $S$ is self-adjoint in $\langle y,z\rangle_g=y^{\mathsf T}\Gamma z$, $\operatorname{spec}(S)\subset\mathbb R$, and
$\sigma\in\operatorname{spec}(S)$ are the stationary values of the Rayleigh quotient
$$r(\delta x)=\frac{\delta x^{\mathsf T}\nabla^2C\,\delta x}{\sum_e D_e\,\delta x_e^2/x_e},\qquad \delta x=x\circ y .$$
*Proof.* $S_{ij}=x_j(\nabla^2C)_{ij}/D_i$, multiply row $i$ by $g_i=x_iD_i$. $\square$

**Lemma 3 (compliance Hessian in the filtered SIMP model).** With $Q:=\operatorname{diag}(\rho)^{-1}PX$
(entrywise $\ge0$, row sums $=1$; $Q=I$ for raw SIMP), $E_\varepsilon=\operatorname{diag}(\varepsilon)$ and the
*energy Gram matrix*
$$G_{ij}=a_ia_j\,(K_iu)^{\mathsf T}K^{-1}(K_ju)\qquad(G=V^{\mathsf T}K^{-1}V,\ V_{:,i}=a_iK_iu),$$
$$X\nabla^2C\,X=p\,Q^{\mathsf T}\big(2pG-(p-1)E_\varepsilon\big)Q,\qquad g=p\,Q^{\mathsf T}\varepsilon,\qquad
S=\operatorname{diag}(Q^{\mathsf T}\varepsilon)^{-1}Q^{\mathsf T}\big(2pG-(p-1)E_\varepsilon\big)Q .$$
*Proof.* $\partial c/\partial\rho_i=-E'(\rho_i)\varphi_i=-p\varepsilon_i/\rho_i$ and
$\partial^2c/\partial\rho_i\partial\rho_j=2u^{\mathsf T}K_i'K^{-1}K_j'u-\delta_{ij}u^{\mathsf T}K_i''u$ with
$K_i'=(pa_i/\rho_i)K_i$, $K_i''=(p(p-1)a_i/\rho_i^2)K_i$, i.e.
$R\,\nabla_\rho^2c\,R=p(2pG-(p-1)E_\varepsilon)$, $R=\operatorname{diag}\rho$. Chain rule
$\nabla^2C=P^{\mathsf T}\nabla^2_\rho c\,P$, $X P^{\mathsf T}R^{-1}=Q^{\mathsf T}$. $\square$
For raw SIMP this is $S=2p\,E_\varepsilon^{-1}G-(p-1)I$.

**Lemma 4 (energy Gram sandwich).** $0\preceq G\preceq\operatorname{diag}(\theta\circ\varepsilon)\preceq E_\varepsilon$.
*Proof.* $G=V^{\mathsf T}K^{-1}V\succeq0$. Write $K_i=L_iL_i^{\mathsf T}$. For $c\in\mathbb R^n$,
$v=Vc=\sum_ic_ia_iL_i\sigma_i$ with $\sigma_i=L_i^{\mathsf T}u$, and
$v^{\mathsf T}K^{-1}v=\sup_y\{2v^{\mathsf T}y-y^{\mathsf T}Ky\}
\le\sum_i\sup_{t\in\mathbb R^{r_i}}\{2c_ia_i\sigma_i^{\mathsf T}t-E_i|t|^2\}=\sum_ic_i^2\,a_i^2|\sigma_i|^2/E_i=\sum_ic_i^2\theta_i\varepsilon_i$
(we dropped nothing: $y^{\mathsf T}Ky=\sum_iE_i|L_i^{\mathsf T}y|^2$, and the sup over the independent
$t_i=L_i^{\mathsf T}y$ can only be larger). $\square$

*Numerical remark (why earlier float runs saw "violations" up to 4.78).* The proof uses **only** that
$V_{:,i}=a_iL_i\sigma_i$ and $\varepsilon_i=a_i|\sigma_i|^2$ come from the *same* element vector $\sigma_i$;
it does not need $\sigma_i=L_i^{\mathsf T}K^{-1}f$ exactly. Evaluating $K_iu_i$ and $u_i^{\mathsf T}K_iu_i$
separately in float64 breaks this consistency when $u_i$ is dominated by rigid-body motion (void
regions): the float $K_i$ annihilates rigid modes only to $\sim10^{-17}$, so $V_{:,i}$ acquires an
unbalanced (net-force) component that $K^{-1}$ amplifies by the stiffness contrast, and $\varepsilon_i$ can
even come out negative. Computing $\sigma_i=\Lambda^{1/2}U_5^{\mathsf T}u_i$ from the 5 non-rigid
eigenpairs of $K_e$ and forming both $V$ and $\varepsilon$ from it removes the artefact (§7, T2).

---------------------------------------------------------------------------------------------------

## 3. Theorem (spectral enclosure; raw SIMP and density filter)

**Theorem.** Under §1 (raw or density filter, exact gradient, any $E_{\min}\in[0,1)$, any $p>0$, any
$K_i\succeq0$, any load, 2D/3D, any filter $H\ge0$ with $H_{ii}>0$):

1. **(real, enclosed)** $\operatorname{spec}(S)\subset\big[-(p-1),\;2p\,\theta_{\max}-(p-1)\big]\subset\big[-(p-1),\;p+1\big]$
   (density filter with $p<1$: lower edge $\min\{0,1-p\}$, because $t\in[0,1]$ in the proof),
   $\theta_{\max}=\max_i a_i/E_i$ (upper edge $\max\{\cdot,0\}$ in the degenerate case
   $\theta_{\max}<\tfrac{p-1}{2p}$). Equivalently, for all $x>0$ and all $\delta x$:
   $$-(p-1)\sum_eD_e\frac{\delta x_e^2}{x_e}\;\le\;\delta x^{\mathsf T}\nabla^2C(x)\,\delta x\;\le\;(p+1)\sum_eD_e\frac{\delta x_e^2}{x_e}.$$
2. **(faces and tangent at fixed points)** Let $F$ be any index set and $m_F>0$ with $m_F\parallel g_F$.
   Then $S_T=\Pi_FS_{FF}|_{T_F}$ is the $g$-orthogonal compression of $S$ onto
   $\{y:\ y_B=0,\ g_F^{\mathsf T}y_F=0\}$, hence real and
   $\operatorname{spec}(S_T)\subset[\lambda_{\min}(S),\lambda_{\max}(S)]\subset[-(p-1),p+1]$.
   At every fixed point of the OC map (bound-clipped, move-limited, strict complementarity) the free
   set satisfies $D_F=\lambda w_F$, i.e. $g_F=\lambda m_F$, so 2 applies to the actual linearization.
3. **(any point, $E_{\min}=0$, nothing clipped)** If $E_{\min}=0$ then $S\mathbf 1=(p+1)\mathbf 1$ at every
   design; if moreover $F=\{1..n\}$, then for **every** weight $m>0$ (in particular off fixed points)
   $\operatorname{spec}(S_T)=\operatorname{spec}(S)\setminus\{p+1\}$ (one copy removed).
4. **(edges)** $p+1$ is attained by the scaling direction $\mathbf 1$ when $E_{\min}=0$; for raw SIMP,
   $\sigma=p+1$ on a direction $y$ iff the perturbation $\delta x=x\circ y$ leaves the element force
   field $a_iK_iu$ unchanged up to a displacement field (equality in Lemma 4 — "statically
   determinate" direction); $\sigma=-(p-1)$ iff $Gy=0$, i.e. $\sum_iy_ia_iK_iu=0$ (the perturbation does
   not change $u$ — "parallel redistribution"). Serial toy: $S=(p+1)I$; parallel toy (one DOF,
   $n$ springs): $S=-(p-1)I+2p\,\mathbf 1t^{\mathsf T}/\mathbf 1^{\mathsf T}t$, $t=x^p$, so the tangent spectrum is
   $\{-(p-1)\}$ with multiplicity $n-1$. Both edges are therefore sharp.

*Proof of 1.* By Lemmas 2–3 with $\zeta=Qy$ and $N(y)=y^{\mathsf T}\operatorname{diag}(Q^{\mathsf T}\varepsilon)y=\tfrac1p y^{\mathsf T}\Gamma y$,
$$r=\frac{2p\,\zeta^{\mathsf T}G\zeta-(p-1)\,\zeta^{\mathsf T}E_\varepsilon\zeta}{N(y)} .$$
Because $Q\ge0$ has unit row sums, Jensen gives $\zeta_i^2=(\sum_jQ_{ij}y_j)^2\le\sum_jQ_{ij}y_j^2$, so
$t:=\zeta^{\mathsf T}E_\varepsilon\zeta/N(y)\in[0,1]$ (for raw SIMP $t\equiv1$). Lemma 4 gives
$0\le\zeta^{\mathsf T}G\zeta\le\theta_{\max}\zeta^{\mathsf T}E_\varepsilon\zeta$, hence
$-(p-1)t\le r\le(2p\theta_{\max}-(p-1))t$. $\square$

*Proof of 2.* On the face, $S_{FF}$ is $g_F$-self-adjoint (principal submatrix of a symmetric pencil).
If $m_F=g_F/\lambda$, the $g_F$-orthogonal complement of $T_F$ is $\operatorname{span}\Gamma_F^{-1}m_F=\operatorname{span}\mathbf 1_F$,
so $\Pi_F$ (projection onto $T_F$ along $\mathbf 1_F$) is the $g_F$-orthogonal projector and
$S_T$ is a Ritz compression. Courant–Fischer (Cauchy interlacing) twice. $\square$

*Proof of 3.* With $E_{\min}=0$, $C(\tau x)=\tau^{-p}C(x)$ (also with the linear density filter), so by
Euler $\nabla^2C(x)\,x=(p+1)D$, i.e. $\Gamma S\mathbf 1=X\nabla^2C\,x=(p+1)\Gamma\mathbf 1$. $S$ leaves
$\operatorname{span}\mathbf 1$ invariant; $\Pi_m$ projects along $\mathbf 1$, so $S_T$ is the operator induced
by $S$ on $\mathbb R^n/\operatorname{span}\mathbf 1$, whose spectrum is $\operatorname{spec}(S)$ minus the
eigenvalue of the invariant line, independently of $m$. $\square$

*Proof of 4.* Edges of Lemma 4 / Jensen; toys by direct differentiation of $c=\sum_ix_i^{-p}$ and
$c=f^2/\sum_ix_i^p$. $\square$

**Remarks.**
(a) *Role of $E_{\min}$.* It never widens the interval. It shrinks the upper edge to
$p+1-2p(1-\theta_{\max})$, which is negligible whenever some element has $\rho\approx1$
($1-\theta_{\max}=E_{\min}/(E_{\min}+(1-E_{\min})\rho_{\max}^p)$). It leaves the lower edge. Its only
qualitative effect is to destroy $S\mathbf 1=(p+1)\mathbf 1$ (residual $O(1)$ on void elements where
$\rho^p\sim E_{\min}$: at $x_{\min}=10^{-3},p=3$, $\theta\approx0.5$), which removes part 3.
(b) *What is not enclosed.* Off fixed points, the tangent operator with the *actual* weights
$m_F=w_F\circ x_F^+$ is an oblique compression; with $E_{\min}>0$ or clipped variables it is not
covered, and it does leave the interval (§7, T3b: up to $6.205=1.034(p+1)$ at $p=5$, $4.0048$ at
$p=3$, all at extreme-contrast non-fixed designs; the excursion vanishes as $E_{\min}\to0$).
The *full* operator $S$ and the $g$-compressed tangent are enclosed at every design.
(c) *Equivalence with Svanberg (1994).* The lower bound is the log-coordinate second-order form of
"compliance is convex in $t=x^p$" (stiffness affine in $t$); the upper bound is "compliance is concave
in $1/t=x^{-p}$" (complementary energy $\min\{\sum_i|s_i|^2/E_i:\sum_iL_is_i=f\}$, and
$1/E_i=y/(E_{\min}y+1-E_{\min})$ is concave in $y=1/t$). Indeed
$S=(1-p)I+p^2\Gamma^{-1}T\,\nabla_t^2C\,T=(1+p)I+p^2\Gamma^{-1}Y\,\nabla_y^2C\,Y$. For the density
filter $C$ is *not* convex in $x^p$; the enclosure survives by the Jensen step (row-stochastic $Q$).
(d) *Scope.* Multi-load weighted compliance: both sides of 1 are additive, so it holds.
Not covered: design-dependent loads, non-power interpolations (RAMP), stress/other responses,
Heaviside projection, and the sensitivity filter (§6).

---------------------------------------------------------------------------------------------------

## 4. Corollaries (all in the linearization at a fixed point $x^*$ with strict complementarity; raw or density filter)

**C1 (universal flip-free band).** Every mode multiplier is $\mu=1-\eta\sigma$ with real
$\sigma\in[-(p-1),p+1]$. For $0<\eta<2/(p+1)$: $\mu>-1$ for all modes — no flip and (spectrum real) no
Neimark–Sacker mode, at **every** fixed point, mesh, load, filter radius. At $\eta=2/(p+1)$: $\mu\ge-1$
(marginal; equality needs a top-edge tangent mode, e.g. the serial toy).
**C2 (local linear convergence).** $\sigma_{\min}(S_T)>0$ iff the reduced Hessian of $C$ on the face
tangent is positive definite (Lemma 2: $S_T$ is the Rayleigh quotient of $\nabla^2C$ in the metric
$\sum_eD_e\delta x_e^2/x_e=\lambda\sum_ew_e\delta x_e^2/x_e$). Then for every
$\eta\in(0,2/(p+1))$, $|\mu|<1$ for all modes: $x^*$ is linearly attracting with rate
$\max\{|1-\eta\sigma_{\min}|,|1-\eta\sigma_{\max}|\}$. If $\sigma_{\min}<0$ (saddle on the face),
$\mu>1$ for every $\eta>0$ (the "negative branch": divergence, not oscillation).
**C3 (deadbeat).** $\eta=1/(p+1)$ annihilates a top-edge mode in one step ($\mu=0$); for
$\eta\le1/(p+1)$ all positive-branch multipliers lie in $[0,1)$ (monotone, over-damped).
**C4 (Groenwold–Etman dictionary, attributed).** OC with damping $\eta$ is dual SAO with exponential
intervening variables $x^a$, $\eta=1/(1-a)$ [Groenwold & Etman 2008]. In log coordinates the separable
model has curvature $(1/\eta-1)\,g=-a\,g$, i.e. *model s-value* $1-a$ on every mode; $\mu=1-\sigma/(1-a)$.
Hence: universally flip-free $\iff a<(1-p)/2$; top-edge deadbeat $\iff a=-p$ (Svanberg's concave
variable $x^{-p}$, exact for statically determinate structures); reciprocal $a=-1$ ($\eta=\tfrac12$) has
model s-value 2 $=\tfrac12(p+1)$ at $p=3$ — exactly the flip edge. For $p=1$ (VTS/trusses) $a=-1$ is
deadbeat, the classical exponent-½ recurrence. The separable model is locally conservative in all
directions at all designs iff $1-a\ge s_{\max}$, guaranteed by $a\le-p$.
**C5 (move limits / clipping).** Neither changes the thresholds (face argument, Lemma 1).
**C6 (non-universality of the optimal $\eta$).** The enclosure bounds the flip threshold from below,
$\eta^*=2/s_{\max}(S_T)\ge2/(p+1)$; it does not predict $\eta^*$ (measured $s_{\max}/(p+1)=0.11$–$0.80$ at
the 45 Astra density-filter fixed points, §7).

---------------------------------------------------------------------------------------------------

## 5. What this says about $(p,\eta)=(3,\tfrac12)$

For exact-gradient OC (raw/density), $\eta=\tfrac12=2/(p+1)$ at $p=3$ is the largest fixed damping that is
flip-free at every fixed point of every problem; it is never unstable by flip, only marginal when a
tangent mode reaches the top edge. The repo's sensitivity-filter saturation law
$s_{\max}/(p+1)=0.961$–$0.995$ is consistent with approaching the top edge from below — but for the
sensitivity filter the edge is *not* a theorem (§6).

---------------------------------------------------------------------------------------------------

## 6. Sensitivity filter (top88, our paper's main regime)

$\tilde D_i=p(H\varepsilon)_i/((H\mathbf 1)_ix_i)$ gives
$$\tilde S=I+p\,\operatorname{diag}(H\varepsilon)^{-1}H(2G-E_\varepsilon)=I+p\,W\!B,\qquad
W=\operatorname{diag}(H\varepsilon)^{-1}HE_\varepsilon,\quad B=2E_\varepsilon^{-1}G-I .$$
$W\ge0$ is row-stochastic (reversible w.r.t. $\pi=\varepsilon\circ H\varepsilon$), $B$ is
$E_\varepsilon$-self-adjoint with spectrum in $[-1,2\theta_{\max}-1]$. Equivalently $\tilde S-I=W(S_{\rm raw}-I)$.

**Proposition (what can be proved).**
(i) $\tilde S\mathbf 1=(p+1)\mathbf 1$ when $E_{\min}=0$ ($G\mathbf 1=V^{\mathsf T}K^{-1}f=\varepsilon$); with no clipped variable the
tangent spectrum is $\operatorname{spec}(\tilde S)\setminus\{p+1\}$ for any weights (same quotient argument).
(ii) **Disc bound.** $\|W\|_{E_\varepsilon}^2\le c_{\max}:=\max_j\sum_iH_{ij}\varepsilon_i/(H\varepsilon)_i\le\max_j\sum_iH_{ij}/H_{ii}$
(Jensen/Schur test), $\|B\|_{E_\varepsilon}\le1$, hence every eigenvalue of $\tilde S$ and of every principal
submatrix $\tilde S_{FF}$ lies in the closed disc $|s-1|\le p\sqrt{c_{\max}}$. Design-independent
version for the top88 hat filter: radius $p\sqrt{\max_j(H\mathbf1)_j/r_{\min}}$ ($=1.17p$ at $r_{\min}=1.1$).
(iii) Nothing better holds in general: $\tilde S$ is not similar to a symmetric matrix
(product of operators self-adjoint in different metrics), complex modes occur, and a disc centred at
1 of radius $\ge1$ contains near-imaginary points with $2\operatorname{Re}s/|s|^2\to0$, so no
$\eta$-universal NS-freeness follows. The tangent projection at a fixed point is oblique in the
$E_\varepsilon$-metric, so the disc bound does not even transfer to $S_T$ on faces.

**Numerical status (§7).** (1) *Operator level / weighted-volume fixed points:* the top edge is
**not** an enclosure for the sensitivity filter: the adversary (T4) finds tangent $s_{\max}$ up to
$6.0622=1.0104(p+1)$ (MBB 20x10 $r_{\min}1.5$, $p=5$) and $4.0248=1.0062(p+1)$ ($p=3$), 33/36
configurations above $p+1$, margins $\le1.04\%$. T6 confirms two of them independently: FD Jacobian of
the nonlinear OC map (6.06216 vs 6.06222; 6.03774 vs 6.03760) and a kick test at
$\eta=0.999\cdot2/(p+1)=0.333$ growing with measured ratio $-1.0187$ (pred. $-1.0187$) and $-1.0105$
(pred. $-1.0105$). These designs are exact fixed points of OC with volume weights $w=\tilde D(x^*)$
(not of the standard problem), and are saddles (negative real branch present). (2) *Linearly stable*
weighted fixed points above the edge also exist (T8: 4/18 starts, best $1.0052(p+1)$ with all
$\operatorname{Re}s>0$, min $\operatorname{Re}s=0.0073$). (3) *Genuine standard-volume fixed points*
reached by OC runs (T5, 65 designs incl. the repo p-sweep): $s_{\max}/(p+1)\le0.9964$,
$(p+1)\min\operatorname{Re}s/|s|^2\ge1.0036$ — no counterexample; complex modes are present at all
54 fresh fixed points (max $|\operatorname{Im}s|=0.12$) but none is critical. (4) The sensitivity filter
also admits complex tangent pairs on the imaginary axis ($s\approx\pm0.06i\ldots\pm0.42i$, Re $\sim10^{-7}$,
found at saddle designs), so "stable positive branch" is not $\eta$-universal there either.
(5) Low-dimensional optima transferred to 40x20/60x20 keep small excursions (max $1.0008/1.0007$).

---------------------------------------------------------------------------------------------------

## 7. Numerical verification (what was run)

**T1 operator identity** (`t1_consistency.py` → `t1_consistency.json`): see §1 (all rel. diffs ≤ 2.2e-05); $\|S\mathbf 1-(p+1)\mathbf 1\|_\infty$ = 1.8e-12/1.3e-13/3.8e-13 (raw/dens/sens, $E_{\min}=0$), 5.4e-07/1.9e-07/3.6e-07 ($E_{\min}=10^{-9}$, x∈[0.2,1]).

**T2 precision** (`t2_precision.py` → `t2_precision.json`; 8x4 MBB r1.5 and cantilever r2.0, raw+density, p∈{3,5}, x log-uniform down to 1e-2/1e-3/1e-4, 24 designs): naive float64 (separate $K_iu_i$ and $u_i^{\mathsf T}K_iu_i$) exceeds $p+1$ in 5 cases (max excess 10.30) and fails (negative $\varepsilon_i$, NaN) in 3; mpmath 50 digits with the exact rational $K_e$: max excess -2.5e-08 (upper), -2.1e-14 (lower) — **no violation**; range-consistent float64 agrees with mpmath to 8.9e-10.

**T3 main verification** (`t3_verify.py 0|1 2` → `t3_verify_{0,1}of2.json`, `t3_summary.py` → `t3_summary.json`): 550 (design, semantics, p) evaluations on 107 distinct design vectors (50 synthetic: log-uniform with floors 0.2/1e-2/1e-3 and smooth random fields, on MBB 30x10 r1.5, cantilever 24x12 r2.0, MBB 45x15 r1.8, cantilever 36x18 r2.25, cantilever 40x20 r2.5; 45 Astra result designs (seed/uniform/continuation/best) at native mesh; 12 repo designs: c4_V3/V4/V5, c5_psweep_ref p1.5–p5, c5_negbranch x_end at 60x20 r1.1/2.4); raw 269, density 281; p ∈ {1.1,1.5,2,2.5,3,3.5,4,4.5,5}.

| operator | n | violations of $[-(p-1),p+1]$ | max excess over $p+1$ | max excess below $-(p-1)$ | note |
|---|---|---|---|---|---|
| full $S$ (Ssym, eigvalsh) | 550 | 0 | -3.9e-09 | -1.8e-15 | refined edge $2p\theta_{\max}-(p-1)$: 0 violations; top gap $(p+1-s_{\max})/(p+1)$ ∈ [1.9e-09, 0.014] |
| $g$-compressed tangent (proved at every design) | 550 | 0 | -3.2e-08 | -5.3e-14 | max $s_T/(p+1)$ = 0.9999999947 (edge sharp even after removing $\mathbf 1$) |
| code tangent, Astra fixed points, native p | 45 | 0 | -0.429 | -0.099 | KKT spread ≤ 6.9e-03; max|Im| = 0.0e+00; |code − g-tangent| top ≤ 1.1e-05; $s_{\max}/(p+1)$ ∈ [0.11, 0.80] |
| code tangent off fixed points (oblique; NOT covered) | 505 | 15 up / 13 low | 0.205 | 2.7e-04 | 14 of the upper excursions are all-free designs; max|Im| 1.10 |

**T3b** (`t3b_offkkt.py` → `t3b_offkkt.json`): the off-fixed-point excursions are real (naive and similarity-scaled evaluations agree to all printed digits) and are an $E_{\min}$ effect: for the same designs the excursion of the all-free tangent vanishes as $E_{\min}\to0$:

| design | p | E_min | code tangent max Re | $\|S\mathbf1-(p+1)\mathbf1\|_\infty$ | $\theta_{\min}$ |
|---|---|---|---|---|---|
| smooth_b6_0 mbb45x15r1.8 raw | 5 | 1e-09 | 6.2049 | 16 | 1.01e-06 |
| smooth_b6_0 mbb45x15r1.8 raw | 5 | 1e-12 | 6.0000 | 25 | 0.00101 |
| smooth_b6_0 mbb45x15r1.8 raw | 5 | 1e-15 | 6.0000 | 4.2 | 0.503 |
| logunif0.01_0 cant40x20r2.5 raw | 5 | 1e-09 | 6.0187 | 5.6 | 0.0975 |
| logunif0.01_0 cant40x20r2.5 raw | 5 | 1e-12 | 6.0000 | 0.0096 | 0.991 |
| logunif0.01_0 cant40x20r2.5 raw | 5 | 1e-15 | 6.0000 | 2.9e-05 | 1 |
| logunif0.01_0 cant24x12r2.0 raw | 5 | 1e-09 | 6.0126 | 6 | 0.0956 |
| logunif0.01_0 cant24x12r2.0 raw | 5 | 1e-12 | 6.0000 | 0.004 | 0.991 |
| logunif0.01_0 cant24x12r2.0 raw | 5 | 1e-15 | 6.0000 | 2.5e-05 | 1 |
| logunif0.001_1 mbb30x10r1.5 raw | 3 | 1e-09 | 4.0048 | 0.83 | 0.526 |
| logunif0.001_1 mbb30x10r1.5 raw | 3 | 1e-12 | 4.0000 | 0.0011 | 0.999 |
| logunif0.001_1 mbb30x10r1.5 raw | 3 | 1e-15 | 4.0000 | 1.9e-06 | 1 |

**T4 sensitivity-filter adversary** (`t4_sens_adv.py 0|1 2` (MBB, DE maxiter 20 + ES 300) and `t4_sens_adv.py 0|1 2 cant 8 200` (cantilever, reduced budget DE maxiter 8 + ES 200 for wall time) → `t4_sens_adv_{0,1}of2.json`, `t4_sens_adv_cant_{0,1}of2.json` (+designs .npz), `t4_summary.py` → `t4_summary.json`): 72 runs, 126072 exact-operator evaluations (20x10, MBB+cantilever, rmin 1.1/1.5/2.4, p 2/3/5, free/faced families; DE on 16 cosine coefficients + per-element (1+1)-ES). r1 > 1 in 33/36 configurations; r2 < 1 in 31/36.

| worst r1 configs | r1 | $s$ (top) | min Re s | found by | r1 with standard-volume weights |
|---|---|---|---|---|---|
| mbb r1.5 p5.0 free | 1.01037 | 6.0622 | -3.954 | es (DE only 1.00133) | 0.99988 |
| cant r1.1 p5.0 faced | 1.00792 | 6.0475 | -3.994 | es (DE only 0.99993) | 1.00783 |
| cant r1.5 p5.0 free | 1.00687 | 6.0412 | -3.924 | es (DE only 1.00012) | 0.99024 |
| cant r2.4 p5.0 free | 1.00627 | 6.0376 | -3.731 | es (DE only 1.00054) | 1.00627 |
| mbb r1.1 p3.0 free | 1.00621 | 4.0248 | -1.554 | es (DE only 1.00038) | 1.00611 |
| mbb r2.4 p3.0 free | 1.00610 | 4.0244 | -0.579 | es (DE only 1.00165) | 1.00522 |

| worst r2 configs | r2 | critical mode $s$ | found by |
|---|---|---|---|
| mbb r1.5 p5.0 free | 0.00001 | 0.00000 ± 0.41666i | es (DE only 0.56939) |
| cant r2.4 p5.0 free | 0.00004 | 0.00000 ± 0.29265i | es (DE only 0.41104) |
| cant r1.5 p5.0 free | 0.00010 | 0.00000 ± 0.12809i | es (DE only 0.27569) |
| cant r1.1 p3.0 free | 0.00029 | 0.00000 ± 0.05883i | es (DE only 0.65535) |
| cant r1.5 p2.0 free | 0.00029 | 0.00000 ± 0.12059i | es (DE only 0.40620) |
| mbb r1.5 p3.0 free | 0.00048 | 0.00000 ± 0.19569i | es (DE only 0.38975) |

Worst r2 whose critical mode sits at the top edge (Re s ≥ 1): mbb r2.4 p3.0 faced r2 = 0.98810, s = 4.0482 ± 0.0000i.

**T6 independent confirmation** (`t6_sens_confirm.py` → `t6_sens_confirm.json`): weighted-volume OC map ($w=\tilde D(x^*)$, so $x^*$ is an exact fixed point), central-FD Jacobian of the nonlinear map and kick tests.

| design | fp residual | dense top s | FD top s | kick at η=0.999·2/(p+1): pred μ / meas | kick at 0.98·2/s_max | kick at 1.02·2/s_max |
|---|---|---|---|---|---|---|
| mbb_r1.5_p5.0_free_r1 | 1.3e-15 | 6.06222+0.00000i | 6.06216+0.00000i | -1.0187 / -1.0187 | -0.9600 / -0.9600 | -1.0400 / -1.0400 |
| cant_r1.5_p5.0_free_r1 (**FD/kick unreliable**: free family floor x=0.01 gives $x^5=10^{-10}<E_{\min}$, map noise ~1e-6) | 6.7e-16 | 6.04124+0.00000i | 22.00560+0.00000i | -1.0117 / -1.1908 | -0.9600 / -1.1067 | -1.0400 / -0.9827 |
| cant_r2.4_p5.0_free_r1 | 4.4e-16 | 6.03760+0.00000i | 6.03774+0.00000i | -1.0105 / -1.0105 | -0.9600 / -0.9600 | -1.0400 / -1.0400 |
| mbb_r1.5_p5.0_free_r2 | 6.7e-16 | 5.99866+0.00000i | 5.92011+0.00000i | -0.9976 / -0.9971 | -0.9600 / -0.9581 | -1.0400 / -1.0400 |
| cant_r2.4_p5.0_free_r2 | 6.7e-16 | 5.99340+0.00000i | 5.99340+0.00000i | -0.9958 / -0.9958 | -0.9600 / -0.9600 | -1.0400 / -1.0400 |

Rows 4–5 kick the *top real* mode of the r2-worst designs (not their near-imaginary pair); their near-imaginary pairs have dense Re s = 1.5e-7 / 1.2e-7 (Im 0.417 / 0.293), i.e. they sit on the imaginary axis, and those designs are saddles (min Re s = −3.13 / −1.26, `t4_summary.json`).

**T7 mesh transfer of the DE (low-dimensional) r1 optima** (`t7_sens_transfer.py` → `t7_sens_transfer.json`): max r1 at 20x10 (DE) 1.00286, at 40x20 1.00084, at 60x20 1.00074; configs with r1>1: 40x20 7/36, 60x20 5/36.

**T5 genuine sensitivity-filter fixed points** (`t5_sens_fixedpoints.py` → `t5_sens_fp.json`; standard volume, code tangent with $m_F=x_F$): 
repo designs (11): r1 ∈ [0.7220, 0.9953], min r2 1.0047, max|Im| 0.037; fresh fixed points (54, 20x10 only — the 40x20 leg was dropped for wall time; MBB/cantilever, rmin 1.1/1.5/2.4, p 2/3/5, η=0.3, move 0.2, 500 its from uniform + 2 random starts; 45/54 reach max|Δx| < 1e-4, worst 8.5e-2): r1 ∈ [0.5405, 0.9964], min r2 1.0036, designs with complex modes 54/54, max|Im| 0.123, r1>1: 0, r2<1: 0.

**T8 constrained adversary from genuine fixed points** (`t8_sens_stable_adv.py` → `t8_sens_stable_adv.json`): 18 starts × 400 evals; best r1 with all Re s > 0: 1.00520; stable-and-above-edge found: 4/18.

**Not run:** 3D meshes; RAMP/other interpolations; multi-load; Heaviside projection; interval/rational certification of any float eigenvalue; mpmath beyond 8x4 meshes; adversarial search beyond 20x10 (only transfer of low-dim optima to 40x20/60x20); gradient-based adversary; sensitivity-filter counterexample at a *standard-volume* fixed point reached by an actual OC run.

---------------------------------------------------------------------------------------------------

## 8. Novelty (literature check, 2026-09-29)

Verdict: **folklore-implied for the raw-SIMP full operator; not found stated anywhere for the OC
s-operator, the tangent/face version, the density-filter extension, or the universal band
$\eta<2/(p+1)$.** Our own preprint (engrXiv 7997, the flip law $\eta^*=2/s_{\max}$) is the only
search hit on an OC stability threshold. Details:

| Source | Verified | What it states | Relation |
|---|---|---|---|
| Svanberg, *On the convexity and concavity of compliances*, Struct. Optim. 7(1–2):42–46, 1994, doi:10.1007/BF01742502 | Crossref metadata + Springer abstract | compliance convex in thickness $t$, concave in $1/t$; used for global convergence of a method for min-weight with compliance constraints | with $t=x^p$ the raw-SIMP enclosure of $S$ is its second-order log-coordinate form (Remark (c)). **Must be cited as the source of the inequality.** Neither OC damping nor $2/(p+1)$ appear in the abstract. |
| Groenwold & Etman, IJNME 73(3):297–316, 2008 (online 2007), doi:10.1002/nme.2071 | Crossref (earlier phase) + abstract | OC with damping $\eta$ ≡ dual SAO with exponential intervening variables; $\eta=0.5$ ≡ reciprocal; exponential variables more efficient | the $\eta=1/(1-a)$ dictionary (C4) is theirs — attribute; no stability threshold in the abstract. |
| Groenwold & Etman, IJNME 82(4):505–524, 2010, doi:10.1002/nme.2774 | abstract | diagonal quadratic approximation of reciprocal/exponential models | model-curvature view of C4 (curvature $-a g$); no threshold stated in abstract |
| Fleury & Braibant, IJNME 23:409–428, 1986; Fleury, CONLIN, Struct. Optim. 1:81–89, 1989 | publisher pages/abstracts | conservative mixed direct/reciprocal convex approximations | conservativeness = model curvature ≥ true curvature, i.e. the over-damped side $a\le -p$ in C4 |
| Berke, *Convergence behavior of optimality criteria based iterative procedures*, AFFDL-TM-72-1-FBR, 1972 | citation only (report not accessible) | convergence of OC recurrences (trusses) | **possible precedent** for the $p=1$ deadbeat/one-step behaviour of exponent ½ on statically determinate structures; unverified |
| Ananiev, arXiv math/0609218, 2006 | arXiv abstract | OC ≡ projected gradient | no spectral/damping analysis |
| Munro | — | — | **not cited** (CLAUDE.md trap 8) |

Web queries run (WebSearch, 2026-09-29): Svanberg 1994 abstract; OC convergence/damping/Jacobian
eigenvalues; "optimality criteria" with "2/(p+1)" or "1/(p+1)"; OC "period-doubling"/"flip
bifurcation"/"spectral radius"; compliance convex in $x^p$/concave in $x^{-p}$; Groenwold–Etman
exponential/reciprocal; Berke 1972; Fleury/CONLIN. No source states
"$\operatorname{spec}\subset[-(p-1),p+1]$" for the OC log-Jacobian, the Jensen extension to the density
filter, or "$\eta<2/(p+1)$ is flip-free at every fixed point". Not searched: paywalled full texts
(Wiley/Springer bodies were 403), Scopus/WoS citation graphs of Svanberg 1994.

**How to state it in the paper:** "The enclosure is the log-coordinate second-order form of Svanberg's
(1994) convexity/concavity of compliance, transported to SIMP by $t=x^p$ and to the density filter by
a Jensen step; its OC consequences (C1–C4) are, to our knowledge, not stated before." Do not claim the
inequality itself as new.

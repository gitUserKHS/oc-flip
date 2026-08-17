# 논문 골격 (사이클 5) — OC 감쇠 사상의 동역학

**날짜:** 2026-08-16 · **상태:** 관문 1(강건성)·2(선행권) 통과 후 확정 골격. 타깃: *Structural and Multidisciplinary Optimization* (SMO) 정규 논문.

---

## 0. 한 줄 요약

감쇠 OC 갱신을 로그좌표 동역학계로 보면 국소 승수는 μ = 1 − η·s (s = 로그-로그 민감도 강성)이고, 정정계 극한에서 s → p+1이므로 표준 (p, η) = (3, ½)은 flip 분기 경계 위에 있다 — 이를 top88에서 연산자 수준으로 실측했고, 밀도필터가 이 불안정을 통째로 해체함을 보인다.

## 1. 제목 후보

1. **"The damped optimality-criteria update as a dynamical system: multiplier law, flip threshold, and the marginality of η = ½"** ← 1순위
2. "Why η = ½ barely works: measured stability spectra of the OC iteration in SIMP topology optimization"
3. "Local multipliers and flip thresholds of the optimality-criteria map"

## 2. 영문 초록 초안 (v0)

> The optimality-criteria (OC) update with numerical damping η is the workhorse of density-based topology optimization, yet the choice η = ½ has remained folklore. We analyze the damped OC map as a discrete dynamical system in logarithmic coordinates and show that its local multiplier obeys the affine law μ = 1 − η·s, where s is the log–log stiffness of the sensitivity field. Exact two-bar models give s = p + 1 in the statically determinate limit, so the standard pair (p, η) = (3, ½) sits precisely on the period-doubling (flip) boundary η* = 2/s; the one-step-convergent damping η = 1/(p+1) is the midpoint of the stability window. On an instrumented 88-line benchmark we verify the law at the operator level: finite-difference Jacobians of the full update yield η-invariant s-spectra whose dominant value saturates near p + 1 (margin 0.003 for the canonical 60×20 MBB problem), the leading eigenmode coincides with the observed oscillation pattern, and sub-threshold decay rates match |1 − η·s_max| to three decimals. A robustness sweep across mesh, boundary conditions, volume fraction, and filter type confirms η* = 2/s_max throughout (s_max ∈ [2.9, 4.0] for sensitivity filtering), while density filtering collapses s_max to 0.37, detuning the flip instability entirely. Combined with the known equivalence of η = ½ to reciprocal intervening variables [Groenwold & Etman 2008], our result identifies the folklore damping as an approximation whose curvature is exactly half the true curvature at p = 3 — the statics and the dynamics of the OC method meet at the same constant. Beyond convergence, saturated oscillations self-organize onto η·ŝ = 2, and designs partially adapt their own capacity 2/s_max under continuation, explaining the filter-dependent quality collapse of over-driven runs.

(길이 ~230 단어. 다이어트 시 마지막 두 문장 축약.)

## 3. 기여 목록 (Introduction 끝에 그대로)

1. **법칙:** 로그좌표 국소 승수 μ = 1 − η·s; 안정창 (0, 2/s); 1-스텝 η = 1/s; 정정계 s = p+1 (직렬 토이 정확해, 병렬 s = −(p−1) → bang-bang).
2. **한계성:** 표준 (3, ½)이 flip 경계 위 — top88 실측 여유 0.003 (R0), 계열 전반 0–10%.
3. **측정 프로그램:** FD 야코비안의 η-불변 s-스펙트럼, 야생 항등식 μ̂ = 1−ηŝ (28실행 3자리), 모드=진동패턴 (|cos| ≥ 0.93), 문턱하 감쇠율 3자리 일치.
4. **동역학 현상:** 포화 진동의 자기조직화 η·ŝ = 2, 설계 용량 적응(자기안정화)과 필터 역설(과구동 시 컴플라이언스 216→346).
5. **실무 결론:** 밀도필터의 flip 소멸(s_max = 0.37, η\* ≈ 5.4) — "진동하면 η ↓ 또는 밀도필터"를 같은 s로 정당화.
6. **종합:** G–E의 "η=½ ≡ 역수 근사"와 결합 — 역수 곡률 = p=3 참곡률의 절반 = 승수 −1. 근사이론과 분기이론의 합류.

## 4. 섹션 플랜 (내용 → 근거 → 그림)

**§1 Introduction** — Bendsøe η 도입, Abaqus 폴클로어 성문화("지수 작을수록 안정·느림"), 이론 부재 지적, 기여 목록. [그림 없음]

**§2 Related work** — 세 갈래로 정리: (a) 문제 수준 수렴(Rietz, Stolpe–Svanberg, Martínez — 우리와 직교), (b) 반복=동역학계(Mueller–Burns 계보, η=1 프레임), (c) η의 정역학(Groenwold–Etman 2008 동치; Svanberg CCSA의 비보수→순환), (d) 현대 수렴 알고리즘(Ananiev, Li–Paulino, SiMPL). 각각과의 차이 한 문장씩.

**§3 Theory** — OC 사상 로그좌표 유도 → μ = 1−η·s → 2-바 정확해(직렬/병렬) → 안정창·1-스텝·flip 경계 → SAO 사전(η = 1/(1−a); a=−p ⇔ 정확·1-스텝; a=−1 ⇔ η=½ ⇔ 곡률 절반 ⇔ μ=−1@p=3) → 이동제한·클리핑의 문턱 불변(진폭만 결정). [Fig 1: 토이 도식 + 분기도, fig_cycle1 재구성]

**§4 Measurement methodology** — 계측 top88, 자유집합 로그 FD 야코비안(ε=1e-5, 부피 널모드 소거), 아놀디 확장(평균중심 deflation), 모드정렬 x-공간 프로브, 고유쌍 잔차 검증 절차. 재현성 서술(코드 공개).

**§5 Results**
- 5.1 야생 항등식: μ̂ = 1−ηŝ, 28실행 3자리. [Fig 2: 산점 y=x]
- 5.2 스펙트럼: η-불변(0.5/0.9 빌드 3자리 동일), s_max = 3.974 ≈ p+1, η\* = 0.503. [Fig 3: 스펙트럼 + 상위값 표]
- 5.3 모드 동정: 최상위 고유모드 = 관측 진동패턴, rmin1.1 국소 패치 vs rmin2.4 광역 2엽 → 품질파괴 기제. [Fig 4: 모드맵 대비]
- 5.4 온셋·자기조직화·용량 적응: full-run 온셋(0.8–0.9) ≫ 고정설계(0.5–0.6) → 자기안정화; 포화 2-주기 η·ŝ = 2 (1.996/1.667×1.2/1.250×1.6); design(η) 용량 궤적(부분 적응 vs 동결)과 필터 역설. [Fig 5: 용량 궤적 + 컴플라이언스]
- 5.5 강건성: R0+V1–V5 표, 예측 vs 온셋 브래킷, 문턱하 3자리 일치. [Fig 6: cycle4 3패널 재구성]
- 5.6 밀도필터: s_max 0.37, flip 소멸, 실무 지침. [Fig 6 하단 또는 Fig 7]

**§6 Discussion** — (a) η 선택 지침: η < 2/s_max, vf 낮을수록 여유↑(V4), (b) 온라인 ŝ 측정 → 적응 감쇠 제어기 가능성(주장 아닌 전망), (c) 정역학-동역학 합류의 의미, (d) 왜 p+1 포화인가(국소 정정 하위구조 가설 — 증명은 미해결로 명시).

**§7 Limitations** — 국소 이론(전역 아님), 2D 컴플라이언스, 단일 시드/브래킷 폭 0.08, V5 정성 검증, V1 크립 꼬리, 복소 모드(비지배) 미탐구.

**§8 Conclusion** — 한 줄 요약 재진술 + "35년 폴클로어에 상수 하나(2/s)를 돌려줌".

## 5. 그림 플랜 (총 6–7개)

| # | 내용 | 소스 |
|---|---|---|
| 1 | 토이 도식·분기도·승수 직선 | fig_cycle1 재작업 |
| 2 | 항등식 산점(28실행) | fig_cycle2 |
| 3 | η-불변 s-스펙트럼 | fig_cycle3 |
| 4 | 고유모드 vs 진동패턴 맵 | fig_cycle3 |
| 5 | 용량 적응 궤적 + 필터 역설 | fig_cycle2c/3 |
| 6 | 강건성(막대·예측vs온셋·3자리 산점) | fig_cycle4 |
| 7 | 밀도필터 설계·스펙트럼 붕괴 (6에 합칠 수도) | fig_cycle4 |

## 6. 주장 다이어트 (하지 않을 주장)

- 전역 수렴 정리 ✗ (국소 선형화만)
- "η=½ ≡ 역수"의 발견 ✗ (G–E 귀속)
- "반복=동역학계" 관점의 최초 ✗ (Mueller–Burns 귀속)
- MMA/GCMMA에 대한 진술 ✗ (범위 밖, future work)
- 1-스텝 η=1/(p+1)의 발견 ✗ (G–E 틀에 암묵 내포 — "재해석·명시화"로 서술)
- 임계지수/보편류 ✗ (β 측정 전까지 보류 — 통계역학 논문 몫)

## 7. 투고 전략

- **1지망 SMO** (Groenwold–Etman·Martínez·Sigmund의 홈그라운드, 독자 적합). 정규 논문 ~25–30쪽.
- 예비: CMAME(방법론 강조 시), IJNME.
- 통계역학 관찰 논문(β, 자기조직 임계성)은 **별도 2편**으로 분리 — 본편에 씨앗 문장만.
- 코드·데이터: GitHub 공개(재현 스크립트 일체) + Zenodo DOI.

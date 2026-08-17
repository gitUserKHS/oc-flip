# 제출 노트 — engrXiv (2026-08-17)

## 최종 상태
- `paper/main.pdf` 23쪽 (본문 15 + 그림 10장 + 부록). TODO 0건, 미해결 참조 0건, bibtex 에러 0건.
- 저자: Hyeongseok Kim, Independent Researcher, Uiwang, Republic of Korea.
- 데이터 가용성: **https://github.com/gitUserKHS/oc-flip 직접 인용** (2026-08-17 공개, 96파일). 아카이브 스냅샷 DOI는 후속 버전에 추가 예정.

## 서지 전수 검증 (2026-08-17, Crossref/출판사 대조)

**수정 1건 (낡은 인용)**
- `keith2024simpl` (arXiv:2409.19341) → **`keith2025simpl`: SIAM J. Optim. 35(2):1134–1164 (2025), doi 10.1137/24M1708863**. 본문 인용 키도 교체.

**보강 (권호·DOI 추가; 기존 값은 전부 정확)**
bendsoe1989 1(4), BF01650949 · sigmund2001top99 s001580050176 · andreassen2011top88 s00158-010-0594-7 · rietz2001 21(2), s001580050180 · stolpe2001 22(2), s001580100129 · mueller2001 52(**12**) · razani1965 **10.2514/3.3355**(AIAA J판; Crossref 최상단은 학회논문판 10.2514/6.1965-76이라 주의) · groenwold2007ise s00158-006-0070-6 · svanberg1987mma nme.1620240207 · svanberg2002ccsa S1052623499362822 · sigmund1998numerical 16(1), BF01214002 · bourdin2001 nme.116

**연도 확인 (온라인 우선공개 vs 인쇄)**
- groenwold2008: Crossref issued 2007이나 **published-print 2008-01-15** → 2008 유지 (본 논문 최대 귀속 인용이라 확인 필수였음).
- andreassen2011top88: online 2010-11-20, **print 2011-01** → 2011 유지.

**그대로 정확** martinez2005 · mueller2002 · liu2003 · groenwold2009gss · ananiev2005 · li2020fixedpoint · kim2025simpl · irons1969 · kuettler2008 · barzilai1988. patnaik1998mfud는 권호·쪽수 확인, DOI는 Crossref에 별칭(.3.co;2-f)만 있어 미기재.

## engrXiv 제출 체크
- 분류: 공학(구조/최적화), 학술 프리프린트 — 조건 충족.
- 저자 표기 정확, 공유 권리 보유 — 조건 충족.
- Declarations/RR/이해충돌/커버레터/추천 리뷰어: 불필요.
- 업로드 파일: `paper/main.pdf` 하나.

## 다음 (v2)
0. ~~코드 공개~~ 완료 — GitHub 96파일 push, gh API로 검증.
1. Zenodo 기탁(`ZENODO.md` 메타데이터 준비 완료, `oc-flip_zenodo.zip` 94파일 6.2MB) → 라이선스만 선택.
2. 기탁 DOI를 데이터 가용성 문단에 넣어 engrXiv v2 갱신(영구 식별자라 링크 안 깨짐).
3. 이후 저널 경로를 재개하면 `notes/review3_note.md`의 리비전 등급 항목부터.

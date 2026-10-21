# Reorganization Plan

## 목적

현재 자산을 ‘selector 우월성 탐색’ 중심이 아니라 ‘selection-stage signal과 downstream learning utility의 연결/불일치에 대한 empirical audit’ 중심으로 재배치한다. 이 문서는 실행 계획만 제시하며 이번 작업에서는 어떤 원본도 이동·이름변경·삭제하지 않았다.

## 제안 논리 구조

| 논문 장 | 핵심 질문 | 우선 자산 | 금지되는 확대 해석 |
|---|---|---|---|
| 1. 서론 | 왜 selection proxy와 learner utility를 분리해야 하는가 | claim boundary, final audit, literature matrix | 특정 selector의 일반적 우월성 |
| 2. 관련 연구 | industrial defect AL, proxy evaluation, training-free screening의 위치 | `study/`, literature evidence matrix | 서지 확인 전 직접 인용 |
| 3. 연구 설계 | dataset/branch/gate/freeze/locked-test 구조 | protocols, manifests, configs, branch decision tables | gate의 predictive accuracy·경제성 |
| 4. 후보 신호 타당성 | VLM/DINO/detector/PatchCore 신호가 무엇을 측정했는가 | validity runs, embeddings, audits | proxy 성능을 downstream 성능으로 동일시 |
| 5. 선택 단계 결과 | discovery, composition, overlap, acquisition stability | GC10/MPDD/VisA/DeepPCB/KSDD2 selection tables | 이질적 지표의 단순 평균 |
| 6. 다운스트림 학습 효용 | learner metric과 minimum-effect gate | GC10/NEU/KSDD2 results | post-hoc rescue·official test 접근 |
| 7. Translation atlas | selection delta와 learner delta의 관계/불일치 | translation atlas, paired tables, claim-gap matrix | benchmark 또는 causal mechanism 주장 |
| 8. 논의/한계 | 무엇이 identifiable하지 않은가 | thesis boundary, audits, STOP records | 실제 비용 절감·생산 lot 일반화 |
| 부록 | manifests, checksums, configs, code, stopped branches | protocols/reports/scripts/checkpoints | 본문 결과처럼 전면 제시 |

## 제안 디렉터리 구조 (승인 후 실행)

```text
thesis_assets/
  00_inventory/          # 이 inventory와 provenance ledger
  01_scope_claims/       # claim boundary, RQ, gap matrix
  02_literature/         # 검증된 논문과 서지 ledger
  03_methods_protocols/  # frozen protocol, manifests, configs
  04_selection_stage/    # discovery/composition/stability tables
  05_downstream_utility/ # learner metrics and minimum-effect decisions
  06_translation_atlas/  # cross-stage joined analysis
  07_figures_tables/     # canonical publication figures/tables only
  08_drafts/             # current canonical thesis text
  09_appendix_repro/     # code/config/checksum/reproduction map
  90_history/            # superseded proposal, mini-paper, decks
  99_quarantine/         # orphan/temp/source-unknown; no deletion
```

## 실행 단계

1. **Canonical 지정:** 유사 버전 목록의 각 계열에 `canonical`, `superseded`, `format derivative`, `unresolved` 상태를 수동 부여한다. 수정 시각만으로 최신본을 결정하지 않는다.
2. **Evidence locator 점검:** 모든 claim-evidence 표/그림에 source run, config, seed, split, metric definition, gate, commit/hash를 연결한다. 연결이 끊기면 claim evidence에서 appendix/orphan으로 강등한다.
3. **Translation join 생성:** dataset × selector × budget/round × acquisition seed 단위로 selection-stage delta와 downstream delta를 결합한다. 지표가 이질적인 branch는 공통 효과크기로 강제 변환하지 않고 별도 panel로 둔다.
4. **Figure/table canonicalization:** 동일 그림의 PNG/SVG/PDF는 하나의 source figure와 배포 파생물로 묶고, caption에 dataset·split·n/seed·metric·status를 넣는다. 의미가 불명확한 그림은 `99_quarantine` 후보로 둔다.
5. **Draft 분리:** mini-paper, 발표본, 교수 보고용 문서는 연구사/communication asset으로 이동 계획만 세우고, 현재 thesis draft와 근거 권한을 분리한다.
6. **Raw/derived 분리:** `data/`는 원천, `datasets/`는 실험용 파생본, `runs/`는 실행 결과로 고정한다. 중복 파일은 삭제하지 말고 manifest 또는 hardlink/symlink 가능성만 기록한다.
7. **Orphan 검역:** `tmp/`, 로컬 환경, 단독 오래된 로그, 출처 불명 모델/그래프는 quarantine 후보 ledger에 생성자·생성 명령·canonical source·보존 이유를 채울 때까지 본문 사용을 금지한다.
8. **최종 claim audit:** 각 본문 문장을 C1–C7 또는 ‘배경/한계/향후 연구’에 매핑한다. ‘우월’, ‘보장’, ‘비용 절감’, ‘일반화’, ‘원인’ 표현은 직접 근거가 없으면 제거 또는 제한한다.

## 이동 전 승인 체크리스트

- [ ] canonical 문서/도표/표의 소유자와 최신성 승인
- [ ] Git dirty 상태의 사용자 변경분 백업/commit 여부 결정
- [ ] dataset 라이선스와 재배포 가능 범위 확인
- [ ] 대규모 `datasets/` 복제본의 hardlink/symlink/manifest 전략 결정
- [ ] official/final test lock 유지 확인
- [ ] orphan 삭제가 아니라 이동/보존 정책 승인
- [ ] 재조직 후 scripts/config 상대경로가 깨지지 않는지 dry-run 계획 승인

## 이번 단계에서 하지 않은 작업

파일 삭제, 이동, rename, deduplication, Git checkout/reset/clean, 논문 본문 작성, 결과 재계산, selector 추가 탐색은 수행하지 않았다.

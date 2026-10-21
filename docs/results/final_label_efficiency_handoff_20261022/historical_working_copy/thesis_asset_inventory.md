# Thesis Asset Inventory

- 스캔 기준 시각: 2026-08-08T14:38:38 (Asia/Seoul 로컬 시스템 시각)
- 프로젝트 루트: `C:\Users\user\Desktop\vlm\Defect_VLM_Project`
- 전수 파일 수: **359,219** (`.git/` 내부 객체·메타데이터, reparse point 하위의 연결된 중복 트리, 본 inventory 생성 파일은 제외)
- 스캔된 디렉터리 수: **13,845** (reparse point 하위 제외), 이 중 빈 디렉터리 **509**
- 최상위 빈 디렉터리: `.agents/`, `final-render/` — 현재 파일 자산은 없으며 결과 존재의 근거로 취급하지 않음
- 확인된 작업용 junction: `tmp/` 아래 7개(`node_modules`/개별 패키지 연결); 연결 대상 파일은 중복 inventory하지 않음
- 파일별 canonical inventory: `thesis_asset_inventory.csv` (UTF-8 BOM, 파일당 1행)
- 원본 파일 이동·수정·삭제: **없음**

## 1. 해석 원칙과 주의사항

이 inventory는 현재 논문 방향인 ‘산업 결함 Active Learning에서 선택 단계 신호와 다운스트림 학습 효용 간 연결에 대한 실증적 평가’에 맞춰 작성했다. 특정 selector의 일반적 우월성, workflow의 예측 정확도, 실제 GPU/annotation 비용 절감, 독립 생산 lot 일반화는 현재 자산만으로 지지되는 주장으로 분류하지 않았다. 파일명·경로만으로 역할을 확정할 수 없거나 바이너리 내용을 개별 확인하지 못한 경우 `확인 필요`로 표시했다. 대규모 이미지·라벨·환경 파일은 경로 규칙으로 전수 분류했으며 개별 시각/내용 검사는 하지 않았다.

## 2. 범주별 파일 수

한 파일이 여러 범주에 속할 수 있으므로 합계는 전체 파일 수보다 클 수 있다.

| 범주 | 파일 수 |
|---|---:|
| appendix candidate | 259,306 |
| claim evidence | 2,005 |
| draft text | 139 |
| experiment log | 8,645 |
| figure | 13,715 |
| processed result | 274,927 |
| raw data | 283,240 |
| table | 2,829 |
| unclear / orphan asset | 50,574 |

## 3. 최상위 위치별 자산 규모

| 위치 | 파일 수 | 논문 작성 관점의 기본 처리 |
|---|---:|---|
| `datasets` | 251,801 | 실험용으로 파생·복제된 학습 데이터/라벨/manifest |
| `.venv` | 42,819 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 |
| `data` | 33,098 | 원천 데이터셋 이미지/주석/분할 또는 외부 다운로드 자산 |
| `runs` | 22,823 | run 메트릭·선택집합·감사표 또는 구조화 결과 |
| `.python311` | 6,915 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 |
| `.codex_doc_deps` | 436 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 |
| `tmp` | 398 | 임시 렌더·검수·재계산·중간 변환 산출물 |
| `scripts` | 360 | 분석/실험/검증 코드 또는 설정 |
| `docs` | 280 | 논문/보고/프로토콜/발표용 서술 자산 |
| `outputs` | 140 | 임베딩·pseudo box·우선순위 점수 등 중간 처리 결과 |
| `tests` | 39 | 분석/실험/검증 코드 또는 설정 |
| `study` | 29 | 관련연구 원문 후보 |
| `protocols` | 26 | 정리표·결과표·manifest 또는 의사결정 레지스트리 |
| `src` | 17 | 분석/실험/검증 코드 또는 설정 |
| `.pytest_cache` | 5 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 |
| `reports` | 5 | 정리표·결과표·manifest 또는 의사결정 레지스트리 |
| `output` | 4 | 논문/보고/프로토콜/발표용 서술 자산 |
| `manifests` | 3 | 정리표·결과표·manifest 또는 의사결정 레지스트리 |
| `Ultralytics` | 3 | 외부 라이브러리 또는 vendored 코드 |
| `configs` | 2 | 실험/통계 분석 설정 또는 frozen config |
| `downloads` | 2 | 원천 데이터셋 이미지/주석/분할 또는 외부 다운로드 자산 |
| `.gitattributes` | 1 | 역할을 경로/확장자만으로 확정하기 어려움 |
| `.github` | 1 | 분석/실험/검증 코드 또는 설정 |
| `.gitignore` | 1 | 역할을 경로/확장자만으로 확정하기 어려움 |
| `init_project.py` | 1 | 분석/실험/검증 코드 또는 설정 |
| `main.py` | 1 | 분석/실험/검증 코드 또는 설정 |
| `pyproject.toml` | 1 | 분석/실험/검증 코드 또는 설정 |
| `README.md` | 1 | 루트 수준 메모/문서 |
| `requirements-analysis.txt` | 1 | 루트 수준 메모/문서 |
| `test_0329.py` | 1 | 분석/실험/검증 코드 또는 설정 |
| `test_model.py` | 1 | 분석/실험/검증 코드 또는 설정 |
| `yolo26n.pt` | 1 | 사전학습 가중치/모델 파일 |
| `yolov8n.pt` | 1 | 사전학습 가중치/모델 파일 |
| `yolov8s.pt` | 1 | 사전학습 가중치/모델 파일 |
| `실행결과 0511.txt` | 1 | 2026-05-11 VLM consistency 다중 데이터셋 초기 실험 로그 |

## 4. 논문 핵심 자산 파일별 우선 검토표

아래는 전수 CSV 중 현재 논문 방향과 직접 연결되는 우선 검토 항목이다. 나머지 모든 파일도 CSV에 동일한 필드로 수록되어 있다.

| 파일 경로 | 파일명 | 파일 유형 | 분류 | 추정 역할 | 연결 장/절 | 연결 주장 | 수동 확인 |
|---|---|---|---|---|---|---|---|
| docs/advisor_decision_brief_evidence_freeze_v2_20260718.md | advisor_decision_brief_evidence_freeze_v2_20260718.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/branch_decision_table_20260718.csv | branch_decision_table_20260718.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/claim_evidence_gap_matrix.csv | claim_evidence_gap_matrix.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/condition_map_master_20260718.csv | condition_map_master_20260718.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/figures/branch_decision_waterfall.png | branch_decision_waterfall.png | PNG image | claim evidence; figure; processed result | 논문/발표용 그림 또는 시각화 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/figures/selection_learning_translation.png | selection_learning_translation.png | PNG image | figure | 논문/발표용 그림 또는 시각화 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/figures/validity_translation_chain.png | validity_translation_chain.png | PNG image | figure | 논문/발표용 그림 또는 시각화 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/final_repository_research_audit_20260730.md | final_repository_research_audit_20260730.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/framework_holdout_confusion_matrix_20260718.csv | framework_holdout_confusion_matrix_20260718.csv | CSV table | table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 3–8장 및 부록 | 논문 서술/도식 후보; 주장 근거 여부는 수동 확인 필요 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/gc10_cost_translation_easy_value_explanation_20260730.md | gc10_cost_translation_easy_value_explanation_20260730.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/gc10_cost_translation_novelty_feasibility_audit_20260730.md | gc10_cost_translation_novelty_feasibility_audit_20260730.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/gc10_cost_translation_protocol_package_report_20260730.md | gc10_cost_translation_protocol_package_report_20260730.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/gc10_cost_translation_protocol_v1_candidate_20260730.md | gc10_cost_translation_protocol_v1_candidate_20260730.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/minimum_additional_experiments.md | minimum_additional_experiments.md | Markdown document | draft text | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | 논문 서술/도식 후보; 주장 근거 여부는 수동 확인 필요 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/presentation_final_v2/slide_translation_atlas.html | slide_translation_atlas.html | HTML document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/presentation_final_v2/slide_translation_atlas.png | slide_translation_atlas.png | PNG image | claim evidence; figure; processed result | 논문/발표용 그림 또는 시각화 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/presentation_final_v2/slide_translation_atlas.png.audit.json | slide_translation_atlas.png.audit.json | JSON data | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/presentation_final_v2/랩미팅_최종수정안_v2_TranslationAtlas.pdf | 랩미팅_최종수정안_v2_TranslationAtlas.pdf | PDF document | draft text | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/presentation_final_v2/랩미팅_최종수정안_v2_TranslationAtlas.pptx | 랩미팅_최종수정안_v2_TranslationAtlas.pptx | PowerPoint deck | draft text | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/presentation_final_v2/랩미팅_최종수정안_v2_TranslationAtlas.pptx.inspect.ndjson | 랩미팅_최종수정안_v2_TranslationAtlas.pptx.inspect.ndjson | NDJSON data | table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/research_asset_reclassification_20260718.csv | research_asset_reclassification_20260718.csv | CSV table | table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 3–8장 및 부록 | 논문 서술/도식 후보; 주장 근거 여부는 수동 확인 필요 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/research_evolution_and_evidence_freeze_v2_20260718.md | research_evolution_and_evidence_freeze_v2_20260718.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/research_evolution_and_evidence_freeze_v3_20260718.md | research_evolution_and_evidence_freeze_v3_20260718.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/research_novelty_and_upgrade_audit.md | research_novelty_and_upgrade_audit.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/current_8seed_interpretation.md | current_8seed_interpretation.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/current_8seed_snapshot.csv | current_8seed_snapshot.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/ksdd2_g5_single_unet_20260727/paired_seed_metrics.csv | paired_seed_metrics.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 6장 KSDD2 bounded downstream translation; 7장 합성 | C1, C4, C5, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/ksdd2_g5_single_unet_20260727/RESULT_ANALYSIS_KO.md | RESULT_ANALYSIS_KO.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 6장 KSDD2 bounded downstream translation; 7장 합성 | C1, C4, C5, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/ksdd2_patchcore_prospective_20260726/RESULT_ANALYSIS_KO.md | RESULT_ANALYSIS_KO.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 6장 KSDD2 bounded downstream translation; 7장 합성 | C1, C4, C5, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/ksdd2_patchcore_prospective_20260726/RESULT_ANALYSIS_KO.md.sha256 | RESULT_ANALYSIS_KO.md.sha256 | checksum manifest | claim evidence; processed result | 역할을 경로/확장자만으로 확정하기 어려움 | 6장 KSDD2 bounded downstream translation; 7장 합성 | C1, C4, C5, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/results/v7_full_curve_20260711_211848/v7_full_curve_summary.md | v7_full_curve_summary.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/selection_learning_translation_analysis_plan.md | selection_learning_translation_analysis_plan.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/thesis_claim_boundary_20260718.md | thesis_claim_boundary_20260718.md | Markdown document | draft text | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | 논문 서술/도식 후보; 주장 근거 여부는 수동 확인 필요 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/thesis_claim_boundary_v3_20260718.md | thesis_claim_boundary_v3_20260718.md | Markdown document | draft text | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | 논문 서술/도식 후보; 주장 근거 여부는 수동 확인 필요 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/three_dataset_condition_map_decision_20260718.md | three_dataset_condition_map_decision_20260718.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 3–8장 및 부록 | C1–C7의 합성 또는 branch 근거 후보; 원자료/run locator와 대조해 사용 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/four_fates_taxonomy.csv | four_fates_taxonomy.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/paper_results_draft.md | paper_results_draft.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/research_introduction_for_nonexperts.md | research_introduction_for_nonexperts.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/translation_atlas_analysis.md | translation_atlas_analysis.md | Markdown document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/translation_atlas_main_figure.pdf | translation_atlas_main_figure.pdf | PDF document | claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/translation_atlas_main_figure.png | translation_atlas_main_figure.png | PNG image | claim evidence; figure; processed result | 논문/발표용 그림 또는 시각화 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/translation_atlas_statistics.json | translation_atlas_statistics.json | JSON data | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| docs/translation_atlas/translation_branch_inventory.csv | translation_branch_inventory.csv | CSV table | claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 7장 선택 신호–학습 효용 연결 분석 | C1, C2, C4, C7 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/acquisition_overlap_matrix.csv | acquisition_overlap_matrix.csv | CSV table | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_event_examples.jsonl | annotation_event_examples.jsonl | JSON Lines data | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_event_schema.json | annotation_event_schema.json | JSON data | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_export_validation_spec.md | annotation_export_validation_spec.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_guide_outline.md | annotation_guide_outline.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_logger_requirements.md | annotation_logger_requirements.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_quality_metric_schema.json | annotation_quality_metric_schema.json | JSON data | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotation_quality_protocol.md | annotation_quality_protocol.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/annotator_information_sheet_draft.md | annotator_information_sheet_draft.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/human_participation_checklist.md | human_participation_checklist.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/image_seed_membership.csv | image_seed_membership.csv | CSV table | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/manifest_build_metadata.json | manifest_build_metadata.json | JSON data | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/manifest_hashes.sha256 | manifest_hashes.sha256 | checksum manifest | appendix candidate; claim evidence; processed result | 역할을 경로/확장자만으로 확정하기 어려움 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/overlap_minimized_seed_sensitivity.csv | overlap_minimized_seed_sensitivity.csv | CSV table | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/package_validation_report.json | package_validation_report.json | JSON data | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/phase_a_annotator_blinded_manifest.csv | phase_a_annotator_blinded_manifest.csv | CSV table | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/phase_a_seed_batch_manifest.csv | phase_a_seed_batch_manifest.csv | CSV table | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/phase_a_unique_image_manifest.csv | phase_a_unique_image_manifest.csv | CSV table | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/phase_b_frozen_candidate_manifest_template.csv | phase_b_frozen_candidate_manifest_template.csv | CSV table | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/source_evidence_hashes.sha256 | source_evidence_hashes.sha256 | checksum manifest | appendix candidate | 역할을 경로/확장자만으로 확정하기 어려움 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/stage0_annotator_blinded_manifest.csv | stage0_annotator_blinded_manifest.csv | CSV table | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/stage0_instrumentation_manifest.csv | stage0_instrumentation_manifest.csv | CSV table | appendix candidate; claim evidence; processed result; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/statistical_analysis_config.template.yaml | statistical_analysis_config.template.yaml | YAML config | appendix candidate; experiment log | 실험/통계 분석 설정 또는 frozen config | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/statistical_analysis_plan.md | statistical_analysis_plan.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/threshold_option_matrix.md | threshold_option_matrix.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| protocols/gc10_cost_translation_v1/unresolved_researcher_decisions.md | unresolved_researcher_decisions.md | Markdown document | appendix candidate; claim evidence; draft text; processed result | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| reports/preflight/gc10_dev_confirm_harness_v1/preflight_report.json | preflight_report.json | JSON data | appendix candidate; table | 정리표·결과표·manifest 또는 의사결정 레지스트리 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |
| reports/preflight/gc10_dev_confirm_harness_v1/preflight_report.md | preflight_report.md | Markdown document | appendix candidate; draft text | 논문/보고/프로토콜/발표용 서술 자산 | 4–6장 GC10 selection/downstream 사례 | C1, C2, C3, C6 | 예 — 문서의 최신성·원자료 locator·문자 인코딩 확인 필요 |

## 5. 연결 가능한 연구 주장 레지스트리

| ID | 현재 자산으로 허용되는 제한적 주장 |
|---|---|
| C1 | 선택 단계의 개선은 다운스트림 학습 효용을 자동으로 보장하지 않는다. |
| C2 | discovery/ranking, composition, stability, learner utility는 분리해 평가해야 한다. |
| C3 | acquisition-seed 재현성과 fixed-set training 재현성은 서로 다른 대상이다. |
| C4 | KSDD2에서 방향성 양의 learner delta가 관찰됐지만 사전 최소효과 기준은 충족하지 못했다. |
| C5 | STOP/authorization 기록은 절차 준수와 locked-test 보존 근거이지 예측 정확도나 실제 비용 절감 근거가 아니다. |
| C6 | 현재 자산은 특정 selector의 보편적 우월성을 지지하지 않는다. |
| C7 | 다중 데이터셋 결과는 공통 benchmark가 아니라 이질적 branch의 empirical translation audit로 해석해야 한다. |

## 6. 이름 유사·중복·버전 충돌 후보 (separate list)

이 목록은 자동 정규화와 제한적 SHA-256 비교로 만든 검토 후보이며, 최신본을 자동 지정하지 않았다. 서로 다른 확장자의 동일 stem은 배포 형식 차이일 수 있고, 같은 계열의 v1/v2/v3·완성본·수정중은 내용 충돌 가능성이 있다.

| 정규화 계열 | 후보 파일 | 조치 |
|---|---|---|
| `full_page_validity_gated_architecture` | `docs/figures/full_page_validity_gated_architecture.pdf`<br>`docs/figures/full_page_validity_gated_architecture.png`<br>`docs/figures/full_page_validity_gated_architecture.svg`<br>`docs/figures/full_page_validity_gated_architecture_v2.html`<br>`docs/figures/full_page_validity_gated_architecture_v2.pdf`<br>`docs/figures/full_page_validity_gated_architecture_v2.png`<br>`docs/figures/full_page_validity_gated_architecture_v2.svg`<br>`docs/slide_assets_20260719/full_page_validity_gated_architecture_v2.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `mini_paper_validity_gated_industrial_al` | `docs/mini_paper_validity_gated_industrial_al_20260718.docx`<br>`docs/mini_paper_validity_gated_industrial_al_20260718.md`<br>`docs/mini_paper_validity_gated_industrial_al_20260718.pdf`<br>`docs/mini_paper_validity_gated_industrial_al_v2_20260718.docx`<br>`docs/mini_paper_validity_gated_industrial_al_v2_20260718.md`<br>`docs/mini_paper_validity_gated_industrial_al_v2_20260718.pdf` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `산업_결함_active_learning_후보_신호의_단계별_타당성_평가` | `docs/산업_결함_Active_Learning_후보_신호의_단계별_타당성_평가_20260726065143.pdf`<br>`docs/산업_결함_Active_Learning_후보_신호의_단계별_타당성_평가_20260726071303.pptx`<br>`docs/산업_결함_Active_Learning_후보_신호의_단계별_타당성_평가_완성본_20260726 수정중.pptx`<br>`docs/산업_결함_Active_Learning_후보_신호의_단계별_타당성_평가_완성본_20260726.pptx`<br>`docs/산업_결함_Active_Learning_후보_신호의_단계별_타당성_평가_완성본_20260726.pptx.inspect.ndjson` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `validity_gated_algorithm_flowchart` | `docs/figures/validity_gated_algorithm_flowchart.pdf`<br>`docs/figures/validity_gated_algorithm_flowchart.png`<br>`docs/figures/validity_gated_algorithm_flowchart.svg`<br>`docs/slide_assets_20260719/validity_gated_algorithm_flowchart.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `랩미팅_최종수정안_translationatlas` | `docs/presentation_final_v2/랩미팅_최종수정안_v2_TranslationAtlas.pdf`<br>`docs/presentation_final_v2/랩미팅_최종수정안_v2_TranslationAtlas.pptx`<br>`docs/presentation_final_v2/랩미팅_최종수정안_v2_TranslationAtlas.pptx.inspect.ndjson` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `acquisition_mechanism_matrix` | `docs/acquisition_mechanism_matrix_20260718.csv`<br>`docs/acquisition_mechanism_matrix_v3_20260718.csv` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `dataset_distribution_by_round` | `docs/results/v7_full_curve_20260711_211848/dataset_distribution_by_round.csv`<br>`docs/results/v7_full_curve_20260711_211848/plots/dataset_distribution_by_round.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `deeppcb_prospective_stop_case` | `docs/figures/deeppcb_prospective_stop_case.png`<br>`docs/slide_assets_20260719/deeppcb_prospective_stop_case.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `discovery_composition_utility_matrix` | `docs/figures/discovery_composition_utility_matrix.png`<br>`docs/slide_assets_20260719/discovery_composition_utility_matrix.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `gc10_research_lineage_audit` | `reports/preflight/gc10_dev_confirm_harness_v1/gc10_research_lineage_audit.json`<br>`reports/preflight/gc10_dev_confirm_harness_v1/gc10_research_lineage_audit.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `hypothesis_transition_matrix` | `docs/hypothesis_transition_matrix_20260718.csv`<br>`docs/hypothesis_transition_matrix_v3_20260718.csv` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `paper_framework_temporal_validation` | `docs/figures/paper_framework_temporal_validation.png`<br>`docs/slide_assets_20260719/paper_framework_temporal_validation.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `preflight_report` | `reports/preflight/gc10_dev_confirm_harness_v1/preflight_report.json`<br>`reports/preflight/gc10_dev_confirm_harness_v1/preflight_report.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `reframed_gt_free_al_workflow` | `docs/reframed_gt_free_al_workflow_20260713.html`<br>`docs/reframed_gt_free_al_workflow_20260713.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `research_evolution_and_evidence_freeze` | `docs/research_evolution_and_evidence_freeze_v2_20260718.md`<br>`docs/research_evolution_and_evidence_freeze_v3_20260718.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `result_analysis_ko` | `docs/results/ksdd2_g5_single_unet_20260727/RESULT_ANALYSIS_KO.md`<br>`docs/results/ksdd2_patchcore_prospective_20260726/RESULT_ANALYSIS_KO.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `seed42_development_gate_report` | `docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.docx`<br>`docs/results/v10b_seed42_documentation_20260712_215841/V10b_Seed42_Development_Gate_Updated_Report.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `shortcut_integrity_audit` | `docs/results/ksdd2_patchcore_prospective_20260726/posthoc_diagnostics/shortcut_integrity_audit.json`<br>`docs/results/ksdd2_patchcore_prospective_20260726/posthoc_diagnostics/shortcut_integrity_audit.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `slide_nonexpert_research_intro` | `docs/presentation_final_v2/slide_nonexpert_research_intro.html`<br>`docs/presentation_final_v2/slide_nonexpert_research_intro.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `slide_translation_atlas` | `docs/presentation_final_v2/slide_translation_atlas.html`<br>`docs/presentation_final_v2/slide_translation_atlas.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `thesis_claim_boundary` | `docs/thesis_claim_boundary_20260718.md`<br>`docs/thesis_claim_boundary_v3_20260718.md` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `translation_atlas_main_figure` | `docs/translation_atlas/translation_atlas_main_figure.pdf`<br>`docs/translation_atlas/translation_atlas_main_figure.png` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `visual_redundancy_by_round` | `docs/results/v7_full_curve_20260711_211848/plots/visual_redundancy_by_round.png`<br>`docs/results/v7_full_curve_20260711_211848/visual_redundancy_by_round.csv` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `vlm_gt_free_al_workflow` | `docs/assets/vlm_gt_free_al_workflow.png`<br>`docs/vlm_gt_free_al_workflow.html` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `석사학위_논문_연구_방향_재정의_경과_및_검토_요청_gc10_검증반영` | `output/documents/석사학위_논문_연구_방향_재정의_경과_및_검토_요청_GC10_검증반영_20260801.docx`<br>`output/pdf/석사학위_논문_연구_방향_재정의_경과_및_검토_요청_GC10_검증반영_20260801.pdf` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |
| `석사학위_논문_연구_진행_경과_및_향후_방향_검토_요청_송부본` | `output/documents/석사학위_논문_연구_진행_경과_및_향후_방향_검토_요청_송부본_20260801.docx`<br>`output/pdf/석사학위_논문_연구_진행_경과_및_향후_방향_검토_요청_송부본_20260801.pdf` | canonical 지정 전 내용·날짜·근거 locator 비교 필요 |

### 내용이 완전히 동일한 파일 후보

| SHA-256 앞 12자 | 동일 파일 경로 |
|---|---|
| `222d6a5f667e` | `docs/figures/discovery_composition_utility_matrix.png`<br>`docs/slide_assets_20260719/discovery_composition_utility_matrix.png` |
| `3c52869f1c04` | `docs/figures/deeppcb_prospective_stop_case.png`<br>`docs/slide_assets_20260719/deeppcb_prospective_stop_case.png` |
| `4a98c34da2f4` | `docs/figures/full_page_validity_gated_architecture_v2.png`<br>`docs/slide_assets_20260719/full_page_validity_gated_architecture_v2.png` |
| `8d73bf8b59b0` | `docs/figures/validity_gated_algorithm_flowchart.png`<br>`docs/slide_assets_20260719/validity_gated_algorithm_flowchart.png` |
| `cffd0b3292ef` | `docs/figures/paper_framework_temporal_validation.png`<br>`docs/slide_assets_20260719/paper_framework_temporal_validation.png` |
| `f6c450c88ce5` | `docs/presentation_final_v2/slide_nonexpert_research_intro.png.audit.json`<br>`docs/presentation_final_v2/slide_translation_atlas.png.audit.json` |

## 7. Orphan asset 분리 목록

다음 묶음은 현재 논문 본문에 직접 쓰기 어렵다. 삭제 대상이라는 뜻은 아니며, canonical source·출처·생성 명령을 확인하기 전 인용/도표 사용을 보류한다.

| 위치/묶음 | 파일 수 | 사유 | 확인 사항 |
|---|---:|---|---|
| `.venv` | 42,819 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 | 아니오 — 환경 재생성 가능 여부만 별도 확인 |
| `.python311` | 6,915 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 | 아니오 — 환경 재생성 가능 여부만 별도 확인 |
| `.codex_doc_deps` | 436 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 | 아니오 — 환경 재생성 가능 여부만 별도 확인 |
| `tmp` | 398 | 임시 렌더·검수·재계산·중간 변환 산출물 | 예 — canonical 원본과 생성 경로 확인 필요 |
| `.pytest_cache` | 5 | 로컬 실행환경/패키지/캐시; 논문 자산 아님 | 아니오 — 환경 재생성 가능 여부만 별도 확인 |
| `Ultralytics` | 3 | 외부 라이브러리 또는 vendored 코드 | 예 — 라이선스·원본 출처·버전 확인 필요 |
| `.gitattributes` | 1 | 역할을 경로/확장자만으로 확정하기 어려움 | 예 — 역할·출처 확인 필요 |
| `.gitignore` | 1 | 역할을 경로/확장자만으로 확정하기 어려움 | 예 — 역할·출처 확인 필요 |
| `tmp/` | 398 | 임시 렌더·검수·재계산물이 섞여 canonical 여부 불명 | 원본 문서/그림과 hash·생성 경로 대조 |
| 루트의 `실행결과 0511.txt` | 1 | 내용은 2026-05-11 VLM consistency 실험 로그로 확인됐으나 참조하는 `scripts/test_0511.py`가 현재 폴더에서 발견되지 않음 | 실행 코드·commit·dataset split을 찾기 전에는 historical appendix로만 사용 |
| 루트의 `test_0329.py`, `test_model.py`, `main.py`, `init_project.py` | 4 | 초기/보조 코드로 보이며 현재 논문 pipeline 소비 여부 불명 | 현행 scripts/src와 중복·호출 여부 확인 |

## 8. 직접 사용 우선순위

1. `docs/translation_atlas/`, `docs/selection_learning_translation_analysis_plan.md`, `docs/claim_evidence_gap_matrix.csv`, `docs/final_repository_research_audit_20260730.md`를 현재 방향의 합성 출발점으로 삼되 raw run locator를 재검증한다.
2. `docs/results/ksdd2_*`, `runs/ksdd2_*`, GC10/NEU 관련 frozen run을 C1–C4의 branch evidence로 연결한다.
3. DeepPCB STOP/authorization 자산은 C5의 절차 근거로만 사용하고 downstream 성공/비용 절감으로 확대하지 않는다.
4. 기존 mini-paper·발표본·workflow 중심 도식은 draft/history로 보존하고, 최신 논문 본문의 근거 권한은 주지 않는다.
5. `tmp/`, 환경 폴더, 중복 렌더, 출처 불명 단독 파일은 quarantine 후보로만 표시한다.

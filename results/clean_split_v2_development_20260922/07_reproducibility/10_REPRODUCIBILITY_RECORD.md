# clean split v2 development 결과 재현 기록

작성일 2026-09-22. 분석 대상은 `FINAL_RESULTS_ONLY_20260922.tar.gz`와 `FINAL_BEST_WEIGHTS_20260922.tar.gz`에서 회수한 **`ANNOTATION_PATCH_v2_0_2` 최종 시도**다. 원본 압축파일 및 실험 산출물은 수정하지 않았다. test는 withheld이며 이 분석에서 test 이미지·라벨·checkpoint 평가를 사용하지 않았다.

## 고정 입력과 provenance

| 파일 | SHA-256 |
|---|---|
| `FINAL_RESULTS_ONLY_20260922.tar.gz` | `b978fd30179ce0136cfac81b59399169de9b64daa889a1dddce8bddd0f06bd18` |
| `FINAL_BEST_WEIGHTS_20260922.tar.gz` | `79b7d9c75973b5bee4e2d37609cef9795dbe1dd535289daa3897b3c67d2ab021` |

최종 package SHA `2c0f74839b320bd6edc3f29963f7185d9f979e9b87e16cbc654d225710435d2d`, runtime patch SHA `571bd4771a60765a962a402043a643bf735dae85e621e5adcc5120998aa49d85`, annotation patch SHA `860af219aa68e8f0c52a656f39308e1a659d69c6df1e6702348af83cdd9cac74`. GC10 split SHA `453ab6eaf22a1439236125364492f4b7ef8d3e8e65c4daa96386b40cb93a243a`, NEU split SHA `a6027b62d12cfe426caaf7d9c178c4dd38354bcb915925069f1dc391c4951403`. source path `analysis_20260922/evidence/final_main_runpod_execution_v2_20260919/`는 최종 결과 압축의 추출 사본이다. 모든 숫자의 run/seed/round/metric 경로는 `02_tables/RAW_SEED_METRICS.csv`에 있다.

이 패키지의 문헌 근거 PDF 5편은 CVF 원문 URL에서 받은 사본으로 `01_literature/sources/`에 보관했고 URL 및 그림/표 번호는 `01_LITERATURE_FIGURE_CONVENTIONS.md`에 있다. 각 파일 SHA는 `OUTPUT_SHA256.csv`에 포함한다.

## 재현 순서

다음 명령은 작업공간 루트(`2026년 논문작성 자료`)에서 실행한다. 로컬 환경에서는 Python 3.14.2, NumPy 2.4.1, Matplotlib 3.10.8, Pillow 12.1.0, Poppler `pdfinfo`/`pdftoppm`을 사용했다. 원문 PDF 텍스트 추출에만 별도 번들 Python 3.12.14 + pypdf 6.10.0을 사용했다.

```powershell
py -3.14 '9월/9월 21일/analysis_20260922/final_thesis_analysis/07_reproducibility/preflight.py'
py -3.14 '9월/9월 21일/analysis_20260922/final_thesis_analysis/07_reproducibility/run_analysis.py'
py -3.14 '9월/9월 21일/analysis_20260922/final_thesis_analysis/07_reproducibility/make_figures.py'
py -3.14 '9월/9월 21일/analysis_20260922/final_thesis_analysis/07_reproducibility/verify_outputs.py'
```

깨끗한 작업공간에서 시작할 때 `analysis_20260922/extract_evidence.py`가 결과 압축에서 계산에 필요한 JSON/CSV/YAML 등을 읽기 전용 추출한다. `extract_preflight_logs.py`는 controller log와 앞선 export 메타데이터를 추출하고, `extract_train_labels.py`는 **최종 `ANNOTATION_PATCH_v2_0_2` overlay의 train 라벨 2,862개만** `07_reproducibility/train_labels/`에 복사한다. 복사한 각 라벨의 SHA를 `TRAIN_DEV_NORMALIZATION.json`의 `after_label_sha`와 대조한다. 빈 라벨 파일은 객체 수 0으로 취급한다. base `data/`의 train 라벨 중 286개는 tar **hardlink**이므로 regular file만 추출하면 R0 라벨이 빠지고 선택 구성 수치가 틀어진다. 최종 선택 구성표는 전부 overlay를 사용해 재계산했다. 원본 archive를 풀어쓴 곳에 계산 결과를 저장하지 않는다. `check_earlier_export.py`는 앞선 분석 export와 겹치는 sealed metric 97개의 SHA가 전부 동일함을 확인한다.

## 계산 규칙

- 평가 raw 값은 최종 시도의 `runs/<run_id>/sealed_dev_metrics.json`의 `overall.metrics/mAP50-95(B)`와 `per_class_AP50_95`. 학습 중 fitness 및 `train/results.csv`는 best epoch 감사용이며 논문 최종 development 수치에 섞지 않는다.
- 5 acquisition seed가 통계적 반복. learner seed `2026091900`은 고정. shared R0 모델은 방법별 곡선에 참조만 한다. Full-data는 데이터셋별 1회 참고값.
- 실제 정수 누적 이미지 수로 trapezoidal nAULC를 계산하고 span `n5−n0`으로 나눈다. Full-data로 정규화하지 않는다. raw AP 0–1을 표시 단계에서 100배 하며 차이는 %p.
- 절대 평균/SD/95% t CI와 paired raw 차이/100,000회 seed-paired percentile bootstrap/exact 양측 sign-flip/4대비 Holm을 `run_analysis.py`에서 재계산한다. 보조 대비는 Holm family 밖이다. test 1차 SAP의 정의를 development에 탐색적으로 적용한 것이므로 test confirmatory 결과가 아니다.
- 클래스 선택량은 train 라벨의 image incidence와 bbox incidence를 신규 `selected_ids` 및 누적 `cumulative_ids`별로 분리한다. split metadata의 test ID는 접근 경계 확인에만 쓰고 성능 계산·feature 추출에 쓰지 않는다.

## 완료 검사

`preflight.py`: 162 scientific COMPLETE, 최종 gate PASS, 모든 162 run의 100 epoch 결과 trace/선택/metric/설정, split train ID, feature coverage, 중첩 선택, test boundary 확인. 앞선 STOP 기록은 부모 시도 이력으로 보존했다. 기존 `shutdown_audit.json`은 best.pt 162개 및 비가중치 2,156개 SHA를 검증했다.

`verify_outputs.py`: active train overlay 라벨 **2,862개**, **180** seed×round×method raw 행, **30** seed nAULC 행, **38** 요약 행, seed별 동일 R0, 독립 trapezoid 계산, mean 재계산, **7개 그림의 PNG/PDF/SVG 세 형식**을 확인했다. 7개 vector PDF는 각각 1쪽이며 Poppler로 다시 렌더하여 레이아웃·축·범례를 육안 점검했다. 300 dpi PNG와 SVG 파일도 확인했다. 수치표의 표시 반올림은 원래 CSV의 전체 정밀도를 바꾸지 않는다.

`OUTPUT_SHA256.csv`는 최종 문서·표·그림·재현 코드의 상대 경로, byte 수, SHA-256이다. 독립적인 test release 및 평가 산출물은 아직 포함되지 않았다.

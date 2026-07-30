# `scripts/old` audit

작성일: 2026-07-12  
범위: `scripts/old/*.py` 22개 파일  
상태: 전 파일 syntax parse 통과. 단, 대부분 현재 Windows/YOLO AL pipeline에서 직접 실행하면 안 됨.

## 한 줄 결론

`scripts/old`는 현재 실험 runner가 아니라, 초기 연구 아이디어의 흔적이다. 핵심 가치는 “VLM explanation consistency / prompt ensemble / Qwen2-VL / SBERT / BERTScore / human review sheet”의 방법론적 출발점을 보존하는 데 있다. 현재 V8/V9 실험에는 직접 연결하지 말고, 필요한 아이디어만 현대화해서 `scripts/01_score_generation` 또는 `scripts/02_active_learning`의 새 유틸로 이관하는 편이 안전하다.

## 공통 문제

- 다수 파일이 macOS 기준 `mps` device를 기본값으로 사용한다.
- 여러 파일에 `/Users/hy/PycharmProjects/PythonProject/Defect_VLM_Project/...` 절대경로가 박혀 있다.
- Qwen2-VL, Florence-2, LoRA 관련 파일은 실행 시 대형 모델 다운로드/로드가 발생할 수 있다.
- 일부 분석 파일은 `if __name__ == "__main__"` guard가 없어 import만 해도 실행된다.
- 현재 V8/V9 protocol의 canonical sampling, split audit, final-test lock, NEU-only filter를 모른다.
- 따라서 현 runner에서 import하거나 재사용하면 protocol leakage 또는 경로 오류가 날 수 있다.

## 파일별 판정

| File | 역할 | 현재 판정 |
|---|---|---|
| `260511_VLM_ActiveLearning_FullExperiment.py` | Qwen2-VL + SBERT/BERTScore 기반 multi-dataset consistency 실험 | 아이디어 보존. 직접 실행 금지. Mac 절대경로/MPS/Qwen 의존. |
| `test_0511.py` | NEU/Kolektor/MVTec multi-dataset prompt consistency 전체 실험 + Excel/plot 생성 | old 폴더에서 가장 풍부한 초기 연구 기록. 실행용이 아니라 참고용. |
| `test2604.py` | 초기 NEU prompt consistency 실험 | `test2604_revised.py`, `test260424.py`, `260511...`에 의해 대체됨. |
| `test2604_revised.py` | test2604 개선판, BERTScore posthoc 포함 | 참고용. 현재 score generation과 직접 연결하지 않음. |
| `test260424.py` | Qwen2-VL + SBERT/BERTScore 실험의 문헌 주석 포함 버전 | 참고용. 현재 runner에는 불필요. |
| `consistency_check.py` | 10장 pilot consistency check | 초기 smoke test. 현재 `01_score_generation`/V7 이후 코드로 대체됨. |
| `consistency_mixed_eval.py` | class별 소량 샘플 mixed consistency 평가 | 초기 pilot. 현재 V7/V8 protocol과 맞지 않음. |
| `consistency_prompt_ensemble.py` | prompt ensemble consistency 생성 | 초기 prompt ensemble 아이디어 참고 가능. 직접 실행은 비추천. |
| `analyze_prompt_ensemble_results.py` | prompt ensemble jsonl 결과 요약 | old log 구조 전용. 필요하면 현대화 가능. |
| `consistency_full_analysis.py` | low/mid/high consistency split 분석 | old log 구조 전용. |
| `select_low_consistency_samples.py` | low consistency sample 선별 | 현재 acquisition logic으로 쓰면 안 됨. old prompt score 확인용. |
| `make_excel.py` | candidate review xlsx 생성 | Mac 절대경로 하드코딩. 참고용. |
| `make_excel_visual_v2.py` | 시각적으로 개선된 review xlsx 생성 | human review sheet 아이디어는 재사용 가능. 경로 현대화 필요. |
| `make_excel_multi_dataset_20260511.py` | multi-dataset review workbook 생성 | old log 기반. 문서/발표용 review sheet 재구성 시 참고. |
| `make_graph_260511.py` | old multi-dataset 결과 그래프 생성 | main guard 없음. import 주의. Mac 절대경로. |
| `vlm_error_correlation_20260511.py` | consistency score와 VLM error correlation 분석 | main guard 없음. old CSV path 하드코딩. 참고용만. |
| `vlm_forced_classification_20260511.py` | VLM forced classification 후 error 분석 | main guard 없음 + top-level Qwen model load. 실행/import 금지에 가까움. |
| `test_qwen.py` | Qwen2-VL 단일 이미지 테스트 | 모델 로드/다운로드 가능. 현재 실험과 무관. |
| `test_qwen_lora.py` | Qwen LoRA adapter 테스트 | old adapter path 필요. 현재 실험과 무관. |
| `test_lora.py` | Qwen2-VL LoRA 학습 테스트 | 모델 학습/쓰기 발생 가능. 현재 pipeline에서는 제외. |
| `train_lora.py` | Qwen2-VL LoRA 학습 | 실행 금지. 현재 연구 방향과 직접 관련 낮음. |
| `test_model.py` | Florence-2 단일 이미지 테스트 | import/top-level에서 모델 로드. 실행/import 금지에 가까움. |

## 특히 조심할 파일

다음 파일은 import만 해도 heavy model load 또는 즉시 분석/plot 저장이 발생할 수 있다.

- `test_model.py`
- `vlm_forced_classification_20260511.py`
- `vlm_error_correlation_20260511.py`
- `make_graph_260511.py`

다음 파일은 대형 VLM/LoRA 학습 또는 다운로드가 발생할 수 있다.

- `train_lora.py`
- `test_lora.py`
- `test_qwen.py`
- `test_qwen_lora.py`
- `260511_VLM_ActiveLearning_FullExperiment.py`
- `test_0511.py`

## 현재 연구에 재사용할 수 있는 아이디어

### 1. Prompt ensemble 정의

`consistency_prompt_ensemble.py`, `test_0511.py`, `260511_VLM_ActiveLearning_FullExperiment.py`에는 rephrase/viewpoint prompt ensemble 구조가 남아 있다. 다만 현재 결론상 explanation consistency는 main acquisition signal이 아니라 failure analysis 또는 auxiliary diagnostic으로만 두는 것이 맞다.

### 2. Human review sheet

`make_excel_visual_v2.py`, `make_excel_multi_dataset_20260511.py`의 review workbook 구성은 여전히 쓸모 있다. V9 이후 “선택된 샘플이 왜 선택됐는지”를 사람이 검토하는 보조 자료를 만들 때 이 아이디어를 현대화할 수 있다.

### 3. Error/correlation 분석

`vlm_error_correlation_20260511.py`, `vlm_forced_classification_20260511.py`는 “VLM response가 실제 class와 얼마나 맞는가”를 보려던 초기 시도다. 지금은 그대로 쓰지 말고, `docs/analysis`용 post-hoc figure로 재구현하는 편이 안전하다.

## 정리 권고

### 유지

old 폴더는 연구 history 보존용으로 유지한다. 단, README 또는 audit 문서에서 “legacy only”임을 명시한다.

### 직접 실행 금지

현재 실험에 다음 파일들을 직접 연결하지 않는다.

- Qwen/LoRA model test/train 파일
- Mac 절대경로 기반 Excel/graph 파일
- main guard 없는 분석 파일

### 이관 후보

필요하다면 다음 기능만 새 코드로 이관한다.

1. prompt ensemble metadata → `scripts/01_score_generation/prompt_families.py`
2. human review workbook generator → `scripts/03_analysis/build_review_workbook_v9.py`
3. VLM forced-classification diagnostic → `scripts/03_analysis/analyze_vlm_class_alignment_v9.py`

## V9 재설계와의 관계

V9 detector-aware pivot에서는 old 코드의 prompt consistency를 acquisition score로 부활시키지 않는다.

대신 old 코드의 역할은 다음으로 제한한다.

- 연구 배경 설명
- 실패 분석
- prompt consistency baseline의 역사적 근거
- human inspection 자료 생성 아이디어

현재 V9의 중심은 다음이어야 한다.

```text
detector pseudo-instance richness
+ predicted-class balance
+ DINO diversity
+ calibrated detector uncertainty
```

old 코드의 가장 큰 교훈은 “VLM 설명 불일치가 흥미로운 semantic signal이긴 하지만, object detector의 localization utility와 자동으로 정렬되지는 않는다”는 점이다.

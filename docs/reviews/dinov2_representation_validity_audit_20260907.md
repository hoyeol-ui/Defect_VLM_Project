# DINOv2 표현 타당성 사후 감사 — 2026-09-07

## 판정

**무학습 정량 감사 완료, 무결성 PASS.** 두 데이터셋의 봉인된 DINOv2 cache와 acquisition-only 객체 라벨을 sample ID로 정확히 연결할 수 있었다. 새 특징 추출, 모델 학습, fine-tuning, 표본 선택 변경은 모두 0회이며 잠긴 test 라벨은 읽지 않았다.

결과는 두 데이터셋 모두에서 라벨 관련 이웃 구조가 있음을 지지한다. 다만 GC10-DET은 클래스별 편차가 크고 8번 클래스 `rolled_pit`의 보존율이 매우 낮다. 따라서 “표현이 모든 결함 클래스를 잘 보존한다”는 결론은 허용되지 않는다.

## 입력 무결성

| 항목 | GC10-DET | NEU-DET |
|---|---:|---:|
| Cache 행 × 차원 | 1,605 × 384 | 1,259 × 384 |
| 저장 형식 | FP32, 행별 L2 정규화 | FP32, 행별 L2 정규화 |
| Cache A/B | bitwise exact | bitwise exact |
| Ordered manifest와 embedding row ID 순서 | exact | exact |
| Cache content SHA-256 | `cc1c9826c692f8b9c9b445504037698692cafa54f62848e79842de29e12a29b8` | `1f6c92f6e54a7ad8134bc6c65eeba6d5b5a51dc8f35a9c5bb1917dd32da9a326` |
| Blind acquisition manifest와 ID 일치 | 1,605/1,605 | 1,259/1,259 |
| Validation 겹침 | 0 | 0 |
| 잠긴 test 겹침 | 0 | 0 |
| 두 개 이상 객체 클래스를 가진 이미지 | 331 | 83 |
| 원 XML SHA-256 및 adapter class-set 대조 | 1,605/1,605 exact | 1,259/1,259 exact |

- DINOv2 revision: `ed25f3a31f01632728cabb09d1542f84ab7b0056`
- Model SHA-256: `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`
- 440-run protocol SHA-256: `6836f452d22c3de26dfa2680c45d0b5cb8f925b1a9274789c4ef81a1770f0b10`
- 440/440 training integrity: PASS, boundary violation 0, unauthorized config change 0.
- 각 acquisition 원 XML의 SHA-256을 sealed private source registry와 대조하고, 유효 bbox의 class set을 다시 파싱해 분석용 YOLO label class set과 전수 일치함을 확인했다.
- Release의 `planned_cells.jsonl`은 원 실행본과 줄바꿈 형식만 CRLF/LF로 달랐고, 440개 과학적 record는 줄 단위로 완전히 같았다. Registry seal도 의미상 동일했다.

## 결과 생성 전 분석 규약 보정

첫 실행은 NEU-DET에 다중 클래스 이미지가 있음을 발견하고 어떤 출력도 만들기 전에 안전 중단됐다. 1,259장 중 83장에 두 개 이상의 객체 클래스가 있으므로, 임의로 한 클래스를 택하거나 이미지를 제외하지 않고 다음처럼 규약을 고정한 뒤 재실행했다.

1. **1차 지표:** 두 데이터셋 모두 전체 객체 annotation class set을 사용한다. 최근접 이웃과 라벨을 하나 이상 공유하는지, 5개 이웃 중 공유 이웃의 비율, class-set Jaccard, class별 보존율을 계산한다.
2. **NEU-DET 2차 지표:** sealed source registry의 원래 파일명에 이미 기록된 대표 이미지 범주를 사용해 same/different category 거리, leave-one-out kNN, silhouette을 계산한다.
3. 결과를 본 뒤 metric, threshold, class를 바꾸지 않았다. 제외된 이미지는 0장이다. 5-NN 다수결 동률은 가장 작은 사전 고정 class index로 결정한다.

이 보정은 outcome-driven 변경이 아니라 실제 라벨 schema에 대한 결과 생성 전 수정이며, `integrity_record.json`에 기록했다.

## 정량 결과

### 객체 annotation class set 기준

| Dataset | 1-NN 라벨 공유 | 5-NN 라벨 공유 평균 | 1-NN class-set Jaccard | 라벨 공유 pair 평균 cosine 거리 | 라벨 비공유 pair 평균 cosine 거리 | 평균 거리 차이 |
|---|---:|---:|---:|---:|---:|---:|
| GC10-DET | 0.8125 | 0.7551 | 0.7383 | 0.3943 | 0.4542 | 0.0599 |
| NEU-DET | 0.9921 | 0.9865 | 0.9641 | 0.3111 | 0.5129 | 0.2019 |

라벨을 공유하는 pair가 공유하지 않는 pair보다 평균적으로 가깝다. NEU-DET에서 분리가 특히 크다. Pair들은 서로 독립인 표본이 아니므로, 이 표에는 독립성 가정이 필요한 p-value를 붙이지 않았다.

이웃 agreement는 class prevalence의 영향을 받는 raw descriptive 값이며 chance-adjusted effect가 아니다. 따라서 단독 성공 판정값으로 쓰지 않고 pairwise 거리와 class별 보존율을 함께 해석한다.

### NEU-DET 대표 이미지 범주 기준 2차 진단

| 지표 | 값 |
|---|---:|
| Leave-one-out 1-NN 정확도 | 0.9921 |
| Leave-one-out 5-NN 정확도 | 0.9905 |
| Cosine silhouette | 0.2995 |
| Same-category 평균 cosine 거리 | 0.2837 |
| Different-category 평균 cosine 거리 | 0.5127 |
| 평균 거리 차이 | 0.2290 |

대표 범주는 sealed source filename의 접두어에서 결정적으로 복원했다. 객체 annotation의 여러 클래스를 하나로 축약한 값이 아니며, 2차 진단에만 사용했다.

## 클래스별 최근접 이웃 보존율

### GC10-DET — 객체 annotation class 기준

| 번호 | Class | Support | 보존율 | 해석 |
|---:|---|---:|---:|---|
| 1 | punching_hole | 230 | 0.7130 | 중간 |
| 2 | welding_line | 358 | 0.7877 | 비교적 높음 |
| 3 | crescent_gap | 185 | 0.7459 | 중간 |
| 4 | water_spot | 217 | 0.7327 | 중간 |
| 5 | oil_spot | 175 | 0.6000 | 낮음 |
| 6 | silk_spot | 508 | 0.8976 | 높음 |
| 7 | inclusion | 141 | 0.6241 | 낮음 |
| **8** | **rolled_pit** | **32** | **0.2500** | **기존 우려와 일치하는 취약 class** |
| 9 | crease | 37 | 0.3243 | 매우 낮음 |
| 10 | waist_folding | 98 | 0.8367 | 높음 |

### NEU-DET — 객체 annotation class 기준

| 번호 | Class | Support | 보존율 |
|---:|---|---:|---:|
| 1 | crazing | 210 | 1.0000 |
| 2 | inclusion | 264 | 0.8750 |
| 3 | patches | 238 | 0.9454 |
| 4 | pitted_surface | 211 | 0.9716 |
| 5 | rolled-in_scale | 210 | 1.0000 |
| 6 | scratches | 210 | 0.9905 |

NEU-DET의 파일명 대표 범주 기준 class별 1-NN 정확도는 0.9762~1.0000이었다. 객체 annotation 기준에서는 `inclusion`이 0.8750으로 가장 낮았으므로, 대표 범주 지표만으로 클래스별 표현 품질을 단정하지 않는다.

## 판단과 논문 사용 범위

### 지지되는 해석

1. 이 exact checkpoint와 preprocessing으로 얻은 이미지 표현에는 두 데이터셋에서 라벨 관련 국소 구조가 있다.
2. NEU-DET의 대표 범주 구조는 1-NN·5-NN과 pairwise distance에서 매우 명확하다.
3. GC10-DET도 평균적으로 라벨 공유 이웃이 우세하지만, 클래스별 보존 정도는 크게 다르다.
4. GC10-DET `rolled_pit`의 낮은 보존율은 기존 representation 우려를 정량적으로 뒷받침한다.

### 지지되지 않는 해석

1. 모든 class에 적합한 표현이라는 주장
2. 이 사후 감사만으로 farthest-first 선택이 성능 차이의 원인이라는 주장
3. DINOv2 표현만의 효과와 acquisition rule의 효과를 분리했다는 주장
4. 잠긴 test 성능을 이용해 selector나 metric을 조정했다는 주장
5. 다른 산업 데이터로 일반화된다는 주장

## 재현 파일

- 분석 코드: `confirmatory_v1_1/posthoc_analysis/analyze_dinov2_representation_validity.py`
- 결과 루트: `confirmatory_v1_1/posthoc_analysis/results/dinov2_representation_validity_20260907/`
- Dataset 지표: `dataset_metrics.csv`
- Pairwise 거리 요약: `pairwise_distance_summary.csv`
- Class별 결과: `classwise_nearest_neighbor_agreement.csv`
- Sample별 최근접 이웃: `per_sample_nearest_neighbors.csv`
- 무결성 기록: `integrity_record.json`
- 출력 해시: `OUTPUT_SHA256.json`

기존 결과를 덮지 않는 재현 명령은 다음과 같다.

```powershell
& 'C:\Users\user\Desktop\vlm\Defect_VLM_Project\.venv\Scripts\python.exe' `
  '.\confirmatory_v1_1\posthoc_analysis\analyze_dinov2_representation_validity.py' `
  --output-dir '.\confirmatory_v1_1\posthoc_analysis\results\dinov2_representation_validity_20260907_reproduction'
```

이 명령은 로컬의 기존 sealed cache를 읽는 CPU 사후 분석이며 특징 추출이나 학습을 수행하지 않는다.

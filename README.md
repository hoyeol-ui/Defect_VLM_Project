# 금속 표면 결함 탐지의 표본 선택과 라벨 효율

**DINOv2-FF · Random · Entropy | YOLO26s | 200 epochs | 획득·학습 시드 5쌍**

같은 수의 라벨 이미지를 사용할 때 표본 선택 방법이 탐지 성능에 어떤 차이를 만드는지, 그 차이가 개별 결함 클래스에서도 같은 방향으로 나타나는지 분석한다. GC10-DET와 NEU-DET에서 세 전략을 동일한 검출기·학습 조건으로 비교한 실험이다.

**2026년 10월 27일 기준: 170개 모델의 학습·최종 Test와 로컬 백업 검증을 완료했다.** 이번 결과에서는 GC10의 평균 이득과 NEU의 혼합된 결과가 구분된다. 수정 원고도 이 결과에 맞춰 갱신했다.

## 결과와 원고 바로 보기

|자료|내용|
|---|---|
|[최종 결과 안내](docs/results/runpod_170_final_20261027/README_KO.md)|실험 결과, 해석 범위, 공개 패키지의 구성|
|[수정 원고 DOCX](docs/results/runpod_170_final_20261027/manuscript/03_JKICS_Revised_Manuscript_170models_20261027.docx) · [웹에서 읽는 원고](docs/results/runpod_170_final_20261027/manuscript/03_JKICS_Revised_Manuscript_170models_20261027.md)|방법·결과·표 9개·그림 5개·논의·결론|
|[예산별 최종 점수표](docs/results/runpod_170_final_20261027/results/TEST_RESULTS_KO.md)|10%, 15%, 20%, 25%, 30%, 35% 누적 이미지 예산의 Test 성능|
|[170모델 지표 CSV](docs/results/runpod_170_final_20261027/results/METRICS_170.csv) · [클래스별 지표 CSV](docs/results/runpod_170_final_20261027/results/CLASS_METRICS_170.csv)|반복별 전체·클래스별 AP, 정밀도, 재현율|
|[통계 요약 JSON](docs/results/runpod_170_final_20261027/results/TEST_SUMMARY.json)|평균·표본 SD, 대응 nAULC 차이, exact 검정, Holm 보정, Full 도달률|
|[실제 적용 설정](docs/results/runpod_170_final_20261027/protocol/EFFECTIVE_CONFIG.json) · [가중치 해시 감사](docs/results/runpod_170_final_20261027/evidence/MODEL_HASH_AUDIT.json)|학습·평가 조건과 170모델 best/last 검증 기록|

## 핵심 결과

아래는 **최종 Test 결과**이며, 평균 ± 표본 표준편차, n=5다. mAP50–95와 nAULC는 % 단위다. nAULC는 실제 10~35% 누적 이미지 수에 따른 학습곡선을 적분하고 관측 구간 길이로 나눈 값이다.

|데이터셋|전략|nAULC (%)|35% 예산 mAP50–95 (%)|
|---|---|---:|---:|
|GC10-DET|Random|23.35 ± 0.58|24.74 ± 0.37|
|GC10-DET|DINOv2-FF|24.62 ± 0.78|26.97 ± 1.22|
|GC10-DET|Entropy|24.17 ± 0.89|26.21 ± 0.92|
|NEU-DET|Random|29.82 ± 0.82|31.18 ± 1.72|
|NEU-DET|DINOv2-FF|29.65 ± 1.19|31.11 ± 0.97|
|NEU-DET|Entropy|29.41 ± 1.18|30.62 ± 1.10|

GC10에서는 DINOv2-FF의 평균 nAULC가 Random보다 **1.27%p 높았지만**, NEU에서는 **0.17%p 낮았다**. 35% 예산의 평균 AP만 보거나 한 데이터셋의 이득을 다른 데이터셋에 일반화하지 않는다.

<img src="docs/results/runpod_170_final_20261027/figures/Fig1.png" alt="GC10-DET와 NEU-DET의 누적 라벨 이미지 수에 따른 최종 Test 학습곡선" width="520">

그림의 선은 반복 평균, 띠는 표본 SD다. 같은 반복의 초기 10% 모델 R0는 세 전략이 공유한다.

### 대응 비교와 통계적 해석

|데이터셋|nAULC 비교|평균 차이 (%p)|양측 exact p|Holm 보정 p|
|---|---|---:|---:|---:|
|GC10-DET|DINOv2-FF − Random|+1.27|0.1250|0.5000|
|GC10-DET|DINOv2-FF − Entropy|+0.45|0.4375|1.0000|
|NEU-DET|DINOv2-FF − Random|−0.17|0.6875|1.0000|
|NEU-DET|DINOv2-FF − Entropy|+0.25|0.5625|1.0000|

검정 단위는 같은 획득·학습 시드 쌍의 대응 차이다. 양측 exact sign-flip의 32개 부호 조합을 열거하고, 사전 지정한 네 비교에 Holm 보정을 적용했다. n=5에서 가능한 최소 양측 p값은 0.0625다. 따라서 이 실험은 지정한 검정으로 통계적 우위를 확립하지 못했으며, 방법 간 동등성을 입증한 결과도 아니다.

### 평균과 클래스별 결과를 함께 보는 이유

GC10에서 DINOv2-FF의 35% 평균 AP가 높아졌어도 inclusion은 Random보다 **1.30%p**, silk spot은 **1.07%p** 낮았다. NEU에서는 scratches가 **4.33%p** 높고 rolled-in scale이 **4.07%p** 낮아 클래스별 방향이 갈렸다. NEU의 최저 클래스 nAULC도 DINOv2-FF가 **0.94%p 낮았다**.

[전체 클래스별 차이 그림](docs/results/runpod_170_final_20261027/figures/Fig4.png)과 [최저 클래스 학습곡선](docs/results/runpod_170_final_20261027/figures/Fig5.png)을 함께 제공한다. 클래스별 변화와 선택 이미지 구성의 관계는 사후 기술적 분석이며, 성능 변화의 원인을 분리한 대조 실험은 아니다.

## 실험 설계

|항목|설정|
|---|---|
|데이터셋|GC10-DET 10클래스, NEU-DET 6클래스|
|고정 Train / Dev / Test|GC10: 1,604 / 344 / 343장 · NEU: 1,258 / 270 / 270장|
|누적 라벨 이미지 예산|약 10%, 15%, 20%, 25%, 30%, 35%|
|GC10 실제 누적 이미지 수|160, 241, 321, 401, 481, 561장|
|NEU 실제 누적 이미지 수|126, 189, 252, 315, 377, 440장|
|학습|YOLO26s, 200 epochs, batch 32, FP32, AMP·TF32 off, SGD, cosine schedule|
|반복|사전 지정한 획득 시드 5개와 학습 시드 5개를 일대일로 대응|
|초기화|모든 예산에서 동일한 공개 사전학습 가중치로 새로 학습|
|모델 선택|Dev mAP50–95로 선택한 best checkpoint|
|최종 평가|고정 Test, 640, FP32, batch 8, Ultralytics 8.4.90|
|모델 수|170개 물리 모델, 공통 R0를 전략별로 연결한 190개 논리 결과 행|

Random은 미선택 이미지에서 비복원 무작위 추출한다. DINOv2-FF는 고정된 DINOv2 ViT-S/14의 정규화 CLS 특징으로 farthest-first 선택을 수행한다. Entropy는 직전 모델의 원래 클래스 점수 벡터를 이용해 정규화한 박스 엔트로피를 이미지별로 평균한다. 특징 전처리, 동점 처리와 획득용 NMS는 [실제 적용 설정](docs/results/runpod_170_final_20261027/protocol/EFFECTIVE_CONFIG.json)에 명시했다.

데이터셋당 공통 R0 5개, 세 전략의 추가 다섯 예산 모델 75개, 전체 Train을 사용하는 Full 참조 5개를 합쳐 85개다. 두 데이터셋의 합이 170개이며, 공유 R0를 중복 추론하지 않았다.

## 해석의 범위

- **반복과 일반화:** 획득·학습 시드를 함께 바꾼 다섯 쌍의 결과다. 두 무작위성의 분산을 따로 추정하지 않으며, 두 공개 자료의 한 고정 분할과 한 검출기에 대한 관측이다.
- **Test의 지위:** 이번 학습·획득에서는 Test를 사용하지 않았다. 원천 자료의 과거 평가·이전 분할 사용 이력은 남아 있어, 연구 전 과정에서 미관찰인 외부 데이터로 부르지 않는다.
- **Full 참조:** 같은 학습 시드의 Full 모델로 반복별 비율을 먼저 계산했다. DINOv2-FF의 35% 도달률 평균은 GC10 **94.58 ± 7.02%**, NEU **89.65 ± 3.26%**다. 이미지 예산만으로 실제 주석 비용 절감률을 산출하지 않는다.
- **P/R와 평가 경로:** P/R는 모델별 평활 평균 F1 최대점의 값이다. 공통 confidence에서의 산업 운영 지표가 아니며, 획득용 클래스별 NMS와 최종 YOLO26 one-to-one NMS-free 평가도 구분한다.
- **실행 환경:** Linux Blackwell/cu129에서 학습하고 Windows RTX3070/cu126에서 평가했다. 장치 간 비트 단위 동일성을 주장하지 않는다.

## 저장된 결과 검증

저장소 루트에서 다음 명령으로 공개 파일 해시와 저장된 지표를 확인할 수 있다. Python 표준 라이브러리만 사용하며 모델을 불러오지 않는다.

```bash
python docs/results/runpod_170_final_20261027/scripts/verify_export.py
```

검증 항목은 파일 SHA-256, 170개 모델과 Test 기록, best/last 340개 해시의 기록 간 일치, 190개 논리 행의 연결, 평균·표본 SD, 실제 이미지 수 기반 nAULC, exact p값·Holm 보정, 대응 Full 비율이다. 가중치 바이트 자체의 재검증은 전체 로컬 백업에서 완료한 별도 절차이며, 이 공개 명령의 범위에는 포함되지 않는다.

[공개 내보내기 출처와 변환 기록](docs/results/runpod_170_final_20261027/PROVENANCE.json), [파일 해시 목록](docs/results/runpod_170_final_20261027/SHA256SUMS), [최종 Test 완료 기록](docs/results/runpod_170_final_20261027/evidence/FINAL_TEST_COMPLETE.json)을 함께 보존한다. 컴퓨터의 절대 경로를 상대 경로로 바꾼 파일은 원본 해시와 공개 파일 해시를 구분했다.

## 공개 자료 구성

```text
docs/results/runpod_170_final_20261027/
├─ README_KO.md          # 최종 결과와 해석 안내
├─ manuscript/          # 수정 DOCX, Markdown, 원고 검증 기록
├─ results/             # 전체·클래스별 지표, 대응 통계, 사후 분석
├─ figures/             # 원고 그림 5개와 PDF 원본
├─ protocol/            # 실제 적용 조건과 고정된 설정 자료
├─ evidence/            # 학습·Test 완료, 백업, 해시 감사 기록
├─ test_records/        # 170모델의 개별 Test 지표
├─ code/                # 실행 당시 학습·평가 소스 참조
├─ scripts/verify_export.py
├─ PROVENANCE.json
└─ SHA256SUMS
```

데이터 원본과 모델 가중치, 전체 서버 백업은 로컬에 보관한다. 공개된 `code/`는 실행 당시 소스 참조이며, 데이터·가중치·봉인된 입력과 실행 환경을 포함한 독립 재실행 키트는 아니다.

## 이전 연구 이력

저장소 이름과 기존 코드에는 초기 VLM 설명 일관성·OWL-ViT 실험의 이력이 남아 있다. 해당 방법·점수·실행 명령은 [이전 VLM 연구 README](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/a189a7252c5b35208d6da2ce16ccf13409c7073d/README.md)에서 확인할 수 있다. 이번 DINOv2-FF·Random·Entropy의 170모델 결과는 위의 날짜별 패키지로 관리한다.

현재 결과 브랜치는 `codex/runpod-170-final-results-20261027`이며, `main` 반영은 [PR #5](https://github.com/hoyeol-ui/Defect_VLM_Project/pull/5)에서 관리한다.

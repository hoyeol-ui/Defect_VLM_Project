# 금속 표면 결함 탐지의 표본 선택과 라벨 효율

**DINOv2-FF · Random · Entropy | YOLO26s | 200 epochs | 획득·학습 시드 5쌍**

같은 수의 라벨 이미지를 사용할 때 표본 선택 방법이 탐지 성능에 어떤 차이를 만드는지, 그 차이가 개별 결함 클래스에서도 같은 방향으로 나타나는지 분석한다. GC10-DET와 NEU-DET에서 세 전략을 동일한 검출기·학습 조건으로 비교한 실험이다.

**170개 모델의 학습·최종 Test와 로컬 백업 검증을 완료했고, 2026년 10월 28일 원고를 보강했다.** 핵심 관측은 평균 탐지 성능의 이득이 모든 결함 클래스의 개선으로 이어지지는 않는다는 점이다. GC10의 평균 이득은 일부 클래스에 집중됐고, NEU의 작은 평균 차이는 클래스별 증감이 상쇄된 결과였다.

## 결과와 원고 바로 보기

|자료|내용|
|---|---|
|[최종 결과 안내](docs/results/runpod_170_final_20261027/README_KO.md)|실험 결과, 해석 범위, 공개 패키지의 구성|
|[최신 개정 원고 DOCX](docs/manuscripts/runpod_170_20261028_v2/03_JKICS_Revised_Manuscript_170models_20261028_v2.docx) · [웹에서 읽는 전체 원고](docs/manuscripts/runpod_170_20261028_v2/03_JKICS_Revised_Manuscript_170models_20261028_v2.md)|2026-10-28 개정, 9쪽, 표 9개·그림 6개, 전체 16클래스 분석|
|[원고 개정 내용과 검증](docs/manuscripts/runpod_170_20261028_v2/README_KO.md)|연구 질문·구성도·클래스별 기여·Full 도달률의 반복별 범위|
|[예산별 최종 점수표](docs/results/runpod_170_final_20261027/results/TEST_RESULTS_KO.md)|10%, 15%, 20%, 25%, 30%, 35% 누적 이미지 예산의 Test 성능|
|[170모델 지표 CSV](docs/results/runpod_170_final_20261027/results/METRICS_170.csv) · [클래스별 지표 CSV](docs/results/runpod_170_final_20261027/results/CLASS_METRICS_170.csv)|반복별 전체·클래스별 AP, 정밀도, 재현율|
|[통계 요약 JSON](docs/results/runpod_170_final_20261027/results/TEST_SUMMARY.json)|평균·표본 SD, 대응 nAULC 차이, exact 검정, Holm 보정, Full 도달률|
|[실제 적용 설정](docs/results/runpod_170_final_20261027/protocol/EFFECTIVE_CONFIG.json) · [가중치 해시 감사](docs/results/runpod_170_final_20261027/evidence/MODEL_HASH_AUDIT.json)|학습·평가 조건과 170모델 best/last 검증 기록|

## 연구 구성과 질문

첫째, 같은 이미지 예산에서 DINOv2-FF의 학습곡선과 대응 효과가 Random·Entropy와 어떻게 다른지 비교한다. 둘째, 전체 평균의 이득이 개별 클래스와 최저 클래스 성능에서도 유지되는지 살펴본다. 공유 R0, 예산별 cold start와 대응 Full 참조가 비교 기준을 맞춘다.

<img src="docs/manuscripts/runpod_170_20261028_v2/figures/Fig1_Study_Architecture.png" alt="고정 데이터 분할에서 세 선택 전략, 누적 라벨 집합, YOLO26s cold start, Dev best 선택과 최종 Test로 이어지는 연구 구성도" width="1000">

직전 검출기의 점선 피드백은 다음 Entropy 순위 계산에만 사용한다. 모든 예산은 동일한 공개 초기 가중치에서 새로 학습하고, Test는 고정된 모델을 평가한다. DINOv2 특징은 선택 표현으로 사용한다. [구성도 SVG](docs/manuscripts/runpod_170_20261028_v2/figures/Fig1_Study_Architecture.svg)와 전체 프로토콜은 최신 원고에서 확인할 수 있다.

## 핵심 결과

아래는 **최종 Test 결과**이며, 평균 ± 표본 표준편차, n=5다. mAP50–95와 nAULC는 % 단위다. nAULC는 실제 10~35% 누적 이미지 수에 따른 학습곡선을 적분하고 관측 구간 길이로 나눈 값이다.

|데이터셋|전략|전체 nAULC (%)|최저 클래스 nAULC (%)|35% 예산 mAP50–95 (%)|
|---|---|---:|---:|---:|
|GC10-DET|Random|23.35 ± 0.58|1.97 ± 0.34|24.74 ± 0.37|
|GC10-DET|DINOv2-FF|24.62 ± 0.78|2.69 ± 0.58|26.97 ± 1.22|
|GC10-DET|Entropy|24.17 ± 0.89|2.84 ± 1.62|26.21 ± 0.92|
|NEU-DET|Random|29.82 ± 0.82|7.26 ± 0.73|31.18 ± 1.72|
|NEU-DET|DINOv2-FF|29.65 ± 1.19|6.32 ± 1.41|31.11 ± 0.97|
|NEU-DET|Entropy|29.41 ± 1.18|7.31 ± 1.71|30.62 ± 1.10|

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

DINOv2-FF는 Entropy보다 전체 nAULC가 GC10 **0.45%p**, NEU **0.25%p** 높았지만, 최저 클래스 nAULC는 각각 **0.15%p**, **0.99%p 낮았다**. 평균 성능과 하위 클래스 성능으로 정한 순서가 달랐다. 최저 클래스 nAULC는 각 반복·예산의 최소 클래스 AP를 적분한 사후 요약이며, 한 클래스를 계속 추적한 곡선이 아니다.

35% 예산의 DINOv2-FF−Random 차이를 클래스별 AP의 산술평균으로 분해하면 다음과 같다. 아래 기여는 클래스 AP 차이를 전체 클래스 수(GC10 10개, NEU 6개)로 나눈 값이다.

|데이터셋|클래스|클래스 AP 차이 (%p)|전체 mAP 차이에 대한 산술 기여 (%p)|
|---|---|---:|---:|
|GC10|crease|+9.37|+0.94|
|GC10|waist folding|+6.01|+0.60|
|NEU|scratches|+4.33|+0.72|
|NEU|rolled-in scale|−4.07|−0.68|
|NEU|crazing|−2.42|−0.40|

GC10의 전체 증가 **2.23%p 중 1.54%p**가 crease와 waist folding의 증가분이다. 동시에 inclusion과 silk spot은 각각 1.30%p, 1.07%p 낮았다. NEU에서는 scratches의 +0.72%p 기여와 rolled-in scale·crazing의 −1.08%p 기여에 나머지 클래스의 증가가 더해져 전체 차이가 **−0.07%p**가 됐다.

최신 원고 표 7은 일부 사례를 넘어 **16개 전체 클래스**의 평균·표본 SD와 같은 반복끼리 뺀 대응 차이를 제시한다. [전체 클래스 차이 그림](docs/results/runpod_170_final_20261027/figures/Fig4.png)과 [최저 클래스 학습곡선](docs/results/runpod_170_final_20261027/figures/Fig5.png)도 함께 확인할 수 있다. 기존 그림 파일 이름은 보존했으며 최신 원고에서는 각각 그림 5·6이다.

선택된 클래스 포함 이미지 수와 AP의 관계는 사후 기술적 분석이다. 예를 들어 GC10 waist folding은 평균 선택 이미지 수가 줄어도 AP는 높아졌다. 이미지 수만으로 성능을 설명하거나 구성의 차이를 인과 효과로 해석하지 않는다. 전체 수치는 [최신 웹 원고](docs/manuscripts/runpod_170_20261028_v2/03_JKICS_Revised_Manuscript_170models_20261028_v2.md)에 있다.

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
- **Full 참조:** 같은 학습 시드의 Full 모델로 반복별 비율을 먼저 계산한다. 전체 Train도 동일한 200epoch 규칙으로 학습한 경험적 참조라 100%를 넘는 반복이 가능하며, 동일 계산량 비교가 아니다. 이미지 수에는 상자 수·주석 난도·특징 추출 비용이 포함되지 않는다.
- **P/R와 평가 경로:** P/R는 모델별 평활 평균 F1 최대점의 값이다. 공통 confidence에서의 산업 운영 지표가 아니며, 획득용 클래스별 NMS와 최종 YOLO26 one-to-one NMS-free 평가도 구분한다.
- **실행 환경:** Linux Blackwell/cu129에서 학습하고 Windows RTX3070/cu126에서 평가했다. 장치 간 비트 단위 동일성을 주장하지 않는다.

### Full 평균 도달률과 반복별 범위

|데이터셋|DINOv2-FF의 35% / Full (%)|다섯 쌍의 관측 범위 (%)|95% 이상인 반복|
|---|---:|---:|---:|
|GC10-DET|94.58 ± 7.02|89.61–105.52|2/5|
|NEU-DET|89.65 ± 3.26|86.02–94.03|0/5|

95%는 사후 기술적 참조선이다. GC10의 평균이 약 95%라는 것과 대부분의 반복이 그 수준에 도달했다는 것은 다르다. 이 비율은 실제 작업 시간이나 주석 비용 절감률이 아니다.

## 저장된 결과 검증

저장소 루트에서 다음 명령으로 공개 파일 해시와 저장된 지표를 확인할 수 있다. Python 표준 라이브러리만 사용하며 모델을 불러오지 않는다.

```bash
python docs/results/runpod_170_final_20261027/scripts/verify_export.py
python docs/manuscripts/runpod_170_20261028_v2/scripts/verify_revision.py
```

검증 항목은 파일 SHA-256, 170개 모델과 Test 기록, best/last 340개 해시의 기록 간 일치, 190개 논리 행의 연결, 평균·표본 SD, 실제 이미지 수 기반 nAULC, exact p값·Holm 보정, 대응 Full 비율이다. 가중치 바이트 자체의 재검증은 전체 로컬 백업에서 완료한 별도 절차이며, 이 공개 명령의 범위에는 포함되지 않는다.

[공개 내보내기 출처와 변환 기록](docs/results/runpod_170_final_20261027/PROVENANCE.json), [파일 해시 목록](docs/results/runpod_170_final_20261027/SHA256SUMS), [최종 Test 완료 기록](docs/results/runpod_170_final_20261027/evidence/FINAL_TEST_COMPLETE.json)을 함께 보존한다. 컴퓨터의 절대 경로를 상대 경로로 바꾼 파일은 원본 해시와 공개 파일 해시를 구분했다.

원고 개정 검증은 새 폴더의 파일 해시, DOCX 원본 일치, 수식·기존 그림 보존과 저장된 지표에서 다시 계산한 클래스 기여·최저 클래스 nAULC·Full 비율을 확인한다. [개정 출처](docs/manuscripts/runpod_170_20261028_v2/PROVENANCE.json)와 [검증 요약](docs/manuscripts/runpod_170_20261028_v2/VERIFICATION.json)을 제공한다. 기존 2026-10-27 패키지의 227개 파일과 해시 목록은 그대로 보존했다.

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

최신 원고와 구성도는 [`docs/manuscripts/runpod_170_20261028_v2/`](docs/manuscripts/runpod_170_20261028_v2/README_KO.md)에 있다. 2026-10-27 폴더의 원고는 이전 공개 버전으로 보존한다.

데이터 원본과 모델 가중치, 전체 서버 백업은 로컬에 보관한다. 공개된 `code/`는 실행 당시 소스 참조이며, 데이터·가중치·봉인된 입력과 실행 환경을 포함한 독립 재실행 키트는 아니다.

## 이전 연구 이력

저장소 이름과 기존 코드에는 초기 VLM 설명 일관성·OWL-ViT 실험의 이력이 남아 있다. 해당 방법·점수·실행 명령은 [이전 VLM 연구 README](https://github.com/hoyeol-ui/Defect_VLM_Project/blob/a189a7252c5b35208d6da2ce16ccf13409c7073d/README.md)에서 확인할 수 있다. 이번 DINOv2-FF·Random·Entropy의 170모델 결과는 위의 날짜별 패키지로 관리한다.

현재 결과 브랜치는 `codex/runpod-170-final-results-20261027`이며, `main` 반영은 [PR #5](https://github.com/hoyeol-ui/Defect_VLM_Project/pull/5)에서 관리한다.

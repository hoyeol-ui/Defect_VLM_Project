# RunPod170모델 최종 결과와 수정 원고

2026년10월27일 완료한 GC10-DET·NEU-DET의 Random / DINOv2-FF / Entropy 비교 결과다. YOLO26s를200epoch, FP32, batch32로 학습하고, 미리 정한 획득·학습 시드5쌍으로 평가했다. **170개 물리 모델의 최종 Test가 모두 PASS**이며, 공통 R0를 전략에 연결한 논리 결과는190행이다. 원 데이터와 가중치, 전체 서버 백업은 로컬에 보관한다.

## 먼저 볼 파일

- [최종 예산별 점수표](results/TEST_RESULTS_KO.md)
- [수정 원고 DOCX](manuscript/03_JKICS_Revised_Manuscript_170models_20261027.docx) / [GitHub에서 읽는 원고](manuscript/03_JKICS_Revised_Manuscript_170models_20261027.md)
- [원시 요약 및 대응 검정](results/TEST_SUMMARY.json), [170모델 지표 CSV](results/METRICS_170.csv), [클래스별 지표 CSV](results/CLASS_METRICS_170.csv)
- [실제로 적용한 설정](protocol/EFFECTIVE_CONFIG.json), [170모델 best/last 감사](evidence/MODEL_HASH_AUDIT.json)
- [내보낸 파일 검증](scripts/verify_export.py): 아래 CPU 명령은 모델을 불러오거나 Test를 재실행하지 않는다.

```sh
python scripts/verify_export.py
```

## 핵심 결과

mAP50-95와 nAULC는% 단위, 차이는%p 단위이며 평균±표본 SD, n=5다. nAULC는 실제10~35% 누적 이미지 수로 적분하고 구간 길이로 나눈다. Full로 정규화하지 않는다.

|자료|전략|nAULC|35% mAP50-95|
|---|---|---:|---:|
|GC10-DET|Random|23.35±0.58|24.74±0.37|
|GC10-DET|DINOv2-FF|24.62±0.78|26.97±1.22|
|GC10-DET|Entropy|24.17±0.89|26.21±0.92|
|NEU-DET|Random|29.82±0.82|31.18±1.72|
|NEU-DET|DINOv2-FF|29.65±1.19|31.11±0.97|
|NEU-DET|Entropy|29.41±1.18|30.62±1.10|

DINO−Random nAULC는 GC10 +1.27%p, NEU −0.17%p다. 양측 exact sign-flip p는0.125와0.6875, 네 비교에 대한 Holm p는0.50과1.00이다. DINO−Entropy는 각각 +0.45%p와 +0.25%p이며 Holm p는 모두1.00이다. n=5 양측 정확 검정의 최소 p=.0625이므로 통계적 우위나 동등성을 입증한 결과로 서술하지 않는다.

GC10의 R5 평균 이득은 모든 클래스에 적용되지 않는다. inclusion은−1.30%p, silk spot은−1.07%p다. NEU의 scratches는+4.33%p, rolled-in scale은−4.07%p이며, 최저 클래스 nAULC도 DINO가 Random보다0.94%p 낮다. 클래스·구성 분석은 사후 기술적 분석이며 인과적 설명이 아니다.

Full5모델 평균은 GC10 28.56±0.80%, NEU 34.72±1.16%다. 같은 학습 시드의 Full과 R5를 먼저 나눈 DINO 도달률 평균은94.58±7.02%,89.65±3.26%다. 전체 자료에서95%를 달성했다거나65%의 실제 주석 비용을 절약했다고 해석하지 않는다.

## 실험 단위와 해석 범위

- Train/Dev/Test는 GC10 1604/344/343장, NEU1258/270/270장이다. 예산별 실제 누적 수와5쌍의 시드는 EFFECTIVE_CONFIG에 있다. 획득·학습 무작위성은 결합 단위이며 두 분산을 따로 추정하지 않는다.
- Test 모델은 Dev mAP50-95로 선택한 best.pt다. 각 모델을 한 번 평가했고, 원천 데이터의 과거 노출 이력은 남아 있다. 새 캠페인의 학습·획득에서 제외한 Test이며 완전히 새로운 외부 자료는 아니다.
- 학습은 Linux Blackwell/cu129, 평가는 Windows RTX3070/cu126이었다. Ultralytics8.4.90, FP32,640, 고정batch8로 최종 평가했다. 장치 간 비트 동일성을 주장하지 않는다.
- 획득 Entropy의 클래스별 NMS와 최종 YOLO26 one-to-one NMS-free 평가는 별개다. P/R는 모델별 평활 평균 F1 최대점의 값이며 공통 confidence 운영점이 아니다.
- common_args의 초기 batch4·기본 시드를 실제 학습 설정으로 읽지 않는다. FROZEN_PLAN과170개 COMPLETE에서 batch32·200epoch·각 learner seed를 검증했다. 초기 단계에서 복구 보존한 모델도 실제 가중치 감사로 포함했다.
- `code/`는 실행 당시 소스 참조다. 원 데이터·가중치·봉인된 로컬 입력과 GPU guard가 이 공개 패키지에 없으므로 독립적인 재학습·Test 실행 키트로 취급하지 않는다. CPU 결과 검증만 이 패키지에서 바로 실행한다.

## 보존과 공개 내보내기

선택한 로컬 백업21,745파일,47,916,387,837바이트의 서버/수신 SHA-256 일치 및 디스크 재읽기를 통과했고 best/last340개를 대조했다. 이 공개 패키지는 그 전체 백업을 대체하지 않는다. 모델·데이터·접속 키·서버 주소·프로세스 목록은 포함하지 않는다. Pod 중지/삭제 API나 재실행 명령도 호출하지 않는다. 종료 여부는 사용자의 별도 RunPod 조작이며 이 기록은21:50 KST 인계 당시 상태다.

`PROVENANCE.json`은 원본 파일 해시와 변환 종류, `SHA256SUMS`는 공개 파일의 해시를 기록한다. TEST_SUMMARY와190행은 원본 바이트를 유지한다. 물리 모델·개별 Test 기록은 컴퓨터의 절대 경로만 백업 기준 상대 경로로 바꾸었으므로 공개 파일 해시는 원본 metrics receipt 해시와 다르다. TEST_RECEIPTS_170에 두 해시를 구분해 저장했고 내보내기 전에 원본170개를 원래 receipt 및 인계 기록과 대조했다.

원고는 이번170모델 결과에 맞춰 방법·표9개·그림5개·논의·결론을 수정했다. 원본 DOCX는 로컬에서 보존했고 수정본8페이지의 배치,138개 수치 셀,4개 편집 가능한 수식을 확인했다. 학술지 투고나 게재 완료를 뜻하지 않는다. 이전 실험 결과는 별도 역사로 남으며 이번 수치와 혼합하지 않는다.

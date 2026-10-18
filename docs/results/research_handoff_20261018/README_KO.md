# GC10-DET 연구 중단 시점 인계 — 2026-10-18

**현재 결정: GPU 실험과 파라미터 탐색을 멈추고, 기존 결과를 정리한다. 새 학습·추론·선택 알고리즘·추가 진단은 승인 없이 시작하지 않는다.**

이 문서는 2026-10-17~18의 단일 시드 개발 실험을 포함한 최신 인계다. 내일 MacBook에서는 아래 문서를 읽고 기존 결과의 해석과 연구 서술을 이어간다. 과거 보고서의 ‘다음 실험’ 문구는 당시 계획이며 현재 실행 지시가 아니다.

## 먼저 읽을 문서

1. [실험 결과 원표와 비교](EXPERIMENT_RESULTS_KO.md)
2. [현재 판단·연구 이력·해석의 한계](DECISIONS_AND_LIMITS_KO.md)
3. [MacBook 재개 안내와 바로 붙여넣을 인계문](MACBOOK_START_HERE.md)

## 현재 기준을 구분한다

- **전체 mAP50-95 최고: CM15 = 31.268124%.** Inclusion AP50-95 7.117179%, AP75 3.033077%.
- **Inclusion 중심 비교 기준: CM20 = 전체 31.063447%, inclusion AP50-95 8.933868%, AP75 5.487196%.**
- CM15는 전체 평균이 CM20보다 +0.204677%p 높지만 inclusion과 water_spot이 모두 하락했다. ‘water_spot 회복 + inclusion 유지’라는 사전 기준은 실패했다. 해당 목적의 기준은 CM20을 유지한다.
- CM20도 water_spot 저하, silk_spot 등의 낮은 절대 AP가 남아 있다. 전 클래스 개선이나 산업 적용 성능 확보를 주장하지 않는다.
- 전체 평균 최고와 취약 클래스 중심 기준을 한 모델의 ‘최고 성능’으로 혼용하지 않는다. R5 561장과 Full1604도 라벨 예산이 다르므로 구분한다.

## 지금까지 확인한 방향

- SGD에서 Cosine만 켠 비교는 전체·inclusion AP에 양의 개발 결과를 보였다.
- 동일 SGD100 best.pt의 1024 추론은 전체와 inclusion AP가 하락했다.
- AdamW+Cosine은 전체 성능 비저하 기준을 충족하지 못했다.
- close_mosaic10→20은 inclusion AP75와 silk_spot·rolled_pit AP50-95가 상승했지만 water_spot이 하락했다.
- close_mosaic20→15는 silk_spot·rolled_pit과 전체 평균은 상승했지만 inclusion 및 water_spot이 하락했다. 중간값이 장점을 결합한다는 가설은 이번 실행에서 지지되지 않았다.
- 이것은 관찰된 조건의 결과이며 ‘더 이상 어떤 실험도 의미 없다’ 또는 데이터셋 성능 상한을 입증한 것은 아니다. **실험 중단은 현재 자원과 연구 정리를 위한 사용자 결정**이다.

## 파일과 보존 범위

- [RESULTS.json](RESULTS.json): 정밀 수치와 비교 기준.
- [tables/](tables/): 전체·클래스별 AP 및 R5 선택 비교 CSV. 수치는 모두 % 단위다.
- [evidence/](evidence/): 완료된 평가 JSON, best epoch/해시 기록, 설정, 학습 CSV와 당시 보고서의 사본.
- [recipes/](recipes/): 실제 사용한 준비·학습·평가 코드와 설정의 참고 사본. **자동 실행하지 않는다.**
- [data_identity/](data_identity/): 고정 train/dev ID, 클래스 순서, 데이터·초기 가중치 해시 원장. 이미지·라벨 파일 자체는 포함하지 않았다.
- [CHECKPOINT_LOCATIONS.csv](CHECKPOINT_LOCATIONS.csv): Windows에 남아 있는 best.pt 위치와 당시 기록된 SHA256. 모델 파일은 Git에 업로드하지 않았다.
- [SOURCE_FILES.csv](SOURCE_FILES.csv): 원본 위치와 사본 SHA256. Windows 원본은 `C:/Users/user/Desktop/vlm/`에 상대 경로를 붙여 찾는다.
- [artifact_manifest.json](artifact_manifest.json): Git 패키지 자체의 파일 해시. `python3 tools/verify_bundle.py`로 MacBook에서 확인할 수 있다.

이 패키지는 문서·경량 수치·참고 코드 인계다. 원본 이미지/라벨, DINO 캐시, checkpoint, 전체 예측 JSON, 가상환경은 Windows에 남는다. Git clone만으로 과거 GPU 실험을 즉시 재실행할 수 있다는 뜻은 아니다. 문서 검토와 저장된 AP 비교는 가능하다.

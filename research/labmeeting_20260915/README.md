# 2026-09-15 랩미팅 후속 분석

**기존 분할의 탐색 결과입니다. 입력 구성과 선택 변화까지 확인했으며, YOLO 재학습이나 detector 성능 개선 검증은 수행하지 않았습니다.**

## 회사 컴퓨터에서 읽을 순서

1. [결과 요약](00_먼저읽기.md)
2. [실험 방법·Entropy·644회 설명](reports/01_실험방법_짧은설명.md)
3. [기존 성능과 본문·부록 그림](reports/02_기존성능_해석.md)
4. [거리·입력 진단과 다음 예비 학습 조건](reports/03_거리와_입력진단_결과.md)

그림 19개를 PNG/SVG로, 수치와 모든 선택 ID를 CSV/JSON/NPY로 제공합니다. 문서의 그림 링크는 다른 폴더에 압축을 풀어도 열리는 상대경로로 바꿨습니다.

## 현재 결론

- GC10 C224에서 박스 40.54%가 완전히 입력 밖으로 빠졌습니다. ‘손실 박스가 있는 이미지 64.74%’는 **박스 하나 이상이 절반 미만 남는 이미지**를 뜻합니다.
- F224와 F518은 박스를 기하학적으로 보존합니다. F518은 GC10에서 잘리기 전 C224와 비슷한 크기를 유지합니다.
- GC10 F518과 C224의 R1 추가 선택 일치율은 5개 시드 평균 39%입니다. 학습 성능 향상 여부는 미검증입니다.
- 다음 후보는 GC10 F518입니다. 분할 정비 후 공통 R0 1회+Random/C224/F518의 R1 3회 예비 학습을 제안합니다.
- 최근접 라벨 구조와 crop 손실이 함께 관찰됐습니다. 배경·중복·크기·여백의 기여를 분리하거나 NEU 취약 클래스 하락의 원인을 확정한 결과는 아닙니다.

## 공개 저장소와 전체 백업의 구성

이 공개 폴더에는 코드·보고서·표·그래프·익명 ID 특징/선택 결과·기존 성능 원자료를 담았습니다. 랩미팅 원문 전사본, 원본 데이터 영상·XML, 모델 가중치는 공개 업로드하지 않았습니다. 전체 이동용 ZIP에는 이 자료들이 별도 폴더로 포함됩니다.

```
전체_백업/
  00_회사에서_먼저읽기.md
  repository/                     # 이 브랜치의 전체 소스 스냅샷
    research/labmeeting_20260915/  # 이 폴더
  private_inputs/                 # 선택 풀 영상·XML, 원본 캐시, 가중치
  original_run/                   # 처음 실행한 결과의 원래 바이트
  context/                        # 전사본·이전 계획·전달 문서
  presentation/                   # 이번 작업에서 수정했던 PPTX/PDF
  GITHUB_UPLOAD.json
  BACKUP_MANIFEST.json
  verify_backup.py
```

## 읽기·추가 분석

문서·PNG·CSV는 Python 없이 열 수 있습니다. Python이 있으면 다음 명령은 표준 라이브러리만 사용합니다.

```powershell
python office.py summary
python office.py verify
```

추가 그래프나 선택 재계산은 `requirements-analysis.txt`와 저장된 [실행 환경](checks/software_and_sources.json)을 참고합니다. 원본 가상환경 전체를 백업한 것은 아닙니다. 기존 GPU 환경을 임의로 업그레이드할 필요는 없습니다.

```powershell
# 현재 폴더에서 실행. 기존 결과를 덮어쓰지 않도록 새 출력 경로를 지정합니다.
python office.py run metrics --output ../../../../office_metrics_recheck
python office.py run select --output ../../../../office_selection_recheck
```

`metrics`는 기존 성능 CSV와 raw JSON 440개의 해시를 대조하고 두 지표를 다시 집계합니다. `select`는 저장된 특징으로 5개 시드의 R1을 다시 계산합니다. 두 명령은 학습이나 새 특징 추출을 실행하지 않습니다.

전체 ZIP의 `private_inputs`가 있으면 다음도 가능합니다.

```powershell
python office.py run distance --output ../../../../office_distance_recheck
python office.py run diagnostics --output ../../../../office_diagnostics_recheck
```

`distance`와 `diagnostics`는 원본 영상을 읽어 사례 그림을 만듭니다. 새 GPU 추출은 자동 실행하지 않습니다. 원본 C224 재현·F224/F518 추출 코드도 `code`에 포함되어 있으며, 환경이 준비된 경우에만 직접 실행하세요.

## 코드와 무결성 구분

- 분석 기준 저장소: `bec9d65dc73ae28216b3009134b4f3b1f86f299b`.
- 당시 실제 실행 코드: [provenance/executed_code](provenance/executed_code). 원래 Windows 경로도 그대로 보존했습니다.
- 이동용 코드: [code](code). 경로 탐색과 기존 지표 원파일 위치만 이동형으로 조정했습니다. 선택 규칙·수치·사전 사례 ID는 바꾸지 않았습니다.
- `provenance/original_run_sha256.json`은 당시 원본 결과에 대한 해시입니다. 이 이동형 폴더의 해시는 `PUBLIC_MANIFEST.json`으로 검증합니다.
- 추가 분석을 수행하면 새 폴더에 기록하고, 기존 분할의 탐색 결과라는 표시와 NEU 분할 중복 한계를 유지하세요.

일부 `checks`의 절대경로는 당시 실행 이력입니다. 회사에서 사용해야 하는 경로가 아닙니다. 선택 시드 2026091501~2026091505를 결과에 따라 교체하지 않습니다.

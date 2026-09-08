# POST-RUN AUDIT: PASS WITH WARNINGS

**INTERIM DESCRIPTIVE ONLY**

**NOT FINAL — 1/10 acquisition trajectories**

완료된 official GC10 a00 / NEU a00 결과만 감사했다. 추가 학습, 통계적 유의성 검정, CI 계산, 최종 우위 주장, protocol 변경은 수행하지 않았다.

| 감사 항목 | 결과 |
|---|---|
| Training cells | 20/20 COMPLETE, 각각 attempt 1 |
| Acquisition jobs | 10/10 COMPLETE, 각각 attempt 1 |
| Round gates | 10/10 PASS |
| Historical 보호 | 1,348개 보호 hash 동일; 440개 셀의 best/last/manifest 재검증 |
| Historical analysis 및 reference 보호 | 2,574개 파일 size/mtime 동일; locked-test 결과 내용은 열지 않음 |
| Locked-test | 입력·동결 코드·authorization·실행 기록 기준 접근 0회. OS syscall 추적은 수행하지 않음 |
| A/B manifests | 같은 round의 A/B 모두 동일 |
| Policy dependency | 10개 acquisition 모두 이전 round A best.pt 사용. B policy edge 0 |
| 중복·재선택·누수 | 모두 0 |
| Freeze / authorization | 모든 결과에서 일치 |
| Integration 재사용 | 없음 |
| 분석 중 변경 | official registry byte hash 및 historical 보호 hash 유지 |

경고는 Ultralytics의 기존 `half` 인자 deprecation뿐이다. FP32 scorer 설정은 변경하지 않았다. Foreground 종료 코드 15는 GC10 R1 A/B seal 후 사용자가 요청한 독립 runner 전환 기록이며 학습 실패가 아니다.

- Official freeze SHA-256: `e40842787a93d2e06304f8da7eea5ea93f2125c7bd15982bbce6610903505e6a`
- Authorization SHA-256: `1cf0b12328878f222be2a3824e945f21e51620240f16723db51b15ef959e3936`

## GC10 interim curve

A/B 평균 development mAP50–95, 0–1 척도. Random/DINO도 같은 a00의 고정 development 평가값이다.

| Round | Labeled n | Entropy | Random | DINO |
|---|---:|---:|---:|---:|
| R0 | 161 | 0.212048 | 0.212048 | 0.212048 |
| R1 | 241 | 0.243611 | 0.237729 | 0.237107 |
| R2 | 321 | 0.255188 | 0.265329 | 0.254184 |
| R3 | 401 | 0.268498 | 0.284806 | 0.276616 |
| R4 | 481 | 0.283047 | 0.296162 | 0.296828 |
| R5 | 561 | 0.292268 | 0.299378 | 0.293206 |

## NEU interim curve

| Round | Labeled n | Entropy | Random | DINO |
|---|---:|---:|---:|---:|
| R0 | 126 | 0.290508 | 0.290508 | 0.290508 |
| R1 | 189 | 0.345017 | 0.332921 | 0.332304 |
| R2 | 252 | 0.366571 | 0.344344 | 0.362964 |
| R3 | 315 | 0.375138 | 0.369393 | 0.382368 |
| R4 | 378 | 0.371808 | 0.376616 | 0.374678 |
| R5 | 441 | 0.394767 | 0.372204 | 0.388752 |

![Interim development curves](C:/Users/user/Desktop/vlm/confirmatory_random_dino_v1/reports/entropy_postrun_a00_20260908/interim_development_curves.png)

## Provisional nAULC

R0–R5의 실제 labeled fraction을 x축으로 사다리꼴 적분한 뒤 `(x_R5 − x_R0)`로 나눴다. A/B learner별 nAULC의 평균과 일치함을 독립 계산으로 확인했다.

| Dataset | Entropy | Random (reference) | DINO (reference) |
|---|---:|---:|---:|
| GC10-DET | 0.260500 | 0.267948 | 0.263472 |
| NEU-DET | 0.360234 | 0.350926 | 0.358389 |

이 표와 곡선은 동일 acquisition index 하나의 참고 기술통계다. 방법 간 우위 또는 일반화에 대한 결론을 내리지 않는다.

## 평가값 출처

Entropy R1–R5는 완료된 official validation-best checkpoint를 원래와 동일한 고정 development 이미지·라벨의 독립 복사본에서 재검증했다. imgsz=640, batch=8, conf=0.001, iou=0.7, max_det=300, split=val을 사용했다. Acquisition의 conf=0.25와 별개의 기존 validation 설정이다. R0 및 Random/DINO는 동일 development membership을 확인한 archived post-training best.pt 평가값이다. GC10 229장, NEU 180장이다.

## 남은 official 작업과 시간

- 남은 official training: **180 cells**. 남은 acquisition: **90 jobs**. 새 실행은 시작하지 않았다.
- 현재 RTX 3070 기준 잔여 중앙 계획치: **29.48시간**.
- 운영 여유 범위: **26.5–35.4시간**. 통계적 CI가 아니다.
- 이번 원래 시작부터 완료까지: 3.341시간. Handoff 대기시간을 제외한 합산 active 실행 추정: 3.276시간.
- 관측 peak device VRAM: 2,373 MiB. GPU 전체 메모리 표본값이다.

각 dataset a00의 round별 실측 시간에 남은 trajectory 9개씩을 곱했다. GC10 R1 B의 process overhead와 수동 R1 gate 시간은 scheduler 인계로 별도 runtime 기록이 없어 다른 실행에서 관측한 median으로 보완했다. 학습·validation 자체 시간은 모두 실측이다. 이미지 구성, 시스템 부하, 온도에 따라 달라질 수 있다.

| Dataset / round | Train + validation (min) | Acquisition (min) | Total active estimate (min) |
|---|---:|---:|---:|
| GC10-DET R1 | 16.84 | 0.76 | 17.94 |
| GC10-DET R2 | 17.99 | 0.58 | 18.85 |
| GC10-DET R3 | 21.09 | 0.53 | 21.90 |
| GC10-DET R4 | 23.71 | 0.49 | 24.50 |
| GC10-DET R5 | 26.21 | 0.47 | 27.15 |
| NEU-DET R1 | 12.15 | 0.33 | 12.75 |
| NEU-DET R2 | 14.35 | 0.31 | 14.90 |
| NEU-DET R3 | 16.61 | 0.30 | 17.16 |
| NEU-DET R4 | 19.18 | 0.29 | 19.73 |
| NEU-DET R5 | 21.13 | 0.28 | 21.68 |

## 재현 자료

- `post_run_audit.json`: 각 셀·round gate와 전체 감사 근거.
- `interim_raw_development.json`: 72개 learner 단위 development 값과 checkpoint/source hash.
- `interim_descriptive_results.json`: 36개 평균 curve point, provisional nAULC 및 계산 정의.
- `runtime_and_remaining_work.json`: round별 시간과 잔여 시간 산출 근거.
- `after_analysis_integrity.json`: 분석 후 official registry 및 historical 보호 무결성.

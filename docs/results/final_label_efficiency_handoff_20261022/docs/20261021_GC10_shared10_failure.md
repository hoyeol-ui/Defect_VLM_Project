# GC10 공유 10% 모델 학습 후 점검 실패

확인일 2026년 10월 21일. 컨트롤러는 22:21:26 KST에 `GC10-DET_Random_pair1_R0` 셀 실패를 감지하여 정지했다. 과학적 설정과 학습/획득 소스는 변경하지 않았고 재학습·최종 테스트를 수행하지 않았다.

## 완료 상태와 시간

- 셀 전체 PASS: 신규 **1/101** (NEU Full pair1), 기존 GC10 Full 재사용 **1**.
- 별도로 GC10 공유 10% pair1도 **100 epoch 학습 자체는 완료**했으나 학습 후 Entropy adapter 점검에서 실패했으므로 완료 셀에 산입하지 않는다.
- GC10 순수 학습 시간 **837.3999046초**(약 13.96분), best 선택 epoch **94**. 정규 모델을 새로 학습해야 하는 미착수 셀은 **99개**다. 공유 10%의 후처리·획득 검증은 미완료다.
- 원장의 잔여 학습+획득 추정은 **41.7634 GPU시간**이다. 실패 셀의 이미 끝난 학습 재활용 여부가 아직 반영되지 않은 보수적 값이며 중단 대기시간과 최종 테스트는 제외한다. 현재 중단 상태이므로 기존 완료 예정 시각은 유효한 일정으로 사용하지 않는다. 복구 후 실측 ETA를 다시 계산한다.

## 보존된 가중치

경로: `research_extensions/final_label_efficiency_20261021_01/runs/GC10-DET_Random_pair1_R0/train/weights/`.

| 파일 | 바이트 | SHA256 |
|---|---:|---|
| best.pt | 80,260,441 | `54507df99a510f6e56dea40639bb12fae0e8bbbcb92e135614be88309da20bbd` |
| last.pt | 80,261,209 | `76fccb4d13ae091562b15aa56e0b66539426f88db5526b7d5e4103243e4ccb3b` |

`TRAINING_COMPLETE.json`, 100 epoch progress, best selection 기록이 남아 있다. NEU Full의 정상 완료와 가중치도 유지된다. `FAILURE.json`과 원래 traceback은 삭제하거나 성공으로 덮어쓰지 않았다.

## 재현된 원인

`cell.py`의 학습 후 점검에서 `entropy_adapter.py:xyxy_to_xywh`가 전체 8,400개 raw 후보에 `x2>=x1, y2>=y1`을 요구하여 실패했다. 모델의 출력은 유한한 수였고 크기는 [14,8400]이었다.

선택된 학습 이미지의 첫 2장만 같은 FP32 GPU 경로로 재현했다. 테스트 영상·라벨을 사용하지 않았다.

| 학습 이미지 ID 앞부분 | 역전 박스 | conf>0.25 후보 | 기준을 넘는 역전 박스 | 역전 박스의 최대 confidence |
|---|---:|---:|---:|---:|
| 01b71abb7ba2 | 5 | 3 | 0 | 0.000002082269 |
| 040bea04e1b9 | 0 | 1 | 0 | 해당 없음 |

첫 이미지의 최대 y 좌표 역전은 약 2.576 픽셀이었다. 즉 실제 Entropy 후보가 되지 않는 낮은 confidence의 배경 출력을 필터 전에 검사한 조건이 중단을 유발했다. 전체 미선택 풀의 모든 후보가 정상이라는 확인은 아직 수행하지 않았다. 별도 진단 시 GPU lock을 독점 사용한 뒤 해제했고 checkpoint 해시가 변하지 않았음을 확인했다.

## 필요한 구현 복구와 영향 검증

1. 변경 전 adapter/entropy 소스와 execution freeze를 별도 보존한다. 최종 과학적 프로토콜 해시는 유지한다.
2. 양수 박스 크기 검사 범위를 기존 confidence>0.25 후보와 일치시키는 방법을 검토한다. `entropy.py`에도 전체 raw 박스 크기 검사가 있으므로 adapter 한 줄만 없애는 것으로는 충분하지 않다. 유효 후보에서 좌표가 역전되면 계속 실패시켜야 한다.
3. 좌표를 임의 clamp하거나 confidence·NMS·Entropy·동점 규칙을 바꾸지 않는다. 필터 전 원시 출력과 원래 anchor ID를 보존해야 한다.
4. 정상 raw 입력에서 기존/수정 경로의 박스·클래스 점수·Entropy·선택 결과가 동일한지 확인한다. 실패 raw의 낮은 confidence 배경 후보가 점수나 선택에 기여하지 않는지 확인한다.
5. **완료된 GC10 학습을 반복하지 않고**, 보존 checkpoint에서 adapter 반복 추론·봉인 개발 평가 등 미완료 후처리만 검증하는 복구 경로를 준비한다. 통과 후에만 셀 완료를 인정하고 공유 10% 우선 순서와 Entropy 획득 검증을 계속한다.

이번 원고 작성 중에는 진단과 기록까지만 수행했다. 연구 소스 수정, execution freeze 교체, 실패 셀 성공 처리, 컨트롤러 재시작은 수행하지 않았다. 최종 테스트는 별도 단계로 남겨 두었다.

근거는 캠페인 원장의 실패 행·events.jsonl·셀 FAILURE/TRAINING_COMPLETE/progress/best_selection 및 `campaign/investigations/20261021_GC10_R0_postcheck.json`이다. 상세 raw 배열과 진단 스크립트는 학술지 작업 폴더 `work`에 별도 보존했다.

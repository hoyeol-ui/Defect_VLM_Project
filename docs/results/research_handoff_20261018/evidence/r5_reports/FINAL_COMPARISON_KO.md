# Class-aware DINO R5 개발 실험 비교

상태: 학습 100 epoch 및 동일 dev 최종 평가 완료.

| 항목 | Random R5 | DINO R5 | Class-aware DINO R5 |
|---|---:|---:|---:|
| 전체 mAP50-95 (%) | 26.476677 | 29.748684 | 27.309846 |
| 전체 mAP50 (%) | 53.224039 | 58.627198 | 56.104608 |
| punching_hole AP50-95 (%) | 47.948717 | 52.198695 | 51.296197 |
| welding_line AP50-95 (%) | 51.641529 | 52.890697 | 47.579486 |
| crescent_gap AP50-95 (%) | 58.528580 | 58.312773 | 56.107517 |
| water_spot AP50-95 (%) | 31.374854 | 35.707430 | 29.003781 |
| oil_spot AP50-95 (%) | 11.410947 | 19.003597 | 15.820775 |
| silk_spot AP50-95 (%) | 9.789162 | 11.494454 | 12.383183 |
| inclusion AP50-95 (%) | 2.709020 | 3.652389 | 3.509474 |
| rolled_pit AP50-95 (%) | 14.979040 | 14.867616 | 13.804073 |
| crease AP50-95 (%) | 4.430870 | 16.496469 | 17.816009 |
| waist_folding AP50-95 (%) | 31.954046 | 32.862722 | 25.777967 |
| inclusion 선택 이미지 수 | 46 | 34 | 33 |
| inclusion 선택 객체 수 | 87 | 67 | 64 |
| silk_spot 선택 이미지 수 | 173 | 95 | 96 |
| silk_spot 선택 객체 수 | 214 | 130 | 131 |
| waist_folding 선택 이미지 수 | 43 | 19 | 23 |
| waist_folding 선택 객체 수 | 43 | 19 | 23 |

공유 R0 160장, 최종 561장. 클래스 보정 60장, FF 341장. 기존 DINO와 공통 544장, 교체 17장.

100장 보정 예산 중 신뢰도 조건을 충족하지 못한 몫은 FF로 이전했다. 사후 GT 결과로 선택·신뢰도 기준을 수정하지 않았다.

클래스별 AP50/AP75/AP50-95, 클래스별 이미지·객체 수 및 DINO/Random 대비 차이는 CLASS_AP_COMPARISON.csv에 기록했다.

전체 mAP50-95: 기존 DINO 대비 -2.438838%p.

신규 best epoch: 83. 최종 dev와 학습 중 fitness는 별도 지표로 보관한다.

클래스별 변화 (DINO 대비):
- punching_hole: 객체 +0, AP50-95 -0.902498%p — AP 하락.
- welding_line: 객체 -3, AP50-95 -5.311211%p — AP 하락.
- crescent_gap: 객체 -1, AP50-95 -2.205256%p — AP 하락.
- water_spot: 객체 +2, AP50-95 -6.703650%p — 커버리지 증가에도 AP 개선 없음.
- oil_spot: 객체 -1, AP50-95 -3.182822%p — AP 하락.
- silk_spot: 객체 +1, AP50-95 +0.888729%p — AP 상승 또는 동률.
- inclusion: 객체 -3, AP50-95 -0.142915%p — AP 하락.
- rolled_pit: 객체 -1, AP50-95 -1.063543%p — AP 하락.
- crease: 객체 +0, AP50-95 +1.319540%p — AP 상승 또는 동률.
- waist_folding: 객체 +4, AP50-95 -7.084756%p — 커버리지 증가에도 AP 개선 없음.

단일 acquisition/learner 시드의 탐색적 dev 결과이다. R0 기반 원샷 선택이며 R1–R5 반복·nAULC 검증이 아니다. 개발셋 재사용 및 프록시 오분류 한계가 있어 일반적 능동학습 효과나 산업 적용 성능을 확정하지 않는다.

실험 설정: YOLO26s 공식 사전학습 cold start, SGD, 100 epoch, batch8, imgsz640, nbs64, seed2026091900; dev343. 독립 test 사용 없음.

새 체크포인트·로그: C:\Users\user\Desktop\vlm\research_extensions\class_aware_dino_ff_20261017_01\runs\GC10_R5_ClassAwareDINO_a2026091901_yolo26s_SGD_e100

새 학습은 이번 한 번만 실행하며, 자동 추가 실험·재학습·설정 조정은 수행하지 않는다.

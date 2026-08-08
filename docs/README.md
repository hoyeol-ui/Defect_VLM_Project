# Defect VLM Active Learning 연구 문서 인덱스

최종 업데이트: 2026-07-12

이 문서 폴더는 “좋아 보이는 결과만 남기기”가 아니라, 현재까지의 실험 실패·원인분리·다음 피벗 설계를 방어 가능하게 정리하기 위한 공간이다.

## 현재 한 줄 결론

초기 가설인 “VLM 설명 일관성만으로 Random보다 강한 GT-free Active Learning 전략을 만들 수 있다”는 현재 결과로는 지지되지 않는다. 다만 실험을 통해 `Consistency-only`의 한계, mixed GC10/NEU 프로토콜의 confound, DINO visual diversity의 보조 신호 가능성, 그리고 detector-aware uncertainty/balance가 필요한 이유는 꽤 명확해졌다.

## 최신 핵심 문서

- [MacBook handoff guide](./macbook_handoff_guide_20260712.md)
  - 맥북에서 GitHub repo를 받아 문서작업을 이어가는 방법, Notion 업데이트 순서, raw run/checkpoint 부재 시 주의사항을 정리.

- [Research context handoff](./research_context_handoff_20260712.md)
  - Codex 대화 맥락이 사라져도 V3~V10b 실험 흐름, 주요 결론, 최신 seed43~46 해석을 복원할 수 있는 핸드오프 문서.

- [Continuation playbook](./continuation_playbook_20260712.md)
  - 맥북 문서작업, GitHub 반영, Windows GPU 머신 재실험 명령을 한 곳에 모은 실전 플레이북.

- [V10c recall-guard plan](./v10c_recall_guard_plan_20260713.md)
  - V10b의 precision gain / recall loss 문제를 깨기 위한 2026-07-13 후보 전략과 실행 기준.

- [V8 NEU-only 5-seed 결과 로그](./v8_neu_only_5seed_result_log_20260712.md)
  - NEU-only 프로토콜에서 Random, DINO Visual, Consistency를 비교한 최신 정리.
  - 중요한 판정: DINO Visual은 Consistency보다 낫지만, Random 대비 독립 전략으로 성공했다고 보기 어렵다.

- [최종 detector-aware pivot protocol](./final_detector_aware_pivot_protocol_20260712.md)
  - 새 실험을 무한 반복하지 않기 위한 마지막 후보 설계.
  - YOLO localization/instance uncertainty + DINO diversity + balance를 결합하되, 사전 성공 기준을 만족하지 못하면 종료한다.

- [최신 분석 시각화 패키지](./analysis/latest_20260712/README.md)
  - V7 mixed와 V8 NEU-only 결과를 비교하는 PNG/CSV 묶음.
  - 발표·논문 초안·연구 로그에 바로 넣을 수 있는 그림 후보를 포함한다.

- [Curated experiment results](./results/README.md)
  - GitHub에 올리기 적합한 경량 결과 파일의 위치와 원본 run 폴더 매핑.

## 현재까지 확정된 사실

1. V7 Stage-A와 full-curve seed42 불일치는 “다른 데이터” 문제가 아니라 row order 의존 sampling 문제였다.
2. canonical sampling 도입 후 같은 population에서는 동일 seed의 initial set을 재현할 수 있다.
3. acquisition/dev/final split 간 image overlap은 발견되지 않았다.
4. 다만 V7 mixed 프로토콜은 GC10 pool이 `crease 48 / waist_folding 1`로 극단적으로 편향되어 있었고, dev/final 도메인 구성도 크게 달랐다.
5. V8 NEU-only로 정리하자 전체 성능은 크게 올라갔지만, Random baseline은 여전히 강했다.
6. Consistency-only는 중단하는 것이 맞고, DINO visual diversity는 독립 전략이 아니라 보조 구성요소로 격하하는 것이 타당하다.

## 문서화 톤

현재 결과를 “성공한 Active Learning 방법”처럼 과장하면 논문 방어가 어렵다. 더 안전한 스토리는 다음이다.

> 본 연구는 VLM explanation consistency를 GT-free industrial defect Active Learning 신호로 검증했으나, consistency-only는 Random baseline을 넘지 못했다. 이후 protocol confound를 분리하고 NEU-only에서 재평가한 결과, DINO visual diversity는 consistency보다 안정적으로 나았지만 Random 대비 이점은 제한적이었다. 따라서 최종 방법은 explanation consistency 단독이 아니라 detector-aware uncertainty, visual diversity, balance를 결합한 제한적 pivot으로 재정의한다.

## GitHub 업로드 원칙

- `runs/` 전체와 `.pt` checkpoint는 기본적으로 올리지 않는다.
- GitHub에는 `docs/`, 핵심 스크립트, selected CSV/PNG/MD만 올린다.
- 대용량 raw 결과가 필요하면 원본 run path를 문서에 남기고, 재현용 config와 summary만 추적한다.

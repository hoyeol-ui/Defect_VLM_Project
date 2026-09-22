# 논문 문장 경계: clean split v2 development

이 문서의 모든 숫자는 withheld test가 아닌 **development** 평가다. 이전 split이나 과거 n=10 자료의 추정치를 합치지 않는다.

| 분류 | 사용 가능한 표현 | 피해야 할 표현·이유 |
|---|---|---|
| GC10 평균 성능 | “GC10 development에서 DINO-F518의 평균 nAULC는 23.69%로 Random 21.85%보다 1.83%p 높았다. 5 seed의 paired 방향은 모두 양수였다.” | “통계적으로 유의하게 우수했다.” 양측 exact p=0.0625, 네 대비 Holm p=0.25. |
| NEU 평균 성능 | “NEU development에서 DINO−Random 평균 nAULC 차이는 +0.09%p였고 R5 AP 차이는 −0.10%p였다.” | “NEU에서도 안정적으로 우월하다.” 라운드별 방향이 바뀐다. |
| Entropy 비교 | “Entropy−Random nAULC 평균 차이는 GC10 −0.12%p, NEU −0.07%p였다.” | “Entropy가 무효/동등하다.” 비유의는 동등성 검정이 아니다. |
| 클래스 | “GC10 `welding_line`에서 DINO−Random R5 평균 class AP 차이는 +12.81%p였으나 `waist_folding`에서는 −2.20%p였다.” | “모든 결함 클래스가 개선됐다.” 반례가 있다. |
| 선택 구성 | “R5 GC10 `welding_line` 이미지 incidence가 DINO에서 더 많았고 AP도 높게 관찰됐다.” | “그 클래스 선택량 증가가 AP 개선을 일으켰다.” 인과 분리 실험이 없다. |
| F518와 NEU | “입력 크기·전처리 민감성은 가능한 설명 요인 중 하나다. 동일 조건 F224/F518 ablation은 없다.” | “NEU 성능 차이는 F518 확대 때문이다.” 직접 대조하지 않았다. |
| Full-data | “GC10 DINO R5 평균과 Full-data 1회 참조값 사이의 절대 gap은 0.65%p다.” | “Full-data와 동등하다” 또는 “full 성능의 X%”. Full-data n=1, 동등성 설계가 없다. |
| 예산 | “35%는 train acquisition pool의 **라벨 이미지 수** 비율이다.” | “라벨링 비용/시간 65% 절감.” 박스 수·인건비를 예산 단위로 측정하지 않았다. |
| 최종 평가 | “이 그림은 봉인된 development 평가다. 사전 정의 1차 test 분석은 남아 있다.” | “test에서 검증된 최종 우월성” 또는 “논문의 confirmatory primary 성공.” test는 withheld다. |

숫자 출처: `02_tables/02_FINAL_NUMERIC_RESULTS.csv`, `03_FINAL_NAULC_RESULTS.csv`, `05_CLASSWISE_RESULTS.csv`; 대비 검정: `05_statistics/PAIRED_NAULC_SUMMARY.csv`. 각 seed의 run ID·metric source는 `RAW_SEED_METRICS.csv`에 있다.

# Statistical Analysis Plan v1

- 상태: **FROZEN BEFORE CONFIRMATORY OUTCOMES**
- 분석 성격: estimation-first
- primary unit: acquisition seed
- GC10/NEU naive pooling: 금지

## 1. Primary endpoint construction

Budget `i=0...5`의 labeled fraction을 `f_i`, mAP50-95를 `m_i`라 할 때:

```text
AULC = [Σ_i ((m_i + m_{i+1}) / 2) (f_{i+1} - f_i)] / (f_5 - f_0)
```

이는 evaluated span의 budget-averaged mAP50-95다. Fully supervised upper bound로 normalize하지 않는다. Six point 중 하나라도 없으면 interpolation/partial AULC를 만들지 않고 cell을 incomplete로 처리한다.

Dataset `d`, acquisition `a`, strategy `s`, training seed `t`의 AULC를 `A_{dast}`라 한다.

```text
M_{das} = (A_{dasA} + A_{dasB}) / 2
Delta_{da} = M_{da,DINO} - M_{da,Random}
```

Dataset별 10개 `Delta`의 산술평균이 primary point estimate다. Training seed 2개를 독립 n으로 세지 않는다.

## 2. Primary uncertainty and test

Dataset별로 다음을 보고한다.

- 10 paired differences 전부
- mean, median, SD, min/max, wins/losses/ties
- acquisition-cluster percentile bootstrap 95% CI, 100,000 resamples
- RNG: NumPy `Generator(PCG64)`
- bootstrap seed: GC10 `421397045`, NEU `1672358496`
- two-sided exact sign-flip p-value: 가능한 `2^10=1024` sign assignments 전부 열거

두 dataset primary p-value를 inferential language에 사용할 때 Holm adjustment를 적용하고 raw/adjusted 값을 모두 보존한다. 그러나 minimum effect threshold는 두지 않으며 p-value 하나로 서사를 결정하지 않는다.

CI가 0을 포함하면 “equivalent”가 아니라 **controlled protocol에서 명확한 downstream advantage가 확립되지 않았다**고 쓴다. Equivalence/noninferiority margin은 외부 근거가 없어 설정하지 않는다.

## 3. Secondary endpoints

- final-budget mAP50-95
- budget별/final recall
- class-wise AP50-95
- GC10 rare macro AP50-95
- NEU macro/worst-class AP50-95
- post-seal discovery와 composition

Final metric은 AULC와 같은 순서로 두 training seeds를 평균하고 acquisition 내 paired contrast를 만든다. Secondary는 estimate/CI를 보고하지만 primary failure를 rescue하는 formal multiplicity claim에 사용하지 않는다.

## 4. Subgroups

### GC10

Curated pre-split image frequency 하위 3개 class를 사전 고정했다: class 8 rolled_pit, 9 crease, 10 waist_folding. 이는 historical 8/9/10과 일치한다. Rare macro는 세 AP50-95의 unweighted mean이다. 세 값 중 하나가 missing이면 zero 대체 없이 cell incomplete다.

### NEU

새 rare group을 만들지 않는다. 여섯 class AP의 unweighted macro와 최소값인 worst-class AP를 계산한다. Worst class의 identity도 descriptive하게 보고하되 결과 후 subgroup 정의를 바꾸지 않는다.

## 5. Discovery/composition

반드시 query/cumulative manifest SHA를 먼저 기록한 뒤 GT를 join한다.

- GC10 target discovery: class 8/9/10 중 하나 이상을 포함한 cumulative image 수
- NEU: 여섯 class별 cumulative image-count vector; 인위적 scalar rare target 없음
- composition: class image-incidence/instance counts, unique class coverage, class image-incidence normalized Shannon entropy

이 audit 결과는 이미 생성된 acquisition trajectory를 변경할 수 없다.

## 6. Missing/failure

- outcome-based seed replacement, imputation, failed-cell 삭제 금지
- infrastructure retry는 같은 cell/config/seed/hash로만 수행하고 attempt를 모두 기록
- primary에 필요한 cell이 닫히지 않으면 해당 dataset primary는 `INCOMPLETE_NOT_ESTIMATED`
- 한 dataset의 incomplete를 다른 dataset과 pooling해 보완하지 않음
- protocol integrity/GT leakage/hash mismatch는 analysis보다 먼저 STOP

## 7. Reporting order

1. protocol integrity와 planned/actual cell registry
2. dataset별 raw acquisition/training/budget rows
3. dataset별 primary curve와 paired Delta 10개
4. secondary/subgroup
5. cross-dataset direction 비교
6. historical evidence와의 bounded discussion

향후 모든 raw row는 기존 master와 분리된 `RAW_CONFIRMATORY_RESULTS_MASTER_V1.md`에 보존한다.


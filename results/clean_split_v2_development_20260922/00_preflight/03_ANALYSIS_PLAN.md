# clean split v2 개발셋 분석 실행안

사전 점검 **PASS** 후 확정한 계산·그림 계획. 상위 근거는 `reference/FINAL_MAIN_EXPERIMENT_PROTOCOL_v2_0_KO.md` 9절이다. 원본 압축, 모델, 평가 기록은 읽기 전용으로 둔다. 최종 test는 봉인 상태이므로 이번 산출물의 머리말·그림 caption·통계표를 모두 **development, exploratory**로 표시한다. v2 계획이 정한 미래의 **test nAULC 1차 검정**을 이 분석으로 대신하지 않는다.

1. `config/cells.json`의 162개 run을 기준으로 `sealed_dev_metrics.json`을 하나씩 읽고 SHA를 `artifacts.json`과 대조한다. shared R0 10개는 각 seed의 세 방법 곡선에 **참조**하되 독립 모델 30개로 세지 않는다. Full-data 2개는 별도 참고값이다. 단위를 raw 0–1과 표시용 0–100%로 구분하여 CSV 컬럼에 기록한다.
2. seed별 R0–R5 곡선에 실제 누적 정수 이미지 수 `n_r`를 x로 쓴다. `nAULC_s = [Σ_(r=1..5) (n_r−n_(r−1))(AP_(s,r)+AP_(s,r−1))/2] / (n_5−n_0)`이다. AP가 0–1이면 nAULC도 0–1이다. 그림·본문의 nAULC(%)는 100배 한다. 여섯 점 중 하나라도 없으면 해당 seed 곡선은 실패로 중단하며 보간하지 않는다.
3. 각 dataset×method×round의 5 seed에 대해 arithmetic mean, sample SD (`ddof=1`) 및 95% t interval(`t_(0.975,4) × SD/√5`)을 계산한다. 이 CI는 seed 간 관측 변동의 기술적 요약이며, 하나의 고정 split/learner에 조건부다. 그림 band는 CI가 아닌 **mean ± SD**다. y=0–100 범위를 벗어나는 band 부분만 시각적으로 clip하고 수치표의 원래 SD는 보존한다.
4. seed를 맞춘 `DINO−Random`, `DINO−Entropy`, `Entropy−Random`의 nAULC 차이를 계산한다. GC10 및 NEU의 앞 두 대비(총 네 개)가 v2 SAP의 사전 정의 대비다. Entropy−Random은 보조 기술 대비다. 원자료, 평균, sample SD, 중앙값, 최소/최대, 승/패/동률을 저장한다. Round별 및 R5 paired 차이는 설명용이다.
5. 네 주요 대비의 exact two-sided sign-flip은 5개 paired 차이에 대한 2^5=32개 부호 조합 전체에서 `|mean| ≥ |observed mean|` 비율로 산출한다. Holm은 네 raw p를 한 family로 조정한다. 원 SAP의 100,000회 paired percentile bootstrap(데이터셋별 RNG seed GC10 `2026091912`, NEU `2026091913`)으로 평균 paired Δ의 95% 구간을 추가한다. bootstrap CI는 n=5에서 탐색적 불확실성 요약이다. 현재 계산은 development 결과에 적용한 **탐색적 복제**이며 미래 test primary inference가 아니다. 5개에서 양측 exact sign-flip 최소 raw p는 0.0625, 네 대비 Holm 최소 조정 p는 0.25다. 보조 대비에 사후 유의성 주장을 붙이지 않는다.
6. R5 클래스별 AP50:95를 전 10/6클래스에 대해 각 방법의 5 seed 평균·SD와 seed별 paired 차이로 작성한다. round별 최저 클래스 AP와 worst-class curve의 nAULC도 보조 파일에 남긴다. 선택 구성은 `selection_manifest.json`의 신규 `selected_ids`와 누적 `cumulative_ids`를 구별한다. **최종 annotation patch overlay의 train 라벨**만을 사용해 클래스별 이미지 incidence와 bbox incidence를 계산하며, AP와의 관계는 서술적 관찰로만 남긴다.
7. 그림은 GC10/NEU 각각의 예산–성능 곡선, dataset별 nAULC 절대값, dataset별 R5 클래스 AP, 보조 paired difference를 만든다. 국제 논문 원문에 맞춘 영문 축·범례, 한국어 caption, 일관된 marker/선종류/색을 사용한다. 각 그림을 PNG 300 dpi, vector PDF, SVG로 저장하고 생성 코드를 보존한다. Full-data는 1회 수평선, 반복 SD/CI 없음. statistical star는 사용하지 않는다.
8. 최종 보고서는 GC10, NEU, 세 쌍의 방법 대비, 예산 효과, Full-data와의 **절대 %p gap**, 클래스 편차, 데이터셋 차이, 허용·불허 주장 순서로 작성한다. F518은 NEU에서 F224와 직접 비교되지 않았으므로 원인으로 단정하지 않는다. 이전 split/과거 n=10 결과를 현재 수치에 합치지 않는다.

원자료 식별자는 모든 seed별 행에 `source_relpath`, `run_id`, `seed`, `round`, `metric_key`, `attempt`로 남긴다. 재현 코드는 `07_reproducibility`에 고정한다. 수치가 기존 요약과 다르면 원인을 추적해 별도 차이 기록에 남기고, 임의로 seed·run·round를 제외하지 않는다.

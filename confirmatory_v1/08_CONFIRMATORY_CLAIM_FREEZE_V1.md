# Confirmatory Claim Freeze v1

- 상태: **FROZEN BEFORE OUTCOMES**
- 변경 규칙: result를 본 뒤 이 표를 바꾸지 않는다.
- 연구 기여: selector novelty가 아니라 selection/discovery/composition/reproducibility/downstream translation의 분리

## Case table

| Case | Observation | 허용 표현 | 금지 표현 |
|---|---|---|---|
| A | DINO AULC > Random | 평가된 dataset/protocol에서 positive paired difference가 관찰됐다. | DINO는 보편적으로 우수하다. |
| B | CI가 0을 포함하거나 direction이 불명확 | controlled protocol에서 명확한 downstream advantage가 확립되지 않았다. | 두 방법은 동등하다; 차이가 전혀 없다. |
| C | DINO AULC < Random | 평가된 조건에서 DINO visual diversity가 Random보다 불리한 paired difference를 보였다. | DINO는 모든 산업 데이터에서 나쁘다. |
| D | GC10/NEU 방향이 다름 | selection-to-learning translation은 dataset-dependent했다. | 한 dataset만 골라 universal 방향을 주장한다. |
| E | discovery positive, AULC null/negative | selection-stage discovery advantage가 downstream learning으로 자동 번역되지 않았다. | discovery metric은 무가치하거나 downstream을 대체한다. |

## Additional boundaries

- Overall positive와 GC10 rare negative가 동시에 나오면 둘 다 보고한다.
- NEU worst-class가 나빠지면 macro/overall과 함께 보고한다.
- Primary null을 secondary metric 하나로 “성공”으로 바꾸지 않는다.
- Positive estimate와 predefined statistical uncertainty를 분리한다.
- Historical result와 새 confirmatory result를 동일 n으로 pool하지 않는다.
- Training stability와 acquisition reproducibility를 같은 것으로 쓰지 않는다.
- Public-data final을 untouched test로 부르지 않는다.

## Scope prohibitions

결과 방향과 무관하게 다음은 주장하지 않는다.

- DINO가 최고의 AL selector다.
- uncertainty/VLM family는 불필요하다.
- annotation cost/time이 절감됐다.
- 산업 현장 또는 모든 defect dataset에 일반화된다.
- 440회 실행 자체가 novelty를 만든다.
- 지도교수가 protocol 또는 thesis framing을 승인했다.

## Null language

본 설계에는 equivalence margin이 없다. 따라서 `approximately null`은 statistical equivalence가 아니라 **평가된 resolution에서 명확한 downstream advantage 미확립**을 뜻한다.


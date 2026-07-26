# 발표자료 삽입 가이드

## 왜 기존 최종본에서 빠져 보였는가

기존 자료에는 G1–G6 설명과 두 장의 알고리즘 규칙이 있었지만, 전체를 한눈에
연결하는 16:9 지도는 약했다. 과거 도식은 세로형으로 지나치게 길고
DeepPCB STOP 사례에 고정되어 있어 현재 KSDD2 `GO → G5` 흐름을 함께
표현하기 어려웠다.

이번 버전은 역할을 두 장으로 분리했다.

- `01_validity_gated_full_architecture.html`: 연구 전체 구성과 데이터·Gate·사례의 관계
- `02_authorization_algorithm_flowchart.html`: PASS/FAIL에 따른 실행 순서와 STOP/GO 규칙

## 권장 삽입 위치

### 1. 전체 아키텍처

파일: `01_validity_gated_full_architecture.html`

권장 위치: 기존 **Slide 6 Experimental Assets 다음**, 현재 Framework
Overview보다 앞.

새 순서:

1. 왜 질문이 이동했는가
2. 연구 chronology와 datasets
3. **전체 아키텍처**
4. Six Validity Gates
5. Authorization Algorithm
6. 개별 결과

발표 설명:

> 지금까지는 왜 연구 질문이 바뀌었는지를 설명했습니다. 이 장부터는 그
> 반증을 어떤 실험 구조로 정리했는지 보여드리겠습니다. 왼쪽의 데이터와
> 후보 신호가 바로 learner로 가지 않고, G0–G4 selection-only 검증을
> 통과한 경우에만 오른쪽의 G5 learner screen 한 번이 허가됩니다. 실제로
> DeepPCB에서는 STOP이, KSDD2에서는 G5 진행 권한이 나왔습니다.

핵심 강조:

- PASS는 성공 예측이 아님
- DeepPCB STOP과 KSDD2 GO가 같은 workflow에서 발생
- Official test는 별도 승인 전 계속 잠김

### 2. 알고리즘 순서도

파일: `02_authorization_algorithm_flowchart.html`

권장 위치: 기존 **Slide 8 Six Validity Gates 다음**.

기존 Slide 9–10은 이 한 장 뒤에서 세부 규칙을 설명하는 보충 장표로
유지하거나 appendix로 이동한다.

발표 설명:

> 알고리즘의 입력은 candidate score이고 출력은 좋은 selector라는 판정이
> 아닙니다. 각 Gate에서 FAIL이면 확장 경로를 닫고, G1–G4가 모두 통과한
> 경우에만 정확히 한 번의 bounded learner screen을 허가합니다. 현재
> KSDD2 G5도 이 규칙에 따라 60개의 training job을 수행하지만 실제 통계
> 단위는 acquisition seed 10개입니다. G5에서도 learning과 safety를 함께
> 통과하지 못하면 final test로 넘어가지 않습니다.

핵심 강조:

- G2 Discovery와 G3 Safety는 동시 PASS 필요
- 60 jobs를 독립 n=60으로 사용하지 않음
- Dice만 상승해도 Pixel AP 또는 FPR이 FAIL이면 확장 중단
- G6 비용 주장은 실제 시간·workload를 측정한 경우에만 가능

## HTML을 슬라이드 이미지로 사용하는 방법

1. Chrome 또는 Edge에서 HTML 파일을 연다.
2. 브라우저를 전체 화면으로 전환한다.
3. 1920×1080 또는 16:9 화면으로 캡처한다.
4. 슬라이드에 `배경 채우기` 또는 전체 화면 이미지로 삽입한다.

HTML은 창 크기에 맞춰 자동 축소되며 내부 캔버스는 1920×1080으로
고정되어 있다. 브라우저 확대/축소는 100%를 권장한다.

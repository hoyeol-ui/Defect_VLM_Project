# 2026-09-07 지도교수 피드백 대응표

> 문서 목적: 완료된 Fact Audit을 반복하지 않고, 2026년 9월 7일 지도교수 미팅의 질문을 `확인 결과 → 판단 → 대응 → 다음 행동`으로 1:1 연결한다. 이번 문서는 의사결정 문서이며 추가 학습, 코드 변경, 새 selector 구현 결과가 아니다. 전사 인용은 자동 전사의 명백한 오인식을 문맥에 맞게 최소한으로 바로잡았다.

### 현재 연구 유지 여부

**보강 후 유지 가능.** 현재 440-run 통제 실험과 작은 평균 개선, 일부 클래스 악화라는 결과는 폐기할 이유가 없다. 다만 제목·서론을 산업 문제와 Task 중심으로 재정렬하고, 100% 학습 기준, DINOv2 표현 적합성 정량 분석, 기하적 다양성과 다른 AL 기준선 1개를 보강해야 한다.

### 제목에 반드시 들어가야 하는 4요소

1. **환경:** 제한된 라벨 이미지 조건
2. **산업 domain:** 산업 결함
3. **Task:** 객체 탐지
4. **목적과 방법:** 탐지 학습에 유용한 표본을 고르기 위한 DINOv2 표현 기반 능동학습

### 가장 추천하는 제목

- 국문: **제한된 라벨 환경의 산업 결함 객체 탐지를 위한 DINOv2 표현 기반 능동학습 표본 선택의 학습 효용 검증**
- 영문: **Validating the Learning Utility of DINOv2 Representation-Based Active Learning Sample Selection for Industrial Defect Object Detection under Limited Label Budgets**

### 교수님 지적 중 설명만 바로잡으면 되는 것

1. 표본은 사람이 고른 것이 아니라 거리 규칙으로 자동 선택됐고, 사람의 육안 확인은 선택 후 점검이었다.
2. 데이터 구조는 `train / validation / test`이며, `best.pt`는 라벨 비율이 아니라 각 단계에서 validation으로 고른 checkpoint다.

### 실제 추가 증거가 필요한 것

1. GC10-DET·NEU-DET에서 DINOv2 표현이 의미 있는 이웃 구조를 만드는지에 대한 정량 분석
2. 동일 split·동일 YOLO lifecycle의 100% 라벨 학습 기준
3. 현재 기하적 다양성 선택과 신호가 다른 object-detection AL 기준선 1개

### 이번 논문에서는 하지 않을 것

1. Domain Generalization로 연구 Task 변경
2. Open-set Recognition으로 연구 Task 변경
3. 새 멀티모달 detector 개발 또는 과거 VLM 실험 전수 복원

---

# MASTER TABLE

| Task | 교수님이 제기한 질문/이슈 | 현재 확인 결과 | 검토 및 판단 | 대응 | 다음 행동 | 우선순위 |
|---|---|---|---|---|---|:---:|
| 01 | “이 연구의 목적이 무엇인가?” — 비교·분석이 목적으로 보임 | 실제 환경은 제한 라벨, Task는 산업 결함 객체 탐지, 목적은 제한된 이미지로 detector 학습 성능을 확보하는 것 | 연구는 유지 가능하나 목적 문장이 잘못 전면화됨 | 목적·환경·Task·방법을 한 문장으로 고정 | 제목·초록·서론·발표 첫 문단 동기화 | P0 |
| 02 | 연구 서사가 “왜 Random을 못 이겼는가”에서 시작함 | 현 서론에 산업 문제도 있으나 곧 연구 변천과 실패 서사가 중심이 됨 | 실험 문제가 아니라 서사 우선순위 문제 | 산업 문제 → 제한 라벨 → 객체 탐지 → AL → 방법 → 질문 순으로 재구성 | 실패 서사는 연구 경과·한계로 이동 | P0 |
| 03 | 제목만 보고 무엇을 하는 연구인지 모름 | 최신 주제목에는 제한 라벨 조건, 정확한 객체 탐지 Task, 공학 목적이 보이지 않음 | 가장 시급한 제목 문제 | 네 요소가 드러나는 제목 후보 5개 중 1순위 제안 | 9/9 전 지도교수 확인 및 변경 가능 여부 확인 | P0 |
| 04 | “사전학습 DINO는 무엇으로 학습됐고 현 데이터에 맞는가?” | DINOv2-small ViT-S/14, LVD-142M 무라벨 사전학습, 384차원 pooled feature, 가중치 고정 사용 | 모델 사실은 확인됐으나 domain 적합성 직접 증거는 부족 | 사전학습 사실과 현재 사용을 분리 설명 | 기존 feature로 이웃·거리 정량 분석 | P1 |
| 05 | “다르다/비슷하다는 판단이 정성평가 아닌가?” | 현재 통제 실험과 MPDD·VisA 선택은 자동이고 육안 확인은 사후 점검 | 교수님은 학생 답변 때문에 수동 선택으로 이해; 설명 오류 | 자동 규칙을 정정하고 정량 표현 audit은 별도 제시 | 다음 미팅에서 즉시 정정; Task 04 분석 추가 | P0/P1 |
| 06 | “DINO가 selector인가? sampling 방법은?” | DINOv2는 feature extractor, 선택은 cosine greedy farthest-first | 둘을 하나의 모델처럼 설명한 것이 문제; Core-set/k-center와 가까움 | representation과 acquisition을 분리해 도식·문장 수정 | 방법론 절의 용어와 수식 정리 | P0 |
| 07 | “100% 데이터를 쓰면 YOLO 성능은?” | 동일 split/lifecycle의 full-data 결과 없음 | 실제 추가 증거 필요 | 최소 4-run reference와 권장 40-run 설계 제안 | 지도교수와 반복 범위 합의 후 실행 | P1 |
| 08 | “Random과 DINO 외 기존 sampling도 비교해야 함” | 현 방법은 기하적 다양성 계열; 별도 Core-set은 중복 가능. 현 protocol의 distinct 결과는 없음 | 교수의 명시적 보강 요구 | detector predictive uncertainty를 1순위 후보로 제안 | 정확한 집계 규칙·200 cells 범위 사전 고정 | P1 |
| 09 | “그래프는 몇 % 데이터를 쓴 결과인가?” | GC10/NEU 모두 약 10%에서 시작해 5%p씩 약 35%까지 증가 | 결과가 없던 것이 아니라 표기·설명 부족 | 모든 표·그림에 이미지 수와 train-pool % 병기 | 캡션 및 발표 문구 수정 | P0 |
| 10 | “표본선택용/개발용/최종평가용 대신 train/validation/test” | 실제 구조가 train pool / validation / locked test임 | 교수 지적이 맞고 즉시 수정 가능 | 표준 용어 뒤에 AL 역할만 한 문장 설명 | 본문·슬라이드 용어 일괄 교정 | P0 |
| 11 | “‘라벨 예산’, ‘누적’이 어렵다” | budget은 실제로 해당 단계까지 학습에 사용한 이미지 수 | 기술적으로 맞지만 전달력이 낮음 | 먼저 이미지 수와 비율로 말하고 전문 용어는 첫 등장에만 병기 | 용어 치환표 적용 | P0 |
| 12 | “텍스트가 왜 빠졌나? 이전 VLM과 연결 가능한가?” | canonical branch에 image-level raw score가 없고 기존 VLM validity도 약함 | 즉시 보강책이 아니며 복원 비용·연구 분산 위험이 큼 | 현재 논문 범위에서 제외하고 실패 근거는 연구 경과로만 사용 | raw artifact 위치 1회 확인 후 없으면 종료 | P2 |
| 13 | “자동차 부싱 VLM-assisted AL과 Task가 비슷하지 않은가?” | 산업 결함·객체 탐지·제한 annotation·AL은 같고, 모델·신호·protocol은 다름 | “Task가 비슷하다”는 지적은 맞지만 방법 동일은 아님 | 가장 가까운 관련연구로 넣고 차별성 표 작성 | 원문 확보 시 세부 detector/baseline만 보정 | P1 |
| 14 | “Novelty가 부족하다” | 현재 기여는 새 알고리즘이 아니라 통제된 downstream 효과와 class-level 경계 | 기여 표현 보강은 가능; Task 전환은 일정상 과도 | 평균 개선과 취약 클래스 악화의 공존, paired-seed 검증을 기여로 한정 | DG/open-set/새 멀티모달 모델은 후속 연구로 이동 | P0/P2 |

---

# 상세 1:1 대응

## Task 01. “이 연구의 목적이 무엇인가?” — 비교·분석이 연구 목적이 되어 있는 문제

### 교수님이 문제 삼은 내용

> “이 연구의 목적이 뭔가요?”

> “분석했다, 비교했다는 것은 토픽이 될 수 없어요. 다른 토픽을 위한 과정일 거든요.”

> “데이터의 패턴을 분석하면 끝이 아니고요. 엔지니어 쪽으로 와야 됩니다.”

### 교수님 입장에서의 핵심 이슈

현재 설명은 연구자가 무엇을 비교했는지는 말하지만, 산업 현장의 어떤 제약 아래 어떤 Task를 더 잘 수행하려는 연구인지는 늦게 나온다. 교수님이 요구한 “기승전결의 결”은 분석 행위가 아니라 **제한된 라벨 이미지로 결함 detector를 학습해야 하는 문제에 어떤 실증적 답을 주는가**다.

### 확인한 내용

- **환경:** 공개 산업 결함 데이터셋에서 학습에 사용할 수 있는 라벨 이미지가 제한된 조건
- **Task:** GC10-DET·NEU-DET의 다중 클래스 산업 결함 객체 탐지
- **목적:** 제한된 라벨 이미지로 학습한 detector의 downstream 성능을 확보·개선하는 데 DINOv2 표현 기반 표본 선택이 실제로 도움이 되는지 검증
- **방법:** 사전학습 DINOv2-small의 이미지 표현과 cosine greedy farthest-first acquisition을 사용하고, Random과 동일 split·동일 budget·반복 seed로 비교

현재 결과는 실제 annotation 시간·금액을 측정하지 않았다. 따라서 “비용을 절감했다”는 성과 주장은 할 수 없다. 말할 수 있는 범위는 **동일한 라벨 이미지 수에서 detector 학습 성능이 얼마나 달라지는가**다.

### 검토 및 판단

교수님 지적은 타당하다. 연구 목적 자체가 없는 것이 아니라, 문서와 발표에서 “표본 선택 신호와 학습 성능의 관계 분석”이 목적처럼 앞에 나온 설명상의 구조 문제다. 현재 실험을 유지하면서 바로잡을 수 있다.

### 대응

초록·서론·발표의 첫 목적 문장을 다음처럼 통일한다.

> 본 연구는 제한된 라벨 이미지 조건의 산업 결함 객체 탐지에서, 사전학습 DINOv2 표현 기반 능동학습 표본 선택이 동일한 이미지 수로 학습한 detector의 성능에 제공하는 효용과 그 한계를 통제 실험으로 검증한다.

비용은 연구 동기일 수 있지만 측정된 결과가 아니므로, “annotation 부담이 존재한다”와 “비용 절감을 달성했다”를 분리한다.

### 다음 행동

1. 제목·국영문 초록·서론 1.1·연구 목적·발표 첫 슬라이드에 위 목적 문장을 동일 의미로 반영한다.
2. 기여는 평균 nAULC 개선과 class-level 악화의 공존, 반복 seed 통제, 적용 범위의 한계로 쓴다.
3. annotation 비용 절감, 보편적 우수성, 모든 클래스 개선 표현을 제거한다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “연구 목적을 비교·분석 자체로 설명한 게 문제였습니다. 실제 목적은 라벨 이미지를 제한적으로만 쓸 수 있는 산업 결함 객체 탐지에서, 어떤 이미지를 먼저 고르느냐가 같은 이미지 수로 학습한 detector 성능에 도움이 되는지 확인하는 것입니다. DINOv2 표현을 이용한 자동 선택을 Random과 통제 비교했고, 실제 비용을 측정하지 않았기 때문에 비용 절감이 아니라 제한된 라벨 조건의 학습 효용으로 범위를 잡겠습니다.”

---

## Task 02. “현재 연구 서사가 왜 Random을 못 이겼는가에서 시작하면 안 된다.”

### 교수님이 문제 삼은 내용

> “이 흐름으로 논문을 쓰시면 안 됩니다.”

> “Random보다 낫지 않더라는 것은 과거 학생의 경험이고, 이 분야 연구의 공통된 문제는 아닙니다.”

> “차라리 라벨이 적은 데이터 상황에서 성능 향상을 위한 방법으로 주제를 가져가는 게 낫습니다.”

### 교수님 입장에서의 핵심 이슈

개인 연구의 시행착오가 산업 AI의 일반 문제처럼 제시되면 연구 필요성이 약해진다. “왜 내 과거 방법이 실패했나”가 아니라 “제한 라벨의 산업 결함 탐지에서 어떤 sampling이 실제 학습에 유용한가”가 공적인 연구 질문이어야 한다.

### 확인한 내용

최신 논문 초안에는 산업 결함 데이터의 희소성·불균형, annotation 부담, 제한된 라벨 수에서의 AL 필요성이 이미 들어 있다. 그러나 그 직후 연구 변천, VLM 신호의 실패, Random의 강함, selection-learning translation 문제가 크게 강조되어 산업 문제보다 연구 경과가 기억에 남는 구성이다.

### 검토 및 판단

내용 부재가 아니라 **배치와 강조의 문제**다. 과거 실패는 현재 방법을 선택한 연구 경과와 동기 보조자료로는 유효하지만 서론의 출발점이나 산업계 공통 한계로 일반화하면 안 된다.

### 대응

서론 논리를 다음 순서로 고정한다.

`산업 결함 탐지의 annotation 부담 → 제한된 라벨 이미지 조건 → 객체 탐지 학습의 필요 → Active Learning → 표본 선택의 downstream 효용 문제 → DINOv2 표현 기반 선택과 연구 질문`

과거 VLM 실험과 Random 미달 경험은 “선행 실험 및 연구 경과” 또는 “문제 구체화 과정”으로 내리고, 현재 연구의 보편적 출발점처럼 쓰지 않는다.

### 다음 행동

1. 초록 첫 3문장과 서론 첫 2~3단락을 위 순서로 재배치한다.
2. “왜 Random을 못 이겼는가”는 연구 목적에서 제외하고 방법 선택 배경 한 단락으로 축소한다.
3. 과거 실험은 현재 통제 실험의 결과와 섞지 않고 historical evidence로 표시한다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “말씀하신 대로 과거에 제가 Random을 못 이겼던 경험을 연구의 출발점처럼 둔 것이 잘못이었습니다. 서론은 산업 결함 데이터의 annotation 부담과 제한된 라벨 조건에서 시작하고, 객체 탐지 성능을 위해 Active Learning 표본 선택을 검증하는 흐름으로 바꾸겠습니다. VLM 실패와 Random이 강했던 경험은 연구 경과로만 뒤에 두겠습니다.”

---

## Task 03. “제목만 보고 무엇을 하는 연구인지 모르겠다.”

### 교수님이 문제 삼은 내용

> “제조 결함에서 어떤 Task인지가 먼저 들어가야 됩니다.”

> “Task라든지 구체적인 방법론이라든지, 어떤 걸 썼는지가 드러나야 됩니다.”

> “제목만 보고도 뭐 하는지 알 수가 있어야 합니다.”

### 교수님 입장에서의 핵심 이슈

제목은 실험 로그의 요약이 아니라, **어떤 조건에서 어떤 산업 문제를 무엇으로 풀고 무엇을 확인하는 연구인지**를 압축해야 한다. “연결”, “분석”, “비교”가 먼저 보이면 Task와 공학 목적이 숨는다.

### 확인한 내용

최신 초안의 주제목은 `산업 결함 Active Learning에서 표본 선택 신호와 학습 성능의 연결`이다. 부제는 이질적 역사 증거와 Random 대비 통제 검증을 전면에 둔다. 이를 제목 요소로 해체하면 다음과 같다.

| 요소 | 현재 제목에서의 상태 | 판단 |
|---|---|---|
| 환경/조건 | 없음 | **제한된 라벨 이미지 조건**을 명시해야 함 |
| 산업 domain | “산업 결함” 있음 | 유지 |
| Task | “학습 성능”만 있고 객체 탐지 불명확 | **산업 결함 객체 탐지** 명시 필요 |
| 연구 목적 | “연결”로 표현 | detector 학습 효용 또는 성능 확보 목적 필요 |
| 핵심 방법 | AL은 있으나 DINOv2 역할이 불명확 | DINOv2 표현 기반 표본 선택 명시 |
| 기여/검증 범위 | 통제 검증은 부제에 있으나 역사 서사가 큼 | 학습 효용의 통제 검증으로 축소 |

### 검토 및 판단

교수님 지적이 정확하며 9월 9일 전에 해결해야 할 P0다. 현재 제목은 연구 과정은 보이지만, 처음 보는 사람이 object detection 연구임을 알기 어렵다. 다만 현재 실험을 바꿀 필요는 없고 제목의 정보 구조를 바꾸면 된다.

### 대응

아래 “지도교수 대표 논문의 제목 구조 분석”과 “현재 연구 제목 후보”를 근거로, 네 요소가 모두 드러나는 1순위 제목을 제출안으로 사용한다. 모델 가중치 고정 여부는 제목의 핵심 가치가 아니므로 제목에서는 빼고 방법론에서 정확히 설명한다.

### 다음 행동

1. 1순위와 2순위 제목을 지도교수에게 함께 제시한다.
2. 9월 9일 제출 전 학과 양식의 국문·영문 제목 길이와 구두점을 점검한다.
3. 접수 후 제목 변경 가능 여부를 대학원/학과 공지와 담당자에게 확인한다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “현재 제목은 산업 결함과 Active Learning은 보이지만, 제한된 라벨 조건과 객체 탐지 Task, 무엇을 위한 연구인지가 빠져 있었습니다. 그래서 제목을 ‘제한된 라벨 환경의 산업 결함 객체 탐지’를 앞에 두고, DINOv2 표현 기반 표본 선택의 학습 효용을 검증한다는 구조로 바꿨습니다. 모델 가중치를 고정했다는 구현 상태는 제목보다 방법론에서 설명하겠습니다.”

---

# 지도교수 대표 논문의 제목 구조 분석

IMAI Lab은 공식 홈페이지에서 센서·영상 등 산업 공정 데이터를 융합해 산업·제조 현장의 문제를 해결하는 것을 연구 방향으로 밝힌다([IMAI Lab 소개](https://imai.knu.ac.kr/)). 아래 분석은 Google Scholar의 불완전한 검색 결과를 조합하지 않고, [IMAI Lab 공식 Publications](https://imai.knu.ac.kr/research/publications)와 연결된 publisher/DOI 서지를 기준으로 했다. 제목에 명시되지 않은 세부 방법은 추정하지 않았다.

| 논문 | 목적/기여 | Task | 환경/조건 | 방법론 | Domain |
|---|---|---|---|---|---|
| [Towards efficient data-driven fault diagnosis under low-budget scenarios via hybrid deep active learning](https://doi.org/10.1016/j.ress.2025.111637) | 효율적 진단 | fault diagnosis | low-budget scenarios | hybrid deep active learning | 산업 설비 진단 |
| [Robust tool wear prediction under novel operating conditions via physics-guided unsupervised domain adaptation](https://doi.org/10.1016/j.aei.2025.103883) | 강건한 예측 | tool wear prediction | novel operating conditions | physics-guided UDA | 가공 공정 |
| [Supervised contrastive learning for multi-source domain generalization: Tool failure prediction in data-deficient cryogenic milling processes](https://doi.org/10.1016/j.asoc.2025.114217) | 다중-source 일반화 | tool failure prediction | data-deficient cryogenic milling | supervised contrastive learning, domain generalization | 극저온 밀링 |
| [Fisher-informed continual learning for remaining useful life prediction of machining tools under varying operating conditions](https://doi.org/10.1016/j.ress.2024.110549) | 조건 변화 대응 | remaining useful life prediction | varying operating conditions | Fisher-informed continual learning | 가공 공구 |
| [Developing a deep learning-based uncertainty-aware tool wear prediction method using smartphone sensors for the turning process of Ti-6Al-4V](https://doi.org/10.1016/j.jmsy.2024.07.010) | 불확실성 인지 예측법 개발 | tool wear prediction | smartphone sensors, Ti-6Al-4V turning | deep learning, uncertainty-aware modeling | 선삭 공정 |
| [Towards holistic machinability estimation of titanium alloy: An integrated approach with enhanced feature extraction and physics-guided deep multi-task learning](https://imai.knu.ac.kr/research/publications) | 총체적 machinability 추정 | machinability estimation | titanium alloy | enhanced feature extraction, physics-guided multi-task learning | 티타늄 가공 |
| [Accurate synthesis of sensor-to-machined-surface image generation in carbon fiber-reinforced plastic drilling](https://imai.knu.ac.kr/research/publications) | 정확한 표면 영상 합성 | sensor-to-surface image generation | CFRP drilling | 제목에는 구체 모델 미표기 | CFRP 드릴링 |
| [Multimodal 1D CNN for delamination prediction in CFRP drilling process with industrial robots](https://imai.knu.ac.kr/research/publications) | 박리 예측 | delamination prediction | industrial-robot CFRP drilling | multimodal 1D CNN | 로봇 드릴링 |
| [A multi-domain mixture density network for tool wear prediction under multiple machining conditions](https://imai.knu.ac.kr/research/publications) | 다중 조건 예측 | tool wear prediction | multiple machining conditions | multi-domain mixture density network | 가공 공구 |

## 반복되는 설계 공식

1. `[목적/효과] + [Task] + under/in [문제가 발생하는 조건] + via/using [방법]`
2. `[방법 또는 연구목표]: [Task] in [구체적인 산업 환경]`
3. 어순은 달라도 **Task와 문제 조건이 거의 항상 제목에서 식별**된다.
4. `low-budget`, `novel/varying operating conditions`, `data-deficient cryogenic milling`, `smartphone sensors`처럼 문제 조건을 구체화한다.
5. 방법 이름은 목적을 대신하지 않는다. 방법은 어떤 산업 Task를 어떤 조건에서 풀기 위한 수단으로 배치된다.

현재 연구에 적용하면 핵심 공식은 다음이다.

> **제한된 라벨 환경 + 산업 결함 객체 탐지 + detector 학습 효용 + DINOv2 표현 기반 능동학습 표본 선택**

---

# 현재 연구 제목 후보 5개 및 냉정한 순위

| 순위 | 제목 | 환경 | Task | 목적 | 방법 | 과장 위험 | 교수 스타일 적합도 | 추천 |
|:---:|---|---|---|---|---|---|:---:|:---:|
| **1** | **제한된 라벨 환경의 산업 결함 객체 탐지를 위한 DINOv2 표현 기반 능동학습 표본 선택의 학습 효용 검증** | 명확 | 명확 | 학습 효용 검증 | DINOv2 표현 기반 AL 선택 | 낮음. “효용”이 우수성 단정이 아니라 검증 대상임 | 매우 높음. 조건·Task·방법·검증 범위가 모두 보임 | **최우선** |
| **2** | **제한된 라벨 환경의 산업 결함 객체 탐지를 위한 능동학습 표본 선택: DINOv2 표현 기반 선택의 통제 검증** | 명확 | 명확 | 통제 검증 | 방법을 콜론 뒤에 배치 | 낮음. 단, “통제”의 의미를 초록에서 설명해야 함 | 매우 높음. Task 전면형이며 연구실 콜론형 제목과도 맞음 | **강력** |
| **3** | **제한된 라벨 환경에서 산업 결함 객체 탐지 성능 확보를 위한 DINOv2 표현 기반 능동학습 표본 선택** | 명확 | 명확 | 성능 확보 | DINOv2 표현 기반 AL 선택 | 중간. ‘확보’가 달성 결과처럼 읽힐 수 있음 | 높음. 목적 중심 공식에 가장 가까움 | 조건부 |
| **4** | **제한된 라벨 환경의 산업 결함 객체 탐지를 위한 DINOv2 표현 기반 능동학습** | 명확 | 명확 | Task 수행 | DINOv2 표현 기반 AL | 중간. 연구가 새 AL 방법 전체를 제안한 것처럼 넓게 읽힘 | 높음. 간결하지만 실제 기여 범위가 덜 보임 | 보조안 |
| **5** | **산업 결함 객체 탐지 능동학습에서 DINOv2 표현 기반 다양성 선택과 무작위 선택의 학습 성능 비교** | 약함 | 명확 | 비교 | 두 arm 직접 표기 | 높음. 교수님이 지적한 ‘비교 자체가 Topic’인 구조로 회귀 | 낮음 | 비추천 |

## 1~3순위 선정 이유

- **1순위:** 사용자가 먼저 제안한 방향을 유지하면서 제외하기로 한 용어만 걷어냈다. 현재 연구가 새 selector의 우월성을 선언하는 것이 아니라 **학습 효용을 검증**한다는 실제 증거 범위가 가장 정확하다.
- **2순위:** 처음 읽는 사람이 제한 라벨·산업 결함·객체 탐지·능동학습을 먼저 이해한다. 방법을 콜론 뒤에 두어 Task가 방법 이름에 묻히지 않는다.
- **3순위:** 교수님의 “무엇을 위해” 요구에는 가장 직접 답하지만, +0.58%p 수준의 평균 nAULC 개선과 일부 class 악화를 고려하면 “성능 확보”가 결과 단정처럼 읽힐 위험이 있다.

1순위 영문안은 다음과 같이 고정한다.

> **Validating the Learning Utility of DINOv2 Representation-Based Active Learning Sample Selection for Industrial Defect Object Detection under Limited Label Budgets**

---

## Task 04. “Pretrained DINO는 무슨 데이터로 학습됐는가?”

### 교수님이 문제 삼은 내용

> “사전학습된 DINO는 어떤 데이터로 학습됐나요?”

> “학습에 사용된 데이터와 지금 넣는 데이터가 너무 다른 패턴이면 특징을 제대로 추출하지 못할 수 있습니다.”

> “현 데이터와 사전학습 데이터가 유사한지 보고, 필요하면 fine-tuning도 해야 합니다.”

### 교수님 입장에서의 핵심 이슈

사전학습 모델이라는 이유만으로 산업 결함 영상에 유효하다고 가정할 수 없다. **무엇으로 학습된 어떤 모델인지**와 **현재 데이터에서 표현이 실제 의미가 있는지**는 별개의 질문이다.

### 확인한 내용

| 구분 | 확인 결과 |
|---|---|
| 정확한 모델 | `facebook/dinov2-small`, DINOv2 ViT-S/14 |
| exact revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| 규모/구조 | 약 21M parameters, patch 14, 12 layers, 6 attention heads |
| 사전학습 데이터 | LVD-142M, 약 1억 4,200만 장의 annotation 없는 이미지 |
| 사전학습 방식 | DINO self-distillation, iBOT masked image modeling, KoLeo regularization; small model은 큰 teacher의 지식을 distillation |
| feature dimension | 384차원 |
| 현재 가중치 상태 | 가중치를 업데이트하지 않고 `eval`·inference mode에서 FP32로 사용; 별도 fine-tuning 없음 |
| feature extraction | RGB 입력을 processor 기본 전처리 후 `pooler_output` 384차원으로 추출하고 각 행을 L2 정규화 |
| 현재 선택에서의 역할 | 이미지 표현만 제공하며, 실제 선택은 별도 거리 기반 acquisition rule이 수행 |

근거는 [DINOv2 논문](https://arxiv.org/abs/2304.07193), [공식 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [공식 backbone 정의](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py), [사용 checkpoint의 exact revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056), 로컬 `confirmatory_v1/confirmatory_protocol_v1.yaml:174-218`에서 교차 확인했다.

현재 확보된 domain 적합성 근거는 두 층위로 나뉜다.

- **있는 근거:** GC10-DET와 NEU-DET의 통제 실험에서 이 표현과 거리 규칙을 함께 사용했을 때 Random 대비 nAULC가 각각 약 +0.581%p, +0.578%p였다. 이는 **표현+선택 규칙 결합의 downstream 효용**을 지지한다.
- **부족한 근거:** 표현 공간 자체가 결함 class·형태를 잘 보존한다는 공통 정량 지표가 없다. GC10에는 과거 retrieval/purity 진단이 있으나 일부 class 실패가 확인됐고, NEU에는 동등한 직접 audit이 없다.

### 검토 및 판단

모델·사전학습·현재 사용 방식은 해결됐다. 그러나 “사전학습 데이터와 산업 결함 데이터가 다른데 왜 그대로 써도 되는가”에 대한 직접 답은 아직 불충분하다. downstream 결과만으로 표현의 의미 타당성을 역추론하면 representation과 acquisition 효과가 섞인다. 지금 바로 fine-tuning으로 연구를 넓히기보다, 기존 feature를 사용한 무학습 정량 분석이 먼저다.

### 대응

다음 미팅에서는 사전학습 사실과 domain 적합성 증거를 분리해 답한다. DINOv2를 산업 결함 전용 모델이라고 부르지 않고, 대규모 일반 영상에서 학습한 범용 시각 표현을 **가설적 representation**으로 사용했다고 설명한다. 적합성은 GC10/NEU에서 이웃 구조를 정량 확인한 뒤 보강한다.

### 다음 행동

1. 기존 또는 재추출한 feature로 same-label 대 different-label cosine distance를 계산한다.
2. leave-one-out kNN label agreement(`k=1,5`)를 계산한다.
3. NEU는 silhouette을 보조 지표로, multi-label 이미지가 있는 GC10은 multilabel-aware nearest-neighbor agreement를 사용한다.
4. class별 결과를 보고 class 8 등 실패 구간을 논문의 한계로 명시한다.
5. 새 학습이나 DINOv2 fine-tuning은 이 분석 전에는 시작하지 않는다.

### 상태

**추가 분석 필요**

### 다음 미팅 답변 (20~40초)

> “사용한 모델은 DINOv2-small, ViT-S/14이고 약 1억 4,200만 장의 라벨 없는 일반 이미지로 사전학습된 모델입니다. 현재 실험에서는 가중치를 바꾸지 않고 384차원 이미지 표현만 뽑았습니다. 다만 일반 이미지와 산업 결함 이미지가 다르다는 지적이 맞아서, downstream 성능만 제시하지 않고 GC10과 NEU에서 같은 class가 실제로 가까이 놓이는지 거리와 kNN 지표로 추가 확인하겠습니다.”

---

## Task 05. “다르다/비슷하다는 것을 어떻게 판단했는가? 정성평가 아닌가?”

### 교수님이 문제 삼은 내용

> “다르다, 비슷하다는 것을 어떻게 봤죠?”

> “대표 이미지를 몇 개 본 정성평가가 다음 과정으로 넘어갈 근거가 되는지 파악이 필요할 것 같습니다.”

> “그럼 선택을 직접 한 거잖아요?”

### 교수님 입장에서의 핵심 이슈

미팅 답변에서 “카테고리별 대표 이미지를 약 20개 열어봤다”와 “제가 골랐다”는 표현이 이어져, 사람이 육안으로 비슷하거나 다른 이미지를 골라 넣은 실험처럼 전달됐다. 그렇다면 Random보다 특정 이미지를 더 찾는 결과는 자명해지고 재현 가능한 algorithmic selection이 아니다.

### 확인한 내용

현재 통제 실험과 확인한 MPDD·VisA 주요 실험에서 **사람이 query 이미지를 직접 고른 과정은 없다.**

- 현재 통제 실험: DINOv2 표현과 cosine greedy farthest-first rule로 자동 선택
- MPDD: `scripts/03_analysis/run_mpdd_selection_only_audit.py:139-186,278-280`에서 자동 선택 후 GT를 audit에만 결합
- VisA: `scripts/03_analysis/run_visa_selection_only_audit.py:99-143,224-225`에서 자동 선택 후 GT를 audit에만 결합
- 사람이 이미지를 본 단계: 자동 선택 결과가 상식적으로 어떤 이미지인지 확인한 post-hoc sanity check

실제 선택 규칙은 다음과 같다.

1. 각 이미지를 DINOv2-small의 384차원 L2-normalized 표현으로 바꾼다.
2. 아직 선택되지 않은 이미지마다 현재 reference set과의 최대 cosine similarity를 구한다.
3. `1 - 최대 유사도`, 즉 현재 선택 집합에서 가장 멀리 떨어진 정도를 계산한다.
4. 그 값이 가장 큰 이미지를 하나 선택하고 reference set에 즉시 추가한다.
5. 필요한 query 수에 도달할 때까지 반복한다. 동률은 sample ID 순서로 결정한다.

이 규칙은 `confirmatory_v1/src/confirmatory_v1/protocol_core.py:265-303`에 구현돼 있다.

### 검토 및 판단

두 문제를 분리해야 한다.

- **수동 선택 의혹:** 구현 문제가 아니라 발표 설명 오류이며 바로 해결 가능하다.
- **표현의 유사/상이성 타당성:** 시각 확인만으로는 부족하므로 Task 04의 정량 분석이 필요하다.

즉 “선택은 자동이었다”가 “DINOv2 거리의 의미가 이미 입증됐다”를 뜻하지는 않는다.

### 대응

다음 미팅에서 먼저 수동 선택으로 오해하게 만든 답변을 명확히 정정한다. 발표 슬라이드에는 `자동 selection`과 `사후 육안 점검`을 서로 다른 상자로 표시한다. 정성 이미지는 예시로만 두고, 결론 근거는 거리·kNN 및 downstream 지표로 제한한다.

### 다음 행동

1. 자동 선택 정정 문장을 발표 노트 첫 페이지에 넣는다.
2. sampling flowchart에 사람 개입 없음, GT 미사용을 표시한다.
3. 대표 이미지 패널에는 “post-hoc illustration; selection에 미사용”이라고 적는다.
4. Task 04 정량 feature audit을 9월 말 원고에 추가한다.

### 상태

**부분 해결**

### 다음 미팅 답변 (20~40초)

> “지난번에 대표 이미지를 열어봤다고 설명드려 제가 직접 골랐던 것처럼 전달된 것 같습니다. 다시 코드를 확인해보니 실제 선택은 전부 자동이었습니다. DINOv2로 특징을 뽑고 현재 선택된 이미지들과 가장 먼 이미지를 거리값으로 하나씩 추가했습니다. 제가 이미지를 본 것은 자동 선택 결과의 사후 확인이었고, 표현이 실제 class 구조를 반영하는지는 별도 정량 지표로 보강하겠습니다.”

---

## Task 06. “DINO가 selector인가? 실제 sampling 방법은 무엇인가?”

### 교수님이 문제 삼은 내용

> “샘플링 방식이 어떻게 되나요?”

> “DINO로 특징을 추출했으면, 추출된 표현을 가지고 기존 Active Learning의 선택 방법으로 고르는 것인가요?”

### 교수님 입장에서의 핵심 이슈

모델 이름과 선택 알고리즘을 한 덩어리로 부르면 무엇이 성능을 만든 것인지 설명할 수 없다. DINOv2가 이미지를 점수화해 직접 selector 역할을 하는지, 아니면 feature 위에 별도 acquisition rule이 있는지를 분리해야 한다.

### 확인한 내용

#### Feature representation

- 사전학습 DINOv2-small ViT-S/14
- 384차원 `pooler_output`
- row-wise L2 normalization
- 현재 학습 과정에서는 가중치 업데이트 없음

#### Acquisition rule

- cosine greedy farthest-first
- 수식: 이미 선택된 reference set을 \(R\), 후보 이미지 표현을 \(z_x\)라 할 때

\[
d(x,R)=1-\max_{r\in R}\cos(z_x,z_r)
\]

- 매 단계에서 \(d(x,R)\)가 가장 큰 후보를 선택하고 즉시 \(R\)에 추가
- clustering이나 class prediction을 사용하지 않음

이 선택은 **greedy k-center/Core-set 계열의 기하적 다양성 원리와 매우 가깝다.** feature 공간에서 현재 선택 집합으로부터 가장 먼 점을 순차적으로 추가하기 때문이다. 엄밀히 같은 논문의 전체 학습 설정을 재현했다는 뜻은 아니지만, acquisition geometry는 동일 계열로 보는 것이 타당하다.

### 검토 및 판단

DINOv2는 selector 자체가 아니라 representation provider다. 실제 selector는 farthest-first acquisition rule이다. 따라서 같은 DINOv2 feature와 같은 cosine distance로 “Core-set”이라는 별도 arm을 추가하면 이름만 다른 중복 비교가 될 가능성이 높다. 교수님의 “다른 sampling 방법” 요구를 실질적으로 충족하지 못한다.

### 대응

방법론과 발표에서 모델 이름 대신 다음 두 줄로 항상 분리한다.

> Representation: 사전학습 DINOv2-small의 384차원 이미지 표현  
> Acquisition: cosine greedy farthest-first를 이용한 순차적 다양성 선택

### 다음 행동

1. 방법론 도식과 용어 표에서 representation/acquisition을 분리한다.
2. `DINO가 이미지를 선택한다` 대신 `DINOv2 표현 위에서 farthest-first가 이미지를 선택한다`고 쓴다.
3. 관련연구에서 Core-set/k-center와의 알고리즘적 근접성을 명시한다.
4. 추가 baseline은 이 기하와 다른 신호에서 선택한다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “DINOv2 자체가 selector는 아닙니다. DINOv2는 각 이미지를 384차원 특징으로 바꾸는 역할만 하고, 실제 표본 선택은 현재 선택된 집합에서 cosine 거리상 가장 먼 이미지를 하나씩 추가하는 farthest-first 방식입니다. 그래서 알고리즘적으로 Core-set이나 k-center 계열과 가깝고, 같은 feature와 거리로 Core-set을 하나 더 넣으면 독립적인 비교가 되지 않습니다.”

---

## Task 07. “100% 데이터를 다 쓰면 YOLO 성능은 얼마인가?”

### 교수님이 문제 삼은 내용

> “Active Learning을 쓰지 않고 모든 데이터를 다 써서 결함 탐지 Task를 수행했을 때 성능이 어떻게 되나요?”

> “모든 데이터를 다 썼을 때와 비교해서, 라벨이 적은 상황에서도 성능이 reasonable하다는 얘기가 필요합니다.”

### 교수님 입장에서의 핵심 이슈

35% 이하의 학습 곡선만 보면 절대적으로 어느 정도 성능 손실을 감수한 것인지 알 수 없다. 100% 학습 결과는 AL 방법끼리의 승패가 아니라, 제한 라벨 조건의 성능이 전체 라벨 학습에 얼마나 가까운지 보여주는 기술적 기준이다.

### 확인한 내용

동일한 split과 동일한 YOLOv8n lifecycle로 수행한 100% 라벨 결과는 현재 없다. 과거 full-supervised script는 별도 split, seed, epoch, batch 설정을 사용하므로 현재 R0~R5 결과와 직접 비교할 수 없다. 현재 동일 protocol의 최종 R5는 다음과 같다.

| Dataset | R5 이미지/Train pool | Random mAP50-95 | DINOv2 표현 기반 선택 mAP50-95 |
|---|---:|---:|---:|
| GC10-DET | 561/1,605 = 34.953% | 0.267444 | 0.273346 |
| NEU-DET | 441/1,259 = 35.028% | 0.380230 | 0.382966 |

이 값은 100% upper bound가 아니라 현재 protocol에서 가장 큰 라벨 이미지 수의 final-test 결과다.

### 검토 및 판단

교수님 질문에 답하려면 실제 추가 학습이 필요하다. legacy 수치를 끌어오면 비교 가능성을 훼손한다. 다만 full-data에서는 acquisition seed가 없으므로 AL의 20 learner-seed 분포를 어느 수준까지 반복할지 선택할 수 있다.

### 대응

#### 최소 실험 설계: 4 runs

- 2 datasets × 2 사전 고정 learner seeds
- 현재와 동일한 train/validation/test split
- YOLOv8n checkpoint/hash, 100 epochs, input 640, optimizer와 augmentation을 동일하게 유지
- train pool의 100% 라벨 이미지를 학습에 사용
- validation으로 `best.pt`를 선택하고 locked test에서 한 번 평가
- 목적: 빠르게 절대 기준을 확보하는 **기술적 reference**
- 한계: 현재 20 unique learner seeds의 분포와 완전히 대응하지 않음

#### 권장 반복 설계: 40 runs

- 2 datasets × 현재와 동일한 20 unique learner seeds
- acquisition seed는 full-data에 의미가 없으므로 반복하지 않음
- 목적: 현재 learner-seed 변동성과 직접 비교 가능한 full-data 분포 확보

#### 비교 방식

1. R5와 100%의 절대 mAP50-95, mAP50, recall을 나란히 제시한다.
2. `R5 / full-data` 비율과 `full-data - R5` gap을 기술적으로 보고한다.
3. 100%보다 높거나 낮다는 결과를 label-cost 절감으로 번역하지 않는다.
4. nAULC는 R0~R5 학습 곡선 요약이므로 100% 단일점과 같은 지표처럼 비교하지 않는다.

### 다음 행동

1. 9월 9일 직후 지도교수에게 4-run과 40-run의 증거 강도·GPU 시간을 제시해 범위를 확정한다.
2. 확정 전에는 학습을 시작하지 않는다.
3. 실행 시 현재 protocol manifest와 별도 full-data manifest를 봉인한다.

### 상태

**추가 실험 필요**

### 다음 미팅 답변 (20~40초)

> “현재 약 35%까지의 결과는 있지만, 같은 split과 같은 학습 조건으로 100%를 쓴 결과는 없습니다. 예전 full-data script는 조건이 달라 가져오면 안 됩니다. 최소로는 두 데이터셋에 두 seed씩 4회, 더 강하게는 현재 learner seed 분포와 맞춘 40회를 제안드립니다. 35% 결과와 절대 mAP, recall, 그리고 100% 대비 gap으로 비교하겠습니다.”

---

## Task 08. “Random과 DINO 말고 기존 Active Learning sampling 방법도 비교해야 한다.”

### 교수님이 문제 삼은 내용

> “Random과 사전학습 DINO 방식 말고 기존 다른 sampling 방법도 무조건 비교해야 합니다.”

> “초반에 정리했던 Active Learning의 선택 기준들과 비교가 되어야 합니다.”

### 교수님 입장에서의 핵심 이슈

Random 하나만으로는 DINOv2 표현 기반 다양성 선택이 단순 무작위보다 나은지는 볼 수 있지만, 기존 AL family 중 어느 위치인지 알 수 없다. 특히 현재 방법이 사실상 k-center/Core-set 기하와 가까우므로 같은 계열을 이름만 바꾼 비교는 방어력이 없다.

### 확인한 내용

현재 release의 정본 결과가 있는 arm은 Random과 DINOv2 표현 기반 farthest-first뿐이다. detector uncertainty, hybrid, BADGE 관련 코드는 과거 개발 흔적이 있지만 현재 split·현재 lifecycle의 통제 결과는 없다. 과거 custom detector-uncertainty signal은 primary gate를 통과하지 못했으므로 그대로 부활시켜 성공 방법처럼 쓰면 안 된다.

| 방법 | 현재 방법과 차별성 | 코드 존재 | 구현비용 | GPU 비용 | 교수 요구 충족도 | 추천 |
|---|---|---|---|---:|:---:|:---:|
| **Detector predictive uncertainty** | 높음. feature geometry가 아니라 현재 detector의 예측 불확실성 사용 | scoring·detector 예측 관련 historical code 있음; current arm 없음 | 중간. image-level aggregation과 no-detection 처리 사전 고정 필요 | full curve 기준 약 **200 training cells** | 매우 높음 | **1순위** |
| Entropy / least-confidence | 높음. 예측 confidence 기반 | entropy 구성요소는 있으나 단순 표준 arm의 current 구현·결과 없음 | 중간. 여러 box를 이미지 점수로 합치는 규칙이 애매함 | 약 200 cells | 높음 | 2순위 후보 |
| Localization uncertainty | 높음. 객체 탐지의 box 안정성에 직접 대응 | flip/localization instability historical code 있음; current arm 없음 | 높음. augmentation·matching·calibration confound 관리 필요 | 약 200 cells | 높음 | 보조 후보 |
| BADGE | 높음. gradient embedding으로 uncertainty와 diversity 결합 | legacy/scratch 흔적만 확인 | 매우 높음. detector용 gradient embedding 정의와 검증 필요 | 최소 200 cells + feature 계산 | 중간 | 비추천 |
| 별도 Core-set/k-center | 낮음. 같은 DINOv2 feature·cosine이면 현 방법과 사실상 중복 | distinct current arm 없음 | 낮음~중간 | 약 200 cells | 낮음 | **추가하지 않음** |

한 개만 선택한다면 **단순하고 사전 고정된 detector predictive uncertainty baseline**이 가장 방어력이 높다. 이유는 현재 방법의 기하적 diversity와 질문이 명확히 다르고, object detection Task의 learner state를 직접 사용하며, 교수님의 “기존 AL sampling” 요구에 가장 직접 답하기 때문이다.

단, 추천은 과거의 복잡한 custom hybrid를 되살리자는 뜻이 아니다. 표준적인 image-level uncertainty를 문헌에 따라 하나 고르고 다음을 결과 보기 전에 고정해야 한다.

- round별 어떤 detector checkpoint로 unlabeled pool을 추론할지
- 여러 detection의 confidence/entropy를 한 이미지 점수로 어떻게 합칠지
- detection이 하나도 없는 이미지를 어떻게 처리할지
- class imbalance를 보정할지 여부
- 동률 처리, seed, query 수

shared R0 학습을 재사용하고 R1~R5만 새로 수행하면 비용은 `2 datasets × 10 acquisition seeds × 2 learner seeds × 5 rounds = 200 training cells`이다.

### 검토 및 판단

교수님 요구는 권고보다 명시적 필수 보강에 가깝다. 그러나 정확한 baseline을 합의하지 않고 급히 돌리면 또 다른 outcome-guided selector 탐색이 된다. **한 개의 distinct baseline을 먼저 문헌·규칙 수준에서 봉인한 뒤** 실행해야 한다.

### 대응

1순위로 detector predictive uncertainty를 제시하되, 단순 least-confidence/entropy 중 어느 variant를 쓸지는 지도교수와 확정한다. 같은 기하를 공유하는 Core-set 재명명 arm은 제외한다. historical 실패 결과는 새 baseline 성공 근거가 아니라 설계 시 피해야 할 복잡성의 근거로 사용한다.

### 다음 행동

1. object-detection AL primary literature에서 재현 가능한 uncertainty aggregation 2개만 비교해 1개를 고른다.
2. 200 training cells와 마감 일정을 지도교수에게 보고한다.
3. 선택한 규칙·seed·checkpoint·stop condition을 protocol에 사전 고정한다.
4. 승인 전에는 코드 변경이나 학습을 시작하지 않는다.

### 상태

**추가 실험 필요**

### 다음 미팅 답변 (20~40초)

> “현재 방식은 DINOv2 feature 공간에서 멀리 떨어진 이미지를 고르는 k-center 계열이라, Core-set을 이름만 바꿔 더하는 것은 중복입니다. 한 개만 추가한다면 detector의 예측 불확실성으로 이미지를 고르는 기준선이 가장 방어력이 높습니다. 다만 여러 box의 confidence를 이미지 단위로 어떻게 합칠지 먼저 문헌에 맞춰 고정해야 하고, 현재 반복 구조로는 약 200개 training cell이 필요해서 범위를 먼저 확인받겠습니다.”

---

## Task 09. “그래프의 결과는 몇 % 데이터를 사용했을 때인가?”

### 교수님이 문제 삼은 내용

> “이 결과는 데이터가 몇 %일 때인가요? 30%인가요?”

> “결과가 언제일 때인지가 중요할 것 같습니다.”

> “‘best’일 때라고 제시하면 안 됩니다. 데이터가 많을수록 좋은 것과 checkpoint 선택은 다른 얘기입니다.”

### 교수님 입장에서의 핵심 이슈

그래프의 x축과 `best`라는 표현이 어떤 데이터 사용량인지 명확하지 않았다. Active Learning round, 현재까지 사용한 이미지 수, validation-selected checkpoint, final test 결과가 섞이면 성능 수치를 해석할 수 없다.

### 확인한 내용

#### GC10-DET

Train/acquisition pool은 1,605장이다.

| Round | 학습에 사용한 이미지 수 | Train pool 비율 |
|:---:|---:|---:|
| R0 | 161 | 10.031% |
| R1 | 241 | 15.016% |
| R2 | 321 | 20.000% |
| R3 | 401 | 24.984% |
| R4 | 481 | 29.969% |
| R5 | 561 | 34.953% |

#### NEU-DET

Train/acquisition pool은 1,259장이다.

| Round | 학습에 사용한 이미지 수 | Train pool 비율 |
|:---:|---:|---:|
| R0 | 126 | 10.008% |
| R1 | 189 | 15.012% |
| R2 | 252 | 20.016% |
| R3 | 315 | 25.020% |
| R4 | 378 | 30.024% |
| R5 | 441 | 35.028% |

#### 혼동하면 안 되는 네 개념

| 용어 | 정확한 의미 | 잘못된 해석 |
|---|---|---|
| `best.pt` | 각 round에서 최대 100 epochs 학습 중 validation 기준으로 선택한 checkpoint | 가장 많은 데이터 또는 가장 좋은 budget |
| final budget | 현재 protocol의 가장 큰 학습 이미지 수인 R5, 약 35% | 100% 데이터 |
| final test | 학습·선택에 쓰지 않은 locked test에서 해당 checkpoint를 평가한 결과 | R5 또는 `best.pt` 자체 |
| nAULC | R0~R5의 mAP50-95 학습 곡선 아래 면적을 x축 범위로 정규화한 요약값 | 특정 round의 mAP 또는 100% 성능 |

### 검토 및 판단

실험 결과가 없는 문제가 아니라 그래프와 구두 설명의 **단위·시점 표기 문제**다. 즉시 해결 가능하다. 특히 “best model 결과” 대신 “R3에서 validation으로 고른 checkpoint의 locked-test 성능”처럼 round와 checkpoint를 동시에 말해야 한다.

### 대응

모든 표·그림에서 x축을 `학습에 사용한 이미지 수 (train pool 대비 %)`로 바꾸고, 캡션에 dataset별 분모를 쓴다. 곡선 전체 요약은 nAULC, 마지막 점은 R5 final-budget mAP50-95라고 구분한다.

### 다음 행동

1. 그림 x축과 표 header에 이미지 수와 %를 병기한다.
2. 본문 첫 결과 설명에 GC10 1,605장, NEU 1,259장이라는 분모를 쓴다.
3. `best` 단독 표기를 없애고 `validation-selected checkpoint`로 바꾼다.
4. R5를 “최종 budget”이라 부를 때 약 35%라는 뜻을 바로 덧붙인다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “그래프는 두 데이터셋 모두 train pool의 약 10%에서 시작해 5%p씩 늘려 약 35%까지 본 결과입니다. GC10은 161장에서 561장, NEU는 126장에서 441장입니다. 제가 ‘best’라고 말한 것은 데이터 비율이 아니라 각 round에서 validation으로 고른 checkpoint를 뜻했습니다. 앞으로는 round의 이미지 수와 비율, checkpoint 선택, test 평가를 따로 표기하겠습니다.”

---

## Task 10. “표본선택용/개발용/최종평가용 대신 train/validation/test라고 하면 된다.”

### 교수님이 문제 삼은 내용

> “표본선택용, 개발용, 최종평가용이 아니라 그냥 train, validation, test set이라고 하면 됩니다.”

### 교수님 입장에서의 핵심 이슈

익숙하지 않은 번역어가 표준적인 split 구조를 오히려 어렵게 만들었다. Active Learning의 특수성은 새 이름이 아니라 train pool 내부의 labeled/unlabeled 상태 변화로 설명하면 된다.

### 확인한 내용

| Dataset | Train/acquisition pool | Validation | Locked test | 합계 |
|---|---:|---:|---:|---:|
| GC10-DET | 1,605 | 229 | 458 | 2,292 |
| NEU-DET | 1,259 | 180 | 359 | 1,798 |

- split seed: `20260822`
- class-stratified split
- canonical pixel identity 기반 duplicate guard 적용
- train pool 안에서만 labeled와 unlabeled 상태가 round에 따라 바뀜
- validation은 각 round의 checkpoint 선택에 사용
- locked test는 최종 평가에만 사용

### 검토 및 판단

이미 구현은 올바른 세 split 구조다. 발표 용어가 이를 가린 것이므로 교수님 권고대로 즉시 바꾸면 된다.

### 대응

권장 한 문장은 다음과 같다.

> 전체 데이터를 train, validation, test로 나눴고, train pool 내부에서만 labeled와 unlabeled가 바뀝니다. validation은 각 단계의 checkpoint 선택에, test는 최종 성능 평가에만 사용했습니다.

### 다음 행동

1. 초록·방법론·표·그림·발표에서 기존 번역어를 표준 split 용어로 치환한다.
2. 첫 등장에만 `train/acquisition pool`이라고 병기한다.
3. data split 도식에서 test가 acquisition과 checkpoint 선택에 들어가지 않음을 표시한다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “말씀하신 대로 실제 구조는 일반적인 train, validation, test입니다. 다만 Active Learning이라 train pool 안에서만 현재 라벨이 있는 이미지와 아직 없는 이미지가 round마다 바뀝니다. validation은 각 round의 checkpoint를 고르는 데 쓰고, test는 선택이나 학습에 관여시키지 않고 최종 평가에만 사용했습니다.”

---

## Task 11. “라벨 예산, 누적 같은 표현이 어렵다.”

### 교수님이 문제 삼은 내용

> “‘누적’이라는 단어가 별로입니다.”

> “학습 때 사용된 이미지 수라고 하면 되지 않나요?”

### 교수님 입장에서의 핵심 이슈

`budget`, `cumulative`는 AL 연구자에게는 익숙하지만, 발표 청중에게는 돈이나 이전 round의 model state 누적처럼 들릴 수 있다. 실제 조작 변수는 단순히 **현재 round에서 학습에 사용 가능한 라벨 이미지의 총수**다.

### 확인한 내용

각 round는 해당 시점까지 선택된 전체 이미지를 labeled training set으로 사용한다. 모델은 round별 cold-start이므로 이전 round model을 이어 학습한 “누적 학습”이 아니다. 늘어나는 것은 학습 가능한 이미지 집합의 크기다.

### 검토 및 판단

기술 용어가 틀린 것은 아니지만 전달력이 낮다. 특히 `누적`은 학습 방식까지 누적된다는 오해를 만든다. 본문 첫 정의 이후에는 이미지 수와 비율을 우선해야 한다.

### 대응

| 기존 표현 | 권장 표현 |
|---|---|
| 라벨 예산 | 학습에 사용한 라벨 이미지 수 |
| 10% budget | train pool의 약 10% 이미지를 학습에 사용 |
| cumulative budget | 현재 단계까지 학습에 사용한 전체 이미지 수 |
| budget round | 이미지 수를 늘린 단계, 또는 R0~R5 |
| final budget | 현재 실험에서 가장 많은 이미지를 쓴 R5, 약 35% |

논문에서는 첫 등장에만 `라벨 이미지 수(label budget)`라고 병기하고, 이후에는 `이미지 수`와 `%`를 쓴다.

### 다음 행동

1. 발표에서 `누적`을 제거한다.
2. 모든 그래프 x축에 절대 이미지 수와 train-pool 비율을 함께 쓴다.
3. cold-start 재학습이라는 점을 방법론과 캡션에 명시한다.

### 상태

**해결 가능**

### 다음 미팅 답변 (20~40초)

> “라벨 예산이라는 말은 실제로는 돈이 아니라 학습에 사용한 이미지 수였습니다. 앞으로는 GC10 161장, train pool의 약 10%처럼 이미지 수와 비율을 먼저 말하겠습니다. 또 ‘누적’은 이전 모델을 이어 학습했다는 뜻이 아니라 선택된 이미지 집합이 커졌다는 뜻이어서, 현재 단계까지 학습에 사용한 전체 이미지 수라고 바꾸겠습니다.”

---

## Task 12. “텍스트는 왜 다 빠졌는가? 이전 VLM 연구와 현재 연구를 연결할 수 없는가?”

### 교수님이 문제 삼은 내용

> “이번에는 텍스트가 다 빠진 건가요?”

> “VLM에서 language가 어디 갔지 하고 있었습니다.”

> “샘플링에는 DINO feature를 쓰고 downstream에서는 텍스트도 같이 써보면 어떨까요?”

### 교수님 입장에서의 핵심 이슈

연구가 VLM 중심에서 image-only representation으로 크게 바뀌었는데 연결 설명이 없었다. 교수님 발언에는 과거 연구를 되살리라는 필수 지시와, novelty를 위해 text/multimodal downstream을 고려해보라는 브레인스토밍이 섞여 있다.

### 확인한 내용

- Qwen/VLM inference·prompt·score code와 aggregate evidence ledger는 남아 있다.
- canonical branch에는 현재 sample 단위로 바로 쓸 수 있는 raw Qwen response/score `.json`, `.jsonl`, `.csv`를 찾지 못했다.
- 과거 audit에서 VLM signal validity는 약했다.
  - consistency-groundedness Spearman: `-0.181199`, 95% CI `[-0.389654, 0.036410]`
  - severe-failure AUC: `0.373433`
  - structured schema completeness: `0.01`
  - 세 Qwen-family model 중 gate pass: `0/3`
- aggregate metric만으로는 image-level acquisition score를 복원할 수 없다.
- 별도 저장소에서 raw artifact를 회수한다면 `legacy filename/path → private_source_registry → canonical sample_id` 연결 가능성은 점검할 수 있다. 다만 split overlap, duplicate collapse, response completeness, final-test leakage audit이 선행돼야 한다.

### 검토 및 판단

현재 논문의 즉시 보강책이 아니다. raw artifact도 없고 기존 신호 타당성도 현재 결합을 지지하지 않는다. 새 Qwen inference, 새로운 multimodal detector, 새 fusion selector는 각각 새로운 연구 설계와 대규모 검증을 요구한다. 교수님 발언은 현재 일정에서는 **novelty 아이디어**로 분류하는 것이 타당하다.

### 대응

현재 연구가 image-only로 바뀐 이유를 숨기지 않고, 과거 VLM 신호가 validity gate를 통과하지 못해 현재 통제 검증에서는 제외했다고 짧게 설명한다. 과거 실패는 연구의 출발점이 아니라 method scope를 정한 경과로 둔다.

### 다음 행동

1. raw VLM artifact의 알려진 저장 위치만 1회 확인한다.
2. 없으면 복원 작업을 종료하고 “canonical raw artifact 없음”으로 기록한다.
3. 있더라도 이번 논문에는 결합하지 않고 sample-ID 연결 가능성만 후속 연구 메모로 남긴다.
4. 새 Qwen inference와 새 text-image detector는 수행하지 않는다.

### 상태

**이번 논문 범위에서 제외**

### 다음 미팅 답변 (20~40초)

> “텍스트가 빠진 것은 설명 없이 Task를 바꾼 것이 아니라, 과거 Qwen 신호가 일관성·grounding 검증을 통과하지 못해서 현재 통제 실험에서 제외한 것입니다. 지금 branch에는 이미지별 raw score도 없어 새 추론 없이 바로 결합할 수도 없습니다. 위치를 한 번만 확인하고, 없으면 이번 논문은 image-only DINOv2 표현 기반 선택으로 마무리하며 멀티모달 downstream은 후속 연구로 두겠습니다.”

---

## Task 13. “자동차 부싱 VLM-assisted Active Learning 연구와 Task가 비슷하지 않은가?”

### 교수님이 문제 삼은 내용

> “이 연구와 지금 Task가 거의 똑같지 않나요?”

> “제목만 봐도 자동차 부싱에서 defect detection을 위해 VLM-assisted Active Learning을 썼다는 것이 나옵니다.”

### 교수님 입장에서의 핵심 이슈

가장 가까운 선행연구와 문제 설정이 겹치는데 이를 인지하지 못하면 novelty와 제목 모두 방어하기 어렵다. 다만 Task가 같다는 사실과 방법·실험 기여가 같다는 주장은 구분해야 한다.

### 확인한 내용

확인된 논문은 Shikun Chen과 Yangguang Liu의 [Vision-Language Model-Assisted Active Learning for High-Accuracy Defect Detection in Automotive Bushing Components](https://doi.org/10.1145/3788731.3788733), EILM 2025, DOI `10.1145/3788731.3788733`이다.

접근 가능한 [서지·초록 record](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3788731.3788733?fields=title,abstract,authors,year,venue,url,externalIds)에 따르면, 해당 연구는 locally deployed Qwen2.5-VL을 LoRA fine-tuning하여 defect classification과 bounding-box prediction을 수행하고, classification confidence와 localization uncertainty를 결합해 annotation 대상을 우선화한다. 15,000장, 4개 defect type의 자동차 부싱 데이터를 사용하며 35% data에서 mAP@50 98.2% 등의 성능을 주장한다. 다만 full text를 확보하지 못했으므로 exact uncertainty 수식, 별도 detector 구조, baseline, seed/statistical protocol은 확인되지 않았다.

| 항목 | 자동차 부싱 연구 | 현재 연구 |
|---|---|---|
| 산업 문제 | 자동차 부싱 결함 검사의 annotation 효율·정확도 | 제한된 라벨 이미지의 산업 결함 탐지 학습 |
| Task | defect classification + bounding-box prediction으로 설명됨 | 다중 클래스 object detection |
| 데이터 | private 15,000 bushing images, 4 defect types | 공개 GC10-DET·NEU-DET |
| VLM | Qwen2.5-VL 사용 | 사용하지 않음 |
| model adaptation | LoRA fine-tuning | DINOv2 가중치 업데이트 없음 |
| text 역할 | multimodal prompt/semantic output에 관여 | 없음 |
| selection signal | classification confidence + localization uncertainty | DINOv2 image representation의 cosine diversity |
| acquisition rule | cross-modal uncertainty 우선순위; exact formula 미확인 | greedy farthest-first |
| downstream learner | full text 미확보로 독립 detector 여부 미확인 | sealed YOLOv8n cold-start learner |
| protocol | abstract 수준만 확인 | 2 datasets, 10 acquisition seeds, 2 learner seeds, 6 rounds, 440 training cells |
| 현재 주장 가능한 novelty | full text 확인 전 제한 | image-only representation 선택의 통제된 downstream 효용과 class-level 경계 |

#### 같은 점

- 산업 결함
- object detection
- 제한된 annotation 환경
- Active Learning

#### 다른 점

- VLM 및 text 사용 여부
- VLM fine-tuning 여부
- uncertainty와 geometric diversity라는 선택 신호 차이
- DINOv2 image representation 사용 여부
- acquisition rule
- downstream learner와 공개 데이터 여부
- 반복 seed·locked-test protocol
- 주장하는 기여의 종류

### 검토 및 판단

교수님의 “Task가 거의 같다”는 판정은 맞다. 두 연구 모두 제한 annotation의 산업 결함 object detection을 Active Learning으로 다룬다. 그러나 `Task가 같다 = 방법론도 같다`는 아니다. 이 논문은 경쟁해서 지워야 할 자료가 아니라, **같은 산업 문제에 대한 가장 가까운 관련연구**이자 현재 연구의 method/protocol 차이를 설명할 기준이다.

### 대응

관련연구에서 이 논문을 가장 가까운 사례로 먼저 소개하고, trained VLM의 cross-modal uncertainty와 현재 image-only representation diversity를 위 표처럼 분리한다. “VLM보다 우수하다”는 직접 비교 주장은 하지 않는다. 데이터와 protocol이 달라 수치도 직접 비교하지 않는다.

### 다음 행동

1. 학교 구독을 통해 full text를 확보한다.
2. exact detector, uncertainty 식, AL baselines, seed protocol만 확인해 표를 보정한다.
3. 확보 전에는 `확인되지 않음`을 유지한다.
4. 연구 차별성 문단에 same Task/different method·evidence structure를 명시한다.

### 상태

**부분 해결**

### 다음 미팅 답변 (20~40초)

> “교수님 말씀대로 문제와 Task는 매우 가깝습니다. 둘 다 제한된 annotation 환경의 산업 결함 객체 탐지에 Active Learning을 씁니다. 다만 부싱 논문은 LoRA로 학습한 Qwen2.5-VL의 classification·localization uncertainty를 쓰고, 제 연구는 text 없이 DINOv2 이미지 표현에서 기하적 다양성을 기준으로 고른 뒤 YOLOv8n 학습 효과를 반복 seed로 검증합니다. 같은 Task의 가장 가까운 관련연구로 넣고 방법과 protocol 차이를 명확히 쓰겠습니다.”

---

## Task 14. “Novelty가 부족하다.”

### 교수님이 문제 삼은 내용

> “기존 다른 연구들과 어떻게 차별점을 둘지, novelty를 고민해야 합니다.”

> “다양한 domain에서의 일반화, open-set 같은 Task로 바꿀 수도 있습니다.”

> “분류할 때 텍스트도 같이 쓰는 방법도 생각해볼 수 있습니다.”

### 교수님 입장에서의 핵심 이슈

현재 방법은 DINOv2 feature와 표준적인 farthest-first geometry를 결합한 것이므로 새 알고리즘이라고 주장하기 어렵다. Random 대비 평균 개선도 작고 일부 class 악화가 있어, novelty를 단순 성능 우월성으로 만들 수 없다.

### 확인한 내용

현재 통제 결과는 다음 범위에서 재현된다.

| Dataset | Random nAULC | DINOv2 표현 기반 선택 nAULC | paired Δ | 95% CI | acquisition win |
|---|---:|---:|---:|---:|---:|
| GC10-DET | 0.244643 | 0.250454 | +0.005812 (+0.581%p) | [0.003970, 0.007576] | 10/10 |
| NEU-DET | 0.351950 | 0.357731 | +0.005782 (+0.578%p) | [0.003100, 0.008119] | 9/10 |

두 dataset의 Holm-adjusted exact sign-flip p-value는 각각 `0.003906`이다. 그러나 NEU의 worst-class nAULC 차이는 `-0.014879`, final-budget worst-class AP 차이는 `-0.031182`로 악화됐다.

따라서 현재 연구에서 방어 가능한 기여는 다음이다.

1. 제한된 라벨 이미지 조건의 산업 결함 객체 탐지에서 representation-based diversity sampling의 downstream 효용을 동일 조건으로 통제 검증
2. acquisition seed와 learner seed를 분리한 paired repeated-seed protocol
3. 작은 aggregate gain과 취약 class degradation이 동시에 발생한다는 empirical evidence
4. selection-stage의 다양성 또는 target discovery가 detector 학습 효용과 동일하지 않다는 경계
5. 보편적 우수성보다 적용 범위와 failure mode를 공개하는 evidence discipline

반대로 현재 증거가 지지하지 않는 주장은 새 알고리즘, SOTA, 모든 class 개선, 실제 비용 절감, 산업 전반 일반화다.

### 검토 및 판단

novelty 표현은 보강해야 하지만, 현재 440-run 연구를 버리고 DG, open-set, 새 multimodal detector로 갈아타는 것은 새로운 논문을 시작하는 수준이다. 전사 뉘앙스와 9월 일정상 이 세 가지는 **가능한 novelty 예시/브레인스토밍**이며 이번 제출의 즉시 필수 요건으로 해석하지 않는다.

현재의 차별성은 algorithmic novelty가 아니라 **엄격한 통제 아래 확인한 작은 평균 이득과 class-level 위험의 공존**이다. 이 범위가 석사논문의 충분한 중심 기여인지 지도교수에게 명시적으로 확인받아야 한다.

### 대응

기여 문장을 “새로운 DINO sampler 제안”에서 “제한 라벨 산업 결함 탐지에서 representation-based sampling의 학습 효용과 class-level 한계를 반복 통제로 검증”으로 바꾼다. distinct baseline과 full-data reference가 추가되면 비교 맥락은 강화되지만, 결과를 본 뒤 novelty 문구를 키우지 않는다.

### 다음 행동

1. 기여 3개를 `downstream controlled validation`, `paired-seed evidence`, `aggregate–worst-class separation`으로 고정한다.
2. 지도교수에게 이 기여 framing이 학위논문의 중심으로 충분한지 확인한다.
3. full-data·feature audit·distinct baseline으로 증거를 보강한다.
4. DG, open-set, 새 multimodal detector는 future work에만 둔다.

### 상태

**부분 해결**

### 다음 미팅 답변 (20~40초)

> “현재 방법을 새로운 알고리즘이나 보편적으로 우수한 selector라고 주장하기는 어렵습니다. 대신 두 데이터셋에서 acquisition seed와 learner seed를 분리해 통제했을 때 평균 nAULC는 약 0.58%p 좋아졌지만, 일부 취약 class는 오히려 나빠졌습니다. 그래서 novelty는 평균 성능만이 아니라 selection 효용과 class-level 위험이 다를 수 있다는 통제된 실증 근거로 잡고, DG·open-set·새 멀티모달 모델은 이번 논문 범위에서는 제외하겠습니다.”

---

# 최종 교수님 피드백 대응표

| Task | 교수님 이슈 | 결론 | 우리가 확인한 근거 | 대응 | 남은 작업 |
|---|---|---|---|---|---|
| 01 | 연구 목적이 비교·분석으로 끝남 | 실제 목적은 제한 라벨 조건의 산업 결함 detector 학습 효용이며 연구 유지 가능 | GC10/NEU object detection, 동일 이미지 수 paired comparison | 목적 문장과 제목을 공학 문제 중심으로 통일 | 초록·서론·발표 반영 |
| 02 | Random 미달 경험이 서론 출발점 | 개인 연구 경과를 산업 공통 문제로 일반화하면 안 됨 | 초안에 산업 문제는 있으나 VLM 실패와 연구 변천이 더 강조됨 | 산업 문제 → 제한 라벨 → Task → AL → 방법 순으로 재구성 | 실패 서사를 후반으로 이동 |
| 03 | 제목만 보고 연구 Task를 알 수 없음 | 제한 라벨·산업 결함·객체 탐지·방법/효용을 모두 명시해야 함 | 현재 주제목에는 조건과 정확한 Task가 없음; IMAI 제목은 조건·Task·방법이 명시적 | 1순위 제목 제출 | 지도교수 승인, 행정상 변경 가능 여부 확인 |
| 04 | DINOv2의 사전학습과 현 domain 적합성 | 모델 사실은 확인; 현 데이터 표현 타당성은 추가 분석 필요 | ViT-S/14, LVD-142M, 384D, 가중치 고정; GC10 일부 audit만 존재 | 사전학습 사실과 domain validity를 분리 설명 | distance·kNN·class별 audit |
| 05 | 유사/상이 판단이 정성·수동인가 | 선택은 자동, 육안은 사후 확인; 표현 타당성은 별도 미해결 | current, MPDD, VisA 코드에서 자동 selection과 post-hoc GT join 확인 | 수동 선택 오해 즉시 정정 | flowchart 수정, 정량 feature audit |
| 06 | DINO가 selector인지 sampling이 무엇인지 불명 | DINOv2는 representation, farthest-first가 acquisition | `protocol_core.py:265-303`, cosine distance와 immediate update | 방법론을 두 단계로 분리 | Core-set/k-center 관계 명시 |
| 07 | 100% 데이터 YOLO 성능 부재 | 동일 lifecycle upper bound가 없음 | legacy 실험은 split/epoch/seed가 달라 비교 불가 | 4-run 최소 또는 40-run 권장 설계 | 반복 범위 합의 후 실행 |
| 08 | Random 외 AL baseline 부재 | 다른 신호의 baseline 1개가 필요 | current result는 2-arm뿐; 별도 Core-set은 중복 | detector predictive uncertainty 제안 | variant 봉인 후 약 200 cells 실행 |
| 09 | 그래프가 몇 % 결과인지 불명 | R0~R5는 약 10%→35%; `best.pt`와 budget은 다름 | GC10 161→561장, NEU 126→441장 | 그림에 절대 이미지 수·%·round를 병기 | 캡션·축·발표 수정 |
| 10 | 비표준 split 명칭 | 구현은 train/validation/test 구조로 정상 | train pool 안에서만 labeled/unlabeled 변화 | 표준 용어 사용, AL 역할 한 문장 보충 | 전 문서 용어 치환 |
| 11 | 라벨 예산·누적 표현이 어려움 | 실제 조작 변수는 학습에 사용한 이미지 수 | 모델은 round별 cold-start, 이미지 집합만 증가 | 이미지 수와 train-pool %를 먼저 제시 | 용어표 적용 |
| 12 | text/VLM이 빠진 이유와 재연결 | 지금 논문의 즉시 보강책이 아님 | raw image-level artifact 없음, 기존 validity 약함 | 제외 이유를 연구 경과로 설명 | 저장 위치 1회 확인 후 종료 |
| 13 | 자동차 부싱 논문과 Task 중복 | same Task, different method/protocol | trained Qwen uncertainty 대 image-only DINOv2 diversity; public data와 440-cell protocol | 가장 가까운 관련연구로 정면 비교 | full text로 미확인 세부만 보정 |
| 14 | novelty 부족 | algorithm novelty 대신 통제된 empirical contribution으로 한정 | 두 dataset 약 +0.58%p nAULC, NEU worst-class 악화, paired seeds | aggregate–class risk 분리를 기여로 제시 | 지도교수 수용 여부 확인, P1 증거 보강 |

---

# 다음 해야 할 일

## A. 9월 9일 제목 제출 전에

| 작업 | 필요 이유 | 예상 산출물 | 실험 필요 여부 |
|---|---|---|:---:|
| 1순위·2순위 제목을 지도교수에게 제시 | 제목이 이번 미팅의 최우선 지적이며 Task가 즉시 보여야 함 | 국문·영문 제목 확정본 | 아니오 |
| 연구 목적 네 줄 확정 | 제목·초록·서론·발표가 서로 다른 연구를 말하지 않게 함 | 환경/Task/목적/방법 4줄 및 1문장 목적 | 아니오 |
| 서론 첫 흐름 재설계 | Random/VLM 연구 경과가 산업 문제보다 앞서는 문제 해결 | 단락별 개요: 산업 문제 → 제한 라벨 → 객체 탐지 → AL → 방법 → 질문 | 아니오 |
| 자동 selection 정정문 준비 | 수동 선택 오해를 즉시 해소 | 20~40초 구두 답변과 1장 flowchart 문안 | 아니오 |
| 이미지 수·split 용어표 확정 | 그래프와 데이터 구조 오해 방지 | R0~R5 이미지 수 표, train/validation/test 한 문장 | 아니오 |
| 제목 변경 가능 여부 확인 | 교수님이 직접 확인을 요청 | 학과 담당자 답변 또는 공지 링크·절차 | 아니오 |

경북대학교 2026학년도 2학기 일반대학원 공지에는 논문제출 신청 시 제목이 가제이며 추후 변경 가능하다고 안내된 사례가 있다([KNU 공지](https://gsee.knu.ac.kr/content/notice.html?fidx=6075&gtid=grad_notice&opt=&page=26&pg=vv&sword=)). 다만 데이터사이언스대학원의 내부 마감·서명·변경 절차는 별도로 확인해 지도교수에게 보고한다.

## B. 지도교수에게 바로 보고/확인할 것

| 확인 항목 | 필요 이유 | 예상 산출물 | 실험 필요 여부 |
|---|---|---|:---:|
| 제목 1순위 대 2순위 | “학습 효용 검증”형과 Task 전면 콜론형 중 교수 선호 확인 | 최종 제출 제목 | 아니오 |
| 현재 empirical novelty framing 수용 여부 | 새 알고리즘이 아닌 통제 검증과 class risk가 학위 중심 기여로 충분한지 확인 | 기여 문장 승인/수정 | 아니오 |
| full-data 반복 범위: 4 대 40 runs | 일정과 증거 강도의 trade-off가 큼 | 실행 횟수 결정 | 보고 단계는 아니오; 승인 후 예 |
| distinct AL baseline 1개와 약 200 cells | 교수의 명시 요구이나 계산량이 큼 | detector uncertainty variant 및 범위 승인 | 보고 단계는 아니오; 승인 후 예 |
| 제목 변경 행정 답변 | 9/9 제출 후 수정 가능성 공유 | 담당자 답변 전달 | 아니오 |

## C. 9월 말 심사용 논문 전에 추가할 것

| 작업 | 필요 이유 | 예상 산출물 | 실험 필요 여부 |
|---|---|---|:---:|
| DINOv2 feature 정량 audit | 일반 사전학습 표현이 산업 결함 class/형태를 반영한다는 직접 근거 필요 | same/different distance, kNN agreement, class별 failure 표·그림 | **추가 학습은 아니오**, feature 분석은 예 |
| 100% labeled YOLO reference | 약 35% 결과의 절대 수준과 gap 설명 | dataset별 full-data mAP50-95/mAP50/recall | **예** |
| distinct AL baseline 1개 | Random 이외 기존 AL과 비교하라는 요구 충족 | R0~R5 learning curve 및 paired comparison | **예** |
| 초록·서론·제목·기여 재작성 | 목적과 Task를 전면화 | 심사용 원고 개정본 | 아니오 |
| 결과 표·그림 용어 교정 | round/budget/checkpoint/test 혼동 제거 | 이미지 수·%가 표시된 표·캡션 | 아니오 |
| 자동차 부싱 논문 full-text 확인 | 가장 가까운 선행연구와의 차별성 정확화 | 관련연구 비교표의 미확인 칸 보정 | 아니오 |
| 제한사항 명시 | 과장 방지 | full-data 여부, 비용 미측정, class 악화, 범위 한계 단락 | 아니오 |

## D. 이번 논문에서는 하지 않을 것

| 제외 작업 | 제외 이유 | 남길 산출물 | 실험 필요 여부 |
|---|---|---|:---:|
| Domain Generalization로 Task 전환 | 새 domain split·학습·평가가 필요한 별도 논문 규모 | future-work 한 문장 | 수행하지 않음 |
| Open-set Recognition로 Task 전환 | label space와 평가 지표를 처음부터 다시 설계해야 함 | future-work 한 문장 | 수행하지 않음 |
| 새 image+text 멀티모달 detector | downstream learner와 연구 질문 자체가 바뀜 | 후속 연구 아이디어 | 수행하지 않음 |
| 과거 VLM 실험 전수 복원 | raw artifact 부재·validity 약함·시간 분산 위험 | 1회 위치 확인 결과 | 수행하지 않음 |
| 새 Qwen inference 또는 fusion selector | 현재 증거가 결합을 정당화하지 않고 새 selector가 됨 | 하지 않는 이유 기록 | 수행하지 않음 |
| DINOv2 fine-tuning | 먼저 무학습 표현 audit으로 필요성을 판단해야 함; 현 protocol과 비교성 훼손 | future-work 조건 | 수행하지 않음 |
| 같은 feature·거리의 별도 Core-set arm | 현 acquisition geometry와 사실상 중복 | 관련연구에서 관계 설명 | 수행하지 않음 |

---

# 9월 9일 의사결정용 최종 요약

1. **제목은 연구 역사보다 문제를 말해야 한다.** 제한된 라벨 환경, 산업 결함, 객체 탐지, DINOv2 표현 기반 AL 표본 선택의 학습 효용을 앞에 둔다.
2. **현재 연구는 버리지 않는다.** 440-run의 통제 결과는 유지하되, 작은 평균 개선과 class 악화를 함께 보고한다.
3. **설명 오류는 오늘 바로 고친다.** 자동 selection, representation/acquisition 구분, train/validation/test, 이미지 수/%를 명확히 한다.
4. **증거 공백은 세 개만 보강한다.** feature 정량 audit, full-data reference, distinct AL baseline.
5. **연구 확장은 멈춘다.** DG, open-set, 새 multimodal detector, 대규모 VLM 복원은 이번 논문에서 하지 않는다.

제목 후보의 요청 방향 대응은 **방향 B가 1순위**, **방법을 부제로 내리는 방향 C가 2순위**, **성능 확보를 전면화한 방향 A가 3순위**다. 제외하기로 한 모델 상태 용어는 세 방향 모두에서 사용하지 않았고, 가중치 업데이트 여부는 방법론에서 설명한다.

---

# 근거 범위와 보수적 해석 원칙

- 내부 사실의 정본: `docs/reviews/professor_feedback_fact_audit_20260907.md`
- 현재 protocol: `confirmatory_v1/confirmatory_protocol_v1.yaml`
- acquisition 구현: `confirmatory_v1/src/confirmatory_v1/protocol_core.py:265-303`
- 외부 제목 근거: [IMAI Lab Publications](https://imai.knu.ac.kr/research/publications)
- DINOv2 근거: [논문](https://arxiv.org/abs/2304.07193), [공식 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [사용 checkpoint revision](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
- 자동차 부싱 연구: [DOI](https://doi.org/10.1145/3788731.3788733), [접근 가능한 abstract record](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3788731.3788733?fields=title,abstract,authors,year,venue,url,externalIds)

이 문서에서 `성능 확보`는 연구 목적을 뜻하며 달성 선언이 아니다. 실제 결과 주장은 두 dataset의 protocol-bounded 평균 nAULC와 class-level 결과로 제한한다. annotation 시간·금액, SOTA, 산업 전반 일반화, 모든 class 개선은 확인되지 않았으며 주장하지 않는다.

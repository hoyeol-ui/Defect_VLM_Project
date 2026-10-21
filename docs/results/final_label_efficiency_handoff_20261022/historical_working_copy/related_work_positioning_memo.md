# Literature-Backed Positioning Memo

> **목적:** 현재 논문의 claim 범위를 유지하면서 학술적 위치, 연구 필요성, 관련 연구와의 공통점·차이, 허용 가능한 기여를 대표 선행연구에 근거해 고정한다.
>
> **문헌 확인 기준:** 2026-08-08 현재 학회·저널·공식 논문 페이지를 우선 확인했다. 아래 목록은 체계적 문헌고찰의 전수 목록이 아니라 네 연구 축을 대표하는 positioning anchor다.
>
> **논문 경계:** 본 연구는 새 selector, 검증된 framework, 산업 benchmark, 비용 절감, 독립 production pool 일반화를 주장하지 않는 **bounded empirical audit**다.

## 1. 포지셔닝 결론

기존 연구는 재현 가능한 AL 평가, object detection 고유의 uncertainty·class balance, VFM/VLM 표현을 이용한 표본 선택, 산업 결함 검사에서의 주석 효율과 human-in-the-loop를 각각 발전시켜 왔다. 그러나 현재 논문이 직접 다루는 질문은 이들 방법을 다시 경쟁시키는 것이 아니다. 평가한 산업 결함 AL 분기에서 **selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety, 사전 최소효과 기준이 같은 판정으로 환원되는지**를 단계별로 확인하고, 관찰되지 않은 연결은 관찰되지 않은 상태로 남기는 것이 핵심이다.

| 연구 축 | 선행연구의 중심 질문 | 현재 논문의 접점 | 현재 논문의 차별점 | 넘지 말아야 할 경계 |
|---|---|---|---|---|
| AL 평가·재현성 | 방법 비교가 seed, 초기 집합, batch, training protocol에 얼마나 민감한가 | EP03의 fixed-set 안정성과 acquisition 반복 구분, Random 기준, CI와 반복 판정 | 재현성을 하나의 값으로 합치지 않고 **어느 단계의 변동성을 반복했는지**를 구분한다 | 표준화 benchmark 또는 재현성 framework 검증으로 부르지 않는다 |
| Object detection AL·rare class | 분류·위치 uncertainty, class balance, rare slice를 acquisition에 어떻게 반영할 것인가 | EP02의 discovery–composition 분리, EP04의 overall–rare metric 충돌 | 희소 클래스 정보를 새 acquisition rule로 해결하기보다 **전체 지표와 rare-class 지표의 판정 불일치**를 보고한다 | 희소 클래스 harm의 일반 법칙, DINO 원인론, rare-class safety 해결을 주장하지 않는다 |
| VFM/VLM 시대 AL | 사전학습 표현이 low-budget selection, diversity, uncertainty, adaptation을 어떻게 바꾸는가 | Frozen DINO·PatchCore 등 사전학습 표현 기반 신호를 selection-stage endpoint로 평가 | 강한 표현이 만드는 selection-stage 양의 방향과 downstream learner 판정을 자동으로 동일시하지 않는다 | VFM/VLM 기반 selector 우월성, VLM adaptation 성능, foundation model 일반화를 주장하지 않는다 |
| 산업 결함 AL | 제한된 expert label로 defect detector를 개선하고 운영자 신뢰·효율을 높일 수 있는가 | 산업 결함 데이터, 발견량·구성·희소 클래스·학습 효용의 분리 평가 | label-efficiency 결과를 단일 성능 곡선으로 끝내지 않고 **선택 집합 구성과 후속 효용의 연결 경계**를 감사한다 | 실제 annotation-time·비용 절감, 생산라인 배포, independent production pool 일반화를 주장하지 않는다 |

## 2. 축 1 — Active Learning 평가와 재현성

### 연구 흐름

Deep AL 평가 연구는 selector의 점 추정치보다 실험 조건과 확률적 변동을 통제하는 문제가 먼저 해결되어야 한다고 지적해 왔다. 이 흐름은 Random baseline, 여러 반복, 초기 labeled set, training randomness, query batch와 실행 환경을 결과 해석의 일부로 취급한다.

| 대표 연구 | 주된 주장 | 현재 논문과 직접 맞닿는 부분 | 현재 논문의 주장 경계 |
|---|---|---|---|
| [Munjal et al., 2022, *Towards Robust and Reproducible Active Learning Using Neural Networks*](https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html) | 동일 조건에서도 uncertainty·diversity·committee 계열의 Random 대비 gain이 일관되지 않을 수 있고, stochasticity와 regularization을 통제한 평가가 필요하다고 보고했다. | Random 비교, 반복 seed, CI, 사전 기준을 함께 보고해야 한다는 근거가 된다. EP03의 acquisition 반복과 EP01의 일반화 제한에 직접 연결된다. | 본 논문이 AL 방법 전반의 재현성 문제를 해결했거나 특정 selector의 불안정성을 보편적으로 입증했다고 쓰지 않는다. |
| [Ji et al., 2023, *Randomness Is the Root of All Evil: More Reliable Evaluation of Deep Active Learning*](https://openaccess.thecvf.com/content/WACV2023/html/Ji_Randomness_Is_the_Root_of_All_Evil_More_Reliable_Evaluation_WACV_2023_paper.html) | 초기 집합, 모델 초기화, query batch, subset selection, 실행 환경 등 서로 다른 randomness source를 분리·통제·해석해야 한다고 제안했다. | 한 fixed selected set의 training-seed 반복과 selector를 다시 실행한 acquisition-seed 반복이 다른 질문이라는 EP03의 위치를 뒷받침한다. | 현재 연구의 반복 설계를 통합적 재현성 표준이나 normative evaluation framework로 제시하지 않는다. |
| [Feng et al., 2022, *ALBench: A Framework for Evaluating Active Learning in Object Detection*](https://arxiv.org/abs/2207.13339) | Object detection AL의 비교 가능성을 위해 동일 training/testing protocol을 제공하는 benchmark framework가 필요하다고 주장했다. | Detector, budget, protocol 차이가 결과 비교를 어렵게 만든다는 점은 본 논문의 no-pooling 원칙과 branch별 native metric 보고를 정당화한다. | 현재 repository는 데이터셋·selector·learner·metric이 이질적이므로 ALBench와 같은 benchmark라고 부를 수 없다. |
| [Werner et al., 2024, *A Cross-Domain Benchmark for Active Learning*](https://papers.nips.cc/paper_files/paper/2024/hash/72de9826cfe582e175520f404eaad473-Abstract-Datasets_and_Benchmarks_Track.html) | AL method의 상대 순위가 domain과 반복 seed에 따라 달라질 수 있으며, 적은 반복만으로는 Random 대비 우열도 뒤집힐 수 있다고 보고했다. | 서로 다른 데이터셋의 native effect를 합산하지 않고, 특정 selector의 일반적 우월성을 금지하는 근거가 된다. | 본 연구의 세 completed branch를 cross-domain benchmark나 method-ranking evidence로 해석하지 않는다. |

### 이 축에서의 논문 위치

본 논문은 기존 재현성 연구의 반복 실험 권고를 완전하게 구현한 benchmark가 아니다. 더 좁게는, **고정 선택 집합을 조건으로 한 learner randomness와 선택 집합 자체가 변하는 acquisition randomness를 분리해 기록하고, 이질적인 분기의 결과를 합산하지 않는 사례 기반 평가**로 위치시킨다.

## 3. 축 2 — Object Detection Active Learning과 rare-class 문제

### 연구 흐름

Object detection AL은 이미지 하나에 여러 객체와 배경이 존재하고 classification과 localization이 동시에 작동하므로, 이미지 분류용 uncertainty를 그대로 확장하기 어렵다. 최근 방법들은 classification·localization uncertainty, augmentation consistency, batch balance, rare slice를 acquisition rule 안에 포함한다.

| 대표 연구 | 주된 주장 | 현재 논문과 직접 맞닿는 부분 | 현재 논문의 주장 경계 |
|---|---|---|---|
| [Choi et al., 2021, *Active Learning for Deep Object Detection via Probabilistic Modeling*](https://openaccess.thecvf.com/content/ICCV2021/html/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.html) | Classification과 localization head의 aleatoric·epistemic uncertainty를 함께 추정해 image informativeness를 구성해야 한다고 제안했다. | Object detection의 selection-stage signal이 단일 분류 점수보다 복합적이라는 배경을 제공한다. 본 논문의 endpoint 분리 필요성과 맞닿는다. | 본 논문이 uncertainty modeling을 개선했거나 detection informativeness의 최적 정의를 제안했다고 쓰지 않는다. |
| [Yu et al., 2022, *Consistency-Based Active Learning for Object Detection*](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Yu_Consistency-Based_Active_Learning_for_Object_Detection_CVPRW_2022_paper.html) | 원본·증강 예측의 consistency, detection subtask, batch-level balanced distribution을 함께 고려하는 CALD를 제안했다. | Informativeness와 selected-set balance가 함께 고려되어야 한다는 흐름은 EP02의 discovery와 composition을 별도 endpoint로 두는 근거가 된다. | EP02를 CALD류 균형 sampling의 우월성 증거나 특정 representation의 category collapse 원인으로 해석하지 않는다. |
| [Kothawade et al., 2022, *TALISMAN: Targeted Active Learning for Object Detection with Rare Classes and Slices*](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4820_ECCV_2022_paper.php) | Overall accuracy가 높더라도 rare·critical slice 성능이 취약할 수 있으며, rare slice를 직접 겨냥한 acquisition과 slice-level AP가 필요하다고 주장했다. | EP04에서 overall mAP와 rare-class macro AP를 병렬 보고하고 rare-class safety를 별도 판정으로 두는 학술적 근거가 된다. | GC10 한 분기의 point estimate를 일반적인 rare-class harm, 산업 안전 위험 또는 TALISMAN식 targeted AL의 검증으로 확대하지 않는다. |
| [Yamani et al., 2024, *Active Learning for Single-Stage Object Detection in UAV Images*](https://openaccess.thecvf.com/content/WACV2024/html/Yamani_Active_Learning_for_Single-Stage_Object_Detection_in_UAV_Images_WACV_2024_paper.html) | 다양한 객체 class의 uncertainty를 집계하고 class별 성능으로 가중해 imbalance를 다루며, 선택 이미지 수 감소가 반드시 labeling cost 감소를 뜻하지 않는다고 지적했다. | EP02·EP04의 class-aware 평가와 5.5의 image-count budget 제한을 동시에 지지한다. | 본 연구는 annotation-time을 측정하지 않았으므로 cost-aware object detection AL 또는 비용 절감을 주장하지 않는다. |

### 이 축에서의 논문 위치

관련 연구가 class balance와 rare slice를 **더 나은 selector 설계의 입력**으로 사용했다면, 본 논문은 이미 관찰된 selection과 learner 결과에서 discovery, composition, aggregate performance, rare-class performance가 **동일한 성공 판정으로 환원되는지**를 감사한다. 이는 방법 제안이 아니라 평가 단위의 분리다.

## 4. 축 3 — Vision Foundation Model / Vision-Language Model 시대의 Active Learning

### 연구 흐름

사전학습된 vision encoder와 VLM은 low-budget 환경에서 강한 표현을 제공하지만, 최근 연구는 그 표현이 기존 AL의 가정을 그대로 보존하지는 않는다고 본다. 이에 따라 representativeness, diversity, uncertainty calibration, implicit class balance, prompt adaptation이 새로운 acquisition 요소로 제안되고 있다.

| 대표 연구 | 주된 주장 | 현재 논문과 직접 맞닿는 부분 | 현재 논문의 주장 경계 |
|---|---|---|---|
| [Xie et al., 2023, *Active Finetuning: Exploiting Annotation Budget in the Pretraining-Finetuning Paradigm*](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Active_Finetuning_Exploiting_Annotation_Budget_in_the_Pretraining-Finetuning_Paradigm_CVPR_2023_paper.html) | Pretraining–finetuning 환경에서 제한된 annotation budget을 위한 subset selection을 별도 문제로 정의하고, pool representativeness와 diversity를 최적화했다. | Frozen representation을 이용한 선택 집합의 구성과 downstream finetuning 효용을 연결해 평가해야 한다는 배경을 제공한다. | 본 논문의 fixed-pool 결과를 ActiveFT류 distribution matching의 검증이나 representation 일반화로 읽지 않는다. |
| [Gupte et al., 2024, *Revisiting Active Learning in the Era of Vision Foundation Models*](https://openreview.net/forum?id=u8K83M9mbG) | DINOv2·OpenCLIP 표현 아래에서 initial pool, diversity, representativeness–uncertainty trade-off를 재검토하고, foundation representation이 기존 AL 관찰을 바꿀 수 있음을 보였다. | Frozen DINO 기반 selection signal을 기존 AL 직관만으로 해석하지 않고 실제 selection endpoint와 learner outcome을 분리해 확인해야 한다는 근거가 된다. | 본 논문은 foundation-model AL의 일반적 성능을 평가하지 않으며 DINO가 endpoint conflict를 일으켰다는 인과 주장도 하지 않는다. |
| [Bang et al., 2024, *Active Prompt Learning in Vision Language Models*](https://openaccess.thecvf.com/content/CVPR2024/html/Bang_Active_Prompt_Learning_in_Vision_Language_Models_CVPR_2024_paper.html) | 기존 AL을 VLM에 단순 적용하면 labeling candidate의 class imbalance로 Random보다 성능이 낮아질 수 있으며, VLM 지식으로 balance를 보정할 수 있다고 보고했다. | 강한 VLM 표현이 자동으로 안전한 selected-set composition이나 downstream utility를 보장하지 않는다는 문제의식과 EP02가 직접 맞닿는다. | 현재 연구는 VLM prompt adaptation, candidate-label balance 해결법, VLM selector 우월성을 검증하지 않는다. |
| [Safaei and Patel, 2025, *Active Learning for Vision Language Models*](https://openaccess.thecvf.com/content/WACV2025/html/Safaei_Active_Learning_for_Vision_Language_Models_WACV_2025_paper.html) | Calibrated entropy와 neighbor-aware uncertainty를 결합해 VLM용 informative sample을 선택하고 여러 classification dataset에서 성능 향상을 보고했다. | VLM/VFM 기반 selection score를 calibration·neighborhood·downstream task와 함께 평가해야 한다는 최근 흐름을 보여준다. | Classification benchmark의 결과를 산업 결함 detection·segmentation으로 일반화하거나 본 논문의 신호를 검증된 VLM AL 방법으로 부르지 않는다. |

### 이 축에서의 논문 위치

VFM/VLM 연구는 더 강한 representation이 새로운 selection opportunity를 만든다는 점을 보여 주지만, 동시에 class imbalance와 adaptation setting이 acquisition outcome을 바꿀 수 있음을 보고한다. 본 논문은 이 흐름을 이어 **강한 selection-stage 신호를 곧바로 learner utility의 대리변수로 취급하지 않고, 선택 결과와 후속 학습 판정을 분리해 기록하는 제한적 사례 연구**로 위치시킨다.

## 5. 축 4 — Industrial Defect Active Learning의 적용 맥락

### 연구 흐름

산업 결함 AL 연구는 전문가 주석 부족, 희소하거나 미세한 결함, 고해상도 이미지, 운영자 피드백을 주된 동기로 삼는다. 다수 연구는 uncertainty·diversity·hybrid sampling으로 적은 label에서 detector 성능을 유지하거나 개선하는 것을 목표로 하며, 최근에는 explainability와 human feedback도 결합한다.

| 대표 연구 | 주된 주장 | 현재 논문과 직접 맞닿는 부분 | 현재 논문의 주장 경계 |
|---|---|---|---|
| [Feng et al., 2017, *Deep Active Learning for Civil Infrastructure Defect Detection and Classification*](https://ascelibrary.org/doi/10.1061/9780784480823.036) | 전문가 label이 제한된 infrastructure defect inspection에서 informative subset을 반복 선택해 detection·classification 성능을 효율적으로 개선할 수 있다고 제안했다. | 산업 결함 영상에서 unlabeled image보다 expert label이 병목이라는 연구 배경을 제공한다. | 현재 논문은 주석 효율의 절대 개선이나 실제 유지보수 효율을 측정하지 않았다. |
| [Lv et al., 2020, *Deep Active Learning for Surface Defect Detection*](https://pubmed.ncbi.nlm.nih.gov/32188066/) | Metallic surface defect detection에 uncertainty sampling과 defect-category별 sampling scale을 적용해 더 적은 labeled data로 목표 성능을 얻는 AL framework를 제안했다. | NEU-DET 등 surface defect 맥락, category-aware selection, Random 대비 downstream detector 평가와 맞닿는다. | 본 연구는 해당 방법을 재검증하거나 fewer labels·동일 성능·annotation saving을 입증하지 않는다. |
| [Nikolakis et al., 2025, *An Explainable Active Learning Approach for Enhanced Defect Detection in Manufacturing*](https://link.springer.com/chapter/10.1007/978-3-031-86489-6_5) | Battery manufacturing에서 LRP explanation과 operator feedback을 결합해 defect model을 갱신하는 human-in-the-loop 접근을 제시했다. | 산업 sampling은 informativeness뿐 아니라 운영자가 확인할 수 있는 선택 근거와 feedback 경로도 중요하다는 점을 보여준다. 본 논문의 inspectable evidence packet·상태 표시와 개념적으로 맞닿는다. | 현재 연구는 human factors, 설명 품질, operator trust, shop-floor feedback을 평가하지 않았다. |
| [Garcia Gonzalez et al., 2026, *Active Learning for Industrial Defect Detection: A Study on Hybrid Sampling Strategies*](https://link.springer.com/article/10.1007/s00170-025-17378-7) | Reflective machined parts의 porosity detection에서 entropy, structured diversity, random exploration을 결합하고 leakage-safe split 아래 label-efficiency를 비교했다. | 불균형과 redundancy를 함께 고려하고 실제 산업 이미지 맥락에서 Random baseline과 downstream performance를 연결하는 최근 사례다. | 본 논문의 fixed pools를 leakage-safe independent production pool로 부르거나 실제 annotation cost·generalization을 입증했다고 쓰지 않는다. |

### 이 축에서의 논문 위치

산업 결함 AL 선행연구가 주로 “적은 label로 성능을 유지·개선하는 sampling strategy”를 중심으로 서술했다면, 본 논문은 **선택 단계에서 무엇이 많이 발견됐는지, 선택 집합이 어떻게 구성됐는지, 선택을 다시 했을 때 결과가 재현됐는지, learner와 희소 클래스 지표가 어떤 판정을 냈는지**를 분리한다. 이 차이는 산업 적용 성능의 우월성이 아니라 evidence granularity의 차이다.

## 6. 기존 연구와 같은 점·다른 점

| 비교 항목 | 기존 연구와 같은 점 | 현재 논문에서 다른 점 |
|---|---|---|
| 기본 문제 | 제한된 annotation budget에서 유용한 표본을 선택한다 | Label efficiency 자체보다 selection signal과 후속 utility의 연결 범위를 감사한다 |
| 비교 기준 | Random sampling을 기본 baseline으로 사용한다 | Random 대비 selection endpoint와 downstream endpoint를 단계별로 분리한다 |
| 다양성·균형 | Diversity, representativeness, class balance를 acquisition에 사용한다 | Discovery gain과 selected-set composition 판정을 같은 성공 지표로 합치지 않는다 |
| 재현성 | 여러 seed와 표준화 protocol의 필요성을 강조한다 | Training-seed 안정성과 acquisition-seed 재현성을 별도 조건부 질문으로 둔다 |
| 희소 클래스 | Rare class 또는 rare slice를 acquisition·평가에 반영한다 | Overall metric과 rare-class metric의 방향 충돌을 별도 안전성 판정으로 보고한다 |
| Foundation representation | Pretrained encoder/VLM을 selection과 adaptation에 활용한다 | Frozen representation 기반 양의 selection signal을 downstream utility의 충분한 근거로 취급하지 않는다 |
| 산업 적용 | 주석 부담과 운영 효율을 문제로 제시한다 | Image-count budget만 관찰했으므로 실제 annotation-time·비용 절감을 주장하지 않는다 |

## 7. 가장 안전한 학술적 포지션

> **본 논문은 평가한 산업 결함 Active Learning 분기에서 selection-stage discovery, selected-set composition, fixed-set training stability, acquisition reproducibility, downstream learner utility, rare-class safety를 분리하고, 이들 사이의 연결이 어디까지 관찰되며 어디서 제한되는지를 정리한 bounded empirical audit이다.**

## 8. 가장 설득력 있는 필요성 서술

Deep AL 평가 연구는 방법의 Random 대비 우열이 seed, 초기 집합, training protocol과 domain에 따라 달라질 수 있음을 보여 주었고, object detection·VFM/VLM 연구는 uncertainty, diversity, class balance, pretrained representation을 결합한 더 정교한 selection signal을 제안해 왔다. 산업 결함 AL 역시 적은 label로 detector 성능을 유지하거나 개선하는 것을 핵심 목표로 삼아 왔다. 그러나 selection-stage의 발견량이나 informativeness가 증가했다는 사실만으로 selected-set composition, acquisition 재현성, downstream 평균 성능, rare-class 성능, 사전 최소효과 기준이 함께 충족되었다고 볼 수는 없다. 따라서 산업 결함 AL에서는 selector의 단일 우승 여부보다 **선택 단계의 신호가 어떤 후속 평가 단위와 일치하고 어디서 불일치하거나 관찰 불가능해지는지**를 구분해 보고할 필요가 있으며, 본 논문은 그 연결을 현재 보유한 평가 분기 안에서 제한적으로 감사한다.

## 9. 가장 방어력 있는 기여 3개

1. **평가 단위 분리:** 평가한 산업 결함 AL 분기에서 discovery, composition, fixed-set learner stability, acquisition reproducibility, aggregate learner performance, rare-class performance, 사전 최소효과 판정을 동일한 success label로 축약하지 않고 별도 평가 단위로 정리한다.
2. **연결 제한의 실증 제시:** MPDD·VisA, NEU-DET, GC10 q20, KSDD2의 관찰을 통해 selection-stage 결과와 후속 판정이 서로 다른 지점에서 불일치할 수 있음을 데이터셋·budget·learner 범위 안에서 제시한다.
3. **증거 상태의 명시:** Post-hoc descriptive arm, fixed-pool perturbation, learner outcome의 `NOT_AVAILABLE`, 사전 기준 미충족을 구분해 실패·부분 양성·관찰 불가능을 서로 바꾸어 쓰지 않는 reporting discipline을 제공한다.

## 10. 삭제 또는 축소해야 할 과장 기여 3개

1. **삭제:** “새로운 selector 또는 VFM/VLM sampling method가 Random보다 일반적으로 우월함을 입증했다.”
2. **삭제:** “선택–학습 translation framework 또는 산업 결함 AL benchmark를 검증했다.”
3. **축소:** “주석 비용과 계산 비용을 절감하고 독립 production pool에 일반화했다”를 “image-count budget과 fixed-pool 평가 분기에서 관찰된 결과를 보고했다”로 낮춘다.

## 11. 제1장·제2장 배치 지침

| 장·절 | 문헌의 역할 | 사용할 축 | 주의점 |
|---|---|---|---|
| 1.1 연구 배경 | 산업 결함 영상의 expert labeling 병목과 pretrained representation 기반 AL의 최근 확장 | 산업 결함 AL; VFM/VLM AL | 비용 절감이 이미 보장된 것처럼 쓰지 않는다 |
| 1.2 연구 문제의 설정 | AL 결과가 seed·protocol·domain에 민감하고 selection metric이 여러 downstream endpoint를 대신하지 못할 수 있다는 문제 제기 | AL 평가·재현성; object detection rare class | 본 연구 결과를 서론에서 미리 보편 법칙으로 선언하지 않는다 |
| 1.3 연구 목적 및 연구 질문 | Selection-stage discovery, composition, reproducibility, learner utility, rare-class safety의 구분 | 네 축 통합 | 새 방법 개발 목적이 아니라 bounded audit 목적을 명시한다 |
| 2.2 객체 탐지 Active Learning | Detection-specific uncertainty, consistency, balance, rare slice | Object detection AL | 방법 성능 순위의 장황한 나열보다 평가 대상 차이를 중심으로 쓴다 |
| 2.3 VLM 및 시각 표현 기반 선택 신호 | Active finetuning, VFM/VLM representation, class balance·calibration | VFM/VLM AL | Classification 결과를 산업 결함 detection·segmentation에 직접 이전하지 않는다 |
| 2.4 Active Learning의 평가와 재현성 | Random baseline, 반복 seed, randomness source, protocol standardization | AL 평가·재현성 | 본 연구를 benchmark로 부르지 않는다 |
| 2.5 본 연구의 위치 | 네 흐름 사이에서 평가 단위 분리와 연결 제한을 다루는 bounded audit | 네 축 통합 | “기존 연구가 다루지 않았다” 대신 “본 논문은 다른 질문에 초점을 둔다”로 쓴다 |

## 12. 참고문헌 앵커 목록

### AL 평가·재현성

1. Munjal, P., Hayat, N., Hayat, M., Sourati, J., & Khan, S. (2022). [Towards Robust and Reproducible Active Learning Using Neural Networks](https://openaccess.thecvf.com/content/CVPR2022/html/Munjal_Towards_Robust_and_Reproducible_Active_Learning_Using_Neural_Networks_CVPR_2022_paper.html). CVPR.
2. Ji, Y., Kaestner, D., Wirth, O., & Wressnegger, C. (2023). [Randomness Is the Root of All Evil: More Reliable Evaluation of Deep Active Learning](https://openaccess.thecvf.com/content/WACV2023/html/Ji_Randomness_Is_the_Root_of_All_Evil_More_Reliable_Evaluation_WACV_2023_paper.html). WACV.
3. Feng, Z., Zhang, S., Takezoe, R., Hu, W., Chandraker, M., Li, L.-J., Narayanan, V. K., & Wang, X. (2022). [ALBench: A Framework for Evaluating Active Learning in Object Detection](https://arxiv.org/abs/2207.13339). AutoML Workshop.
4. Werner, T., Burchert, J., Stubbemann, M., & Schmidt-Thieme, L. (2024). [A Cross-Domain Benchmark for Active Learning](https://papers.nips.cc/paper_files/paper/2024/hash/72de9826cfe582e175520f404eaad473-Abstract-Datasets_and_Benchmarks_Track.html). NeurIPS Datasets and Benchmarks Track.

### Object detection AL·rare class

5. Choi, J., Elezi, I., Lee, H.-J., Farabet, C., & Alvarez, J. M. (2021). [Active Learning for Deep Object Detection via Probabilistic Modeling](https://openaccess.thecvf.com/content/ICCV2021/html/Choi_Active_Learning_for_Deep_Object_Detection_via_Probabilistic_Modeling_ICCV_2021_paper.html). ICCV.
6. Yu, W., Zhu, S., Yang, T., & Chen, C. (2022). [Consistency-Based Active Learning for Object Detection](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/html/Yu_Consistency-Based_Active_Learning_for_Object_Detection_CVPRW_2022_paper.html). CVPR Workshops.
7. Kothawade, S., Ghosh, S., Shekhar, S., Xiang, Y., & Iyer, R. (2022). [TALISMAN: Targeted Active Learning for Object Detection with Rare Classes and Slices Using Submodular Mutual Information](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4820_ECCV_2022_paper.php). ECCV.
8. Yamani, A., Alyami, A., Luqman, H., Ghanem, B., & Giancola, S. (2024). [Active Learning for Single-Stage Object Detection in UAV Images](https://openaccess.thecvf.com/content/WACV2024/html/Yamani_Active_Learning_for_Single-Stage_Object_Detection_in_UAV_Images_WACV_2024_paper.html). WACV.

### VFM/VLM 시대 AL

9. Xie, Y., Lu, H., Yan, J., Yang, X., Tomizuka, M., & Zhan, W. (2023). [Active Finetuning: Exploiting Annotation Budget in the Pretraining-Finetuning Paradigm](https://openaccess.thecvf.com/content/CVPR2023/html/Xie_Active_Finetuning_Exploiting_Annotation_Budget_in_the_Pretraining-Finetuning_Paradigm_CVPR_2023_paper.html). CVPR.
10. Gupte, S. R., Aklilu, J., Nirschl, J. J., & Yeung-Levy, S. (2024). [Revisiting Active Learning in the Era of Vision Foundation Models](https://openreview.net/forum?id=u8K83M9mbG). TMLR.
11. Bang, J., Ahn, S., & Lee, J.-G. (2024). [Active Prompt Learning in Vision Language Models](https://openaccess.thecvf.com/content/CVPR2024/html/Bang_Active_Prompt_Learning_in_Vision_Language_Models_CVPR_2024_paper.html). CVPR.
12. Safaei, B., & Patel, V. M. (2025). [Active Learning for Vision Language Models](https://openaccess.thecvf.com/content/WACV2025/html/Safaei_Active_Learning_for_Vision_Language_Models_WACV_2025_paper.html). WACV.

### 산업 결함 AL

13. Feng, C., Liu, M.-Y., Kao, C.-C., & Lee, T.-Y. (2017). [Deep Active Learning for Civil Infrastructure Defect Detection and Classification](https://ascelibrary.org/doi/10.1061/9780784480823.036). Computing in Civil Engineering.
14. Lv, X., Duan, F., Jiang, J.-J., Fu, X., & Gan, L. (2020). [Deep Active Learning for Surface Defect Detection](https://pubmed.ncbi.nlm.nih.gov/32188066/). Sensors, 20(6), 1650.
15. Nikolakis, N., Catti, P., & Alexopoulos, K. (2025). [An Explainable Active Learning Approach for Enhanced Defect Detection in Manufacturing](https://link.springer.com/chapter/10.1007/978-3-031-86489-6_5). Lecture Notes in Mechanical Engineering.
16. Garcia Gonzalez, D., Nascimento, R., Rocha, C. D., Silva, M. F., & Filipe, V. (2026). [Active Learning for Industrial Defect Detection: A Study on Hybrid Sampling Strategies](https://link.springer.com/article/10.1007/s00170-025-17378-7). International Journal of Advanced Manufacturing Technology.

## 제1장에서 바로 쓸 문제의식 문장 5개

1. Deep Active Learning의 성능 비교는 selector 자체뿐 아니라 초기 labeled set, 학습 seed, query batch와 실행 protocol에 민감하므로, 단일 실행의 Random 대비 우위만으로 방법의 안정성을 판단하기 어렵다.
2. Object detection Active Learning에서는 이미지별 결함 발견량이 증가하더라도 선택 집합의 범주 구성과 희소 클래스 성능이 함께 보존된다고 가정할 수 없다.
3. Vision foundation model과 vision-language model은 강한 사전학습 표현을 selection signal로 제공하지만, 그 신호의 양의 방향이 특정 downstream learner의 효용을 자동으로 보장하는지는 별도로 확인해야 한다.
4. 산업 결함 Active Learning은 제한된 expert annotation을 효율적으로 사용하는 문제를 다루지만, 선택 이미지 수의 감소와 실제 주석 시간·비용의 감소는 동일한 명제가 아니다.
5. 따라서 산업 결함 Active Learning의 평가는 selection-stage discovery, selected-set composition, acquisition reproducibility, downstream utility와 rare-class safety를 분리하고 그 연결 범위를 명시할 필요가 있다.

## 제2장에서 바로 쓸 관련연구 대비 문장 5개

1. 기존 재현성 연구가 seed와 protocol 변화에 따른 AL 방법 순위의 변동을 분석했다면, 본 연구는 한 산업 결함 평가 패키지 안에서 fixed-set training stability와 acquisition reproducibility를 구분한다.
2. 기존 object detection AL 연구가 uncertainty, consistency와 class balance를 새로운 acquisition rule에 통합했다면, 본 연구는 discovery와 selected-set composition이 실제 평가에서 같은 판정을 만드는지를 별도 endpoint로 확인한다.
3. Rare-slice AL 연구가 희소 범주를 직접 겨냥한 선택 전략을 제안했다면, 본 연구는 새로운 rare-aware selector를 제안하지 않고 aggregate metric과 rare-class metric의 방향을 병렬 보고한다.
4. VFM/VLM 기반 AL 연구가 pretrained representation을 이용해 low-budget selection과 adaptation 성능을 개선하는 데 초점을 두었다면, 본 연구는 frozen representation 기반 selection signal과 downstream learner 판정의 연결 한계를 감사한다.
5. 산업 결함 AL 연구가 label efficiency와 detector 성능 개선을 중심으로 평가해 왔다면, 본 연구는 평가한 분기에서 발견량·구성·재현성·학습 효용·희소 클래스 판정이 동일한 success label로 환원되지 않는지를 정리한다.

## 심사위원이 물을 가능성이 높은 비교 질문 5개와 답변 포인트

| 예상 질문 | 답변 포인트 |
|---|---|
| **“재현성 연구가 이미 많은데 EP03은 무엇이 다른가?”** | 기존 연구의 일반 권고를 새로 제안하는 것이 아니다. NEU-DET 한 사례에서 결과 관찰 후 선택된 favorable fixed set의 training-seed 반복과 selector를 다시 실행한 acquisition-seed 반복이 다른 조건부 질문임을 `POSTHOC_DESCRIPTIVE` 지위와 함께 보여 주는 bounded evidence다. |
| **“TALISMAN이나 class-balanced AL과 비교하면 새로운 rare-class 방법이 없지 않은가?”** | 맞다. 새 rare-aware selector는 기여가 아니다. 기여는 GC10 q20에서 overall metric과 rare-class metric의 상반된 방향을 별도 판정으로 보존해, aggregate improvement를 rare-class safety로 번역하지 않는 평가 방식이다. |
| **“VFM/VLM AL 선행연구와 비교할 때 왜 method paper가 아닌가?”** | 본 연구는 DINO·PatchCore 등 pretrained representation을 selection signal로 사용하지만, prompt learning이나 active finetuning algorithm을 제안하지 않는다. 질문은 representation 기반 selection positivity가 관찰된 후 learner·composition·rare-class endpoint가 어떤 판정을 냈는지다. |
| **“산업 결함 AL 연구들이 이미 label efficiency를 보였는데 이 논문이 필요한가?”** | 기존 연구의 label-efficiency 결과를 부정하지 않는다. 다만 image-count budget, discovery, composition, learner performance, rare-class metric과 practical threshold가 서로 다른 평가 단위이므로, 산업 적용 주장을 하기 전에 이 연결을 분리해 보고할 필요가 있다는 제한적 실증 근거를 제공한다. |
| **“왜 benchmark나 framework라고 부르지 않는가?”** | 데이터셋·task·selector·budget·learner·metric이 이질적이고 공통 protocol, independent production pool, 모든 branch의 counterfactual learner outcome이 없기 때문이다. 따라서 현재 결과는 method ranking이나 generalization benchmark가 아니라 evaluated branches에 한정된 empirical audit다. |

# 지도교수 피드백 Fact Audit — 2026-09-07

> 목적: 9월 9일 제목 결정에 필요한 사실을 먼저 확정하고, 9월 말 심사용 논문에 필요한 최소 보강과 이번 논문에서 버릴 아이디어를 분리한다. 이 문서는 코드·설정·결과·논문 원문을 읽어 작성한 상태 감사이며, 새 실험 설계서가 아니다.
>
> 감사 기준 시점: 2026-09-07 (Asia/Seoul)  
> 판정: **보강 후 유지 가능**
>
> 2026-09-07 미팅 전사 교차확인 반영: `C:/Users/user/.codex/attachments/31cc23dd-cd71-4291-86bf-979ac2e881af/pasted-text.txt`. STT 오류가 있는 전사이므로 교수 발언의 방향과 요구 강도를 확인하는 근거로 사용하고, 모델·실험의 기술적 사실은 코드/config/result를 우선했다. `C:/Users/user/.codex/attachments/18c4c33b-e7fc-469a-9a4f-5498d615d516/pasted-text.txt`는 미팅 후 해석문이므로 독립 사실 근거가 아니라 교차검토 자료로만 사용했다.

## 1. Executive Summary

### 지금 당장 제목 결정 전에 알아야 할 것

1. 최신 DOCX와 육안 확인한 최신 PDF의 표지·국문초록 제목은 `산업 결함 Active Learning에서 표본 선택 신호와 학습 성능의 연결 — 이질적 역사 증거의 단계적 분석과 Random-Frozen DINO 통제 검증`으로 일치한다. 다만 PDF metadata에는 `산업 결함 Active Learning에서 선택 신호는 언제 학습 성능으로 이어지는가`라는 이전 제목이 남아 있다.
2. 현재 제목은 `산업 결함`이라는 환경, `Active Learning`과 `Frozen DINO`라는 방법은 보이지만, **정확한 Task인 object detection**, **제한된 라벨 조건**, **탐지 성능 확보라는 공학적 목적**이 제목에 직접 나타나지 않는다. 대신 `연결·신호·분석`이 중심이어서 9월 7일 지도교수 지적과 충돌한다.
3. 가장 작은 제목 수정 방향은 다음과 같다.
   - 권고안: **`제한된 라벨 환경에서 산업 결함 객체 탐지 성능 확보를 위한 Frozen DINO 표현 기반 능동학습 표본 선택`**
   - 결과 경계를 더 드러내는 안: **`제한된 라벨 환경의 산업 결함 객체 탐지를 위한 능동학습 표본 선택: Frozen DINO 표현 기반 선택의 학습 효용 통제 검증`**
   - 어느 안을 택하든 `DINO sampling` 대신 **Frozen DINOv2-small 표현 + cosine greedy farthest-first**라고 방법을 설명해야 한다.
4. 논문을 한 문장으로 설명할 때는 다음이 현재 증거와 가장 잘 맞는다.  
   **“제한된 라벨 예산으로 산업 결함 객체 탐지기를 학습할 때, Frozen DINO 표현 기반 표본 선택이 Random보다 detector의 budget-averaged 성능을 얼마나 안정적으로 확보하는지 통제 검증하고, 그 이득이 모든 클래스에 보장되지 않는 경계를 밝힌 연구다.”**
5. 전사에서 교수는 제목 제출 기간이 9월 9~11일이라고 확인하면서 **제목을 나중에 변경할 수 있는지 행정적으로 반드시 확인해 알려 달라**고 요청했다. 공개된 경북대 2026-2학기 공지에는 신청서의 논문 제목은 **가제목이며 제출 이후 수정 가능**하다고 명시되어 있다. 다만 데이터사이언스대학원 내부 제출 마감과 실제 수정 절차는 소속 행정실에 한 번 더 확인해야 한다.

### 현재 이미 답할 수 있는 교수 질문

- 정확한 Task는 GC10-DET·NEU-DET의 **다중 클래스 object detection**이고 learner는 YOLOv8n이다.
- 현재 모델은 DETR 계열의 DINO detector나 Grounding DINO가 아니라 **Meta DINOv2의 ViT-S/14 이미지 backbone**인 `facebook/dinov2-small`이다.
- DINO는 selector 전체가 아니라 **동결된 feature extractor**다. 실제 acquisition은 L2-normalized 384-D pooled feature에서 cosine distance를 쓰는 greedy farthest-first다.
- 현재 440-run은 사람의 육안 선택이 아니라 완전 자동 선택이다. MPDD·VisA selection-only audit도 자동 선택 후 GT를 사후 평가에만 사용했다.
- train/validation/test에 대응하는 실제 구조와 수는 GC10 `1,605/229/458`, NEU `1,259/180/359`이고, validation은 `best.pt` 선택, locked final은 최종 평가에만 사용했다.
- label budget은 train/acquisition pool의 약 10%에서 시작해 원래 pool의 약 5%씩 다섯 번 추가하여 약 35%까지 간다.
- primary 결과는 한 endpoint의 “best”가 아니라 **10~35%의 여섯 budget 전체 locked-final mAP50-95 curve의 nAULC**다. GC10은 `+0.005812`, NEU는 `+0.005782`였으나 NEU worst-class는 악화됐다.

### 아직 답할 수 없는 교수 질문

- 동일 split·동일 YOLO lifecycle의 **100% labeled full-data upper bound**는 없다. 판정은 `Full-data upper-bound missing`이다.
- DINO feature가 GC10과 NEU의 결함 의미를 일반적으로 보존한다는 직접 정량 근거는 충분하지 않다. GC10의 과거 representation 진단은 일부 있으나 실패 경계가 중심이고, NEU의 kNN/silhouette/same-vs-different-class 분석은 없다.
- 자동차 부싱 논문의 full text를 확보하지 못해 정확한 uncertainty 식, 독립 downstream detector, 비교 baseline 명칭은 `확인되지 않음`이다.
- canonical branch에는 과거 Qwen raw response/score 파일이 없어, 새 추론 없이 현재 sample ID에 즉시 결합할 수 있는 상태가 아니다.

### 심사용 논문 제출 전 최소 보강 필요 항목

1. 제목과 서론 첫 목적 문장을 산업 문제→Task→목적→방법 순으로 정렬하고, 연구 변천사는 배경/한계로 내린다. 본문 자동 수정은 이번 audit에서 하지 않았다.
2. 가능하면 full-data YOLO upper bound를 최소 4 runs(2 datasets × 2 사전 고정 learner seeds)로 산출한다. 현재 20-seed learner 분포와 직접 맞추려면 40 runs가 필요하다.
3. 기존 DINO cache와 GT를 이용해 학습 없이 same-class/different-class cosine distance, kNN label agreement, silhouette을 산출한다. canonical release에 cache 본체가 없으므로 원 실행 환경의 cache A를 회수하거나 동일 sealed checkpoint로 feature inference만 재생성해야 한다.
4. 전사에서 교수는 Random과 Frozen DINO 외에 기존 sampling 방법을 추가해야 한다고 명시적으로 요구했다. 현재 DINO farthest-first와 기하적으로 중복되지 않는 detector-aware uncertainty 등 **최소 한 가지 distinct baseline**을 P1 필수 후보로 정해야 한다. 완전 동일 protocol이면 새 학습 cell은 200개다.

### 미팅 전사로 새로 확정된 요구 강도

| 미팅 발언의 요지 | 전사 기반 판정 | 이 audit에서의 처리 |
|---|---|---|
| 분석·비교는 연구 목적이 아니라 과정이며, 환경·Task·목적·방법이 제목에 보여야 함 | 명시적 필수 | P0 유지 |
| DINO pretraining data와 현 결함 domain의 유사성/feature 타당성을 확인해야 함 | 명시적 필수 | P0 fact sheet + P1 정량 audit |
| AL을 쓰지 않고 모든 data를 쓴 YOLO 성능이 필요함 | 직접 질문 및 보강 권고 | P1 full-data upper bound |
| Random/DINO 외 기존 sampling 방법을 추가해야 함 | 전사상 “무조건”에 가까운 명시적 요구 | 조건부가 아니라 P1 필수로 상향 |
| 대표 이미지 육안 확인만으로 유사성/차이를 넘기면 안 됨 | 명시적 문제 제기 | post-hoc visualization과 실제 selection을 분리하고 정량근거 추가 |
| train/validation/test, 사용 이미지 수/%처럼 일반적인 용어 사용 | 즉시 수정 가능한 표현 요구 | P0 문서/발표 용어 |
| Domain Generalization, open-set, text-input detector | novelty를 만드는 예시/브레인스토밍 | P2 유지; 이번 제출 연구로 전환하지 않음 |

전사 전체에는 “현재 연구를 폐기하라”는 지시가 없다. 반대로 지금까지의 실험을 학위논문 수준의 공학적 질문과 비교 구조로 닫기 위해 무엇이 부족한지를 지적한 미팅으로 해석하는 것이 발언 흐름과 일치한다.

행정 일정은 [경북대학교 2026 학사력](https://www.knu.ac.kr/wbbs/wbbs/user/yearSchedule/index.action?menu_idx=4)에 학위논문 제출예정자 신청 `2026-09-09~09-11`, 심사용 논문 접수 `2026-09-30~10-02`로 공개돼 있다. [경북대 전자전기공학부의 2026-2학기 제출예정자 공지](https://gsee.knu.ac.kr/content/notice.html?fidx=6075&gtid=grad_notice&opt=&page=26&pg=vv&sword=)는 대학원 공통 신청 안내를 첨부하면서 “신청서에 작성한 논문제목은 가제목으로 제출 이후 수정 가능”이라고 명시한다. 해당 학부의 내부 접수는 9월 9~10일이므로, 데이터사이언스대학원 자체 내부 마감과 수정 방법은 소속 행정실 확인이 필요하다.

## 2. 현재 연구 정본 기록

| 항목 | 확인 결과 | 근거 |
|---|---|---|
| 정본 worktree | `C:/Users/user/Desktop/vlm/Defect_VLM_Project_thesis_release` | 로컬 경로 확인 |
| branch | `al-selector` | `git branch --show-current` |
| HEAD | `55342a6251014fbf7e54b823a7e816737fd52130` | `git rev-parse HEAD`; 2026-08-28 17:01:04+09:00 |
| remote 상태 | `al-selector...origin/al-selector`, ahead/behind 표시 없음 | `git status --short --branch` |
| audit 전 미커밋 변경 | 결과 figure 6개와 `plot_final_results.py` 1개 수정 | 아래 목록; 이번 audit에서 손대지 않음 |
| protocol/config | `confirmatory_v1/confirmatory_protocol_v1.yaml`, `confirmatory_v1_1/confirmatory_protocol_v1_1.yaml` | `THESIS_FILE_INDEX.md:28-38` |
| 440 training runner | `confirmatory_v1_1/full_run/src/run_full_440_orchestrator.py`, cell runner `run_full_cell.py` | `THESIS_FILE_INDEX.md:35`; orchestrator source |
| locked-final runner | `confirmatory_v1_1/final_analysis/src/authorize_final_evaluation.py`, `run_final_evaluation.py` | final-analysis source directory |
| split manifests | `confirmatory_v1/data_freeze/*_acquisition_universe_blind.csv`, `*_development_locked_manifest.csv`, `*_final_locked_manifest.csv` | `confirmatory_protocol_v1.yaml:35-104` |
| acquisition/learner seed | 데이터셋별 acquisition 10개, acquisition마다 learner A/B 2개 | `confirmatory_protocol_v1.yaml:228-260`; `data_freeze/seed_manifest.csv` |
| DINO checkpoint | `facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, model SHA-256 `ae1e99fc…d14be1` | `confirmatory_protocol_v1.yaml:174-218` |
| YOLO checkpoint | YOLOv8n, SHA-256 `f59b3d83…fc83b36` | `confirmatory_protocol_v1.yaml:262-326` |
| 440-run/locked-final 정본 | `confirmatory_v1_1/full_run/`, `confirmatory_v1_1/final_analysis/` | `THESIS_FILE_INDEX.md`; `final_analysis/FINAL_CONTROLLED_ANALYSIS.md:11-17` |
| 공개 결과 사본 | `results/thesis_final/` | `results/thesis_final/README.md:3-33`; authoritative source는 위 final-analysis package |
| 최신 논문 DOCX | `C:/Users/user/Desktop/vlm/0__260829/산업결함_AL_석사학위논문_실험결과 반영 수정버전 초안.docx` | 수정 2026-08-29 19:18:27; SHA-256 `24D400…696F1` |
| 최신 검증 PDF | `C:/Users/user/Desktop/vlm/dvlm-20260822T112232Z-1-001/dvlm/tmp/thesis_80p_20260829/render_v3_verified3/산업결함_AL_석사학위논문_본심사최종패치_v3.pdf` | 수정 2026-08-29 19:10:34; 65쪽; SHA-256 `2CD1E6…7EB96` |
| 최신 요약본 | `final_analysis/FINAL_RESULT_BRIEF.md` | `THESIS_FILE_INDEX.md:11` |
| 발표자료 | workspace 아래 `.ppt/.pptx` 없음 | 2026-09-07 재귀 파일 검색; `확인되지 않음`이 아니라 **해당 workspace에는 없음** |

감사 시작 전 존재하던 변경사항은 다음과 같다.

- `confirmatory_v1_1/final_analysis/figures/confirmatory_classwise_results.{pdf,svg}`
- `confirmatory_v1_1/final_analysis/figures/confirmatory_classwise_results_600dpi.png`
- `confirmatory_v1_1/final_analysis/figures/confirmatory_primary_results.{pdf,svg}`
- `confirmatory_v1_1/final_analysis/figures/confirmatory_primary_results_600dpi.png`
- `confirmatory_v1_1/final_analysis/src/plot_final_results.py`

이 audit은 reset, checkout, stash, commit을 하지 않았으며 위 파일을 수정하지 않았다.

## 3. MASTER TABLE

| 교수님 질문/지적 | 현재 실제 상태 | 증거 위치 | 판정 | 필요한 조치 | 우선순위 |
|---|---|---|---|---|---|
| 연구 목적이 무엇인가 | 실제 공학 문제는 제한된 이미지 라벨 예산에서 산업 결함 detector 성능을 확보하는 것. 문서 전면은 신호–성능 관계 분석이 지배 | 최신 DOCX P0098-P0102, P0132-P0144, P0506-P0511 | 일부 확인 | 제목·서론 첫 문단의 목적 우선순위 수정 | P0 |
| Task와 목적이 제목에 보이는가 | `object detection`, 제한 라벨, 탐지 성능 목적이 직접 없음 | DOCX P0001-P0003, P0501; PDF 표지/국문초록 | 설명과 구현 불일치 | 목적 중심 제목 확정 | P0 |
| 현재 제목이 문서 간 일치하는가 | DOCX와 PDF visible title은 일치, PDF metadata는 다른 제목 | 최신 DOCX; 최신 PDF page 1/61 및 `pdfinfo` | 일부 확인 | PDF metadata도 최종 제목으로 갱신 | P0 |
| 9월 9~11일 접수 후 제목 변경이 가능한가 | 경북대 2026-2학기 공지는 신청 제목을 가제목으로 보고 제출 후 수정 가능하다고 명시. 데이터사이언스대학원 내부 절차는 별도 확인 필요 | 미팅 전사; 경북대 2026-2학기 공지 | 일부 확인 | 공식 공지 링크와 함께 교수에게 우선 회신하고 소속 행정실에 수정 절차/내부 마감 확인 | P0 |
| DINO가 무엇으로 사전학습됐는가 | DINOv2 ViT-S/14, LVD-142M, annotation 없이 DINO+iBOT+KoLeo; ViT-g teacher에서 distilled | local protocol; Meta model card | 확인됨 | 논문 방법·발표에 2문장 명시 | P0 |
| DINO가 selector인가 | feature extractor이고 selection은 cosine greedy farthest-first | `frozen_dino_ff_v1.py:93-127`, `protocol_core.py:265-303` | 확인됨 | `DINO sampling` 표현 교정 | P0 |
| Core-set/k-center와 다른가 | 현재 rule은 metric-space greedy k-center/Core-set 계열과 매우 가까움 | `protocol_core.py:265-303`; protocol `:211-217` | 확인됨 | 별도 Core-set arm을 새 방법처럼 중복 추가하지 않음 | P0 |
| 사람 눈으로 다른 이미지를 골랐는가 | 코드는 current/MPDD/VisA 모두 자동. 그러나 미팅에서는 대표 이미지 육안 확인과 실제 selection이 섞여 설명되어 교수가 수동 선택으로 이해함 | MPDD script `:139-186,278-280`; VisA `:99-143,224-225`; 미팅 전사 | 설명과 구현 불일치 | 다음 설명에서 자동 selection과 post-hoc sanity check를 명시적으로 분리 | P0 |
| 100% data 성능은 얼마인가 | 같은 split/lifecycle의 결과 없음 | full-data script 존재, current result 없음 | 근거 부족 | P1에서 최소 4-run upper bound 검토 | P1 |
| Random 외 baseline은 있는가 | 현재 protocol 결과는 Random과 FrozenDINO뿐. 다수 legacy code/ledger는 비동일 protocol | protocol `:148-150`; `CODE_INDEX.md`; evidence ledger; 미팅 전사 | 일부 확인 | 교수의 명시적 요구에 따라 distinct baseline 최소 1개를 사전 고정 | P1 |
| 결과가 몇 %일 때인가 | primary는 10~35% 전체 nAULC; R5는 GC10 561장/34.953%, NEU 441장/35.028% | budget config `:125-146`; budget/result CSV | 확인됨 | 모든 그림 caption에 round/count/% 병기 | P0 |
| best와 final이 같은가 | `best.pt`는 각 budget에서 validation으로 고른 checkpoint; `final budget`은 R5 최대 labeled count; 서로 다름 | protocol `:313-323`; DOCX P0201, P0247, P0284 | 확인됨 | 용어 표준화. 현재 최신 본문에서 직접 혼동은 발견되지 않음 | P0 |
| train/validation/test 구조 | acquisition/development/locked-final의 70/10/20 고정 구조 | protocol `:35-104` | 확인됨 | 발표에서는 train pool/validation/locked test로 병기하고 역할 설명 | P0 |
| initial/query rule | 동일 R0, train pool 10% initial, 원 pool 5%씩 5회, GT 미사용 | protocol `:119-171` | 확인됨 | 실제 장수와 rounding rule 제시 | P0 |
| DINO feature의 도메인 의미 | GC10 historical representation metric 일부, NEU 직접 분석 없음; current 결과는 representation+rule 묶음 효과 | evidence ledger E010-E023; current selection results | 일부 확인 | 무학습 post-hoc feature audit | P1 |
| 옛 Qwen 텍스트를 바로 쓸 수 있는가 | code와 aggregate ledger는 있으나 canonical branch에 raw responses/scores 없음 | `scripts/01_score_generation`, `scripts/03_analysis`, evidence ledger E001-E005 | 근거 부족 | raw artifact를 찾기 전 새 결합 보류 | P1 |
| 부싱 논문과 거의 같은 Task인가 | 제한 라벨 산업 defect object detection+AL이라는 problem/task는 유사. VLM 학습·multimodal uncertainty라는 방법/novelty는 다름 | DOI/Crossref/abstract; 현재 protocol | 일부 확인 | 공통 문제와 방법 차이를 관련연구에서 명시 | P1 |
| DG/open-set/new multimodal detector | 기존 연구 연속성을 크게 벗어남 | 지도교수 미팅 범위 | 확인됨 | 후속연구로만 기록 | P2 |

## 4. Title & Problem Definition Audit

### 4.1 현재 정확한 최신 제목

최신 DOCX(P0001-P0003, P0501)와 최신 PDF visible page 1/61에서 확인한 제목은 다음과 같다.

**국문 주제목**  
`산업 결함 Active Learning에서 표본 선택 신호와 학습 성능의 연결`

**국문 부제**  
`이질적 역사 증거의 단계적 분석과 Random-Frozen DINO 통제 검증`

**영문 제목**  
`Linking Acquisition Signals to Downstream Learning in Industrial Defect Active Learning`

**영문 부제**  
`A Stage-Wise Analysis of Heterogeneous Historical Evidence and a Random-Frozen DINO Controlled Validation`

문서 불일치가 하나 있다. 최신 PDF의 metadata Title은 `산업 결함 Active Learning에서 선택 신호는 언제 학습 성능으로 이어지는가`, Subject는 `이질적 역사 증거와 Random-Frozen DINO 통제 검증`이다. visible title과 국문초록 제목은 DOCX와 일치하므로 **현재 내용상 정본은 visible/DOCX 제목**이지만, 최종 PDF export 때 metadata를 함께 고쳐야 한다.

### 4.2 교수님 기준으로 제목 분해

| 요소 | 현재 제목에 있는가 | 감사 판단 |
|---|---|---|
| Task | 부분적 | `산업 결함 Active Learning`만 있어 defect classification/anomaly detection/object detection 중 무엇인지 제목만으로 알 수 없다. 실제 Task는 object detection이다. |
| 환경/문제 조건 | 부분적 | `산업 결함`은 있으나 제한된 라벨/적은 학습 이미지 조건이 없다. |
| 연구 목적 | 부족 | 탐지 성능 확보·개선이라는 목적보다 `신호와 성능의 연결`이라는 분석 질문이 전면에 있다. |
| 핵심 방법 | 있음 | Active Learning, Random-Frozen DINO 통제 검증이 보인다. 다만 실제 방법은 DINO representation + farthest-first다. |
| 기여/최종 메시지 | 부분적 | 증거 경계를 분석했다는 메시지는 보이지만, 왜 산업적으로 필요한 연구인지는 제목만으로 약하다. |

**Flag:** 제목의 중심어가 `연결`, `신호`, `분석`, `검증`이다. 이들은 실험 과정과 증거 해석에는 정확하지만, 지도교수가 요구한 공학적 연구 목적을 대신하고 있다. `[환경/문제 조건] + [Task/목적] + [핵심 방법론]` 기준에서 **Task·조건·목적이 빠진 상태**다.

### 4.3 제목 결정 권고

현재 연구를 바꾸지 않고 문제를 고치는 순서는 다음이다.

1. `제한된 라벨 환경`을 명시한다.
2. Task를 `산업 결함 객체 탐지`로 고정한다.
3. 목적을 `탐지 성능 확보` 또는 `학습 효용 검증`으로 명시한다.
4. 방법은 `Frozen DINO 표현 기반 능동학습 표본 선택`으로 쓴다.
5. 역사 증거·관계 분석은 필요하면 부제로 내린다.

가장 목적 중심인 권고안은 다음과 같다.

> **제한된 라벨 환경에서 산업 결함 객체 탐지 성능 확보를 위한 Frozen DINO 표현 기반 능동학습 표본 선택**

현재 논문의 혼합·제한 결과까지 더 정직하게 보존하는 대안은 다음과 같다.

> **제한된 라벨 환경의 산업 결함 객체 탐지를 위한 능동학습 표본 선택: Frozen DINO 표현 기반 선택의 학습 효용 통제 검증**

`라벨 효율 향상`이라는 표현은 실제 annotation 시간·금전 비용을 측정하지 않았으므로 제목에서 단정하지 않는 편이 안전하다. 현재 입증된 것은 **image-count budget 10~35% 구간의 detector 성능**이다.

### 4.4 서론·초록의 산업적/공학적 문제 정의

산업 문제 정의 자체는 존재한다.

- 최신 DOCX P0098: “산업 현장에서 … 자동으로 탐지하려면 … 충분한 학습 데이터가 필요하다. 그러나 산업 결함 데이터는 … 희소하고 … 불균형하다. … annotation의 형태와 부담도 서로 다르다.”
- P0100: “Active Learning(AL)은 … 모든 데이터를 한 번에 라벨링하는 대신 … 제한된 라벨 예산을 효율적으로 사용하는 방법이다.”
- P0101: “결함 탐지 학습에 더 가치 있는 이미지를 먼저 선택할 수 있다면 동일한 이미지 수로 더 나은 detector를 학습할 수 있다.”
- 국문초록 P0506: “산업 결함 탐지에서 Active Learning은 제한된 라벨 예산으로 학습 가치가 높은 이미지를 선택하는 방법…”

따라서 질문 “왜 제한된 labeled data가 문제인가?”에는 **예**라고 답할 수 있다. 다만 P0102부터 곧바로 “연결”과 다섯 단계 분석으로 중심이 옮겨지고, 국문초록 P0507도 historical evidence 조직을 먼저 길게 설명한다. 처음 읽는 사람은 산업 문제를 한 문장으로 이해할 수 있지만, **연구 목적이 그 문제 해결보다 evidence audit로 읽힐 위험이 크다.**

### 4.5 개인 연구 변천사가 주된 문제 정의가 된 위치

- P0132 제목: `연구의 출발점과 재구성`
- P0133: 현재 구조로 처음부터 시작하지 않았음을 전면 진술한다.
- P0134: “초기 연구의 목적은 VLM을 활용하여 … 새로운 표본 선택 기준을 만드는 것이었다.”
- P0137: 질문이 “어떤 selector가 Random보다 더 좋은가?”에서 “좋아 보이는 신호는 어떤 조건에서 실제 학습 성능으로 이어지는가?”로 재구성됐다고 쓴다.
- P0138-P0139: historical evidence와 사후적 다섯 단계 조직을 연구 구성의 중심으로 둔다.
- 통합 RQ P0144: selection-stage 신호와 discovery/composition/reproducibility/downstream performance의 “관계”를 묻는다.

정확히 “Random을 못 이겨서 시작했다”라는 단일 문장은 최신 본문에서 발견되지 않았다. 그러나 P0133-P0139는 독자가 그렇게 해석할 수 있는 연구 변천 서사다. 이 내용은 삭제할 필요까지는 없지만, **산업 문제→연구 목적→방법→연구 질문 다음의 배경/연구 경과**로 위치를 낮춰야 한다.

## 5. DINO Fact Sheet

### 5.1 2분 설명용 정리

| 질문 | 확인된 사실 |
|---|---|
| 어떤 DINO인가 | `facebook/dinov2-small`; DINOv2 ViT-S/14, registers 없는 base backbone |
| 다른 DINO와 혼동 여부 | DETR 기반 DINO detector 및 Grounding DINO가 아님 |
| library/load | Hugging Face Transformers `AutoImageProcessor`, `AutoModel`; local snapshot only |
| exact revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| checkpoint hash | `model.safetensors` SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| backbone | ViT-S/14, 12 transformer layers, 6 heads, 21M parameters, patch 14, embedding 384 |
| pretraining | label/annotation을 사용하지 않은 self-supervised DINOv2 계열; LVD-142M 142M images |
| objective | DINO multi-crop self-distillation + iBOT masked-image modeling + KoLeo CLS regularization; ViT-S는 frozen ViT-g teacher에서 distilled |
| 현재 freeze | `model.eval()`, `torch.inference_mode()`, gradient/fine-tuning 없음; FP32, autocast/TF32 off |
| input | RGB, shortest edge 256 bicubic resize, 224×224 center crop, 1/255 rescale, ImageNet mean/std normalization |
| feature | Hugging Face `outputs.pooler_output`; 384-D float32; row-wise L2 normalization |
| acquisition | 현재 cumulative labeled/reference set까지의 최대 cosine similarity를 구해 `1-max cosine`이 가장 큰 표본을 하나씩 추가하는 greedy farthest-first |
| 사람 개입 | 없음 |

로컬 근거는 `confirmatory_v1/confirmatory_protocol_v1.yaml:174-218`과 `confirmatory_v1/src/confirmatory_v1/frozen_dino_ff_v1.py:93-127`이다. 모델의 외부 pretraining 사실은 [Meta DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md), [Meta official backbone code](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py), [DINOv2 paper](https://arxiv.org/abs/2304.07193)에서 확인했다. exact HF snapshot은 [revision `ed25f3a…`](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056), architecture config는 [Hugging Face config](https://huggingface.co/facebook/dinov2-small/blob/main/config.json)에서 확인 가능하다.

### 5.2 Representation과 acquisition rule의 분리

**Representation**  
각 이미지를 frozen DINOv2-small ViT-S/14의 `pooler_output` 384-D vector로 나타내고 row L2-normalize한다.

**Acquisition rule**  
각 후보 (x)에 대해 현재 reference set (R)과의 최소 cosine distance와 같은

`d(x,R) = 1 - max_{r in R} cosine(z_x, z_r)`

를 계산한 뒤 가장 먼 후보를 하나 고르고 즉시 (R)에 추가한다. 동점은 `1e-12` tolerance 안에서 `sample_id` 오름차순이다. 구현은 `protocol_core.py:265-303`, freeze 명세는 protocol `:211-217`이다.

따라서 정확한 설명은 다음이다.

> 현재 방법은 단순한 “DINO sampling”이라기보다 **frozen DINOv2-small pooled representation + cosine greedy farthest-first acquisition** 방식이다.

clustering이나 class prediction을 수행하지 않는다. 알고리즘 구조는 metric representation에서 labeled set을 중심으로 covering radius를 줄이는 **greedy k-center/Core-set 계열**과 매우 가깝다. 따라서 동일 feature·동일 distance로 `Core-set` arm을 별도로 추가하면 이름만 다른 중복 비교가 될 가능성이 높다. 별도 baseline이 필요하다면 detector uncertainty처럼 acquisition 신호가 실제로 다른 방법을 택해야 한다.

### 5.3 사람 개입 감사

- 현재 controlled Random/DINO 440-run: blind manifest에는 opaque `sample_id`만 노출되고, DINO embedding과 seed가 자동 trajectory를 만든다. 사람 선택 없음.
- MPDD: `run_mpdd_selection_only_audit.py:139-186`에서 blind pool과 embedding으로 자동 선택하며, `:278-280`에 GT/mask/bbox는 selector에 사용하지 않고 사후 audit에만 join했다고 명시한다.
- VisA: `run_visa_selection_only_audit.py:99-143`, `:224-225`도 동일하다.
- GC10 D2R historical audit: `SelectedFeedbackOracle`이라는 이름이 있으나 이는 사람이 아니라 이미 선택되어 라벨이 공개된 표본의 class feedback을 코드가 사용하는 객체다. `run_gc10_discovery_representation_audit.py:109-224,306-324`; unselected GT는 selector에 사용하지 않았다고 `:591,617`에 명시한다.
- `scripts/build_human_inspection_set.py`와 이미지 시각화는 사후 확인 자산이다. 주요 실험 경로에서 사람이 anomaly/target을 직접 골라 selector output으로 사용한 증거는 발견되지 않았다.

다만 미팅 전사에는 학생 설명으로 “대표 이미지를 … 랜덤으로 한 20개씩 열어보면서 체크”했다는 말과, STT상 “제가 골랐습니다”로 기록된 대목이 이어진다. 이 때문에 교수는 MPDD·VisA 결과를 사람이 target을 알고 직접 고른 것으로 이해하고 “그러면 당연히 Random보다 많이 나오지 않느냐”고 반문했다. 이는 **실험 구현의 수동 selection 증거가 아니라 발표 설명과 구현의 불일치**다. 다음 미팅에서는 아래처럼 정정해야 한다.

> “대표 이미지 약 20장을 본 것은 자동 selection 결과의 사후 sanity check였습니다. 실제 selected set은 blind manifest와 frozen embedding/selection rule로 코드가 자동 생성했고, MPDD·VisA의 GT·mask·bbox는 선택이 끝난 뒤 audit metric 계산에만 결합했습니다.”

판정: **사람의 육안 판단이 실제 표본 선택에 개입한 것으로 확인된 주요 실험은 없다.** 단, deprecated scratch script 전체의 행위까지 전수 복원한 것은 아니므로 “모든 과거 임시 시도에서 절대 없었다”는 주장은 하지 않는다.

### 5.4 현재 도메인에서 DINO feature의 정량근거

판정은 **2. 일부 있으나 부족함**이다.

- current controlled study는 DINO representation+farthest-first의 downstream 효과를 직접 측정한다. 이는 조합의 효용 근거이지, feature 공간의 class semantics를 분리 검증한 근거는 아니다.
- current public selection audit은 class incidence, entropy, instance count와 rare target count를 보고하지만 kNN agreement/silhouette/same-vs-different distance는 없다: `results/thesis_final/tables/selection_composition_*.csv`.
- historical GC10에는 within-class cosine distance와 representation target hit/purity 코드가 있다: `scripts/03_analysis/run_gc10_discovery_representation_audit.py:239-286`, `analyze_gc10_d2r_representation_failure.py:156-312`.
- frozen evidence ledger E023은 class 8 nearest-target top-1 hit가 `0.153846`으로 gate를 실패했다고 기록한다. `docs/gc10_d2r_representation_branch_closure_20260715.md`에는 centroid-nearest overall target hit `0.494`, class 8 `0.116809`, top-32 target purity `0.049947`이 기록되어 있다. 이는 오히려 global pooled representation의 한계를 보인다.
- NEU에 대한 동등한 직접 feature-quality 분석은 발견되지 않았다.
- 위 historical raw output 경로는 evidence ledger에 기록되어 있으나 canonical branch에는 raw directory가 없고 ledger/요약만 남아 있다.

추가 학습 없이 할 최소 분석은 데이터셋별로 (1) same-label vs different-label cosine-distance 분포, (2) leave-one-out kNN label agreement(k=1,5), (3) single-label NEU silhouette 및 GC10은 multilabel-aware nearest-neighbor agreement를 보고하는 것이다. threshold tuning 없이 사전 고정하고, 전체/희소 class를 함께 보고해야 한다.

## 6. Budget, 결과 시점, split 감사

### 6.1 Split은 train/validation/test로 어떻게 말할 수 있는가

| dataset | train/acquisition pool | validation/development | locked final test | curated total |
|---|---:|---:|---:|---:|
| GC10-DET | 1,605 (70.03%) | 229 (9.99%) | 458 (19.98%) | 2,292 |
| NEU-DET | 1,259 (70.02%) | 180 (10.01%) | 359 (19.97%) | 1,798 |

- split seed는 `20260822`다.
- deterministic rare-label-first iterative multilabel stratification과 global largest-remainder rounding을 사용했다.
- split unit은 canonical pixel identity group이다. 동일 pixel+동일 GT는 하나로 collapse하고 conflicting GT identity group은 제외하여 duplicate leakage를 방지했다.
- selector는 train/acquisition pool의 `sample_id`만 본다. validation/final membership과 GT는 보지 않는다.
- train pool 내부에서 R0 labeled set을 만들고 나머지를 unlabeled로 둔 뒤, 각 round에서 query를 cumulative labeled set으로 이동한다.
- validation/development는 각 training cell의 `best.pt` 선택에 사용한다.
- locked-final은 440/440 training과 integrity PASS 뒤 별도 authorization으로 평가했고 training/selection/checkpoint choice에는 쓰지 않았다.
- 공개 원자료가 이전 연구에 쓰였기 때문에 논문은 이를 “historically untouched test”가 아니라 **prospectively locked confirmatory evaluation split**으로 제한해 부른다.

따라서 발표에서 train/validation/test라는 익숙한 용어로 바꾸는 것은 가능하지만, 단순 이름 변경만으로는 부족하다. `train pool 안의 labeled/unlabeled 이동`, `validation의 best.pt 선택`, `locked test의 비개입`을 한 문장씩 설명해야 한다.

### 6.2 Initial set과 budget 증가 규칙

- R0는 canonical sample ID 순서 위에서 PCG64 uniform without replacement로 생성한다.
- acquisition seed만 쓰며 GT class와 learner seed는 쓰지 않는다.
- 같은 dataset/acquisition seed에서 Random과 DINO는 동일 R0를 공유한다.
- initial count는 원래 고정 train pool의 10%를 Decimal `ROUND_HALF_UP`으로 한 번 반올림한다.
- 매 query는 **remaining pool의 5%가 아니라 원래 train pool의 5%**를 같은 방식으로 정한다.
- 다섯 query round 후 약 35%가 된다. 10% + 5% × 5 = 35%이기 때문이다.

| round | GC10 학습 이미지 | GC10 train 대비 | NEU 학습 이미지 | NEU train 대비 |
|---:|---:|---:|---:|---:|
| R0 | 161 | 10.031% | 126 | 10.008% |
| R1 | 241 | 15.016% | 189 | 15.012% |
| R2 | 321 | 20.000% | 252 | 20.016% |
| R3 | 401 | 24.984% | 315 | 25.020% |
| R4 | 481 | 29.969% | 378 | 30.024% |
| R5 | 561 | 34.953% | 441 | 35.028% |

근거: `confirmatory_protocol_v1.yaml:119-171`.

### 6.3 그래프의 각 점과 primary 결과 연결

각 값은 acquisition seed 10개 × learner seed A/B를 먼저 평균한 budget-level locked-final mAP50-95다. R0는 두 strategy가 동일 selected set이므로 같다.

| dataset | round | images / train % | Random mAP50-95 | Frozen DINO mAP50-95 | DINO−Random |
|---|---:|---:|---:|---:|---:|
| GC10 | R0 | 161 / 10.031% | 0.205067 | 0.205067 | 0.000 pp |
| GC10 | R1 | 241 / 15.016% | 0.229121 | 0.231580 | +0.246 pp |
| GC10 | R2 | 321 / 20.000% | 0.239521 | 0.250944 | +1.142 pp |
| GC10 | R3 | 401 / 24.984% | 0.251855 | 0.257079 | +0.522 pp |
| GC10 | R4 | 481 / 29.969% | 0.266462 | 0.273462 | +0.700 pp |
| GC10 | R5 | 561 / 34.953% | 0.267444 | 0.273346 | +0.590 pp |
| NEU | R0 | 126 / 10.008% | 0.306194 | 0.306194 | 0.000 pp |
| NEU | R1 | 189 / 15.012% | 0.330632 | 0.338749 | +0.812 pp |
| NEU | R2 | 252 / 20.016% | 0.347246 | 0.358155 | +1.091 pp |
| NEU | R3 | 315 / 25.020% | 0.366282 | 0.370161 | +0.388 pp |
| NEU | R4 | 378 / 30.024% | 0.372376 | 0.377010 | +0.463 pp |
| NEU | R5 | 441 / 35.028% | 0.380230 | 0.382966 | +0.274 pp |

근거: `results/thesis_final/tables/budget_level.csv`.

Primary는 R5가 아니라 여섯 점 전체의 budget-span-normalized AULC다.

| dataset | Random nAULC | DINO nAULC | paired Δ | 95% CI | acquisition 방향 |
|---|---:|---:|---:|---:|---:|
| GC10 | 0.244643 | 0.250454 | +0.005812 | [0.003970, 0.007576] | 10/10 positive |
| NEU | 0.351950 | 0.357731 | +0.005782 | [0.003100, 0.008119] | 9/10 positive |

두 Holm-adjusted p-value는 `0.003906`이다. 단, NEU worst-class nAULC는 `-0.014879`, final worst-class AP50-95는 `-0.031182`로 10/10 acquisition에서 악화됐다. 근거는 `results/thesis_final/tables/locked_final_primary.csv`, `secondary_subgroup.csv`, `final_analysis/FINAL_CONTROLLED_ANALYSIS.md:5-26`이다.

R5에서 요구된 보조 지표의 절대 평균은 다음과 같다. 각 arm당 conceptual row 20개(10 acquisition × learner A/B) 평균이다.

| dataset | strategy | mAP50-95 | mAP50 | precision | recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| GC10 | Random | 0.267444 | 0.558170 | 0.599348 | 0.561004 | 0.578120* |
| GC10 | Frozen DINO | 0.273346 | 0.567268 | 0.610081 | 0.567578 | 0.587178* |
| NEU | Random | 0.380230 | 0.699253 | 0.678473 | 0.668448 | 0.672916* |
| NEU | Frozen DINO | 0.382966 | 0.694703 | 0.664238 | 0.672550 | 0.667775* |

`*` F1은 canonical CSV에 직접 저장된 metric이 아니며, 이 audit에서 각 row의 precision/recall harmonic mean을 구한 뒤 평균한 **설명용 post-hoc 값**이다. confirmatory claim으로 쓰면 안 된다. 원자료는 `results/thesis_final/tables/locked_final_conceptual_480.csv`다.

`best.pt`는 각 budget training에서 validation fitness로 선택된 checkpoint이고, `final budget`은 가장 많은 labeled image를 사용한 R5다. 최신 DOCX P0201/P0247/P0284/P0297은 이 둘을 구분하고 있어 직접 혼동 문장은 발견되지 않았다. 다만 그림의 `final-budget`에는 항상 `R5, 561/441 images, 34.953/35.028%`를 병기하는 것이 안전하다.

## 7. Baseline Matrix

상태 정의: A=코드+현재 protocol 결과, B=코드는 있으나 현재 protocol 결과 없음, C=과거 결과만 있음, D=코드도 결과도 없음. 코드와 historical ledger가 모두 있지만 현재 protocol 결과가 없으면 B로 분류하고 비고에 과거 근거를 썼다.

| 방법 | 상태 | 현재/과거 근거 | 판정 |
|---|:---:|---|---|
| Random-PCG64-v1 | A | current protocol, 440 results | 정본 baseline |
| FrozenDINO-FF-v1 | A | current protocol, 440 results | 정본 비교 arm; representation+FF 묶음 |
| greedy k-center/Core-set family로서의 현 방법 | A | `protocol_core.py:265-303` | 현재 DINO FF가 이 계열의 구현 |
| 별도 Core-set/k-center arm | D | distinct current arm 없음; deprecated code에 coreset 명칭 일부 | 같은 feature/distance면 중복이라 신규 추가 타당성 낮음 |
| detector uncertainty/entropy | B | `scripts/02_active_learning`, historical protocol/docs | current controlled 결과 없음; distinct baseline 후보 |
| margin sampling | D | 통계 margin 등 동명이인 문구만 있고 acquisition 구현/결과 미확인 | 추가 근거 없음 |
| least confidence | D | 검색 결과 없음 | 추가 근거 없음 |
| BADGE | B | root/`scripts/old` 구현 흔적 | current protocol 결과 없고 production-ready 여부도 불명 |
| DINO cluster coverage/diversity | B | historical scripts/ledger E019-E022 | current 동일 protocol 아님; clustering은 현 FF에 사용되지 않음 |
| VLM consistency | B | score/audit code, ledger E001-E002 | validity gate 실패; current protocol 결과 없음 |
| groundedness/pseudo-groundedness | B | score/audit code, ledger E003-E005 | validity gate 실패; current protocol 결과 없음 |
| VLM+DINO/detector hybrid | B | legacy scripts/docs | current protocol 결과 없음; raw artifact도 canonical branch에 없음 |
| full-data YOLO upper bound | B | `scripts/02_active_learning/run_yolo_full_supervised_baseline.py`, `run_yolo_upper_bound_v7.py` | code만 있고 동일 split/lifecycle 결과 없음 |

historical `run_yolo_full_supervised_baseline.py`는 Mac 경로, 3 seeds(42–44), 30 epochs, batch 4, 별도 80/20 split을 전제로 하므로 현재 100-epoch 70/10/20 protocol과 바로 비교할 수 없다. `docs/research_logs/VLM_GTFree_Active_Learning_Research_Log_v1.md`의 99-image priority-pool 결과도 full-data upper bound가 아니다.

**Full-data 판정: `Full-data upper-bound missing`.**

최소 제안은 다음과 같다.

- 설명 가능한 최소: 4 GPU training runs = 2 datasets × 2 사전 고정 learner seeds. same split, YOLOv8n checkpoint/hash, 100 epochs, imgsz 640, SGD, validation-selected `best.pt`, locked-final evaluation을 그대로 쓴다. 이는 기술적 reference이며 현재 20-seed learner 분포와 완전 직접 비교는 아니다.
- 현재 learner-seed 분포까지 맞춘 비교: 40 runs = 2 datasets × 20 기존 unique learner seeds. acquisition seed는 full-data에서 의미가 없으므로 반복하면 안 된다.
- distinct AL baseline 한 가지를 current full curve로 추가: shared R0를 재사용한다면 2 datasets × 10 acquisition seeds × 2 learner seeds × 5 query rounds = **200 새 training cells**. 전사에서 교수가 Random/DINO 외 방법 추가를 명시했으므로 P1 필수 요구로 본다. 다만 현재 DINO FF와 사실상 같은 Core-set 이름의 중복 arm이 아니라 detector-aware uncertainty처럼 신호가 다른 방법을 사전 고정해야 한다.

## 8. VLM Reuse Check

### 8.1 현재 남아 있는 것

- Qwen/VLM inference·prompt·score code: `scripts/01_score_generation/`, `scripts/00_prompt_validation/`, `scripts/03_analysis/audit_*vlm*`, `compare_paired_vlm_model_gates.py`.
- legacy code: `scripts/old/`의 Qwen, consistency, LoRA, BADGE/hybrid 계열.
- frozen aggregate evidence: `runs/evidence_freeze_v3_20260718/research_evidence_ledger.csv` E001-E005.
- 확인된 aggregate 결과: consistency-groundedness Spearman `-0.181199` (95% CI `[-0.389654, 0.036410]`), inconsistency severe-failure AUC `0.373433` (`[0.246360, 0.508573]`), structured schema completeness `0.01`, oracle crop presence/evidence/median IoU `0/0/0`, 세 Qwen-family 중 gate pass `0/3`.

### 8.2 현재 남아 있지 않은 것

- canonical branch에서 tracked raw Qwen/VLM response `.json/.jsonl/.csv` 파일을 찾지 못했다.
- `audit_legacy_vlm_consistency_validity.py`가 기대하는 `outputs/grounded_experiment_20260601/pilot_grounded_consistency_results.json`은 이 worktree에 없다.
- ledger가 가리키는 `runs/vlm_consistency_groundedness_validity/...` raw result directory도 이 branch에는 없다.
- 과거 VLM selector→YOLO의 current split/current lifecycle paired 결과는 없다. legacy script가 있다는 사실만으로 결과 존재를 주장할 수 없으므로 **확인되지 않음**이다.

### 8.3 새 추론 없이 현재 pipeline에 연결 가능한가

**현재 canonical branch만으로는 불가능하다.** aggregate metric은 image-level acquisition score로 되돌릴 수 없기 때문이다.

과거 raw response/score를 별도 저장소에서 회수할 수 있다면 가능성은 있다. legacy row는 `image_name`/`image_path`를 저장했고, current `confirmatory_v1/data_freeze/private_source_registry.csv`는 원 filename 경로를 canonical pixel SHA `sample_id`로 연결한다. 따라서 새 Qwen inference 없이 다음 join을 검증할 수 있다.

`legacy image_name/path → normalized relative filename → private_source_registry → current sample_id`

단, 중복 pixel collapse, old split과 current acquisition split의 교집합, response completeness, locked-final leakage를 먼저 audit해야 한다. raw 파일이 없는 현재는 구현하지 않는다.

raw score가 회수되고 validity 한계를 명시한다는 조건에서 가장 작은 결합은 다음 두 후보뿐이다.

- 후보 A: pre-normalized VLM uncertainty score + DINO diversity를 사전 고정 방식으로 결합.
- 후보 B: VLM score 상위 후보군을 만든 뒤 그 안에서 현재 DINO farthest-first 적용.

둘 다 새 selector가 되므로 weight/cutoff를 결과를 보고 정하면 안 되며, 이번 P0 작업에는 포함하지 않는다. E001-E005가 VLM signal validity를 지지하지 않았기 때문에 **논문의 즉시 보강책으로는 권고하지 않는다.**

## 9. Automotive Bushing Paper Comparison

### 9.1 식별

정확한 논문은 다음으로 확인된다.

> Shikun Chen and Yangguang Liu, **“Vision-Language Model-Assisted Active Learning for High-Accuracy Defect Detection in Automotive Bushing Components,”** Proceedings of EILM 2025, pp. 9–14, DOI `10.1145/3788731.3788733`.

서지사항은 [공식 DOI/ACM landing](https://doi.org/10.1145/3788731.3788733)과 [Crossref record](https://api.crossref.org/works/10.1145/3788731.3788733)로 확인했다. Crossref상 proceedings publication date는 2025-12-19이고 record 생성은 2026-02-14다. repository, BibTeX, literature review, 로컬 PDF에서는 이 논문을 찾지 못했다.

접근 가능한 [Semantic Scholar record/abstract](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3788731.3788733?fields=title,abstract,authors,year,venue,url,externalIds)에 따르면 저자들은 locally deployed Qwen2.5-VL을 LoRA fine-tuning하여 defect classification과 bounding-box prediction을 함께 수행하고, classification confidence와 localization uncertainty를 합친 cross-modal uncertainty로 annotation 대상을 우선순위화했다. 15,000 bushing images, 네 defect type에서 35% training data로 mAP@50 98.2%, annotation requirement 68% 감소, best fully supervised baseline 대비 5.9%p를 주장한다.

단, ACM full text를 확보하지 못했으므로 다음은 `확인되지 않음`이다.

- cross-modal uncertainty의 정확한 수식과 weight
- 별도 downstream detector의 architecture 또는 VLM 자체가 최종 detector인지
- “best fully supervised baseline”의 정확한 모델/설정
- 다른 AL baseline의 명칭과 seed/statistical protocol
- text prompt·label description의 정확한 구성

### 9.2 비교

| 항목 | 자동차 부싱 연구 | 과거 Qwen 연구 | 현재 DINO 연구 |
|---|---|---|---|
| 산업 목적 | bushing 검사 정확도와 annotation 효율 | VLM이 어려워하는 결함 이미지를 찾는 selector 후보 | 제한 image budget의 산업 결함 detector 학습 효용 검증 |
| domain | automotive rubber-metal bushing, private 15k images | GC10/NEU 중심 공개 산업 결함 | GC10-DET, NEU-DET 공개 benchmark |
| task | spatially annotated defect detection | consistency/grounding signal audit 및 일부 AL 시도 | multi-class object detection |
| VLM 역할 | Qwen2.5-VL을 LoRA fine-tune해 classification+bbox | Qwen 응답 consistency/groundedness score | 없음; DINOv2는 image-only SSL backbone |
| text 역할 | multimodal prompt/semantic output에 관여; 세부는 미확인 | 여러 prompt의 응답/설명 일관성과 pseudo grounding | 전혀 사용하지 않음 |
| acquisition | classification confidence+localization uncertainty 결합 | VLM uncertainty/groundedness/hybrid 후보; validity 실패 | cosine greedy farthest-first |
| detector | 정확한 독립 detector `확인되지 않음` | YOLO historical pathway가 있으나 current paired 결과 없음 | sealed YOLOv8n cold-start |
| baseline | exact identities `확인되지 않음`; abstract가 fully supervised best를 언급 | Random/legacy variants, 비동일 protocol | Random-PCG64-v1 |
| 핵심 novelty | trained VLM의 cross-modal detection uncertainty로 annotation 우선순위화 | text response를 acquisition proxy로 삼을 수 있는지 탐색 | frozen image representation 선택의 통제된 downstream 효과와 경계 |
| 현재 증거 강도 | abstract claim만 확인; full protocol 미확인 | signal validity가 지지되지 않음 | 440 lifecycle, acquisition n=10/dataset, locked final |

교수님이 “Task가 거의 비슷하지 않느냐”고 한 이유는 정확하다. 두 연구 모두 **제한된 라벨/annotation 비용 아래 산업 결함 object detection을 위한 Active Learning**을 다룬다. 가장 중요한 차이는 자동차 부싱 논문은 **VLM을 학습하고 text-cross-modal uncertainty를 핵심 방법으로 사용하여 정확도·annotation 효율 향상을 직접 목표로 하는 반면**, 현재 연구는 **frozen image-only DINO representation과 표준적인 geometric farthest-first의 작은 aggregate 효용 및 class-level 실패 경계를 통제 검증**한다는 점이다. 현재 논문의 novelty를 “VLM-assisted selector”나 “새로운 DINO sampler”로 주장해서는 안 된다.

## 10. Minimum Work Before Thesis Submission

### 반드시 할 것

1. **P0 제목 확정:** `제한된 라벨 환경 + 산업 결함 객체 탐지 + 성능 확보/학습 효용 + Frozen DINO 표현 기반 AL`이 보이도록 바꾼다.
2. **P0 연구 목적 문장 고정:** 서론·초록·발표 첫 문단에 동일한 한 문장을 두고, historical failure/Random 비교는 연구 경과로 내린다.
3. **P0 기술 설명 교정:** DINOv2 ViT-S/14, LVD-142M SSL, pooled 384-D, frozen feature extractor와 cosine farthest-first를 분리한다.
4. **P0 그림 설명 교정:** 모든 결과에 dataset, R0-R5, 실제 이미지 수, train-pool %, nAULC인지 final-budget인지 명시한다.
5. **P0 제한사항 공개:** full-data upper bound missing, annotation time/cost 미측정, NEU worst-class 악화를 숨기지 않는다.
6. **P1 full-data reference:** 지도교수 질문에 직접 답하려면 최소 4 runs를 동일 lifecycle로 수행한다. 이번 audit에서는 실행하지 않았다.
7. **P1 DINO feature post-hoc audit:** 기존 embedding+GT로 3개 지표를 산출한다. 학습은 필요 없다.
8. **P1 distinct AL baseline:** 전사에서 명시적으로 요구됐으므로 Random과 현 DINO/Core-set 계열 이외의 대표 방법 최소 1개를 사전 고정한다. current full protocol 적용 시 200 새 training cells이므로 실행 범위는 교수와 즉시 합의한다.

### 하면 좋은 것

1. 자동차 부싱 논문 full text를 학교 구독으로 확보하여 detector와 baseline만 확인하고 관련연구 표를 보정한다.
2. 과거 VLM raw response 위치를 한 번만 확인한다. 없으면 복원 프로젝트를 즉시 종료하고 current thesis의 historical negative evidence만 유지한다.
3. final PDF metadata와 `[공식 학위명 확인 필요]`, 영문 성명/학과 placeholder를 제목 확정 후 행정적으로 정리한다.

### 지금 하면 안 되는 것

- DINO fine-tuning/새 DINO 학습
- Qwen 재추론 또는 raw artifact가 없는 상태의 VLM selector 재건
- 같은 DINO feature+distance의 Core-set 이름 바꾸기 실험
- 결과를 보고 weight/quota/threshold/seed를 고르는 일
- Open-set, Domain Generalization, 새 image+text multimodal detector로 연구 중심을 갈아타는 일
- 이번 audit 결과를 근거로 기존 440 결과나 논문 본문을 자동 수정하는 일

> “지도교수 미팅에서 novelty 확보의 예시로 제시되었으나, 현재 9월 학위논문 일정과 기존 440-run 연구의 연속성을 고려하면 사실상 연구 범위를 크게 변경하는 선택지이므로 이번 제출을 위한 P0/P1 작업에서는 제외한다.”

## 11. 마지막 12개 질문에 대한 직접 답변

### 1. 현재 논문의 정확한 최신 제목은 무엇이며 무엇이 빠졌는가?

`산업 결함 Active Learning에서 표본 선택 신호와 학습 성능의 연결 — 이질적 역사 증거의 단계적 분석과 Random-Frozen DINO 통제 검증`이다. 산업 환경과 AL/DINO는 있으나 정확한 **object detection Task**, **제한된 라벨 환경**, **탐지 성능 확보 목적**이 빠졌다. PDF metadata에는 이전 제목도 남아 있다.

### 2. 초록/서론은 왜 제한된 labeled data가 문제인지 설명하는가?

예. P0098-P0101과 P0506이 희소·불균형 결함, annotation 부담, 제한 라벨 예산과 동일 이미지 수의 detector 성능을 명확히 말한다. 다만 직후부터 관계/분석과 연구 변천사가 전면에 나와 목적의 우선순위가 흐려진다.

### 3. Frozen DINO는 정확히 어떤 모델이며 무엇으로 사전학습되었는가?

`facebook/dinov2-small` exact revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, DINOv2 ViT-S/14(21M, patch 14, 384-D)다. LVD-142M의 142M images를 annotation 없이 사용한 DINOv2 학습 계열이며 DINO self-distillation, iBOT masked image modeling, KoLeo regularization을 쓴다. ViT-S는 ViT-g teacher에서 distillation됐다.

### 4. DINO는 selector 자체인가 feature extractor인가?

**Feature extractor**다. selector는 그 feature 위의 별도 acquisition rule이다.

### 5. acquisition rule은 어느 계열인가?

L2-normalized pooled feature에서 현재 labeled/reference set까지의 minimum cosine distance가 최대인 표본을 순차 선택하는 greedy farthest-first다. 방법론적으로 greedy k-center/Core-set 계열과 매우 가깝다.

### 6. 사람의 육안 판단이 실제 표본 선택에 개입한 실험은 무엇인가?

현재 controlled 실험과 조사한 MPDD·VisA·GC10 주요 historical 경로에서는 **없다**. 육안 이미지는 사후 확인용이고, MPDD/VisA GT는 selection 후 audit에만 join됐다. 미팅에서는 이 구분이 흐리게 설명되어 교수가 수동 selection으로 이해했으므로 설명을 정정해야 한다. deprecated scratch 전체를 전수 복원하지 않았으므로 모든 임시 시도에 대한 절대 부정은 하지 않는다.

### 7. DINO feature가 GC10/NEU에서 의미 있다는 정량 증거가 존재하는가?

**일부 있으나 부족하다.** GC10 historical within-class/retrieval/purity 진단과 current downstream/composition 결과는 있다. class 8 representation failure도 기록되어 있다. NEU direct feature-space validation과 두 데이터셋 공통 kNN/silhouette 분석은 없다.

### 8. 100% labeled-data YOLO baseline은 존재하는가?

아니다. `Full-data upper-bound missing`. legacy script와 비동일 protocol 결과는 현재 비교치가 아니다.

### 9. Random 외 현재 재사용 가능한 AL baseline은 무엇인가?

current result까지 재사용 가능한 것은 FrozenDINO-FF-v1뿐이다. detector uncertainty, VLM, cluster, BADGE 등은 code/historical 흔적은 있지만 current protocol 결과가 없다. 별도 Core-set은 현재 방법과 사실상 중복될 수 있다.

### 10. 과거 Qwen/VLM 결과를 새 추론 없이 연결할 수 있는가?

현재 canonical branch만으로는 불가능하다. raw image-level response/score가 없고 aggregate ledger만 있다. raw 파일이 회수되면 filename→`private_source_registry`→canonical `sample_id` mapping으로 새 추론 없이 연결 가능성을 검증할 수 있다.

### 11. 자동차 부싱 VLM-assisted AL 연구와 가장 중요한 차이는 무엇인가?

공통 문제/Task는 제한 annotation의 산업 결함 object detection이다. 부싱 연구는 LoRA-tuned Qwen2.5-VL의 classification+localization uncertainty라는 multimodal method로 정확도와 annotation 효율을 직접 노린다. 현재 연구는 frozen image-only DINO representation+geometric farthest-first의 protocol-bounded downstream 효과와 worst-class 경계를 검증한다.

### 12. 현재 연구를 가장 적게 변경하면서 피드백을 충족하려면 무엇을 해야 하는가?

P0는 제목·목적을 산업 문제/Task 중심으로 재정렬하고 DINO/acquisition/budget/split과 자동 selection을 정확히 설명하며 제목 변경 가능 여부를 행정적으로 확인하는 것이다. P1은 동일 lifecycle full-data reference, 무학습 DINO feature audit, 그리고 교수 요구에 따른 distinct AL baseline 최소 한 개다. VLM 재건, DG, open-set, 새 multimodal detector는 하지 않는다.

## 12. 최종 우선순위 출력

### P0 — 9월 9일 제목 결정 전에 해결할 것

1. 제목에 제한 라벨 환경, 산업 결함 객체 탐지 Task, 탐지 성능 목적, Frozen DINO 표현 기반 AL 방법을 넣는다.
2. 서론·초록의 첫 목적을 공학 문제 중심 한 문장으로 고정하고 연구 변천/관계 분석을 뒤로 내린다.
3. DINO feature extractor와 cosine farthest-first acquisition, 자동 selection과 육안 사후 확인, R0-R5 이미지 수/비율, train-validation-locked test 역할을 정확히 설명하고 접수 후 제목 변경 가능 여부를 확인한다.

### P1 — 9월 말 심사용 논문 전에 해결할 것

1. 동일 split/lifecycle의 100% labeled YOLO upper bound를 최소 4 runs로 확보한다.
2. 기존 DINO feature와 GT의 same/different distance, kNN agreement, silhouette을 무학습 사후 분석한다.
3. 교수의 명시적 비교 요구에 따라 현 Core-set 계열과 다른 detector-aware baseline 최소 1개를 사전 고정한다.

### P2 — 이번 논문에서는 하지 않을 것

1. Domain Generalization.
2. Open-set Recognition.
3. 새로운 image+text multimodal detector 또는 과거 VLM 전수 복원.

### 현재 가장 치명적인 미확인 사항

동일 split·동일 YOLO lifecycle에서 100% labeled training을 했을 때의 GC10/NEU 성능이 없다.

### 현재 연구를 유지할 수 있는지에 대한 판정

**보강 후 유지 가능.** 전사에는 연구 폐기 지시가 없고, 440-run의 integrity와 Random 대비 작은 aggregate nAULC 이득은 유지할 수 있다. 다만 제목·연구 목적을 object detection의 제한-label 공학 문제로 재정렬하고, full-data reference, DINO feature의 직접 정량 근거, distinct AL baseline을 P1로 보강해야 한다. Frozen DINO의 보편적 우월성, 모든 class의 개선, 실제 annotation 비용 절감은 주장할 수 없다.

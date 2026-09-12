# ANALYSIS VALIDATION REPORT

**PASS — 전체 로컬 development 분석 검증.**

- Physical 입력 644 = historical 440 + Entropy 200 + Full-data 4. 과거 원격 속도 확인 4회와 실패 attempt는 독립 실험으로 합산하지 않음.
- 두 dataset × 세 AL 방법 × a00–a09 × A/B × R0–R5 = conceptual 720행. missing 0 / duplicate 0.
- 공통 historical R0 40 reference의 checkpoint/manifest/args identity 확인. 같은 R0가 세 방법의 conceptual 행에 나타나지만 새 학습으로 세지 않음.
- A/B를 seed·round별로 먼저 평균: 360행. 이어 동일 frozen 정의로 nAULC 60행. A/B별 적분 후 평균과도 수치 일치.
- GC10 budgets [161,241,321,401,481,561], NEU [126,189,252,315,378,441]. pool 1605/1259, validation 229/180. train/validation overlap 없음. seed 수는 각각 10.
- 100-epoch CSV 최대 mAP50–95를 모든 physical cell에서 직접 확인. 과거 별도 best-checkpoint evaluation 수치와 혼합하지 않음. 이미 선택에 쓰인 validation의 최댓값이므로 test 성능 추정치가 아님.
- frozen trapezoid: x=labels/pool, 면적/(x5−x0). 같은 간격이므로 [0.1,0.2,0.2,0.2,0.2,0.1] weighted sum과 독립 대조 PASS.
- paired bootstrap 100,000회, PCG64 seed GC10 421397045 / NEU 1672358496. 차이의 평균·중앙값·구간·양/음 seed 수 6개 대비 전부 독립 재계산 PASS. confirmatory p-value·새 multiplicity family 없음.
- Full-data는 dataset별 A/B 실제 2개. n=10으로 복제하지 않음. R5 비율·차이 산술 PASS. Reference의 uncertainty를 AL paired CI에 섞지 않음.
- Entropy 100 query를 registry COMPLETE artifact 및 score SHA로 연결. 직전 learner A best lineage와 selected/candidate membership 확인. GT를 사용한 새 acquisition 없음.
- 클래스 구성은 sealed acquisition train IDs만으로 label hash를 대조. 원본 train label 2864개 이외의 image/validation/final label은 열지 않음. multi-label image prevalence로 집계하며 query round를 독립 반복으로 취급하지 않음.
- Historical geometry class명 연결 16개 PASS. NEU의 geometry class index와 YOLO class ID 순서가 달라 이름으로 대조. 기존 class support와 R5 DINO/Random 누적 선택 평균도 원본 manifest에서 대조 PASS.
- 원래 분석에 읽은 입력 5751개 SHA/size 재확인: 변화 0. 별도 git blob 5개 SHA 일치. 미방문 checkpoint 전체나 repo 전체를 재검증했다고 주장하지 않음.
- Figure 5종 × 3형식 존재. PNG 육안 점검에서 하단 설명 및 일부 class label 간격을 조정함. 표 8종의 CSV/Markdown 모두 존재.
- 기존 n=9 intermediate 파일은 대표 수치의 입력이 아님. development와 historical locked-final 표는 분리. 이번 locked-test 접근 0, checkpoint inference 0, training/acquisition 0, RunPod/SSH 0, GitHub push/branch 생성 0.
- 요청한 기록 용어 검사 PASS. 원래 frozen snapshot/source를 수정하지 않고 새 실험 기록에 적용함.

## 해석상 남는 제한

Endpoint identity는 A등급이다. 전체 비교는 GPU·OS 차이를 분리하지 못해 B등급이다. 과거 비교 참조 feasibility record 4개에는 완전한 package field가 없고 해당 결측은 environment_comparison.csv에 남겼다. 기록된 package field에서는 차이가 없으며 missing을 일치로 처리하지 않았다.

Remote compact acquisition JSON은 inference_config를 각 record에 반복 저장하지 않는다. Local a00는 실제 record의 config를 확인했고 remote는 동결 worker의 guard/source와 이전 승인 감사에 근거했다. 기존 COMPLETE/gate 상태를 사용하는 것이며, 새 inference 실행으로 검증한 것이 아니다.

Entropy validation은 parent learner/checkpoint 선택 과정에도 관여한다. 동일 development split이라는 사실이 독립 holdout 비교를 뜻하지 않는다. 점별 95% CI는 새 6개 대비 전체에 대해 familywise coverage를 보장하지 않으며 0 포함을 동등성으로 해석하지 않는다. 새로운 classwise validation AP는 없어 classwise detector 우월성을 판단하지 않는다.

추가 AL training을 닫는 것과 최종 평가를 끝내는 것은 별개다. 신규 204 checkpoint의 동일 locked-final 평가 scope/statistics/authorization은 후속 별도 단계로 남긴다.

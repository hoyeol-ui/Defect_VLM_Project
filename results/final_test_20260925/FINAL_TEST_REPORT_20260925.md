# Final held-out TEST evaluation — 2026-09-25

## 1. 최종평가 실행 개요

- 실행 완료 시각 UTC: 2026-09-25T07:12:32.095647+00:00
- 저장소: `C:\Users\user\Desktop\vlm\Defect_VLM_Project_final_main_results_20260922`; branch `results/final-main-v2-20260922`; commit `a189a7252c5b35208d6da2ce16ccf13409c7073d`
- GPU: NVIDIA GeForce RTX 3070; Python 3.11.9; PyTorch 2.12.1+cu126; torchvision 0.27.1+cu126; Ultralytics 8.4.90
- Scientific checkpoint SHA 검증: 162/162. `TECHNICAL_NEU-DET` 제외.
- Test 이미지: GC10-DET 344, NEU-DET 270.
- YOLOv8n; imgsz 640; batch 8; conf 0.001; NMS IoU 0.70; max_det 300; 각 `best.pt` 1회 평가.

## 2. 사전 검증

- 고정 split의 train/dev/test ID, RGB SHA, 파일 SHA, component 중복 0.
- 원본 test 이미지·XML SHA, RGB identity, bbox, class mapping, native label parser: PASS.
- GC10 DINO R5 dev dry-run: 저장된 전체 및 클래스별 AP와 최대 절대 차이 0.0; PASS.
- Test completeness: 162/162 SUCCESS, missing/duplicate/unexpected/technical 0.

## 3. GC10-DET test 결과

### R0–R5 mAP50–95 (%)

|Round|Labeled|Random|DINOv2-FF|Entropy|
|---:|---:|---:|---:|---:|
|R0|160|16.905|16.905|16.905|
|R1|241|17.934|19.057|18.435|
|R2|321|18.631|20.336|19.027|
|R3|401|21.039|21.956|19.739|
|R4|481|21.834|24.673|19.937|
|R5|561|21.706|23.280|21.776|

### nAULC 및 비교

|Method|Mean nAULC (%)|Sample SD (%p)|
|---|---:|---:|
|Random|19.743|0.664|
|DINOv2-FF|21.215|0.801|
|Entropy|19.292|0.351|

|Comparison|Paired mean (%p)|95% bootstrap CI (%p)|Positive seeds|Exact p|Holm p|
|---|---:|---:|---:|---:|---:|
|DINOv2-FF - Random|+1.472|[+0.941, +1.953]|5/5|0.06250|0.25000|
|DINOv2-FF - Entropy|+1.923|[+1.437, +2.317]|5/5|0.06250|0.25000|
|Entropy - Random|-0.451|[-0.873, -0.096]|1/5|0.12500|—|

Full-data reference test mAP50–95: **24.586%**. DINO R5 평균과의 절대 차이: **-1.305%p** (기술적 참조).

## 4. NEU-DET test 결과

### R0–R5 mAP50–95 (%)

|Round|Labeled|Random|DINOv2-FF|Entropy|
|---:|---:|---:|---:|---:|
|R0|126|26.619|26.619|26.619|
|R1|189|28.399|29.524|27.906|
|R2|252|30.245|30.508|30.343|
|R3|315|30.848|31.636|31.270|
|R4|377|32.508|31.606|31.438|
|R5|440|32.670|33.063|32.627|

### nAULC 및 비교

|Method|Mean nAULC (%)|Sample SD (%p)|
|---|---:|---:|
|Random|30.325|0.406|
|DINOv2-FF|30.620|0.517|
|Entropy|30.112|0.667|

|Comparison|Paired mean (%p)|95% bootstrap CI (%p)|Positive seeds|Exact p|Holm p|
|---|---:|---:|---:|---:|---:|
|DINOv2-FF - Random|+0.295|[-0.338, +0.967]|3/5|0.56250|0.56250|
|DINOv2-FF - Entropy|+0.508|[+0.189, +0.841]|5/5|0.06250|0.25000|
|Entropy - Random|-0.213|[-0.890, +0.675]|2/5|0.62500|—|

Full-data reference test mAP50–95: **37.167%**. DINO R5 평균과의 절대 차이: **-4.103%p** (기술적 참조).

## 5. 통계 결과

표의 CI는 seed 5개의 대응 차이를 100,000회 percentile bootstrap으로 재표본 추출한 값이다. Exact two-sided sign-flip은 32개 부호 배치를 열거했다. 네 primary comparison에만 Holm 보정을 적용했다. n=5에서 최소 가능한 양측 p는 0.0625다.

## 6. 클래스별 결과

아래 값은 R5에서 seed 5개의 평균 AP50–95이며 단위는 %. Selected composition은 봉인된 R5 누적 선택 이미지 중 해당 클래스를 포함한 이미지 수의 평균이다.

|Dataset|Class|Random AP|DINO AP|Entropy AP|DINO−Random (%p)|Random selected|DINO selected|Entropy selected|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|GC10-DET|punching_hole|45.773|47.237|48.575|+1.464|78.4|113.2|66.8|
|GC10-DET|welding_line|30.695|42.015|28.322|+11.320|122.4|205.4|90.6|
|GC10-DET|crescent_gap|51.943|53.093|49.290|+1.151|61.6|101.4|49.4|
|GC10-DET|water_spot|30.343|31.155|29.387|+0.813|73.6|86.4|120.0|
|GC10-DET|oil_spot|22.349|23.551|24.662|+1.202|62.4|67.6|90.8|
|GC10-DET|silk_spot|2.534|3.660|3.906|+1.125|180.2|98.0|102.0|
|GC10-DET|inclusion|3.032|2.818|3.512|-0.214|52.0|34.2|99.8|
|GC10-DET|rolled_pit|6.401|5.443|6.936|-0.958|12.8|20.4|23.2|
|GC10-DET|crease|8.076|8.135|11.403|+0.059|13.6|26.6|25.2|
|GC10-DET|waist_folding|15.919|15.697|11.764|-0.222|36.4|17.0|12.0|
|NEU-DET|crazing|13.130|11.187|14.024|-1.943|70.8|31.0|102.4|
|NEU-DET|inclusion|33.561|32.189|34.256|-1.372|92.8|116.2|81.2|
|NEU-DET|patches|50.251|52.180|50.047|+1.929|80.8|86.2|92.6|
|NEU-DET|pitted_surface|35.310|36.900|36.191|+1.590|74.8|106.0|69.4|
|NEU-DET|rolled-in_scale|25.656|24.273|25.723|-1.382|74.4|32.8|76.6|
|NEU-DET|scratches|38.112|41.651|35.524|+3.538|74.6|121.2|59.4|

Worst-class nAULC (%), 매 round의 최저 클래스 AP로 구성:

|Dataset|Random|DINOv2-FF|Entropy|
|---|---:|---:|---:|
|GC10-DET|1.563|1.667|1.894|
|NEU-DET|12.007|10.371|12.582|

## 7. Development vs Test

|Dataset|Comparison|Dev difference (%p)|Test difference (%p)|Direction|
|---|---|---:|---:|---|
|GC10-DET|DINOv2-FF - Random|+1.832|+1.472|방향은 같으나 크기가 달라짐|
|GC10-DET|DINOv2-FF - Entropy|+1.952|+1.923|방향은 같으나 크기가 달라짐|
|NEU-DET|DINOv2-FF - Random|+0.086|+0.295|방향은 같으나 크기가 달라짐|
|NEU-DET|DINOv2-FF - Entropy|+0.159|+0.508|방향은 같으나 크기가 달라짐|

## 8. 논문에서 수정해야 할 핵심 수치

국문초록·영문초록·3장 평가 설명·4장 결과·5장 논의·6장 결론에서 기존 ‘최종평가 미실시’ 문구와 development 전용 수치를 검토한다. 교체에 사용할 정확한 test 수치는 `statistics/` CSV와 `raw/results.csv`에 보존했다. 기존 문장은 여기서 임의로 다시 쓰지 않는다.

## 9. 최종평가 결론

- **GC10-DET:** DINOv2-FF의 test 평균 nAULC 차이는 Random 대비 +1.472%p, Entropy 대비 +1.923%p. Development 대비 Random 방향: 방향은 같으나 크기가 달라짐. 클래스별 양·음 차이 공존: 예. Worst-class 값은 위 표와 원장 참조. n=5와 Holm 결과를 넘는 우월성 주장은 하지 않는다.
- **NEU-DET:** DINOv2-FF의 test 평균 nAULC 차이는 Random 대비 +0.295%p, Entropy 대비 +0.508%p. Development 대비 Random 방향: 방향은 같으나 크기가 달라짐. 클래스별 양·음 차이 공존: 예. Worst-class 값은 위 표와 원장 참조. n=5와 Holm 결과를 넘는 우월성 주장은 하지 않는다.

## 10. 생성 파일 목록

- `preflight/`: test 데이터·checkpoint 검증, dev dry-run, native label 검사, 고정 protocol, completeness
- `raw/runs/<run_id>/`: 모델별 config, raw metrics JSON, console·Ultralytics 로그 및 원래 validator 출력
- `raw/results.csv`, `raw/class_ap.csv`, `raw/results.json`: 논문 수치의 machine-readable 원장
- `statistics/`: budget, seed별 nAULC, 대응 비교·bootstrap·Holm, class AP·composition, worst-class, full-data, dev/test 비교
- `figures/`: 제목과 파일명에 TEST가 표시된 PNG 4개
- `FINAL_TEST_REPORT_20260925.md`: 사람이 읽는 요약

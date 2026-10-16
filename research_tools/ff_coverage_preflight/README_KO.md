# 집에서 실행할 FF 재현·교체 가능성 점검

**현재 상태: 코드 준비 및 합성 입력 검증. 원본 GC10 실행 결과는 아직 없다.**
범위는 GC10-DET / 획득 seed 2026091901 / 기존 FF R4→R5다. 기존 `preflight.py`는 CPU·NumPy로 CLS만 읽는다. 추가 진단은 기존 Patch cache를 읽고, 분리된 사후 명령만 R4 정답·검토 이미지를 읽는다. 모델·특징 추출·새 획득 집합 생성·YOLO 학습 기능은 없다. 상세 순서는 아래5절에 있다.

## 먼저 알아둘 사항

- 원래 R4 481장을 고정하고 마지막 80장을 재계산한다. **80장 집합과 실제 획득 순서가 모두 일치**해야 진단이 열린다. R5 누적 목록을 정렬하거나 R5−R4를 임의로 나열해 원래 획득 순서라고 넣으면 안 된다.
- 현재 구현은 공개된 최종 규약을 옮긴 **독립 재계산기**다. 실제 최종 실행 코드의 복사본이 아니다. 원본 소스와 캐시 생성 기록에서 아래 계산 규칙을 확인한 뒤 설정의 `original_protocol_review`를 채운다. 다른 버전의 FF 코드로 대신 확인하지 않는다.
- 규칙: 저장된 float32 CLS를 float64로 변환, `1−dot`, 재정규화·거리 clipping·대각선 0 강제 없음, 매 선택 즉시 최근접 거리 갱신, 최대값과 absolute tolerance 1e-12 내 동점은 opaque ID 문자열 오름차순. 반경 판정에는 tolerance를 더하지 않는다.
- 새로운 계산기는 전체 행렬 곱을 사용한다. 원본의 연산 순서/BLAS가 다르면 근접 동점에서 차이가 생길 수 있다. **불일치하면 중단하고 원인을 조사한다. 통과시키려고 tolerance나 ID를 바꾸지 않는다.** NumPy·BLAS·thread 환경은 결과에 기록된다.
- 해시 일치만으로 파일의 역사적 정당성을 증명하지는 않는다. 원래 cache seal, 실행 manifest, 선택 로그와 대조한 출처를 `hash_evidence`에 남긴다. 현재 파일의 해시만 새로 계산해 정식 캐시로 간주하지 않는다.
- 최초 1대1 허용쌍 0은 이 초기 상태와 계산 규칙에 대한 결과다. 다중 교체·다른 제약까지 불가능하다는 결론은 아니다.
- **8/80은 후속 탐색을 판단할 실무 참고값**이다. 이 도구는 성공/실패 판정에 사용하지 않는다. 8장의 배경 변화보다 5장의 분명한 결함 형태 차이가 더 흥미로울 수 있다. 이 도구 자체는 교체 집합을 만들지 않으므로 순교체 수도 아직 계산하지 않는다.

## 1. 기존 Windows 실험 폴더와 별도 위치에 받기

PowerShell에서 새 폴더 이름을 골라 실행한다. 이미 있는 작업 디렉터리를 덮어쓰거나 reset하지 않는다.

```powershell
git clone --single-branch --branch research/ff-coverage-preflight-20261016 https://github.com/hoyeol-ui/Defect_VLM_Project.git Defect_FF_Preflight
cd Defect_FF_Preflight
python -m unittest discover -s research_tools/ff_coverage_preflight -p "test_*.py" -v
```

NumPy가 설치된 Python 환경을 사용한다. 테스트는 합성 데이터만 사용하며 원본 캐시를 탐색하지 않는다. NumPy가 없으면 별도 venv에 설치하고 버전을 기록한다. 과거 실험 환경의 패키지를 업그레이드하지 않는다.

## 2. 원본 연결 — 가장 중요한 단계

`config.example.json`을 같은 폴더의 `config.local.json`으로 복사한다. 모든 `path`, `sha256`, `hash_evidence`와 ID 파일의 `format`/`keys`를 실제 파일에 맞춰 채운다. JSON 경로는 `/` 또는 `\\`를 사용한다. 상대 경로는 config 파일 위치 기준이다. 예제는 경로를 자동 추측하지 않도록 비워져 있다.

| 입력 | 필요한 실제 내용 |
|---|---|
| features | 정식 GC10 F518 train CLS `(1604,384)` float32 `.npy`; labmeeting의 다른 cache와 혼동 금지 |
| cache_ids | feature 행 순서와 정확히 대응하는 opaque ID 목록. 절대 재정렬 금지 |
| train_ids | 최종 clean split의 train 1604 ID. dev/test 제외를 증명할 manifest에서 확인 |
| r4 / r5 | seed2026091901 DINO_F518의 누적481/561 ID, 원본 `selection_manifest.json`의 `cumulative_ids` 형식은 공개 감사 코드에서 확인됨 |
| new_order | 원본 R5의 **실제 획득 순서**80 ID. 해당 필드명은 집에서 확인 필요. 순서가 저장되지 않았다면 원본 실행 소스로 복원·대조하는 별도 확인부터 진행 |
| selector_source | 실제 사용한 최종 선택 소스 파일. 읽기/해시만 수행하며 임의 import나 실행하지 않음 |
| feature_manifest | 원래 cache seal/manifest. 캐시 SHA·전처리·모델·행 ID 대응을 사람이 확인한 기록 |

ID는 JSON 배열(필요시 `keys: ["cumulative_ids"]`), 문자열형 `.npy` 또는 한 줄 한 ID인 `.txt`를 지원한다. CSV·객체형 pickle은 자동 해석하지 않는다. 원본이 다른 형식이면 **별도 파생 파일**로 내보내고 원본 파일 hash와 순서 보존 근거를 기록한다. 원본을 수정하지 않는다. `.npy` features가 아닌 경우도 원본 비트·행 대응을 보존한 변환이 먼저 필요하다.

현재 맥에서 원본 경로와 schema 전체를 검증할 수 없었으므로 자동 연결은 제공하지 않는다. 참고할 원본 상대 경로는 기존 공개 감사 코드의 `state_attempts/ANNOTATION_PATCH_v2_0_2/features/GC10-DET/manifest.json`, `runs/GC10_v2_a2026091901_DINO_F518_R4/selection_manifest.json`, R5의 같은 파일이다. 이것은 기존 export 내부 상대 경로이며 Windows의 절대 위치는 집에서 찾아야 한다.

## 3. FF만 재현하기

```powershell
python research_tools/ff_coverage_preflight/preflight.py verify --config research_tools/ff_coverage_preflight/config.local.json --out research_tools/ff_coverage_preflight/outputs/verify_01
```

`ff_reproduction.json`에서 마지막80장 집합, 획득 순서, 누적 R5 집합 일치를 확인한다. step별 점수·동점 수·최초 순서 불일치·누락/추가 ID도 저장된다. `blocked.json`이면 입력부터 해결한다. `PASS`는 **이 cache에서 이 전이의 재현**만 의미하며 전처리 재추출이나 모든 라운드의 재현을 뜻하지 않는다. 먼저 결과를 검토하고 다음 명령의 실행 여부를 결정한다.

## 4. 통과한 경우 최초 1대1 교체만 검사

```powershell
python research_tools/ff_coverage_preflight/preflight.py screen --config research_tools/ff_coverage_preflight/config.local.json --out research_tools/ff_coverage_preflight/outputs/screen_01 --seconds 1800
```

1800초는 제안한 **screen 루프 시간 상한**이며 예상 소요시간이 아니다. 파일 검사/FF 재현/행렬 생성 시간은 별도다. 새 실행도 입력 hash와 FF를 다시 검사한다. FF 불일치 시 screen으로 넘어갈 수 없다. 최초 상태의80×1043=83,440쌍을 검사하고, R4는 제거하지 않는다. 원래 전체561장 기준 최대 반경을 유지하는 쌍만 CSV에 기록한다. 원본 집합은 바꾸지 않는다.

- `swap_feasibility.json`: 검사 수/전체 수, 허용쌍 수, 고유 유입·유출 수, 원래 반경·평균·95분위, 시간, 완료 여부.
- `feasible_initial_pairs.csv`: 각 허용쌍의 유입·유출 ID와 반경 변화. 미완료일 때는 **부분 결과**다.
- `provenance.json`: 입력·도구 hash와 환경. 로컬 절대 경로가 포함되므로 결과 공유 전 확인한다.
- 종료코드0=정상 완료(허용쌍0도 포함), 2=입력/재현 차단, 3=시간 상한으로 미완료. 양의 반경 차이가1e-12 이하여도 허용하지 않고 별도로 센다.

허용쌍이 있으면 Patch 비용·표현·정성 점검 설계를 다시 확인한다. 자동으로 Patch나 YOLO를 실행하는 후속 단계는 없다. 0이거나 미완료라면 각각 다른 의미로 보고한다. 논문 성과/노블티/AP 개선은 이 결과로 판정하지 않는다.

## 5. 추가된 최소 검증: Patch 점수가 결함과 관련되는가?

**준비 코드이며 실제 원본 실행은 아직 없다.** 단계는 `verify → screen → freeze → (별도 결정 후 준비된 cache 사용) score → evaluate-r4`다. 실제 교체 결과가 생긴 뒤에만 `pairs → render-pairs`를 실행한다. 원래 FF 실행기는 수정하지 않았다.

| 프로그램/명령 | 하는 일 | 접근하지 않는 정보 |
|---|---|---|
| `patch_signal.py freeze` | 통과한 screen의 입력 hash를 재확인하고 R4 24장·진단 규칙·코드 hash 봉인 | GT·이미지·Patch |
| `patch_signal.py score` | 기존 Patch cache로 R4 leave-one-out 기여 계산, 상위 최대3개 지역 고정 | GT·이미지 |
| `patch_posthoc.py evaluate-r4` | 봉인된 점수를 고정 R4 표본의 box와 사후 비교; 선택적으로 HTML | 미라벨 GT·선택 알고리즘 |
| `patch_signal.py pairs` | 외부에서 봉인된 Patch/임의교체 최종 집합의 반경 확인, 쌍 매칭·기여 진단 | 모든 GT·이미지 |
| `patch_posthoc.py render-pairs` | 정답 없는 A/B 검토 화면·빈 기록표 생성 | 모든 GT; annotation 인자를 거부 |

GT를 읽는 파일은 `patch_posthoc.py` 하나다. 선택/점수 측은 이를 import하지 않는다. `used_gt:false` 같은 메타데이터는 출처의 선언을 검사하는 것으로, 과거 추출·선택의 정보 사용을 코드가 소급 증명하는 것은 아니다. 집에서 실제 exporter/selector 소스의 정보 경계를 확인해야 한다.

### 5-1. GT를 열기 전에 R4 24장 고정

```powershell
$taskRoot = "research_tools/ff_coverage_preflight"
python "$taskRoot/patch_signal.py" freeze --config "$taskRoot/config.local.json" --screen-dir "$taskRoot/outputs/screen_01" --out "$taskRoot/outputs/patch_plan_01"
```

`plan.json`과 `plan.sha256`이 생긴다. 표본은 ID-hash 순서24장이며 클래스·결함 크기·좋아 보이는 예시를 보고 고르지 않는다. 24장은 탐색용 제안값이며 통계적 검정력을 확보한 표본 수가 아니다. R4가 이미 FF로 선택된 집합이므로 모집단을 대표하는 독립 검증셋도 아니다. 반복 실행으로 다른 표본을 고르지 않는다.

### 5-2. 필요한 cache: 아직 없으면 여기서 중단

**이 패키지에 DINO 추출기·prototype 학습기·Patch 교체 selector는 포함되지 않는다.** CLS에서 Patch를 복원하지 않는다. 다음 파일이 이미 준비돼 있거나, 별도로 추출·구성하기로 결정된 뒤에만 후속 명령을 사용할 수 있다. 진단 때문에 YOLO는 필요 없다.

`patch_cache.npz`의 허용 필드는 아래 네 개뿐이다(`allow_pickle=False`). GT·class·bbox 필드를 넣지 않는다.

| 필드 | 형식 |
|---|---|
| `ids` | Unicode `[1604]`, 정식 train ID와 일치; feature 행 대응 유지 |
| `q` | float32 `[1604,16,384]`, 각 이미지의 정규화된4×4 지역 descriptor |
| `valid` | bool `[1604,16]`, 유효 지역 여부; invalid q는0 |
| `prototypes` | float32 `[256,384]`, 동일 train pool에서 GT 없이 만든 정규화 prototype |

지역 번호는 **padding을 제외한 원본 영상의4×4 격자, row-major 0..15**다. 전체 F518 canvas의4×4를 쓰면 안 된다. 저장된 마지막 norm 이후 `[37,37,384]` token이 있다면 `pool_saved_tokens(tokens, valid_xyxy)`가 token과 각 지역의 교차 면적을 가중해 평균·L2 정규화한다. `valid_xyxy`는 실제 resize·padding 기록에서 얻은 **518 canvas 기준 정규화 유효 영상 경계**다. GT box를 여기에 넣지 않는다. 완전 padding token은 제외하지만 self-attention에 유입된 padding 문맥까지 제거하는 것은 아니다.

이 함수는 이미 저장된 token용 CPU 변환이며 모델 추론을 하지 않는다. 원래 token과 집계 표현의 차이를 더 보고 싶다면 별도의 진단으로 기록한다. 이번 자동 겹침 점검은 실제 선택에 쓰려는4×4 표현만 검증하며, 원래 모든 token의 결함 관련성까지 검증했다고 주장하지 않는다.

프로토타입은 기존 제안대로 train 지역 descriptor의 ID-hash reservoir 최대10,000개, spherical k-means256개, seed2026101600을 사용한다. 실제 fitting 구현·종료 조건·reservoir tie 규칙은 추출 준비 시 GT를 보기 전에 기록하고 hash로 봉인해야 한다. 이 진단 도구가 fitting을 실행하거나 과거 prototype을 동일하게 재현해 주는 것은 아니다.

`patch_cache_meta.local.json` 예시(placeholder는 실제 증거로 교체):

```json
{
  "policy": "F518_lastnorm_unpadded_4x4_area_weighted_L2_v1",
  "used_gt": false,
  "cls_reproduction_passed": true,
  "canonical_cls_sha256": "ORIGINAL_CLS_CACHE_SHA256",
  "train_ids_digest": "DIGEST_OF_FROZEN_TRAIN_IDS",
  "prototype_fit_ids_digest": "DIGEST_OF_SAME_ALLOWED_TRAIN_POOL",
  "prototype_seed": 2026101600,
  "prototype_reservoir_limit": 10000,
  "extractor_sha256": "VERIFIED_EXTRACTOR_SHA256",
  "weights_sha256": "VERIFIED_WEIGHTS_SHA256",
  "preprocessing_sha256": "VERIFIED_PREPROCESSING_SHA256",
  "cls_check_evidence": "원본 CLS 재현·token 좌표·padding 처리 확인 결과와 보고서 경로를 기록",
  "prototype_fit_evidence": "fitting 소스 hash, reservoir 규칙, 반복/종료 조건을 기록"
}
```

ID digest는 `patch_signal.digest_ids(ids)`로 계산한다. 허용 pool의 digest이며 모든 ID가 reservoir에 실제 포함됐다는 뜻은 아니다. `cls_reproduction_passed:true`는 새 추출 경로의 CLS를 canonical cache와 대조한 뒤에만 설정한다. 기존 FF 재현 통과와는 별도 확인이다.

### 5-3. 정답 없이 점수부터 봉인

```powershell
$planFile = "$taskRoot/outputs/patch_plan_01/plan.json"
$planHash = (Get-FileHash $planFile -Algorithm SHA256).Hash.ToLowerInvariant()
$cacheFile = "$taskRoot/local_inputs/patch_cache.npz"
$cacheHash = (Get-FileHash $cacheFile -Algorithm SHA256).Hash.ToLowerInvariant()
$metaFile = "$taskRoot/patch_cache_meta.local.json"
$metaHash = (Get-FileHash $metaFile -Algorithm SHA256).Hash.ToLowerInvariant()
python "$taskRoot/patch_signal.py" score --plan $planFile --plan-sha $planHash --cache $cacheFile --cache-sha $cacheHash --meta $metaFile --meta-sha $metaHash --out "$taskRoot/outputs/r4_signal_01"
```

위 hash 계산은 전달 중 변경 확인용이다. hash를 매번 계산한다고 과거 출처나 사전 봉인을 증명하지는 않는다. 최초 값과 파일을 보존하고, box를 본 뒤 cache/prototype/상위지역 수를 바꿔 같은 실험으로 보고하지 않는다.

R4의 각 표본 x에 대해 **나머지 R4 전체480장**을 참조한다. 표본24장끼리만 비교하지 않는다.

`gain_j(x) = max(0, a_xj − max_{y∈R4\{x}} a_yj) / 256`

prototype별 이득을 해당 prototype과 가장 유사한 유효 지역에 배분한다. 동점은 작은 지역 번호다. 지역별 기여 합은 `F(R4)−F(R4\{x})`와 일치한다. 자기 자신을 참조에 넣어 인위적으로 기여를0으로 만들지 않는다. 양의 기여 지역 중 최대3개를 고정하며 전체 기여≤1e-9면 **포화/극소**로 남기고 임의 상위지역을 그리지 않는다. 이는 선택 목적함수와 연결된 진단이지만, R5의 실제 획득 점수는 아니다.

### 5-4. 고정 R4 표본만 사후 평가

`r4_boxes.local.json`은 **plan의24 ID만** 담는다. 다른 R4 ID나 미라벨/dev/test ID가 섞여도 거부한다. 외부 파일의 자동 검색·XML/YOLO parsing은 하지 않으므로 집에서 읽기 전용 원본으로부터 별도 export한다.

```json
{
  "coordinate_system": "original_image_normalized_xyxy",
  "boxes": {
    "REPLACE_WITH_FROZEN_R4_ID": [[0.25, 0.25, 0.5, 0.5]]
  }
}
```

예시 box는 실행용 정답이 아니다. 원본 픽셀 xyxy는 실제 width/height로 정규화하고, YOLO cx/cy/w/h는 xyxy로 변환해야 한다. F518 padded 좌표를 넣지 않는다. `[]`는 확인된 box 없음이며 누락 annotation을 대신하지 않는다. box/class를 feature 생성·점수·prototype에 넘기지 않는다.

선택적으로 원본 이미지 경로·hash를 `images.local.json`에 기록하면 HTML까지 만든다. 상대 경로는 이 JSON 위치 기준이며, 이미지별 SHA도 검사한다. EXIF 회전이 있으면 자동 회전하지 않고 좌표 확인을 위해 중단한다.

```json
{"images": {"REPLACE_WITH_ID": {"path": "C:/actual/image.jpg", "sha256": "ORIGINAL_IMAGE_SHA256"}}}
```

```powershell
$scoreFile = "$taskRoot/outputs/r4_signal_01/scores.json"
$scoreHash = (Get-FileHash $scoreFile -Algorithm SHA256).Hash.ToLowerInvariant()
$boxFile = "$taskRoot/r4_boxes.local.json"
$boxHash = (Get-FileHash $boxFile -Algorithm SHA256).Hash.ToLowerInvariant()
$imageFile = "$taskRoot/images.local.json"
$imageHash = (Get-FileHash $imageFile -Algorithm SHA256).Hash.ToLowerInvariant()
python "$taskRoot/patch_posthoc.py" evaluate-r4 --plan $planFile --plan-sha $planHash --scores $scoreFile --scores-sha $scoreHash --annotations $boxFile --annotations-sha $boxHash --images $imageFile --images-sha $imageHash --out "$taskRoot/outputs/r4_posthoc_01"
```

HTML 생성에만 Pillow가 필요하다. 정량 결과만 필요하면 `--images`, `--images-sha`를 생략한다. 설치가 필요하면 기존 훈련 환경 대신 별도 환경을 쓴다.

`box_overlap.json`은 강조된 지역 면적 중 GT box **합집합**이 차지하는 비율을 보고한다. 겹치는 box를 중복 계산하지 않는다. 동일 이미지의 유효 지역에서 같은 개수 k를 균일·비복원 추출하는200회 대조를 ID별 고정seed로 계산하며, 그 기대값은 전체 유효 지역 겹침률의 평균으로 정확히 계산한다. Monte Carlo 5/95분위는 **무작위 공간 대조 분포**이며 성능 신뢰구간이나 통계적 유의성 검정이 아니다. 선택 이미지 무작위 대조군3개와도 다른 대조다.

포화·box 없음의 수를 별도 보고하며 전체 표본을 다른 표본으로 교체하지 않는다. 평균 차이는 양의 기여와 box가 모두 있는 표본만의 기술통계로 분모를 명시한다. 큰 box·배경 포함·4×4의 거친 해상도·이미지 간 의존성 때문에 겹침 상승을 결함 이해나 AP 향상으로 해석하지 않는다.

`review.html`의 주황은 기여 지역, 초록은 R4 box다. `review_sheet.csv`에 형태 차이/배경·조명/padding/중복/불명확을 기록한다. 정답을 보고 수정하면 새 탐색 버전이다. 같은24장을 반복 튜닝한 뒤 독립 검증으로 보고하지 않는다.

### 5-5. 실제 교체 결과가 생긴 뒤 정답 없는 A/B 검토

이 단계는 **미리 봉인된 최종 선택 결과를 읽는 adapter**다. 새 선택 알고리즘이나 임의교체 생성기는 구현하지 않았다. 원본 FF, Patch, 임의 허용교체3개를 생성·봉인하는 별도 단계가 준비되기 전에는 실행할 입력이 없다.

`selections.local.json` schema:

```json
{
  "plan_sha256": "FROZEN_PLAN_SHA256",
  "cache_sha256": "SAME_PATCH_CACHE_SHA256",
  "meta_sha256": "SAME_META_SHA256",
  "used_gt": false,
  "arms": {
    "patch": {"new_ids": ["80_FINAL_NEW_IDS"], "accepted_swaps": 0},
    "random_1": {"new_ids": ["80_FINAL_NEW_IDS"], "accepted_swaps": 0, "seed": 2026101601},
    "random_2": {"new_ids": ["80_FINAL_NEW_IDS"], "accepted_swaps": 0, "seed": 2026101602},
    "random_3": {"new_ids": ["80_FINAL_NEW_IDS"], "accepted_swaps": 0, "seed": 2026101603}
  }
}
```

`new_ids`는 누적561장이 아니라 최종80장 신규분이다. `accepted_swaps`는 실제 수락 횟수(최대80), 순교체 수와 다르다. 임의 대조군은 같은 수락 횟수를 목표로 하지만 도달하지 못하면 그대로 기록하고 동등 조건으로 해석하지 않는다. seed는 원래 획득seed와 별개의 진단 대조seed다.

```powershell
$selectionFile = "$taskRoot/selections.local.json"
$selectionHash = (Get-FileHash $selectionFile -Algorithm SHA256).Hash.ToLowerInvariant()
python "$taskRoot/patch_signal.py" pairs --config "$taskRoot/config.local.json" --plan $planFile --plan-sha $planHash --cache $cacheFile --cache-sha $cacheHash --meta $metaFile --meta-sha $metaHash --selections $selectionFile --selections-sha $selectionHash --out "$taskRoot/outputs/pairs_signal_01"
$pairFile = "$taskRoot/outputs/pairs_signal_01/scores.json"
$pairHash = (Get-FileHash $pairFile -Algorithm SHA256).Hash.ToLowerInvariant()
python "$taskRoot/patch_posthoc.py" render-pairs --plan $planFile --plan-sha $planHash --scores $pairFile --scores-sha $pairHash --images $imageFile --images-sha $imageHash --out "$taskRoot/outputs/pairs_review_01"
```

각 arm의 최종 집합에 대해 동일 CLS에서 ε=0 반경·고정R4·80장 예산을 재확인한다. 최종 순유입 ID 오름차순으로 아직 미매칭인 순유출 중 CLS 거리가 가장 작은 것과 묶는다(동점 ID순). 전역 최적 매칭이나 실제 교체 trace는 아니다. ID-hash로 arm당 최대20쌍을 고정하고, 방법·유입/유출 표시는 가린 A/B 화면과 별도 `unblind_key.json`을 만든다. 키를 보기 전에 CSV를 작성한다. 이 절차는 편향을 줄이는 보조 수단이며, 이미지를 아는 연구자까지 완전히 맹검화하지는 못한다.

기여 지도는 두 이미지에 공통으로 `원래 R5 − 대응 유출 이미지`를 참조한 주변 기여다. 실제 순차 선택 당시의 참조 집합과 다를 수 있으므로 **선택의 인과적 이유/실제 step trace라고 부르지 않는다.** 실제 step 설명에는 별도로 봉인된 선택 trace가 필요하다. 쌍 검토에는 미라벨 GT를 열지 않는다.

**판단:** 겹침·국소 형태 관찰이 일관되면 추가 검증 근거, 배경 위주면 표현 재검토, 포화면 목적함수의 민감도 문제로 기록한다. 8장이나 특정 겹침률을 성공 기준으로 사용하지 않는다. 5장 변화도 검토 가치가 있을 수 있다. YOLO 학습·새 특징 추출을 자동 실행하는 경로는 없다.

## 집의 Codex에 전달할 요청

> 이 폴더의 README와 DECISION_KO.md를 읽고 원본 GC10 정식 CLS cache, 행 ID, train manifest, seed2026091901 FF R4/R5 목록과 실제 R5 획득 순서, 최종 실행 선택 소스를 읽기 전용으로 찾아 연결해 주세요. 과거 다른 F518 cache로 대체하지 마세요. 실제 원본 규칙과 새 checker의 차이를 먼저 보고하고 config.local.json을 채워 주세요. 새 특징 추출·학습·원본 수정은 하지 마세요. 우선 verify만 실행해 마지막80장의 집합과 순서 재현 결과를 보고해 주세요. screen은 그 결과를 검토한 뒤 결정하겠습니다. Patch 관련 단계는 README 5절의 사전 봉인·GT 분리 규칙을 따르며, cache와 prototype이 없으면 추출을 자동 실행하지 말고 필요한 준비를 보고해 주세요. R4 GT는 표본·점수를 봉인한 후 사후 평가에서만 읽어 주세요.

## 이번 맥북 검증 기록

2026-10-16, Python 3.12.14, NumPy 2.3.5에서 합성 입력 테스트15개 통과. 빠른 교체 판정과 독립적인 직접 전수 재계산의 일치, 동점/즉시 갱신, 행-ID 대응, 허용 반경 경계, 시간 상한, 실제 설정 schema 로딩, 해시 변조, 비정규화 cache, 순서 불일치 차단, 출력 덮어쓰기 방지를 검사했다. `git diff --check`도 통과했다. Windows와 원본 GC10 자료에 대한 검증은 귀가 후 남아 있다.

추가 준비 검증: Patch 진단은 합성 배열·합성 이미지에서만 검사했다. 원본 특징 추출·prototype fitting·표본 선택·원본 GT 분석·학습은 실행하지 않았다. Python 3.12.14 / NumPy 2.3.5 / Pillow 12.3.0에서 기존15개와 새16개, 총31개 테스트가 통과했다. 기여 합과 목적함수 차이의 일치, padding 면적 가중, GT ID 차단, 좌표·box 합집합, 무작위 공간 대조 재현, hash 변경 차단, cache→점수→사후 평가 연결 및 합성 HTML 생성을 확인했다.

# Asset Manifest

## 범위

이 manifest는 노션 재업로드 패키지에 포함된 대표 개요 도식과 기존 본문 그림만 기록한다. `thesis_notion_v0_2_asset_index.csv`에서 `asset_type=figure`인 네 파일을 포함했으며, source CSV·JSON은 본문의 표·경로 참조로 유지하고 패키지 자산으로 복제하지 않았다.

## 포함 자산

| packaged_path | asset_role | source_origin | bytes | SHA-256 | 상태 |
|---|---|---|---:|---|---|
| `assets/overview_research_architecture.svg` | 대표 연구 개요 벡터 도식 | 자기 규정·RQ1–RQ4·EP01–EP05 및 Appendix A의 확정 표현으로 코드 생성 | 11753 | `6d3a90aeabe6213014b9d663edc7a565dfeec5d8686d480b00d5f41e5bfb9627` | 포함 |
| `assets/overview_research_architecture.png` | 대표 연구 개요 PNG | 위 SVG를 1920×1440으로 렌더링 | 266063 | `03f010c2b252c7b983ae1801b4e52ae1cd81c902279c13ee44ead2bad2aeb55e` | 포함 |
| `assets/overview_research_architecture_caption.md` | 대표 도식 캡션·해석 경계 | 대표 도식과 함께 신규 작성 | 1316 | `d7e63e83bf1ec16e7f4a023c09fa426b7999189c9fb84823b1e50a600113d842` | 포함 |
| `assets/figure04_gc10_endpoint_conflict.png` | 제4장 4.4 기존 그림 | `docs/notion_professor_update_20260803/assets/figure04_gc10_endpoint_conflict.png` | 136513 | `a982a9a584c66d4f12d9d6981d2dd3b461335f375bdd96a3c4e7981a31973335` | 포함·원본 해시 일치 |
| `assets/figure05_neu_two_reproducibilities.png` | 제4장 4.3 기존 그림 | `docs/notion_professor_update_20260803/assets/figure05_neu_two_reproducibilities.png` | 193457 | `b91894c35cfcc441426f058dcc7616186e6de325a8824d0f39e054672d92e59f` | 포함·원본 해시 일치 |
| `assets/figure06_discovery_composition_contrast.png` | 제4장 4.2 기존 그림 | `docs/notion_professor_update_20260803/assets/figure06_discovery_composition_contrast.png` | 221319 | `e2d2652a979864f669410aeaa46b6d998571fd58d034e9b7b2c34729e44f1025` | 포함·원본 해시 일치 |
| `assets/figure08_ksdd2_positive_below_threshold.png` | 제4장 4.5 기존 그림 | `docs/notion_professor_update_20260803/assets/figure08_ksdd2_positive_below_threshold.png` | 143188 | `d4ea4c7a3222eabd2a3a18b1eadcba9d65177a18303ec843458831ee18e9d3d0` | 포함·원본 해시 일치 |

## 기존 공개 URL 매핑

- [figure06_discovery_composition_contrast.png](https://raw.githubusercontent.com/hoyeol-ui/Defect_VLM_Project/137e53e05fd7f710bd001e311ec3af0676a513dc/docs/notion_professor_update_20260803/assets/figure06_discovery_composition_contrast.png)
- [figure05_neu_two_reproducibilities.png](https://raw.githubusercontent.com/hoyeol-ui/Defect_VLM_Project/137e53e05fd7f710bd001e311ec3af0676a513dc/docs/notion_professor_update_20260803/assets/figure05_neu_two_reproducibilities.png)
- [figure04_gc10_endpoint_conflict.png](https://raw.githubusercontent.com/hoyeol-ui/Defect_VLM_Project/137e53e05fd7f710bd001e311ec3af0676a513dc/docs/notion_professor_update_20260803/assets/figure04_gc10_endpoint_conflict.png)
- [figure08_ksdd2_positive_below_threshold.png](https://raw.githubusercontent.com/hoyeol-ui/Defect_VLM_Project/137e53e05fd7f710bd001e311ec3af0676a513dc/docs/notion_professor_update_20260803/assets/figure08_ksdd2_positive_below_threshold.png)

기존 업로드 로그상 네 URL은 HTTP 200과 `image/png`로 검증됐으며, 이 패키지에서는 외부 URL 의존을 줄이기 위해 동일 해시의 로컬 사본을 사용한다.

## 의도적으로 포함하지 않은 자산

- 4.1 전용 외부 그림: 현재 장 경계를 만족하는 확정 그림이 없어 자동 대체하지 않았다.
- `translation_atlas_main_figure.png`, `figure03_branch_evidence_map.png`: GC10 K40/140 또는 STOP 분기가 제4장에 유입될 위험 때문에 제외했다.
- source CSV·JSON: clean manuscript의 Markdown 표와 출처 경로를 유지하며 이미지 자산으로 복제하지 않았다.

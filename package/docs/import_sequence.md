# Notion 재업로드 순서

## 권장 순서

1. ZIP을 로컬에서 풀고 `package/` 구조가 유지되는지 확인한다.
2. `manuscript/thesis_notion_import_clean.md`를 Notion의 Markdown import 대상으로 선택한다.
3. 4.2–4.5의 그림 네 개가 정상 표시되는지 확인한다.
4. 누락된 그림이 있으면 `assets/`의 PNG를 아래 순서대로 해당 캡션 위치에 수동 첨부한다.
5. `assets/overview_research_architecture.png`를 제3장 3.1 말미 또는 별도 연구 개요 위치에 수동 첨부한다.
6. `overview_research_architecture_caption.md`의 캡션과 해석 경계를 이미지 아래에 배치한다.
7. Appendix A의 NUM-111 문장이 상태-only 표현인지 확인한다.
8. 전체 목차·각주·표의 Notion 렌더링을 점검한 뒤 기존 페이지와 교체하거나 새 페이지로 게시한다.

## 기존 그림 수동 첨부 순서

1. 4.2 — `figure06_discovery_composition_contrast.png`
2. 4.3 — `figure05_neu_two_reproducibilities.png`
3. 4.4 — `figure04_gc10_endpoint_conflict.png`
4. 4.5 — `figure08_ksdd2_positive_below_threshold.png`

## 이미지가 누락될 때

Notion importer가 `../assets/` 상대경로를 자동 처리하지 못하면 원고의 캡션과 원본 경로 표기를 그대로 두고, 위 네 PNG를 수동으로 끌어다 놓는다. 파일명이 고유하므로 충돌용 순번은 필요하지 않다. 기존 commit SHA 기반 공개 URL은 `assets/asset_manifest.md`에 보관되어 있다.

## 표·CSV·JSON 처리

본문의 표는 Markdown 표로 유지한다. 원문에 적힌 CSV·JSON 경로는 출처 추적용 텍스트이며 Notion 데이터베이스로 자동 변환하지 않는다. 별도 계산이나 값 변환을 수행하지 않는다.

## 직접 수정 금지 항목

재업로드 과정에서 자기 규정 3문장, RQ1–RQ4, EP01의 6.1 단일 배치, branch별 증거 상태 및 수치 자릿수를 수정하지 않는다. 실제 Notion 페이지의 편집은 이 패키지 생성 범위에 포함되지 않는다.

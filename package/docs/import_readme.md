# Notion Import Package 2026-08-08

## 이 ZIP의 목적

이 패키지는 NUM-111 상태-only 패치가 반영된 v0.2 원고를 Notion에 다시 올리기 위한 import source와 그림 자산을 한 묶음으로 제공한다. 논문 본문, 핵심 claim EP01–EP05, 자기 규정 3문장 및 수치는 새로 수정하거나 확장하지 않았다.

## 가장 먼저 열 파일

`manuscript/thesis_notion_import_clean.md`

이 파일이 Notion 재업로드의 기준 원고다. 패치된 Notion-ready 원고를 바탕으로 했으며, 기존 그림 네 개의 참조만 `../assets/` 상대경로로 정리했다. `manuscript/thesis_full_clean.txt`는 내용 대조용 보조 사본이다.

## 대표 이미지와 본문의 관계

`assets/overview_research_architecture.png`와 SVG 원본은 연구 전체의 평가 단위, RQ1–RQ4, EP01–EP05 및 Appendix A 보조 분기를 한 장에 요약한다. 이 도식은 신규 실험 결과나 claim이 아니며, 제4장의 기존 결과 그림을 대체하지 않는다. 권장 위치는 제3장 3.1 전체 연구 설계 말미 또는 별도 연구 개요 블록이다.

캡션과 해석 경계는 `assets/overview_research_architecture_caption.md`에 있다.

## 패키지 구성

- `manuscript/`: clean Markdown, 검증용 TXT, cleaning log
- `assets/`: 대표 도식 SVG·PNG·캡션, 기존 제4장 그림 네 개, 자산 manifest
- `docs/`: import 순서, 검증 결과, 전체 포함 파일 표

## 중요 경계

- `NOT_AVAILABLE`은 downstream 실패를 뜻하지 않는다.
- NEU-DET fixed-set 결과는 `POSTHOC_DESCRIPTIVE`이다.
- acquisition reproducibility는 동일 fixed pool 범위다.
- KSDD2의 +0.05는 결과 확인 전에 고정한 기준이며 외부 보정 임계값이 아니다.
- Appendix A의 GC10 K40/140은 상태-only 문장과 `expansion not authorized` 지위를 유지한다.
- 이 패키지는 특정 selector의 일반적 우월성, framework 검증, 비용 절감, 산업 benchmark 또는 독립 production pool 일반화를 주장하지 않는다.

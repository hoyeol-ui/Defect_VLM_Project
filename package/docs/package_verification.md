# Package Verification

## 최종 상태

`SUCCESS`

## 포함 현황

- 예상 package 파일 수: 16개
- manuscript: 3개
- assets: 9개
- docs: 4개
- 기존 그림: 4개 모두 포함
- 대표 개요 이미지: SVG와 PNG 모두 생성
- asset manifest: Markdown과 CSV 모두 생성
- included files table: CSV 생성

## Clean manuscript 검증

- 기준: `drafts/thesis_notion_v0_2_notion_ready_num111patched.md`
- 이물 블록 검사: 발견 없음
- 제거된 논문 문단: 없음
- 변경 범위: 기존 그림 네 개의 URL을 패키지 상대경로로 치환
- TXT 검증 사본: 기준 TXT와 내용 동일

## NUM-111 검증

- 기존 `selected-set coverage 차이가 +0.015로 기록되어` 표현: clean 원고에 없음
- 상태-only 문장: clean Markdown과 TXT에 각각 존재
- 유지 문장: `GC10 K40/140 분기에서는 K40 후보가 holdout의 all-class coverage 기준을 충족한 것으로 기록되었다.`
- 영향 범위: Appendix A 보조 평가에 한정, EP01–EP05 무변경

## 대표 이미지 검증

- SVG: 1920×1440 viewBox, 코드 기반 벡터 도식
- PNG: 1920×1440 렌더링 완료
- 육안 검사: 글자 잘림, 박스 중첩, 깨진 화살표 없음
- 포함 요소: 자기 규정 요약, RQ1–RQ4, 평가 단위 흐름, MPDD·VisA·NEU-DET·GC10 q20·KSDD2·Appendix A, EP01 통합 문장
- 금지 표현 검사: selector superiority, framework validation, 비용 절감 입증, 산업 benchmark, production pool 일반화 주장을 도식의 결론으로 사용하지 않음

## 기존 이미지 검증

네 PNG의 package 사본 SHA-256은 `thesis_notion_v0_2_upload_index.csv`의 로컬 원본 해시와 일치한다. 파일명 충돌이나 누락은 없다.

## 남는 수동 확인 사항

1. Notion importer가 `../assets/` 상대경로 그림을 자동 첨부하는지 확인해야 한다.
2. 대표 개요 이미지는 본문에 자동 삽입하지 않았으므로 위치를 사람이 선택해야 한다.
3. 기존 NEU-DET 그림 내부의 구 용어는 본문·캡션 경계를 우선해 해석해야 한다.
4. 4.1에는 확정 경계를 만족하는 외부 그림이 없어 `자료 없음` 상태가 유지된다.
5. Notion에서 직접 수정한 문장이 생기면 로컬 clean source와 다시 동기화해야 한다.

## ZIP 검증 원칙

최종 ZIP에는 최상위 `package/` 폴더와 위 16개 파일만 포함하며, 원본 drafts·runs·docs 파일이나 임시 빌더는 넣지 않는다.

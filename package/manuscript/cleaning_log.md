# Manuscript Cleaning Log

## 기준 파일

- Markdown 기준 원고: `drafts/thesis_notion_v0_2_notion_ready_num111patched.md`
- TXT 검증 원고: `drafts/thesis_v0_2_full_draft_num111patched.txt`
- 생성 일자: 2026-08-08

## 이물 블록 검사

다음 문자열을 기준으로 외부 패치 로그·CSV 내용이 논문 원고 안에 유입됐는지 확인했다.

- `"target_file","found_sentence"`
- `drafts/thesis_v0_2_full_draft_num111patched.txt`
- `num111_patch_application`
- `replacement_status`
- `patch application table`

검사 결과 두 기준 원고에서 해당 이물 블록은 발견되지 않았다. 따라서 논문 본문 문장이나 문단을 삭제하지 않았다.

## 수행한 변경

1. Markdown 기준 원고를 `thesis_notion_import_clean.md`로 복사했다.
2. 패키지 내부에서 함께 제공되는 기존 그림 4개의 원격 URL을 `../assets/` 상대경로로 바꿨다.
3. TXT 검증 원고는 내용 변경 없이 `thesis_full_clean.txt`로 복사했다.
4. Appendix A의 NUM-111 상태-only 문장과 나머지 본문 문장은 유지했다.

## 제거된 이물 블록

없음.

## 본문 보존 원칙

이미지 참조 경로 외에는 문장, 수치, claim, 장절 순서, 각주 및 캡션을 변경하지 않았다. 새로 생성한 대표 개요 이미지는 clean manuscript 본문에 자동 삽입하지 않고 패키지 자산으로만 제공한다.

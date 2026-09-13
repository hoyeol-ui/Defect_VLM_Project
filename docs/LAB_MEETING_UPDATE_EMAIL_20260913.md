# 교수님 공유용 이메일 초안

제목: 학위논문 추가 실험과 original-final 평가 결과 공유드립니다

교수님, 안녕하세요. 이호열입니다.

지난번에 말씀드렸던 학위논문 연구의 추가 실험과 평가 결과를 정리해 다시 공유드립니다. 지난 자료에서는 Entropy와 Full-data 학습까지 정리했는데, 이번에는 Entropy 200개와 Full-data 4개 checkpoint의 original test 기준 평가까지 완료했습니다. 기존 프로토콜을 끝까지 확인한 exploratory 결과로, 독립적인 논문 주결과와 구분해 정리했습니다.

Random, DINO, Entropy를 비교하면 두 데이터셋 모두 전체 학습곡선 nAULC에서는 DINO가 가장 높았고, 마지막 R5에서는 Entropy가 가장 높았습니다. 다만 차이가 아주 크지는 않았고, 한쪽이 계속 우세하게 나오지는 않았습니다. 전체 train pool을 사용한 Full-data original-final mAP50–95 평균은 GC10 약 32.04%, NEU 약 42.97%였습니다.

업데이트한 랩미팅 자료와 GitHub 연구 기록을 함께 공유드립니다.

첨부: 이호열_학위논문_랩미팅_연구진행보고_20260913_최신실험결과.docx

[GitHub 연구 기록](https://github.com/hoyeol-ui/Defect_VLM_Project/tree/research/development-3way-fullref-20260913)

이번 미팅에서는 현재 비교 범위와 결과 해석으로 논문을 정리하는 방향이 적절한지, 추가 실험이 더 필요한지 교수님 의견을 듣고 싶습니다.

감사합니다.
이호열 드림

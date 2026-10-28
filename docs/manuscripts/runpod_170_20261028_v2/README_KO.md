# 170모델 논문 개정본 2026년 10월 28일

[개정 원고 DOCX](03_JKICS_Revised_Manuscript_170models_20261028_v2.docx) · [웹에서 읽는 전체 원고](03_JKICS_Revised_Manuscript_170models_20261028_v2.md) · [저장된 최종 결과](../../results/runpod_170_final_20261027/README_KO.md)

이 개정본은 170개 모델의 기존 최종 Test를 바탕으로 연구 구성과 해석을 보강한 9쪽 원고다. 표 9개와 그림 6개, 원본과 동일한 네 개의 편집 가능한 수식을 포함한다. DOCX에는 주요 편집 지점을 설명하는 Word 검토 댓글 3개가 있다.

## 연구 질문과 개정 내용

동일한 이미지 예산과 학습 조건에서 선택 전략의 학습곡선이 어떻게 달라지는지, 전체 평균의 이득이 개별·최저 클래스에서도 유지되는지를 연결했다. 그림 1은 선택 표현, 누적 라벨 집합, 예산별 학습 초기화, Dev checkpoint 선택과 최종 Test 경로를 보여준다. DINOv2 특징은 표본 선택에 사용하며 YOLO26s 검출기는 공통이다.

![표본 선택과 검출기 학습 및 평가의 연구 구성도](figures/Fig1_Study_Architecture.png)

[구성도 SVG 원본](figures/Fig1_Study_Architecture.svg)

전체 nAULC에서 DINOv2-FF−Entropy는 GC10 +0.45%p, NEU +0.25%p였지만 최저 클래스 nAULC에서는 −0.15%p, −0.99%p였다. 전체 평균과 낮은 클래스 성능은 서로 다른 평가 정보를 준다. 최저 클래스는 예산·반복마다 달라질 수 있다.

표 7은 GC10 10개와 NEU 6개 클래스 전체의 AP와 대응 차이를 제시한다. GC10의 DINOv2-FF−Random 평균 AP 증가 2.23%p 중 crease와 waist folding의 산술 기여는 1.54%p다. NEU에서는 scratches의 +0.72%p 기여와 rolled-in scale·crazing의 −1.08%p 기여 및 다른 클래스의 증가가 합쳐져 전체 차이가 −0.07%p였다. 이는 클래스 AP 차이를 클래스 수로 나눈 분해이며 원인을 분리한 실험이 아니다.

표 9는 Full 도달률의 평균과 함께 관측 범위와 95% 이상인 반복 수를 추가했다. DINOv2-FF의 GC10 도달률은 94.58±7.02%, 범위 89.61–105.52%, 95% 이상 2/5쌍이다. NEU는 89.65±3.26%, 범위 86.02–94.03%, 0/5쌍이다. 95%는 사후 기술적 참조선이다.

새 분석은 저장된 결과의 산술 재계산이다. 사전 지정한 네 가지 전체 nAULC 비교의 검정과 결론은 그대로이며, 다섯 쌍의 양측 exact 검정과 Holm 보정으로 통계적 우위를 확립하지 못했다. 표현 공간과 클래스 변화의 연결은 후보 가설로 서술했다. 실제 주석 비용 절감이나 공통 confidence 운영 성능을 입증한 결과로 해석하지 않는다.

## 버전과 검증

2026년 10월 27일 공개 결과 패키지는 파일과 해시 목록을 그대로 보존한다. 이 폴더의 DOCX는 검토한 로컬 개정본과 바이트 단위로 동일하다. 웹 원고는 본문·표·그림의 문서 순서를 따르며 네 수식은 GitHub 수식 표기로 옮겼다. 기존 다섯 결과 그림은 원래 패키지의 이미지에 연결한다.

- [개정 출처와 변경 범위](PROVENANCE.json)
- [검증 요약](VERIFICATION.json)
- [기존 지표에서 계산한 추가 분석](DERIVED_ANALYSIS.json)
- [이 개정 폴더의 파일 해시](SHA256SUMS)

저장소 루트에서 읽기 전용 검증을 실행할 수 있다. Python 표준 라이브러리만 사용하며 모델을 불러오지 않는다.

```sh
python docs/results/runpod_170_final_20261027/scripts/verify_export.py
python docs/manuscripts/runpod_170_20261028_v2/scripts/verify_revision.py
```

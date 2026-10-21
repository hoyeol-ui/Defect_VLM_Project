# Full1604 YOLO26s SGD100: Cosine 단독 변경

기존 Full1604 SGD100과 동일한 데이터·라벨·공식 사전학습 가중치에서 새로 시작한다.
변경은 cos_lr=False→True 하나뿐이다.
SGD,lr0=.01,lrf=.01,100epoch,batch8,nbs64,imgsz640,seed2026091900,AMP 유지.
Warmup3,close_mosaic10,기존 증강·손실 가중치 및 학습 루프도 유지한다.
기존 SGD best.pt로 이어 학습하지 않는다. Test는 읽지 않는다.
학습 중 dev batch16과 최종 FP32 dev batch8,전체 dev mAP50-95 best 선정 기준을 유지한다.
로그와 checkpoint는 runs/GC10_FULL1604_yolo26s_SGD_Cosine_e100_b8_i640에 기록된다.
완료 후 SGD_VS_SGD_COSINE_KO.md 및 클래스별 AP 비교 CSV를 자동 생성한다.
재시도·배치변경·후속 실험·중간 에이전트 모니터링은 없다.

재현 명령: Defect_VLM_Project/.python311/python.exe -I -B research_extensions/full_data_26s_sgd_cosine_640_20261018_01/controller.py
실행 이력이 있는 디렉터리는 재사용할 수 없다.

# 정규 실험 중단 보고

시간: 2026-10-21T22:21:26+09:00

과학적 설정을 변경하거나 자동 재시도하지 않았습니다. 실패 로그·가중치·실행 원장을 보존했습니다.

```text
Traceback (most recent call last):
  File "C:\Users\user\Desktop\vlm\research_extensions\final_label_efficiency_20261021_01\campaign_controller.py", line 146, in main
    before=time.perf_counter();owned_process(cmd,C/'logs'/(row['cell_id']+'.log'),rows,row,'TRAIN')
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\Desktop\vlm\research_extensions\final_label_efficiency_20261021_01\campaign_controller.py", line 94, in owned_process
    if code!=0:raise RuntimeError(f'{action} failed exit={code}; log={log_path}')
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: TRAIN failed exit=1; log=C:\Users\user\Desktop\vlm\research_extensions\final_label_efficiency_20261021_01\campaign\logs\GC10-DET_Random_pair1_R0.log

```

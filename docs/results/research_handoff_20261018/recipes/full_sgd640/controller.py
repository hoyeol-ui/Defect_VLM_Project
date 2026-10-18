"""One user-authorized full-data100 run; writes dev comparison and never launches another experiment."""
from pathlib import Path
import os,sys,json,csv,hashlib,subprocess,traceback,collections,time
P=Path(__file__).resolve().parent;BASE=P.parents[1]
sys.path.insert(0,str(P))
PY=BASE/'Defect_VLM_Project/.python311/python.exe'
DINO=BASE/'research_extensions/yolo26s_sgd_20261017_01/runs/GC10_R5_yolo26s_SGD_e100'
RANDOM=BASE/'research_extensions/random_dino_26s_640_PREPARED_20261017_01/runs/GC10_R5_Random_a2026091901_yolo26s_SGD_e100'
RUN_ID='GC10_FULL1604_yolo26s_SGD_e100_b8_i640'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def table(p,rows):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(out):
 m=read(out/'sealed_dev_metrics.json');reference=read(DINO/'sealed_dev_metrics.json');r=read(out/'RESULT.json')
 args=read(out/'resolved_args.json');old=read(DINO/'resolved_args.json')
 diff={k:[old.get(k),args.get(k)] for k in set(old)|set(args) if old.get(k)!=args.get(k)}
 assert set(diff)<=set(['model','data','project','save_dir']),diff
 assert read(out/'training_backend.json')['initial_model_state_sha']==read(DINO/'training_backend.json')['initial_model_state_sha']
 assert r['train_n']==1604 and r['dev_n']==343 and r['epochs']==100 and r['train_batches']==20100 and not r['test_evaluated']
 put(out/'ACTUAL_SETTING_MATCH.json',dict(resolved_differences=diff,initial_model_state_matches_DINO=True,train_n=1604,not_active_learning_budget=True))
 with (BASE/'research_extensions/yolo26s_sgd_300_20261017_01/diagnostics/SGD_e100_b8_CLASS_AP.csv').open(encoding='utf-8-sig',newline='') as f:
  base_ap={row['class_name']:row for row in csv.DictReader(f)}
 classes=[]
 for name,a in m['all_AP_by_class_and_IoU'].items():
  classes.append(dict(class_name=name,AP50_percent=a[0]*100,AP75_percent=a[5]*100,AP50_95_percent=sum(a)*10,
   DINO561_AP50_95_percent=float(base_ap[name]['AP50_95_percent']),delta_Full_minus_DINO_pp=sum(a)*10-float(base_ap[name]['AP50_95_percent'])))
 table(out/'CLASS_AP_FULL_VS_DINO561.csv',classes)
 comparisons=[]
 for label,folder,n in [('DINO561',DINO,561),('Random561',RANDOM,561),('Full1604',out,1604)]:
  v=read(folder/'sealed_dev_metrics.json');b=read(folder/'best_selection.json')
  comparisons.append(dict(condition=label,train_n=n,mAP50_95_percent=v['overall']['metrics/mAP50-95(B)']*100,mAP50_percent=v['overall']['metrics/mAP50(B)']*100,best_epoch=b['epoch']))
 table(out/'DATA_BUDGET_COMPARISON.csv',comparisons)
 # Reuse the established CPU matching function for a fixed operating-point diagnostic.
 from failure_core import failures
 gt=read(BASE/'research_extensions/yolo26s_sgd_300_20261017_01/diagnostics/DEV_GT.json')
 names=read(P/'datasets/GC10-DET/names.json');preds=collections.defaultdict(list)
 for row in read(out/'sealed_dev/predictions.json'):
  x,y,w,h=row['bbox'];preds[str(row['image_id'])].append(dict(cls=int(row['category_id'])-1,conf=float(row['score']),box=[x,y,x+w,y+h]))
 counts,_=failures(gt,preds,names);operating=[]
 for i,name in enumerate(names):
  c=counts[i];tp=c['TP50'];fp=sum(v for k,v in c.items() if k.startswith('FP_'));n=c['GT_boxes']
  operating.append(dict(class_name=name,confidence=.25,IoU=.5,GT=n,TP=tp,FN=n-tp,FP=fp,
   recall_percent=100*tp/n if n else None,precision_percent=100*tp/(tp+fp) if tp+fp else None,FP_per_dev_image=fp/343))
 table(out/'DEV_FIXED_CONF25_RECALL_FP.csv',operating)
 delta=(m['overall']['metrics/mAP50-95(B)']-reference['overall']['metrics/mAP50-95(B)'])*100
 put(out/'FINAL_SUMMARY.json',dict(status='COMPLETE',comparisons=comparisons,delta_Full_minus_DINO_pp=delta,
  inclusion=next(x for x in classes if x['class_name']=='inclusion'),test_used=False,additional_experiments_started=False,
  fixed_conf_diagnostic='conf>=.25 and class-correct IoU>=.5; saved rounded predictions; dev only, not field qualification.',
  inference_speed_context='Ultralytics dev batch8 local timing; not production end-to-end throughput.',
  interpretation='Single learner seed, full labels diagnostic at fixed100 epochs. More examples and more updates; not a dataset ceiling or active-learning budget result.'))
 lines=['# Full-data YOLO26s·SGD100 완료','', '| 조건 | 학습 이미지 | dev mAP50-95 (%) | dev mAP50 (%) | best epoch |','|---|---:|---:|---:|---:|']
 for a in comparisons:lines.append(f"|{a['condition']}|{a['train_n']}|{a['mAP50_95_percent']:.6f}|{a['mAP50_percent']:.6f}|{a['best_epoch']}|")
 lines+=['',f'Full1604−DINO561: {delta:+.6f}%p.', '', '클래스 AP50/AP75/AP50-95는 CLASS_AP_FULL_VS_DINO561.csv, 고정conf.25의 dev 재현율·오탐은 DEV_FIXED_CONF25_RECALL_FP.csv에 있다.',
  '561장보다 향상되면 예산·표본 구성 및 추가 update가 중요한 후보가 된다. 비슷하거나 낮아도 모델·입력·라벨 원인을 확정하지 않고 다음 한 변수 실험의 우선순위를 정한다.',
  '절대 성능 상한, 독립 test 검증 또는 산업 적용 적합성을 주장하지 않는다. 새 실험은 자동 실행하지 않았다.']
 (P/'FINAL_REPORT_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
 prep=read(P/'PREPARATION.json');lock=BASE/'research_extensions/GPU_TRAINING_ACTIVE.lock'
 assert all(sha(P/'src'/n)==h for n,h in prep['source_hashes'].items())
 assert sha(P/'config/common_args.json')==prep['config_sha']
 with lock.open('x',encoding='utf-8') as f:json.dump(dict(controller_pid=os.getpid(),workspace=str(P),run_id=RUN_ID),f)
 proc=None;job=None;out=P/'runs'/RUN_ID
 try:
  with (P/'RUN_ONCE.json').open('x',encoding='utf-8') as f:json.dump(dict(authorization=prep['authorization'],controller_pid=os.getpid()),f)
  out.mkdir(parents=True,exist_ok=False)
  from windows_job import WindowsJob
  job=WindowsJob()
  cmd=[str(PY),'-I','-B',str(P/'bootstrap.py'),'--dataset','GC10-DET','--model','yolo26s','--phase','train','--run-id',RUN_ID,'--controller-pid',str(os.getpid())]
  with (out/'console.log').open('x',encoding='utf-8') as log:
   proc=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
   job.assign(proc);put(out/'JOB_READY.json',dict(controller_pid=os.getpid(),worker_pid=proc.pid))
   put(P/'STATUS.json',dict(status='TRAINING',controller_pid=os.getpid(),worker_pid=proc.pid,output=str(out),epochs=100,train_n=1604,followup='No automatic further experiment'))
   code=proc.wait()
  assert code==0,f'Worker failed ({code}); no retry/batch fallback'
  summarize(out);put(P/'STATUS.json',dict(status='COMPLETE',output=str(out),report=str(P/'FINAL_REPORT_KO.md'),additional_experiments_started=False))
 except BaseException:
  put(P/'FAILURE.json',dict(error=traceback.format_exc(),automatic_retry=False));put(P/'STATUS.json',dict(status='FAILED',output=str(out),automatic_retry=False));raise
 finally:
  if job:job.close()
  if proc and proc.poll() is None:proc.kill();proc.wait(timeout=30)
  if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()
if __name__=='__main__':main()

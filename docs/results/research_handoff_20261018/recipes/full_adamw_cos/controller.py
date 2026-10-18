"""Detached owner of exactly one authorized training and its normal final dev evaluation."""
from pathlib import Path
import os,sys,json,csv,hashlib,subprocess,traceback,collections
P=Path(__file__).resolve().parent;B=P.parents[1]
sys.path.insert(0,str(P))
PY=B/'Defect_VLM_Project/.python311/python.exe'
BASELINE=B/'research_extensions/full_data_26s_sgd_640_20261017_01/runs/GC10_FULL1604_yolo26s_SGD_e100_b8_i640'
RUN_ID='GC10_FULL1604_yolo26s_AdamW_Cosine_e100_b8_i640'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def table(p,rows):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(out):
    old=read(BASELINE/'sealed_dev_metrics.json');new=read(out/'sealed_dev_metrics.json')
    oldargs=read(BASELINE/'resolved_args.json');args=read(out/'resolved_args.json')
    diffs={k:[oldargs.get(k),args.get(k)] for k in set(oldargs)|set(args) if oldargs.get(k)!=args.get(k)}
    assert set(diffs)<=set(['optimizer','lr0','cos_lr','model','data','project','save_dir']),diffs
    assert diffs['optimizer']==['SGD','AdamW'] and diffs['lr0']==[.01,.001] and diffs['cos_lr']==[False,True]
    assert read(out/'training_backend.json')['initial_model_state_sha']==read(BASELINE/'training_backend.json')['initial_model_state_sha']
    assert read(out/'RESULT.json')['train_batches']==20100
    assert read(out/'loader_ids.json')==read(BASELINE/'loader_ids.json')
    put(out/'SETTING_MATCH.json',dict(differences=diffs,same_initial_model_state=True,same_train_dev_ids=True))
    classes=[]
    for name,a in old['all_AP_by_class_and_IoU'].items():
        b=new['all_AP_by_class_and_IoU'][name]
        classes.append(dict(class_name=name,SGD_AP50=a[0]*100,AdamW_AP50=b[0]*100,delta_AP50_pp=(b[0]-a[0])*100,
            SGD_AP75=a[5]*100,AdamW_AP75=b[5]*100,delta_AP75_pp=(b[5]-a[5])*100,
            SGD_AP50_95=sum(a)*10,AdamW_AP50_95=sum(b)*10,delta_AP50_95_pp=(sum(b)-sum(a))*10))
    table(out/'CLASS_AP_SGD_VS_ADAMW_COSINE.csv',classes)
    from failure_core import failures
    gt=read(B/'research_extensions/yolo26s_sgd_300_20261017_01/diagnostics/DEV_GT.json')
    names=read(P/'datasets/GC10-DET/names.json');pred=collections.defaultdict(list)
    for v in read(out/'sealed_dev/predictions.json'):
        x,y,w,h=v['bbox'];pred[str(v['image_id'])].append(dict(cls=int(v['category_id'])-1,conf=v['score'],box=[x,y,x+w,y+h]))
    counts,_=failures(gt,pred,names);op=[]
    for i,name in enumerate(names):
        c=counts[i];tp=c['TP50'];fp=sum(v for k,v in c.items() if k.startswith('FP_'))
        op.append(dict(class_name=name,confidence=.25,IoU=.5,GT=c['GT_boxes'],TP=tp,FN=c['GT_boxes']-tp,FP=fp))
    table(out/'DEV_FIXED_CONF25_RECALL_FP.csv',op)
    inc=next(c for c in classes if c['class_name']=='inclusion')
    delta=(new['overall']['metrics/mAP50-95(B)']-old['overall']['metrics/mAP50-95(B)'])*100
    summary=dict(status='COMPLETE',SGD_overall=old['overall'],AdamW_overall=new['overall'],delta_mAP50_95_pp=delta,
        inclusion=inc,inclusion_conf25=op[6],SGD_best_epoch=read(BASELINE/'best_selection.json')['epoch'],
        AdamW_best_epoch=read(out/'best_selection.json')['epoch'],test_used=False,additional_experiments=False,
        candidate=inc['delta_AP75_pp']>0 and inc['delta_AP50_95_pp']>0 and delta>=0,
        interpretation='Single seed dev recipe bundle comparison; not isolated AdamW causality.')
    put(out/'FINAL_SUMMARY.json',summary)
    lines=['# Full1604: SGD100 vs AdamW + Cosine100','',
        '동일 사전학습 가중치·데이터·평가 경로. 변경은 optimizer=AdamW,lr0=.001,cos_lr=True. 재학습/추가 실험 자동 실행 없음.','',
        '| 지표(%) | SGD100 | AdamW+Cosine100 | 차이(%p) |','|---|---:|---:|---:|']
    for label,key in [('전체 mAP50','metrics/mAP50(B)'),('전체 mAP50-95','metrics/mAP50-95(B)')]:
        a=old['overall'][key]*100;b=new['overall'][key]*100;lines.append(f'|{label}|{a:.6f}|{b:.6f}|{b-a:+.6f}|')
    for label,key in [('Inclusion AP50','AP50'),('Inclusion AP75','AP75'),('Inclusion AP50-95','AP50_95')]:
        lines.append(f"|{label}|{inc['SGD_'+key]:.6f}|{inc['AdamW_'+key]:.6f}|{inc['delta_'+key+'_pp']:+.6f}|")
    lines+=['',f"Best epoch: SGD {summary['SGD_best_epoch']}, AdamW+Cosine {summary['AdamW_best_epoch']}.",
        f"Inclusion conf≥.25/IoU≥.5: 기존TP16/FP44 → TP{op[6]['TP']}/FP{op[6]['FP']}. CPU 매칭이며 공식AP와 구분한다.",
        '', '| 클래스 | SGD AP50-95(%) | AdamW AP50-95(%) | 차이(%p) |','|---|---:|---:|---:|']
    for c in classes:lines.append(f"|{c['class_name']}|{c['SGD_AP50_95']:.6f}|{c['AdamW_AP50_95']:.6f}|{c['delta_AP50_95_pp']:+.6f}|")
    lines+=['',f"사전 기준(inclusion AP75·AP50-95 상승,전체 mAP50-95 비저하) 충족: {summary['candidate']}.",
        '학습 레시피 묶음의 단일 시드 개발 결과다. 신뢰도 상승만으로 개선을 판정하지 않으며 클래스별 하락도 함께 읽는다.']
    (P/'SGD_VS_ADAMW_COSINE_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    prep=read(P/'PREPARATION.json');lock=B/'research_extensions/GPU_TRAINING_ACTIVE.lock'
    assert all(sha(P/'src'/n)==h for n,h in prep['source_hashes'].items())
    assert sha(P/'config/common_args.json')==prep['config_sha']
    assert not (P/'RUN_ONCE.json').exists(),'Already attempted; no retry'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(controller_pid=os.getpid(),workspace=str(P),run_id=RUN_ID),f)
    proc=None;job=None;out=P/'runs'/RUN_ID
    try:
        with (P/'RUN_ONCE.json').open('x',encoding='utf-8') as f:json.dump(dict(authorization=prep['authorization'],controller_pid=os.getpid()),f)
        out.mkdir(parents=True,exist_ok=False)
        from windows_job import WindowsJob
        job=WindowsJob()
        cmd=[str(PY),'-I','-B',str(P/'bootstrap.py'),'--dataset','GC10-DET','--model','yolo26s','--phase','train','--run-id',RUN_ID,'--controller-pid',str(os.getpid())]
        with (out/'console.log').open('x',encoding='utf-8') as log:
            proc=subprocess.Popen(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
            job.assign(proc);put(out/'JOB_READY.json',dict(controller_pid=os.getpid(),worker_pid=proc.pid))
            put(P/'STATUS.json',dict(status='TRAINING',controller_pid=os.getpid(),worker_pid=proc.pid,output=str(out),epochs=100,train_n=1604,automatic_followup=False))
            code=proc.wait()
        assert code==0,f'Worker failed ({code}); no retry or batch fallback'
        summarize(out)
        put(P/'STATUS.json',dict(status='COMPLETE',output=str(out),report=str(P/'SGD_VS_ADAMW_COSINE_KO.md'),additional_experiments=False))
    except BaseException:
        put(P/'FAILURE.json',dict(error=traceback.format_exc(),automatic_retry=False))
        put(P/'STATUS.json',dict(status='FAILED',output=str(out),automatic_retry=False));raise
    finally:
        if job:job.close()
        if proc and proc.poll() is None:proc.kill();proc.wait(timeout=30)
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()

if __name__=='__main__':main()

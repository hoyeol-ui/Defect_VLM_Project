"""Prepare exactly one Full1604 AdamW/cosine comparison; no ML runtime imports."""
from pathlib import Path
import ast,json,hashlib,shutil
P=Path(__file__).resolve().parent;B=P.parents[1]
OLD=B/'research_extensions/full_data_26s_sgd_640_20261017_01'
RUN=OLD/'runs/GC10_FULL1604_yolo26s_SGD_e100_b8_i640'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    for name in ('src','config','assets','evidence','runs'):(P/name).mkdir(exist_ok=False)
    # Byte copies avoid sharing any files with the completed SGD baseline.
    shutil.copytree(OLD/'datasets',P/'datasets')
    for f in (OLD/'src').glob('*.py'):shutil.copyfile(f,P/'src'/f.name)
    for name in ('bootstrap.py','windows_job.py','failure_core.py'):shutil.copyfile(OLD/name,P/name)
    shutil.copyfile(OLD/'assets/yolo26s.pt',P/'assets/yolo26s.pt')
    args=read(OLD/'config/common_args.json');args.update(optimizer='AdamW',lr0=.001,cos_lr=True)
    put(P/'config/common_args.json',args)
    delta={k:[v,args[k]] for k,v in read(OLD/'config/common_args.json').items() if args[k]!=v}
    assert delta=={'optimizer':['SGD','AdamW'],'cos_lr':[False,True],'lr0':[.01,.001]}
    source=P/'src/modern_trainer.py';text=source.read_text(encoding='utf-8')
    old="require(isinstance(self.optimizer,torch.optim.SGD),'First architecture experiment requires SGD')"
    assert text.count(old)==1
    new="""require(isinstance(self.optimizer,torch.optim.AdamW),'Authorized recipe requires AdamW')
  require(self.args.lr0==.001 and self.args.cos_lr is True,'AdamW/cosine recipe drift')
  require(abs(self.lf(25)-(.01+.99*(1+np.cos(np.pi*.25))/2))<1e-10,'Actual cosine scheduler drift')
  put(self.artifact_dir/'OPTIMIZER_SCHEDULER_ACTUAL.json',{'optimizer':type(self.optimizer).__name__,'defaults':self.optimizer.defaults,'cos_lr':self.args.cos_lr,'lr0':self.args.lr0,'lrf':self.args.lrf,'lr_multipliers':{str(e):float(self.lf(e)) for e in (0,25,50,75,100)},'nbs':self.args.nbs,'warmup_epochs':self.args.warmup_epochs,'warmup_bias_lr':self.args.warmup_bias_lr})"""
    source.write_text(text.replace(old,new),encoding='utf-8')
    for name in ('core.py','strict_loop.py','trainer_adapter.py','amp_runtime.py','amp_compat.py','worker.py'):
        assert sha(P/'src'/name)==sha(OLD/'src'/name)
    ids=read(P/'datasets/GC10-DET/loader_ids.json')
    assert len(ids['train'])==1604 and len(ids['dev'])==343 and not set(ids['train'])&set(ids['dev'])
    files={f.relative_to(P).as_posix():sha(f) for d in ('datasets','assets') for f in (P/d).rglob('*') if f.is_file()}
    oldseal=read(OLD/'evidence/INPUT_SEAL.json')
    assert files==oldseal['files'],'Data or pretrained identity differs from SGD'
    assert files['assets/yolo26s.pt']=='646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
    put(P/'evidence/INPUT_SEAL.json',dict(files=files,downloaded_weights={},test_present=False))
    put(P/'PREPARATION.json',dict(status='READY',authorization='User authorized one Full1604 YOLO26s 640 100epoch batch8 AdamW lr0=.001 cosine run, then stop reporting after start.',
        baseline=str(RUN),changed_settings=delta,train_n=1604,dev_n=343,epochs=100,seed=2026091900,
        same_pretrained=True,pretrained_sha=files['assets/yolo26s.pt'],same_data_bytes=True,
        baseline_best_used_for_initialization=False,test_used=False,additional_experiments=False,
        source_hashes={f.name:sha(f) for f in (P/'src').glob('*.py')},config_sha=sha(P/'config/common_args.json')))
    for f in [*P.glob('*.py'),*(P/'src').glob('*.py')]:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
    (P/'README_KO.md').write_text('''# Full1604 YOLO26s AdamW + Cosine 단일 비교

기존 Full1604 SGD100과 동일한 데이터·라벨·사전학습 가중치에서 새로 시작한다.
변경: optimizer SGD→AdamW, lr0 0.01→0.001, cos_lr False→True.
유지: 100epoch,batch8,nbs64,imgsz640,seed2026091900,AMP,기존 warmup/증강/손실 가중치.
공식 사전학습 yolo26s.pt에서 시작하며 SGD best.pt를 이어 학습하지 않는다.
기존 학습 루프·GradScaler·FP32 checkpoint 저장·전체 dev mAP50-95 best 선정·최종 Dev343 평가를 재사용한다.
기존 epoch dev batch16과 최종 FP32 dev batch8도 유지한다. Test는 읽지 않는다.

실행: Defect_VLM_Project/.python311/python.exe -I -B research_extensions/full_data_26s_adamw_cosine_640_20261018_01/controller.py
이미 실행한 폴더의 재실행은 금지된다. 완료 후 자동 재시도·추가 학습은 없다.
학습 중 console.log,train/results.csv,amp_events.jsonl 및 체크포인트는 기존 방식으로 저장한다.
완료 후 SGD_VS_ADAMW_COSINE_KO.md와 클래스별 AP 비교 CSV를 자동 생성한다.
이번 비교는 AdamW·학습률·스케줄의 묶음 효과를 보는 단일 시드 dev 실험이다.
''',encoding='utf-8')
    print(json.dumps(dict(status='READY',changes=delta,same_pretrained=True,train_n=1604,dev_n=343)))

if __name__=='__main__':main()

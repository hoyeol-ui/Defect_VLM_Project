"""Prepare one SGD100 cosine-only experiment without changing prior experiments."""
from pathlib import Path
import ast,json,hashlib,shutil
P=Path(__file__).resolve().parent;B=P.parents[1]
OLD=B/'research_extensions/full_data_26s_sgd_640_20261017_01'
RUN=OLD/'runs/GC10_FULL1604_yolo26s_SGD_e100_b8_i640'
PREVIOUS=B/'research_extensions/full_data_26s_adamw_cosine_640_20261018_01'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1048576),b''):h.update(z)
    return h.hexdigest()
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def replace_once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def main():
    for name in ('src','config','assets','evidence','runs'):(P/name).mkdir(exist_ok=False)
    shutil.copytree(OLD/'datasets',P/'datasets')
    for f in (OLD/'src').glob('*.py'):shutil.copyfile(f,P/'src'/f.name)
    for name in ('bootstrap.py','windows_job.py','failure_core.py'):shutil.copyfile(OLD/name,P/name)
    shutil.copyfile(OLD/'assets/yolo26s.pt',P/'assets/yolo26s.pt')
    args=read(OLD/'config/common_args.json');args['cos_lr']=True
    put(P/'config/common_args.json',args)
    delta={k:[v,args[k]] for k,v in read(OLD/'config/common_args.json').items() if args[k]!=v}
    assert delta=={'cos_lr':[False,True]}
    assert args['optimizer']=='SGD' and args['lr0']==.01 and args['lrf']==.01
    # Add observation/assertions only; preserve SGD optimizer and original numerical loop.
    source=P/'src/modern_trainer.py';text=source.read_text(encoding='utf-8')
    needle="require(isinstance(self.optimizer,torch.optim.SGD),'First architecture experiment requires SGD')"
    extra="""
  require(self.args.lr0==.01 and self.args.lrf==.01 and self.args.cos_lr is True,'Cosine-only recipe drift')
  require(abs(self.lf(25)-(.01+.99*(1+np.cos(np.pi*.25))/2))<1e-10,'Actual cosine scheduler drift')
  put(self.artifact_dir/'OPTIMIZER_SCHEDULER_ACTUAL.json',{'optimizer':type(self.optimizer).__name__,'cos_lr':self.args.cos_lr,'lr0':self.args.lr0,'lrf':self.args.lrf,'lr_multipliers':{str(e):float(self.lf(e)) for e in (0,25,50,75,100)},'nbs':self.args.nbs,'warmup_epochs':self.args.warmup_epochs,'warmup_bias_lr':self.args.warmup_bias_lr,'parameter_groups':[{k:v for k,v in g.items() if k in ('lr','initial_lr','momentum','weight_decay','param_group')} for g in self.optimizer.param_groups]})"""
    source.write_text(replace_once(text,needle,needle+extra),encoding='utf-8')
    for name in ('core.py','strict_loop.py','trainer_adapter.py','amp_runtime.py','amp_compat.py','worker.py'):
        assert sha(P/'src'/name)==sha(OLD/'src'/name)
    ids=read(P/'datasets/GC10-DET/loader_ids.json')
    assert len(ids['train'])==1604 and len(ids['dev'])==343 and not set(ids['train'])&set(ids['dev'])
    files={f.relative_to(P).as_posix():sha(f) for d in ('datasets','assets') for f in (P/d).rglob('*') if f.is_file()}
    assert files==read(OLD/'evidence/INPUT_SEAL.json')['files']
    assert files['assets/yolo26s.pt']=='646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
    put(P/'evidence/INPUT_SEAL.json',dict(files=files,downloaded_weights={},test_present=False))
    # Reuse the completed campaign's lifecycle and report code, adapting only recipe checks and report labels.
    controller=(PREVIOUS/'controller.py').read_text(encoding='utf-8')
    controller=replace_once(controller,
        "assert set(diffs)<=set(['optimizer','lr0','cos_lr','model','data','project','save_dir']),diffs",
        "assert set(diffs)<=set(['cos_lr','model','data','project','save_dir']),diffs")
    controller=replace_once(controller,
        "assert diffs['optimizer']==['SGD','AdamW'] and diffs['lr0']==[.01,.001] and diffs['cos_lr']==[False,True]",
        "assert diffs['cos_lr']==[False,True] and args['optimizer']=='SGD' and args['lr0']==.01 and args['lrf']==.01")
    controller=replace_once(controller,
        '동일 사전학습 가중치·데이터·평가 경로. 변경은 optimizer=AdamW,lr0=.001,cos_lr=True. 재학습/추가 실험 자동 실행 없음.',
        '동일 사전학습 가중치·데이터·평가 경로. 변경은 cos_lr=True 하나뿐. SGD,lr0=.01,lrf=.01 유지. 추가 실험 자동 실행 없음.')
    controller=controller.replace('AdamW_Cosine','SGD_Cosine').replace('ADAMW_COSINE','SGD_COSINE').replace('AdamW','SGD_Cosine')
    controller=controller.replace('Single seed dev recipe bundle comparison; not isolated SGD_Cosine causality.',
        'Single seed dev comparison; only cosine schedule changed from the SGD100 baseline.')
    controller=controller.replace('학습 레시피 묶음의 단일 시드 개발 결과다.','Cosine 스케줄만 변경한 단일 시드 개발 결과다.')
    controller=controller.replace('SGD_Cosine + Cosine100','SGD + Cosine100').replace('SGD_Cosine+Cosine100','SGD+Cosine100')
    assert "RUN_ID='GC10_FULL1604_yolo26s_SGD_Cosine_e100_b8_i640'" in controller
    assert 'AdamW' not in controller and 'ADAMW' not in controller
    (P/'controller.py').write_text(controller,encoding='utf-8')
    put(P/'PREPARATION.json',dict(status='READY',authorization='User authorized exactly one Full1604 YOLO26s SGD100 run changing cos_lr=True only; stop agent reporting after training starts.',
        baseline=str(RUN),changed_settings=delta,train_n=1604,dev_n=343,epochs=100,seed=2026091900,
        same_pretrained=True,pretrained_sha=files['assets/yolo26s.pt'],same_data_bytes=True,
        baseline_best_used_for_initialization=False,test_used=False,additional_experiments=False,
        source_hashes={f.name:sha(f) for f in (P/'src').glob('*.py')},config_sha=sha(P/'config/common_args.json')))
    for f in [*P.glob('*.py'),*(P/'src').glob('*.py')]:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
    (P/'README_KO.md').write_text('''# Full1604 YOLO26s SGD100: Cosine 단독 변경

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
''',encoding='utf-8')
    print(json.dumps(dict(status='READY',changes=delta,same_pretrained=True,train_n=1604,dev_n=343)))

if __name__=='__main__':main()

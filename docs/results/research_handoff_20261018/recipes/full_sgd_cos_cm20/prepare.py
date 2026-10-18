"""Prepare exactly one close_mosaic20 experiment; no ML runtime or GPU imports."""
from pathlib import Path
import ast,json,hashlib,shutil
P=Path(__file__).resolve().parent;B=P.parents[1]
OLD=B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01'
BASELINE=OLD/'runs/GC10_FULL1604_yolo26s_SGD_Cosine_e100_b8_i640'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for z in iter(lambda:f.read(1048576),b''):h.update(z)
    return h.hexdigest()
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    assert read(OLD/'STATUS.json')['status']=='COMPLETE'
    for name in ('src','config','assets','evidence','runs'):(P/name).mkdir(exist_ok=False)
    shutil.copytree(OLD/'datasets',P/'datasets')
    for f in (OLD/'src').glob('*.py'):shutil.copyfile(f,P/'src'/f.name)
    for name in ('bootstrap.py','windows_job.py','failure_core.py'):shutil.copyfile(OLD/name,P/name)
    shutil.copyfile(OLD/'assets/yolo26s.pt',P/'assets/yolo26s.pt')
    args=read(OLD/'config/common_args.json');args['close_mosaic']=20
    delta={k:[v,args[k]] for k,v in read(OLD/'config/common_args.json').items() if args[k]!=v}
    assert delta=={'close_mosaic':[10,20]}
    assert args['optimizer']=='SGD' and args['cos_lr'] is True and args['lr0']==.01 and args['lrf']==.01
    put(P/'config/common_args.json',args)
    # Only an observation/assertion is added. Numerical training loop is byte-identical.
    modern=P/'src/modern_trainer.py';text=modern.read_text(encoding='utf-8')
    needle="require(self.args.lr0==.01 and self.args.lrf==.01 and self.args.cos_lr is True,'Cosine-only recipe drift')"
    assert text.count(needle)==1
    text=text.replace(needle,needle+"\n  require(self.args.close_mosaic==20,'close_mosaic20 drift')\n  put(self.artifact_dir/'MOSAIC_SCHEDULE.json',{'close_mosaic':self.args.close_mosaic,'epochs':self.args.epochs,'first_epoch_without_mosaic':self.args.epochs-self.args.close_mosaic+1,'other_augmentations_unchanged':True})")
    modern.write_text(text,encoding='utf-8')
    for name in ('core.py','strict_loop.py','trainer_adapter.py','amp_runtime.py','amp_compat.py','worker.py'):
        assert sha(P/'src'/name)==sha(OLD/'src'/name)
    ids=read(P/'datasets/GC10-DET/loader_ids.json')
    assert len(ids['train'])==1604 and len(ids['dev'])==343 and not set(ids['train'])&set(ids['dev'])
    files={f.relative_to(P).as_posix():sha(f) for d in ('datasets','assets') for f in (P/d).rglob('*') if f.is_file()}
    assert files==read(OLD/'evidence/INPUT_SEAL.json')['files']
    assert files['assets/yolo26s.pt']=='646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'
    put(P/'evidence/INPUT_SEAL.json',dict(files=files,downloaded_weights={},test_present=False))
    # Reuse the same detached, one-shot lifecycle, with a focused comparison reporter.
    previous=(OLD/'controller.py').read_text(encoding='utf-8')
    header=previous[:previous.index('def summarize(out):')]
    lifecycle=previous[previous.index('def main():'):]
    controller=header+'from compare_results import summarize\n\n'+lifecycle
    controller=controller.replace("RUN_ID='GC10_FULL1604_yolo26s_SGD_Cosine_e100_b8_i640'",
        "RUN_ID='GC10_FULL1604_yolo26s_SGD_Cosine_CM20_e100_b8_i640'")
    controller=controller.replace('SGD_VS_SGD_COSINE_KO.md','SGD_COSINE_CM10_VS_CM20_KO.md')
    (P/'controller.py').write_text(controller,encoding='utf-8')
    put(P/'PREPARATION.json',dict(status='READY',authorization='User authorized exactly one Full1604 YOLO26s SGD+Cosine100 run changing close_mosaic10 to20 only; stop agent reporting after training starts.',
        baseline=str(BASELINE),changed_settings=delta,train_n=1604,dev_n=343,epochs=100,seed=2026091900,
        same_pretrained=True,pretrained_sha=files['assets/yolo26s.pt'],same_data_bytes=True,
        baseline_best_used_for_initialization=False,test_used=False,additional_experiments=False,
        source_hashes={f.name:sha(f) for f in (P/'src').glob('*.py')},config_sha=sha(P/'config/common_args.json')))
    for f in [*P.glob('*.py'),*(P/'src').glob('*.py')]:ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
    (P/'README_KO.md').write_text('''# Full1604 SGD+Cosine100: close_mosaic20

변경은 close_mosaic10→20 하나뿐이다. 81epoch부터 Mosaic을 해제한다.
동일 공식 사전학습 yolo26s.pt에서 새로 시작한다. 기존 best.pt로 이어 학습하지 않는다.
Full1604/Dev343,YOLO26s,SGD,cos_lr=True,lr0=.01,lrf=.01,100epoch,batch8,nbs64,imgsz640,seed2026091900 유지.
warmup3,손실 가중치,나머지 증강,AMP,학습 루프,전체 dev mAP50-95 best 선정,최종 FP32 dev batch8 평가를 유지한다.
독립Test 접근·추가 학습·자동재시도·배치변경·중간 에이전트 모니터링은 없다.
완료 후 SGD_COSINE_CM10_VS_CM20_KO.md와 클래스별 AP 비교 CSV를 자동 생성한다.

실행: Defect_VLM_Project/.python311/python.exe -I -B research_extensions/full_data_26s_sgd_cosine_cm20_640_20261018_01/controller.py
이미 실행한 폴더에서는 재실행할 수 없다.
''',encoding='utf-8')
    print(json.dumps(dict(status='READY',changes=delta,same_pretrained=True,train_n=1604,dev_n=343)))

if __name__=='__main__':main()

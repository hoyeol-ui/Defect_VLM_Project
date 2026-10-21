"""Recompute one Entropy acquisition from the preceding YOLO26 checkpoint, without GT."""
from pathlib import Path
import os,sys,json,argparse,hashlib,time
P=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',choices=['GC10-DET','NEU-DET'],required=True)
    ap.add_argument('--pair',type=int,choices=[1,2,3],required=True)
    ap.add_argument('--round',type=int,choices=range(1,6),required=True)
    a=ap.parse_args()
    sys.path.insert(0,str(P))
    from integrity import verify
    verify()
    cfg=read(P/'config/protocol.json');seed=cfg['seed_pairs'][a.pair-1]['acquisition']
    assert read(P/'evidence/readiness.json')['status']=='READY_LOCAL_TRAIN_DEV'
    # Technical STATE only locates the inherited scoring implementation; no labels are opened here.
    os.environ.update(MODERN_YOLO_RUN=str(P/'runs'/'entropy_adapter'),CUBLAS_WORKSPACE_CONFIG=':4096:8',YOLO_AUTOINSTALL='false',HF_HUB_OFFLINE='1')
    sys.path.insert(0,str(P/'src'))
    import torch
    from ultralytics import YOLO
    from entropy_adapter import image_tensor,score_one2one
    prior_method='Random' if a.round==1 else 'Entropy'
    prev=read(P/'selections'/a.dataset/str(seed)/prior_method/f'R{a.round-1}.json')['cumulative_ids']
    prior=P/'runs'/f'{a.dataset}_{prior_method}_pair{a.pair}_R{a.round-1}'
    done=read(prior/'RESULT.json');assert done['status']=='PASS' and done['technical_only'] is False
    checkpoint=prior/'train/weights/best.pt';assert sha(checkpoint)==done['best_sha']
    target=P/'selections'/a.dataset/str(seed)/'Entropy'/f'R{a.round}.json'
    assert not target.exists(),'Acquisition already exists; inspect it instead of overwriting'
    idx=read(P/'inputs'/a.dataset/'index.json')
    assert sha(P/'inputs'/a.dataset/'index.json')==read(P/'evidence/preparation.json')['datasets'][a.dataset]['index_sha']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model=YOLO(str(checkpoint)).model.float().cuda().eval();scores=[];previous=set(prev)
    candidates=[r for r in idx['train'] if r['id'] not in previous]
    diagnostics=P/'acquisition_diagnostics'/a.dataset/str(seed)
    diagnostics.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    for k,r in enumerate(candidates):
        assert sha(r['image'])==r['image_sha']
        result=score_one2one(model,image_tensor(r['image']),len(idx['names']))
        scores.append({'id':r['id'],'uncertainty':result['uncertainty'],'raw_sha':result['raw_sha'],'detections':result['detections']})
        if k%32==0 or k==len(candidates)-1:
            temp=diagnostics/f'R{a.round}_progress.tmp'
            temp.write_text(json.dumps({'scored':k+1,'eligible':len(candidates),'seconds':time.perf_counter()-started}),encoding='utf-8')
            os.replace(temp,diagnostics/f'R{a.round}_progress.json')
    scores.sort(key=lambda r:(-r['uncertainty'],r['id']))
    q=cfg['counts'][a.dataset][a.round]-len(prev);chosen=[r['id'] for r in scores[:q]]
    assert len(set(prev+chosen))==cfg['counts'][a.dataset][a.round]
    from entropy_diagnostics import describe
    report=describe(scores,chosen,q)
    # Independent repeat forward checks from different positions in the candidate pool.
    lookup={r['id']:r for r in candidates};repeat=[]
    for id in sorted({scores[0]['id'],scores[-1]['id'],candidates[len(candidates)//2]['id']}):
        r=lookup[id];again=score_one2one(model,image_tensor(r['image']),len(idx['names']))
        original=next(v for v in scores if v['id']==id)
        assert all(again[k]==original[k] for k in ['uncertainty','raw_sha','detections'])
        repeat.append({'id':id,'raw_sha':again['raw_sha'],'repeat_equal':True})
    report.update(dataset=a.dataset,acquisition_seed=seed,learner_seed=cfg['seed_pairs'][a.pair-1]['learner'],round=a.round,
        parent_best_sha=done['best_sha'],repeat_checks=repeat,seconds=time.perf_counter()-started,
        no_candidate_GT_opened=True,ranking_validated=True,
        selection_within_remaining_pool=set(chosen)<=set(lookup),
        retained_rule='confidence>0.25; normalized class-score entropy; class NMS IoU0.7; max300; no predictions=0; exact score then ID')
    report_path=diagnostics/f'R{a.round}_diagnostics.json'
    assert not report_path.exists(),'Diagnostic already exists; preserve the prior attempt'
    temp=report_path.with_suffix('.tmp')
    with temp.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    os.replace(temp,report_path)
    # Preserve the complete ranking even if the validity gate stops the campaign.
    with (diagnostics/f'R{a.round}_ranking.json').open('x',encoding='utf-8') as f:json.dump(scores,f)
    assert report['gate']=='PASS','Entire acquisition ranking tied; stop and report without changing selector'
    manifest={'dataset':a.dataset,'method':'Entropy','seed':seed,'round':a.round,
        'selected_ids':chosen,'cumulative_ids':prev+chosen,'parent_best_sha':done['best_sha'],
        'head':'one2one_before_topk','score':'ENTROPY_NCS_MEAN_V2','no_candidate_GT_opened':True,'ranking':scores,
        'diagnostics_path':str(report_path),'diagnostics_sha':sha(report_path)}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as f:json.dump(manifest,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'PASS','output':str(target),'new_images':q,'candidates_scored':len(scores)}))

if __name__=='__main__':
    lock=P.parents[1]/'research_extensions/GPU_TRAINING_ACTIVE.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'controller_pid':os.getpid(),'action':'entropy_acquisition'},f)
    try:main()
    finally:
        if lock.exists() and read(lock).get('controller_pid')==os.getpid():lock.unlink()

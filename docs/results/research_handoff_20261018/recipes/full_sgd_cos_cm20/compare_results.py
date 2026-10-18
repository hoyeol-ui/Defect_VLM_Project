"""Compare this one completed run to saved SGD+Cosine CM10; no extra inference."""
from pathlib import Path
import json,csv,collections
P=Path(__file__).resolve().parent;B=P.parents[1]
BASELINE=B/'research_extensions/full_data_26s_sgd_cosine_640_20261018_01/runs/GC10_FULL1604_yolo26s_SGD_Cosine_e100_b8_i640'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def table(p,z):
    with p.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(z[0]));w.writeheader();w.writerows(z)

def summarize(out):
    old=read(BASELINE/'sealed_dev_metrics.json');new=read(out/'sealed_dev_metrics.json')
    a=read(BASELINE/'resolved_args.json');b=read(out/'resolved_args.json')
    diffs={k:[a.get(k),b.get(k)] for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    assert set(diffs)<=set(['close_mosaic','model','data','project','save_dir']),diffs
    assert diffs['close_mosaic']==[10,20] and b['optimizer']=='SGD' and b['cos_lr'] is True and b['lr0']==.01 and b['lrf']==.01
    assert read(out/'training_backend.json')['initial_model_state_sha']==read(BASELINE/'training_backend.json')['initial_model_state_sha']
    assert read(out/'loader_ids.json')==read(BASELINE/'loader_ids.json')
    r=read(out/'RESULT.json');assert r['train_batches']==20100 and r['epochs']==100 and not r['test_evaluated']
    put(out/'SETTING_MATCH.json',dict(differences=diffs,same_initial_model_state=True,same_train_dev_ids=True))
    classes=[]
    for name,aa in old['all_AP_by_class_and_IoU'].items():
        bb=new['all_AP_by_class_and_IoU'][name]
        classes.append(dict(class_name=name,CM10_AP50=aa[0]*100,CM20_AP50=bb[0]*100,delta_AP50_pp=(bb[0]-aa[0])*100,
            CM10_AP75=aa[5]*100,CM20_AP75=bb[5]*100,delta_AP75_pp=(bb[5]-aa[5])*100,
            CM10_AP50_95=sum(aa)*10,CM20_AP50_95=sum(bb)*10,delta_AP50_95_pp=(sum(bb)-sum(aa))*10))
    table(out/'CLASS_AP_CM10_VS_CM20.csv',classes)
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
    silk=next(c for c in classes if c['class_name']=='silk_spot')
    rolled=next(c for c in classes if c['class_name']=='rolled_pit')
    delta=(new['overall']['metrics/mAP50-95(B)']-old['overall']['metrics/mAP50-95(B)'])*100
    candidate=delta>=0 and inc['delta_AP75_pp']>0 and inc['delta_AP50_95_pp']>0 and silk['delta_AP50_95_pp']>0 and rolled['delta_AP50_95_pp']>0
    summary=dict(status='COMPLETE',CM10_overall=old['overall'],CM20_overall=new['overall'],delta_mAP50_95_pp=delta,
        inclusion=inc,silk_spot=silk,rolled_pit=rolled,inclusion_conf25=op[6],
        CM10_best_epoch=read(BASELINE/'best_selection.json')['epoch'],CM20_best_epoch=read(out/'best_selection.json')['epoch'],
        candidate=candidate,test_used=False,additional_experiments=False,
        interpretation='Single seed dev comparison: close_mosaic10 to20 only; not a generalization guarantee.')
    put(out/'FINAL_SUMMARY.json',summary)
    lines=['# Full1604 SGD+Cosine100: close_mosaic10 vs20','',
        '동일 사전학습 가중치·데이터·평가 경로. 변경은 close_mosaic10→20 하나뿐. 추가 실험 자동 실행 없음.','',
        '| 지표(%) | CM10 | CM20 | 차이(%p) |','|---|---:|---:|---:|']
    for label,key in [('전체 mAP50','metrics/mAP50(B)'),('전체 mAP50-95','metrics/mAP50-95(B)')]:
        aa=old['overall'][key]*100;bb=new['overall'][key]*100;lines.append(f'|{label}|{aa:.6f}|{bb:.6f}|{bb-aa:+.6f}|')
    for label,key in [('Inclusion AP50','AP50'),('Inclusion AP75','AP75'),('Inclusion AP50-95','AP50_95')]:
        lines.append(f"|{label}|{inc['CM10_'+key]:.6f}|{inc['CM20_'+key]:.6f}|{inc['delta_'+key+'_pp']:+.6f}|")
    lines+=['',f"Best epoch: CM10 {summary['CM10_best_epoch']}, CM20 {summary['CM20_best_epoch']}.",
        f"Inclusion conf≥.25/IoU≥.5: 기준TP13/FP27 → TP{op[6]['TP']}/FP{op[6]['FP']}. CPU 매칭이며 공식 AP와 구분한다.",
        '', '| 클래스 | CM10 AP50-95(%) | CM20 AP50-95(%) | 차이(%p) |','|---|---:|---:|---:|']
    for c in classes:lines.append(f"|{c['class_name']}|{c['CM10_AP50_95']:.6f}|{c['CM20_AP50_95']:.6f}|{c['delta_AP50_95_pp']:+.6f}|")
    lines+=['',f'사전 채택 기준 충족: {candidate}.',
        '기준: 전체 mAP50-95 비저하, inclusion AP75·AP50-95 상승, silk_spot·rolled_pit AP50-95 상승.',
        '단일 시드 개발 결과다. 신뢰도 상승만으로 개선을 판정하지 않으며 다른 클래스의 하락도 함께 읽는다.']
    (P/'SGD_COSINE_CM10_VS_CM20_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

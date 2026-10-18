"""Saved JSON-only paired comparison, with frozen 640 size bins; no model imports."""
from pathlib import Path
import os,sys,json,csv,collections,importlib.util,math
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True;sys.stdout.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parent;B=P.parents[1]
RUN=B/'research_extensions/full_data_26s_sgd_640_20261017_01/runs/GC10_FULL1604_yolo26s_SGD_e100_b8_i640'
OLD=B/'research_extensions/full_data_inclusion52_cpu_20261018_01'
GT=B/'research_extensions/yolo26s_sgd_300_20261017_01/diagnostics/DEV_GT.json'

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(name,z):
    with (P/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(z[0]));w.writeheader();w.writerows(z)
def put(name,x):
    with (P/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    assert read(P/'VALIDATION.json')['status']=='PASS'
    assert not (P/'COMPARISON.json').exists()
    spec=importlib.util.spec_from_file_location('saved_R5_matching',B/'research_extensions/inclusion_object_errors_cpu_20261017_01/analyze.py')
    rule=importlib.util.module_from_spec(spec);spec.loader.exec_module(rule)
    gt=read(GT);assert len(gt)==343
    old=[r for r in rows(OLD/'FULL_INCLUSION_OBJECT_ERRORS.csv') if float(r['confidence_threshold'])==.001]
    geom={(r['image_id'],int(r['gt_index'])):r for r in old};assert len(geom)==52
    assert collections.Counter(r['size_bin'] for r in old)=={'Small_<16':15,'Medium_16_to_<32':15,'Large_>=32':22}
    assert {(sid,j) for sid,gg in gt.items() for j,g in enumerate(gg) if g['cls']==6}==set(geom)
    out=[];totalpred={};totals=[];sizes=[]
    for resolution,path in [(640,RUN/'sealed_dev/predictions.json'),(1024,P/'sealed_dev_1024/predictions.json')]:
        raw=read(path);byid=collections.defaultdict(list)
        assert all(str(p['image_id']) in gt for p in raw)
        for idx,p in enumerate(raw):
            x,y,w,h=p['bbox'];s=float(p['score']);c=int(p['category_id'])-1
            assert w>=0 and h>=0 and 0<=s<=1 and 0<=c<10 and all(math.isfinite(v) for v in (x,y,w,h,s))
            byid[str(p['image_id'])].append(dict(index=idx,cls=c,score=s,box=[x,y,x+w,y+h]))
        for conf in (.001,.25):totalpred[resolution,conf]=sum(p['cls']==6 and p['score']>=conf for pp in byid.values() for p in pp)
        for sid in sorted({key[0] for key in geom}):
            gi=sorted(j for s,j in geom if s==sid)
            pp0=byid.get(sid,[])
            for conf in (.001,.25):
                pp=sorted([p for p in pp0 if p['score']>=conf],key=lambda p:(-p['score'],p['index']))
                mat=rule.ious([gt[sid][j]['box'] for j in gi],[p['box'] for p in pp])
                for threshold in (.5,.75):
                    used=set();pairs={}
                    for k,p in enumerate(pp):
                        if p['cls']!=6:continue
                        candidates=[r for r in range(len(gi)) if r not in used and mat[r,k]>=threshold]
                        if candidates:
                            r=max(candidates,key=lambda r:(float(mat[r,k]),-gi[r]));pairs[r]=k;used.add(r)
                    assert len(set(pairs.values()))==len(pairs)
                    for r,j in enumerate(gi):
                        same=[k for k,p in enumerate(pp) if p['cls']==6]
                        other=[k for k,p in enumerate(pp) if p['cls']!=6 and mat[r,k]>=threshold]
                        best=max(same,key=lambda k:(float(mat[r,k]),pp[k]['score'],-pp[k]['index'])) if same else None
                        iou=float(mat[r,best]) if best is not None else 0.
                        kind='A_correct' if r in pairs else 'B_classification_confusion' if other else 'C_localization_error' if .1<=iou<threshold else 'D_missed_or_other'
                        m=pairs.get(r);g=geom[sid,j]
                        out.append(dict(resolution=resolution,confidence_threshold=conf,matching_iou_threshold=threshold,
                            object_id=g['object_id'],image_id=sid,gt_index=j,size_bin_640=g['size_bin'],
                            nominal_height640=float(g['nominal_height640']),error_type=kind,
                            matched_prediction_row=pp[m]['index'] if m is not None else '',
                            matched_confidence=pp[m]['score'] if m is not None else '',
                            matched_iou=float(mat[r,m]) if m is not None else '',
                            matched_box_xyxy=json.dumps(pp[m]['box']) if m is not None else '',
                            best_inclusion_iou=iou,best_inclusion_confidence=pp[best]['score'] if best is not None else '',
                            best_inclusion_box_xyxy=json.dumps(pp[best]['box']) if best is not None else '',
                            has_other_class_overlap=bool(other)))
    table('PAIRED_INCLUSION_OBJECTS.csv',out)
    # Independently reproduce the previous Full640 .5 matching, not a new inference.
    oldall=rows(OLD/'FULL_INCLUSION_OBJECT_ERRORS.csv')
    for r in [x for x in out if x['resolution']==640 and x['matching_iou_threshold']==.5]:
        prev=next(x for x in oldall if x['object_id']==r['object_id'] and float(x['confidence_threshold'])==r['confidence_threshold'])
        assert r['error_type']==prev['error_type']
        assert str(r['matched_prediction_row'])==prev['matched_prediction_row']
    for resolution in (640,1024):
        for conf in (.001,.25):
            for threshold in (.5,.75):
                z=[r for r in out if r['resolution']==resolution and r['confidence_threshold']==conf and r['matching_iou_threshold']==threshold]
                assert len(z)==52
                for size in ('ALL','Small_<16','Medium_16_to_<32','Large_>=32'):
                    zz=z if size=='ALL' else [r for r in z if r['size_bin_640']==size]
                    cc=collections.Counter(r['error_type'] for r in zz)
                    row=dict(resolution=resolution,confidence_threshold=conf,matching_iou_threshold=threshold,size_bin_640=size,
                        GT=len(zz),correct=cc['A_correct'],confusion=cc['B_classification_confusion'],
                        localization=cc['C_localization_error'],missed_or_other=cc['D_missed_or_other'],
                        correct_conf_ge25=sum(r['error_type']=='A_correct' and r['matched_confidence']>=.25 for r in zz),
                        correct_conf_lt25=sum(r['error_type']=='A_correct' and r['matched_confidence']<.25 for r in zz),
                        iou_ge75_subset_of_matching=sum(r['error_type']=='A_correct' and r['matched_iou']>=.75 for r in zz))
                    sizes.append(row)
                    if size=='ALL':totals.append(dict(**row,inclusion_predictions_all_dev=totalpred[resolution,conf],
                        nonmatched_inclusion_predictions_all_dev=totalpred[resolution,conf]-cc['A_correct']))
    table('CPU_MATCHING_SUMMARY.csv',totals);table('CPU_SIZE_SUMMARY.csv',sizes)
    m={640:read(RUN/'sealed_dev_metrics.json'),1024:read(P/'sealed_dev_metrics.json')}
    classes=[]
    for name,a in m[640]['all_AP_by_class_and_IoU'].items():
        b=m[1024]['all_AP_by_class_and_IoU'][name]
        classes.append(dict(class_name=name,AP50_640=sum(a[:1])*100,AP50_1024=b[0]*100,delta_AP50_pp=(b[0]-a[0])*100,
            AP75_640=a[5]*100,AP75_1024=b[5]*100,delta_AP75_pp=(b[5]-a[5])*100,
            AP50_95_640=sum(a)*10,AP50_95_1024=sum(b)*10,delta_AP50_95_pp=(sum(b)-sum(a))*10))
    table('CLASS_AP_COMPARISON.csv',classes)
    comparison=dict(official_metrics={str(k):v['overall'] for k,v in m.items()},classes=classes,
        cpu_summary=totals,cpu_size_summary=sizes,previous640_matching_reproduced=True,
        no_new_size_bins=True,no_new_training=True,evaluation_calls=1)
    put('COMPARISON.json',comparison)
    def get(res,conf=.001,iou=.5,size='ALL'):
        return next(r for r in sizes if r['resolution']==res and r['confidence_threshold']==conf and r['matching_iou_threshold']==iou and r['size_bin_640']==size)
    lines=['# Full-data YOLO26s: 640 vs 1024 추론 해상도 민감도','',
        '기존 Full1604·SGD100의 동일 best.pt(best epoch61)를 고정하고 Dev343을1024로 정확히1회 평가했다.640은 완료된 평가 산출물을 재사용했고 재추론하지 않았다. 새 학습·DINO·test·GT크롭·자동 후속실험은 없다.','',
        '## 공식 Ultralytics AP','', '| 지표 (%) | 640 | 1024 | 변화 (%p) |','|---|---:|---:|---:|']
    for label,key in [('전체 mAP50','metrics/mAP50(B)'),('전체 mAP50-95','metrics/mAP50-95(B)')]:
        a=m[640]['overall'][key]*100;b=m[1024]['overall'][key]*100
        lines.append(f'|{label}|{a:.6f}|{b:.6f}|{b-a:+.6f}|')
    inc=next(c for c in classes if c['class_name']=='inclusion')
    for label,key in [('Inclusion AP50','AP50'),('Inclusion AP75','AP75'),('Inclusion AP50-95','AP50_95')]:
        lines.append(f"|{label}|{inc[key+'_640']:.6f}|{inc[key+'_1024']:.6f}|{inc['delta_'+key+'_pp']:+.6f}|")
    lines+=['','## 동일52개 GT의 CPU 매칭','',
        '공식 AP와 별개다. 원본 JSON의3소수 좌표·5소수score를 이용하며 conf↓,JSON행↑ 순서로 같은 클래스의 미사용GT 최대IoU를 탐욕 매칭한다. IoU.5/.75는 각각 독립 재매칭했다. 공식 Ultralytics의 IoU기반 매칭·PR 적분 및 반올림 전 텐서와 일치하는 지표로 취급하지 않는다.','',
        '| conf | IoU | 정답52 중640 | 정답52 중1024 | 비매칭 inclusion예측640 | 비매칭1024 |','|---|---|---:|---:|---:|---:|']
    for conf in (.001,.25):
        for threshold in (.5,.75):
            a=next(r for r in totals if r['resolution']==640 and r['confidence_threshold']==conf and r['matching_iou_threshold']==threshold)
            b=next(r for r in totals if r['resolution']==1024 and r['confidence_threshold']==conf and r['matching_iou_threshold']==threshold)
            lines.append(f"|{conf}|{threshold}|{a['correct']}|{b['correct']}|{a['nonmatched_inclusion_predictions_all_dev']}|{b['nonmatched_inclusion_predictions_all_dev']}|")
    lines+=['','## 크기별 변화','',
        '크기는 기존640 환산 높이 구간으로 고정했다. 다음 표는 conf≥.001,IoU≥.5이며 모든 수치는640→1024다. 저신뢰도는 올바르게 매칭된 후보 중score<.25다. 클래스혼동은 다른 클래스의 겹침을 이용한 진단이며 실제 억제의 증거는 아니다.','',
        '| 640높이 구간 | GT | 정상score≥.25 | 저신뢰도 정상 | 클래스중첩 진단 | 위치오류 | 미검출·기타 | .75 독립매칭 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for size in ('Small_<16','Medium_16_to_<32','Large_>=32','ALL'):
        a=get(640,size=size);b=get(1024,size=size)
        pair=lambda key:f"{a[key]}→{b[key]}"
        lines.append(f"|{size}|{a['GT']}|{pair('correct_conf_ge25')}|{pair('correct_conf_lt25')}|{pair('confusion')}|{pair('localization')}|{pair('missed_or_other')}|{get(640,iou=.75,size=size)['correct']}→{get(1024,iou=.75,size=size)['correct']}|")
    lines+=['','## 10개 클래스 공식 AP50-95','', '| 클래스 | 640 (%) | 1024 (%) | 변화 (%p) |','|---|---:|---:|---:|']
    for c in classes:lines.append(f"|{c['class_name']}|{c['AP50_95_640']:.6f}|{c['AP50_95_1024']:.6f}|{c['delta_AP50_95_pp']:+.6f}|")
    overall_delta=(m[1024]['overall']['metrics/mAP50-95(B)']-m[640]['overall']['metrics/mAP50-95(B)'])*100
    sm0=get(640,size='Small_<16');sm1=get(1024,size='Small_<16')
    lines+=['','## 결론과 한계','',
        f"전체 mAP50-95 {overall_delta:+.6f}%p; inclusion AP50 {inc['delta_AP50_pp']:+.6f}%p,AP75 {inc['delta_AP75_pp']:+.6f}%p,AP50-95 {inc['delta_AP50_95_pp']:+.6f}%p.",
        f"작은15개 정상IoU.5 매칭 {sm0['correct']}→{sm1['correct']},위치오류 {sm0['localization']}→{sm1['localization']}, 독립IoU.75 매칭 {get(640,iou=.75,size='Small_<16')['correct']}→{get(1024,iou=.75,size='Small_<16')['correct']}.",
        '이 고정 모델의 추론 입력만1024로 확대하는 방법은 성능 개선안으로 채택하지 않는다. 작은 객체의 낮은임계값 매칭은 소폭 좋아졌지만, inclusion의 공식AP50/AP75/AP50-95가 모두 하락했고 전체mAP도 악화됐다. 작은 객체의 conf≥.25 정답은4→2개,전체 저신뢰도 정상후보는26→32개로 변했다. 더 크게 입력했다고 더 정확하고 자신 있게 검출한 것은 아니다.',
        'conf≥.001,IoU.5에서 비매칭inclusion은2088→3151개로 늘었다. conf≥.25에서는44→31개로 줄었지만 정답도16→8개로 감소했으므로 오탐감소만으로 개선이라 판단할 수 없다.10개클래스 중punching_hole만 AP50-95가 상승했고,welding_line은-33.121072%p로 가장 크게 하락했다.',
        '이전 Full52 보고서의9개는IoU.5에서 선택된 정답예측 중IoU≥.75인 부분집합이다.이번표의20개는IoU.75를기준으로처음부터독립재매칭한수다.동일정의의부분집합은9→8개,독립재매칭은20→15개이며혼용하지않는다.',
        '고정checkpoint의 추론 해상도에 대한 단일dev 탐색이다.1024 학습 효과, 데이터셋 절대상한, 산업 적용 성능, 손실함수의 인과적 병목을 입증하지 않는다.1024 이후 다른해상도·재학습·새알고리즘·자동진단은 실행하지 않는다.','',
        '## 재현 및 보존','',
        '실행: `Defect_VLM_Project/.python311/python.exe -I -B research_extensions/full_data_1024_sensitivity_20261018_01/run_once.py` (이미 완료된 폴더에서는 재실행 방지). CPU 비교: 같은Python으로compare_cpu.py. 자동재시도/배치변경없음.',
        'REQUEST.json과 ACTUAL_EVALUATION.json에요청/실제조건,ENVIRONMENT.json에버전/코드SHA,VALIDATION.json에평가1회/메모리/완료시간/기존입력불변을기록했다. 새dev캐시와예측은전부현재폴더에만저장됐다.독립test를가리키는항목은없다.']
    (P/'FULLDATA_640_VS_1024_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(overall_delta_pp=overall_delta,inclusion=inc,small640=sm0,small1024=sm1,cpu=totals),ensure_ascii=False))

if __name__=='__main__':main()

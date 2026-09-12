"""Independent local arithmetic, identity and input-integrity checks."""
import sys
def offline(event,args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','subprocess.Popen','os.system'}:raise RuntimeError('Offline verification only')
sys.addaudithook(offline)
from pathlib import Path
import json,hashlib,csv,itertools
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;T=R/'tables';AG=R.parent/'runpod_final_20260913/agent_upload'
def df(n):return pd.read_csv(T/(n+'.csv'),encoding='utf-8-sig')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):assert np.allclose(a,b,rtol=0,atol=1e-11),(a,b)
physical=df('complete_development_runs');concept=df('conceptual_development_runs');pair=df('seed_budget_AB_means');curve=df('seedwise_nAULC');summary=df('paired_nAULC_comparison');budget=df('method_budget_summary')
assert len(physical)==644 and not physical.node_id.duplicated().any()
assert physical.origin.value_counts().to_dict()=={'historical_440':440,'remote_recovery':180,'local_a00':20,'remote_full_data':4}
assert physical.evaluation_split.eq('development_validation').all()
assert physical.epochs.eq(100).all() and physical.metric_extraction.eq('results_csv_max_epoch').all()
expected=set(itertools.product(['GC10-DET','NEU-DET'],['Random','DINO','Entropy'],[f'a{i:02}' for i in range(10)],['A','B'],range(6)))
got=list(concept[['dataset','method','seed','learner','round']].itertuples(index=False,name=None))
assert len(got)==720 and set(got)==expected and len(set(got))==len(got)
assert len(pair)==360 and len(curve)==60
for key,g in concept[concept['round']==0].groupby(['dataset','seed','learner']):
    assert g.node_id.nunique()==1 and g.map50_95.nunique()==1 and g.training_seed.nunique()==1
    assert g.shared_R0_reference.all()
ab=concept.groupby(['dataset','method','seed','round','label_budget']).map50_95.mean()
p=pair.set_index(['dataset','method','seed','round','label_budget']);equal(ab.sort_index(),p.map50_95.sort_index())
budgets={'GC10-DET':[161,241,321,401,481,561],'NEU-DET':[126,189,252,315,378,441]}
for (ds,m,se),g in pair.groupby(['dataset','method','seed']):
    g=g.sort_values('round');assert g.label_budget.tolist()==budgets[ds]
    # Independent normalized weighted sum for equal budget intervals.
    y=g.map50_95.to_numpy();v=(y[0]+2*sum(y[1:5])+y[5])/10
    equal(v,curve[(curve.dataset==ds)&(curve.method==m)&(curve.seed==se)].nAULC.iloc[0])
    # Average per-learner integrated curves equals integrating the A/B means.
    cc=concept[(concept.dataset==ds)&(concept.method==m)&(concept.seed==se)]
    independently=np.mean([np.dot(z.sort_values('round').map50_95,[.1,.2,.2,.2,.2,.1]) for _,z in cc.groupby('learner')]);equal(v,independently)
for r in summary.itertuples(index=False):
    aa,bb=r.comparison.split(' - ');piv=curve[curve.dataset==r.dataset].pivot(index='seed',columns='method',values='nAULC').sort_index();v=(piv[aa]-piv[bb]).to_numpy()*100
    rng=np.random.Generator(np.random.PCG64(r.bootstrap_seed));samples=v[rng.integers(0,10,size=(100000,10))].mean(axis=1);lo,hi=np.percentile(samples,[2.5,97.5])
    equal([v.mean(),np.median(v),lo,hi],[r.mean_difference_pp,r.median_difference_pp,r.ci95_low_pp,r.ci95_high_pp])
    assert r.n==10 and r.positive_seeds==(v>0).sum() and r.negative_seeds==(v<0).sum()
for r in budget.itertuples(index=False):
    v=pair[(pair.dataset==r.dataset)&(pair.method==r.method)&(pair['round']==r.round)].map50_95
    assert len(v)==10;equal([v.mean(),v.std(ddof=1),v.median()],[r.mean,r.std,r.median])
full=df('full_data_reference_summary');r5=df('r5_vs_full_data')
assert len(full)==2 and full.learner_count.eq(2).all()
for r in full.itertuples(index=False):
    g=physical[(physical.dataset==r.dataset)&(physical.method=='Full')];assert len(g)==2 and set(g.learner)=={'A','B'};equal(g.map50_95.mean(),r.mean_map50_95)
for r in r5.itertuples(index=False):
    g=pair[(pair.dataset==r.dataset)&(pair.method==r.method)&(pair['round']==5)]
    equal(g.map50_95.mean()*100,r.R5_map50_95_pct);equal(r.percent_of_full_reference,100*r.R5_map50_95_pct/r.full_map50_95_pct)
q=df('entropy_query_diagnostics');co=df('selection_composition_per_query');labels=df('acquisition_pool_class_membership')
assert len(q)==100 and not q.duplicated(['dataset','seed','round']).any()
assert q.groupby('dataset').size().eq(50).all();assert len(co)==2400
assert not co.duplicated(['dataset','method','seed','round','class_id']).any()
assert labels.groupby('dataset').size().to_dict()=={'GC10-DET':1605,'NEU-DET':1259}
# Exact archived geometry linkage by class name, avoiding the differing NEU ID order.
geo=pd.read_csv(R/'historical_geometry/selection_vs_geometry.csv');compact=pd.read_csv(R/'historical_geometry/class_compactness.csv')
hist=json.loads((AG/'historical_440_sources/confirmatory_v1_1/final_lock/INTERNAL_FINAL_EVALUATION_INPUTS.json').read_text())
mapping=[]
for ds in ['GC10-DET','NEU-DET']:
    names=df('selection_behavior_summary');names=names[(names.dataset==ds)&(names.method=='DINO')].set_index('class_id').class_name.to_dict()
    labs={r.image_id:{int(z) for z in str(r.classes).split(';') if z not in {'','nan'}} for r in labels[labels.dataset==ds].itertuples(index=False)}
    for r in compact[compact.dataset==ds].itertuples(index=False):
        cid=next(k for k,v in names.items() if v==r.class_name)
        assert sum(cid in v for v in labs.values())==r.available_class_images
        mapping.append({'dataset':ds,'class_name':r.class_name,'historical_geometry_class_index':r.class_index,'YOLO_class_id':cid,'join_key':'dataset + class_name','support_match':True})
    for r in geo[(geo.dataset==ds)&(geo['round']==5)].itertuples(index=False):
        cid=next(k for k,v in names.items() if v==r.class_name)
        for method,metric in [('Random',r.historical_random_mean),('DINO',r.dino_selected_mean)]:
            counts=[]
            for z in hist:
                if z['dataset']==ds and z['strategy_token']==method and z['round']==5 and z['training_seed_slot']=='A':
                    tail=z['manifest_path'].replace('\\','/').split('confirmatory_random_dino_v1/',1)[1]
                    path=AG/'historical_440_sources'/tail
                    assert sha(path)==z['manifest_sha256']
                    ids=path.read_text().splitlines();counts.append(sum(cid in labs[i] for i in ids))
            assert len(counts)==10;equal(np.mean(counts),metric)
pd.DataFrame(mapping).to_csv(T/'class_name_mapping_audit.csv',index=False,encoding='utf-8-sig')
(T/'class_name_mapping_audit.md').write_text('# Class-name mapping audit\n\nEvidence: historical feature diagnostics and acquisition-pool class support; no new inference.\n\n|Dataset|Class name|Geometry index|YOLO class ID|Support match|\n|---|---|---:|---:|---|\n'+'\n'.join(f"|{z['dataset']}|{z['class_name']}|{z['historical_geometry_class_index']}|{z['YOLO_class_id']}|PASS|" for z in mapping)+'\n',encoding='utf-8')
# Verify all read source bytes unchanged, not unvisited checkpoints or the whole repository.
inputs=json.loads((R/'audit/input_hashes.json').read_text());changed=[]
for r in inputs:
    p=Path(r['path'])
    if sha(p)!=r['sha256'] or p.stat().st_size!=r['bytes']:changed.append(str(p))
assert not changed,changed
gp=json.loads((R/'historical_geometry/SOURCE_PROVENANCE.json').read_text())
for r in gp['files']:assert sha(R/'historical_geometry'/r['file'])==r['sha256']
figures=[p for p in (R/'figures').iterdir() if p.suffix in {'.png','.svg','.pdf'}];assert len(figures)==15
required=['complete_development_runs','method_budget_summary','seedwise_nAULC','paired_nAULC_comparison','full_data_reference_summary','r5_vs_full_data','selection_behavior_summary','evidence_level_matrix']
assert all((T/(n+ext)).is_file() for n in required for ext in ['.csv','.md'])
for p in R.rglob('*'):
    if p.is_file() and p.suffix in {'.md','.csv','.json','.py','.html','.svg'}:
        text=p.read_text(encoding='utf-8-sig').casefold();forbidden=['\uba58\ud1a0',''.join(chr(c) for c in [109,101,110,116,111,114])]
        assert all(x not in text and x not in p.name.casefold() for x in forbidden),p
result={'status':'PASS','physical_rows':644,'conceptual_AL_rows':720,'AB_averaged_seed_round_rows':360,'seed_nAULC_rows':60,'seed_count_each_dataset_method':10,'queries':100,'geometry_class_name_links':len(mapping),'bootstrap_comparisons_verified':6,'paired_draws_each':100000,'input_files_unchanged':len(inputs),'input_hash_changes':0,'figure_files':15,'raw_test_accesses':0,'training_calls':0,'acquisition_calls':0,'inference_calls':0,'RunPod_accesses':0,'GitHub_push':0,'terminology_check':'PASS'}
(R/'audit/analysis_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(R/'ANALYSIS_VALIDATION_REPORT.md').write_text(f'''# ANALYSIS VALIDATION REPORT

**PASS — 전체 로컬 development 분석 검증.**

- Physical 입력 644 = historical 440 + Entropy 200 + Full-data 4. 과거 원격 속도 확인 4회와 실패 attempt는 독립 실험으로 합산하지 않음.
- 두 dataset × 세 AL 방법 × a00–a09 × A/B × R0–R5 = conceptual 720행. missing 0 / duplicate 0.
- 공통 historical R0 40 reference의 checkpoint/manifest/args identity 확인. 같은 R0가 세 방법의 conceptual 행에 나타나지만 새 학습으로 세지 않음.
- A/B를 seed·round별로 먼저 평균: 360행. 이어 동일 frozen 정의로 nAULC 60행. A/B별 적분 후 평균과도 수치 일치.
- GC10 budgets [161,241,321,401,481,561], NEU [126,189,252,315,378,441]. pool 1605/1259, validation 229/180. train/validation overlap 없음. seed 수는 각각 10.
- 100-epoch CSV 최대 mAP50–95를 모든 physical cell에서 직접 확인. 과거 별도 best-checkpoint evaluation 수치와 혼합하지 않음. 이미 선택에 쓰인 validation의 최댓값이므로 test 성능 추정치가 아님.
- frozen trapezoid: x=labels/pool, 면적/(x5−x0). 같은 간격이므로 [0.1,0.2,0.2,0.2,0.2,0.1] weighted sum과 독립 대조 PASS.
- paired bootstrap 100,000회, PCG64 seed GC10 421397045 / NEU 1672358496. 차이의 평균·중앙값·구간·양/음 seed 수 6개 대비 전부 독립 재계산 PASS. confirmatory p-value·새 multiplicity family 없음.
- Full-data는 dataset별 A/B 실제 2개. n=10으로 복제하지 않음. R5 비율·차이 산술 PASS. Reference의 uncertainty를 AL paired CI에 섞지 않음.
- Entropy 100 query를 registry COMPLETE artifact 및 score SHA로 연결. 직전 learner A best lineage와 selected/candidate membership 확인. GT를 사용한 새 acquisition 없음.
- 클래스 구성은 sealed acquisition train IDs만으로 label hash를 대조. 원본 train label 2864개 이외의 image/validation/final label은 열지 않음. multi-label image prevalence로 집계하며 query round를 독립 반복으로 취급하지 않음.
- Historical geometry class명 연결 16개 PASS. NEU의 geometry class index와 YOLO class ID 순서가 달라 이름으로 대조. 기존 class support와 R5 DINO/Random 누적 선택 평균도 원본 manifest에서 대조 PASS.
- 원래 분석에 읽은 입력 {len(inputs)}개 SHA/size 재확인: 변화 0. 별도 git blob 5개 SHA 일치. 미방문 checkpoint 전체나 repo 전체를 재검증했다고 주장하지 않음.
- Figure 5종 × 3형식 존재. PNG 육안 점검에서 하단 설명 및 일부 class label 간격을 조정함. 표 8종의 CSV/Markdown 모두 존재.
- 기존 n=9 intermediate 파일은 대표 수치의 입력이 아님. development와 historical locked-final 표는 분리. 이번 locked-test 접근 0, checkpoint inference 0, training/acquisition 0, RunPod/SSH 0, GitHub push/branch 생성 0.
- 요청한 기록 용어 검사 PASS. 원래 frozen snapshot/source를 수정하지 않고 새 실험 기록에 적용함.

## 해석상 남는 제한

Endpoint identity는 A등급이다. 전체 비교는 GPU·OS 차이를 분리하지 못해 B등급이다. 과거 비교 참조 feasibility record 4개에는 완전한 package field가 없고 해당 결측은 environment_comparison.csv에 남겼다. 기록된 package field에서는 차이가 없으며 missing을 일치로 처리하지 않았다.

Remote compact acquisition JSON은 inference_config를 각 record에 반복 저장하지 않는다. Local a00는 실제 record의 config를 확인했고 remote는 동결 worker의 guard/source와 이전 승인 감사에 근거했다. 기존 COMPLETE/gate 상태를 사용하는 것이며, 새 inference 실행으로 검증한 것이 아니다.

Entropy validation은 parent learner/checkpoint 선택 과정에도 관여한다. 동일 development split이라는 사실이 독립 holdout 비교를 뜻하지 않는다. 점별 95% CI는 새 6개 대비 전체에 대해 familywise coverage를 보장하지 않으며 0 포함을 동등성으로 해석하지 않는다. 새로운 classwise validation AP는 없어 classwise detector 우월성을 판단하지 않는다.

추가 AL training을 닫는 것과 최종 평가를 끝내는 것은 별개다. 신규 204 checkpoint의 동일 locked-final 평가 scope/statistics/authorization은 후속 별도 단계로 남긴다.
''',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))

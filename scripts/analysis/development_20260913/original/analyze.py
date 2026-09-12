"""Local, model-free integrated development analysis. Inputs are immutable."""
import sys
def offline(event, args):
    if event in {'socket.connect','socket.connect_ex','socket.getaddrinfo','subprocess.Popen','os.system'}:
        raise RuntimeError('Offline analysis only')
sys.addaudithook(offline)
from pathlib import Path
import json, hashlib, io, ast, sqlite3, re
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent; W=R.parent
AG=W/'runpod_final_20260913/agent_upload'
H=AG/'historical_440_sources'; LOCAL=W/'confirmatory_random_dino_v1'
T=R/'tables'; T.mkdir(parents=True,exist_ok=True)
P=R/'audit'; P.mkdir(exist_ok=True)
EVID='Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within seed; Locked-final: not used'
POOL={'GC10-DET':1605,'NEU-DET':1259}; BUD={'GC10-DET':[161,241,321,401,481,561],'NEU-DET':[126,189,252,315,378,441]}
METHODS=['Random','DINO','Entropy']; BOOT={'GC10-DET':421397045,'NEU-DET':1672358496}
sources={}
sealed={x['relative_path']:x for x in json.loads((AG/'FILE_MANIFEST.json').read_text())['files']}
def read(p,binary=False):
    p=Path(p).resolve(); assert AG in p.parents or LOCAL in p.parents or R in p.parents, p
    if not p.exists() and H in p.parents:
        p=LOCAL/p.relative_to(H)
    b=p.read_bytes(); h=hashlib.sha256(b).hexdigest()
    if AG in p.parents:
        k=p.relative_to(AG).as_posix(); x=sealed[k]; assert x['sha256']==h and x['size']==len(b),(k,'package seal')
    sources[str(p)]={'path':str(p),'sha256':h,'bytes':len(b)}
    return b if binary else b.decode('utf-8-sig')
def load(p):return json.loads(read(p))
def frame(p):
    x=pd.read_csv(io.StringIO(read(p)));x.columns=x.columns.str.strip();return x
def sha(p):return hashlib.sha256(read(p,binary=True)).hexdigest()
def jwrite(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def table(name,rows,evidence=EVID):
    df=rows.copy() if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(T/(name+'.csv'),index=False,encoding='utf-8-sig')
    def val(x):
        if isinstance(x,float):return '' if pd.isna(x) else f'{x:.6g}'
        return str(x).replace('|','/').replace('\n',' ')
    md=['# '+name,'',evidence,'','| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']
    md += ['| '+' | '.join(val(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)]
    (T/(name+'.md')).write_text('\n'.join(md)+'\n',encoding='utf-8');return df
def mapped(s):
    s=s.replace('\\','/');assert '..' not in s.split('/'),s
    if s.startswith('<RUNPOD_NETWORK_VOLUME>/'):return AG/'remote'/s[len('<RUNPOD_NETWORK_VOLUME>/'):]
    marker='confirmatory_random_dino_v1/'; assert marker in s,s
    tail=s.split(marker,1)[1]
    if tail.startswith('entropy_extension_v1/'):return AG/'local_a00'/tail.split('/',1)[1]
    return H/tail
def yaml(p):
    # Only top-level scalar args and the observed simple numeric class-name map.
    out={}; names={}
    for line in read(p).splitlines():
        if line and not line[0].isspace() and ':' in line:
            k,v=line.split(':',1);out[k]=v.strip().strip('\"\'')
        elif line.startswith('  ') and ':' in line:
            k,v=line.strip().split(':',1)
            if k.strip('\"\'').isdigit():names[int(k.strip('\"\''))]=v.strip().strip('\"\'')
    if names:out['names']=[names[i] for i in range(len(names))]
    return out
def idlist(p):
    ids=[Path(s.replace('\\','/')).stem for s in read(p).splitlines() if s.strip()]
    assert len(ids)==len(set(ids)),p
    return set(ids)
def dataids(a):
    d=yaml(mapped(a['data']));root=d.get('path','')
    assert 'test' not in d,'test endpoint in training YAML'
    def resolve(v):return mapped(v if v.startswith('<RUNPOD_NETWORK_VOLUME>/') or ':' in v else root.rstrip('/')+'/'+v)
    return idlist(resolve(d['train'])),idlist(resolve(d['val'])),d['names']
index=frame(AG/'analysis_inputs/TRAINING_CELL_INDEX_644.csv')
assert len(index)==644 and not index.node_id.duplicated().any()
assert index.evaluation_split.eq('development_validation').all()
plan=load(H/'confirmatory_v1_1/final_lock/INTERNAL_FINAL_EVALUATION_INPUTS.json');assert len(plan)==440
records={x['cell_id']:x for x in plan}; histargs={}; manifests={}; pools={}; vals={}; classes={}; checks=[]; diffs=[]; env=[]
for ds in POOL:
    stem='gc10' if ds=='GC10-DET' else 'neu'
    pool=set(frame(H/f'confirmatory_v1/data_freeze/{stem}_acquisition_universe_blind.csv').sample_id)
    val=set(frame(H/f'confirmatory_v1/data_freeze/{stem}_development_locked_manifest.csv').sample_id)
    ix=load(AG/f'remote/defect_al_entropy/inputs/manifests/datasets/{ds}.json')
    assert pool==set(ix['train_ids']) and val==set(ix['validation_ids']) and not pool&val
    assert len(pool)==POOL[ds] and ix['locked_test_images_indexed']==0
    pools[ds]=pool;vals[ds]=val;classes[ds]=list(ix['names'].values()) if isinstance(ix['names'],dict) else ix['names']
for r in plan:
    cid=r['cell_id'];row=index[index.node_id==cid].iloc[0];a=yaml((AG/row.results_path).with_name('args.yaml'));histargs[cid]=a
    m=mapped(r['manifest_path']);assert sha(m)==r['manifest_sha256'];ids=idlist(m);manifests[cid]=ids
    tr,va,na=dataids(a);assert tr==ids and va==vals[r['dataset']] and na==classes[r['dataset']] and tr<=pools[r['dataset']]
    assert len(tr)==r['labeled_count']==BUD[r['dataset']][r['round']]
    assert a['epochs']=='100' and a['resume']=='false' and a['seed']==str(r['training_seed'])
    if r['round']>0:assert r['training_metadata']['epochs_completed']==100
    checks.append({'node_id':cid,'kind':'historical','train_count':len(tr),'validation_count':len(va),'split_seed_manifest':'PASS'})
r0=load(AG/'local_a00/historical_r0.json');assert len(r0)==40
for cid,x in r0.items():
    r=records[cid];assert x['read_only'] and not x['new_training_cell']
    assert x['best_sha256']==r['best_pt_sha256'] and x['manifest_sha256']==r['manifest_sha256'] and x['training_seed']==r['training_seed']
    assert sha(mapped(x['training_args_path']))==x['training_args_sha256']
nonpath={'data','model','project','name','save_dir'}
for row in index[index.method.isin(['Entropy','Full'])].itertuples(index=False):
    x=load(AG/row.result_path);a=yaml((AG/row.results_path).with_name('args.yaml'))
    assert x['status']=='COMPLETE' and x['epochs_completed']==100
    isfull=row.method=='Full';seed=0 if isfull else int(row.acquisition_index);rr=1 if isfull else int(row.round)
    ref=records[f'{row.dataset}_a{seed:02d}_Random_{row.learner}_R{rr}'];oa=histargs[ref['cell_id']]
    assert a['seed']==str(ref['training_seed'])
    for k in set(a)|set(oa):
        if k not in nonpath and a.get(k)!=oa.get(k):diffs.append({'cell':row.node_id,'key':k,'historical':oa.get(k),'new':a.get(k)})
    tr,va,na=dataids(a);assert va==vals[row.dataset] and na==classes[row.dataset] and not tr&va and tr<=pools[row.dataset]
    assert len(tr)==int(row.label_budget)
    if not isfull:
        assert not x['locked_test_accessed'] and x['cold_start'] and not x['resume'] and x['checkpoint_selection_split']=='validation'
        assert x['acquisition_seed']==ref['acquisition_seed'] and x['training_seed']==ref['training_seed'] and x['initialization_sha256']==ref['checkpoint_sha256']
        assert sha(mapped(x['manifest_path']))==x['manifest_sha256'] and idlist(mapped(x['manifest_path']))==tr
        ah=next(f['sha256'] for f in x['files'] if f['path'].replace('\\','/').endswith('/args.yaml'))
        assert sha((AG/row.results_path).with_name('args.yaml'))==ah
        oe=ref['training_metadata'].get('environment',{});ne=x['environment']['packages']
        present={k:oe[k] for k in ['torch','ultralytics','numpy','pillow'] if k in oe}
        assert all(v==ne[k] for k,v in present.items())
        env.append({'node_id':row.node_id,'new_gpu':x['environment']['gpu'],'historical_python':oe.get('python','not recorded in selected feasibility metadata'),'new_python':x['environment']['python_version'],'package_match':len(present)==4,'missing_historical_package_fields':','.join(k for k in ['torch','ultralytics','numpy','pillow'] if k not in oe),'os_gpu_bitwise_equivalence':'not established'})
        manifests[row.node_id]=tr
    else:
        assert tr==pools[row.dataset] and x['validation_completed'];assert sha((AG/row.results_path).with_name('args.yaml'))==x['args_sha256']
    checks.append({'node_id':row.node_id,'kind':row.method,'train_count':len(tr),'validation_count':len(va),'split_seed_manifest':'PASS'})
table('configuration_differences',diffs or [{'status':'NO_NONPATH_ARGUMENT_DIFFERENCES'}],'Evidence: configuration and identity audit; no performance inference')
assert not diffs, 'Configuration mismatch; stop before metric calculation'
# Registry-selected attempts only. Original failures remain excluded.
jobs=[]
for p,local in [(AG/'local_a00/runtime/official/registry.sqlite',True),(AG/'remote/defect_al_entropy_recovery/outputs/entropy/registry.sqlite',False)]:
    sha(p);db=sqlite3.connect(p.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row
    rows=[dict(z) for z in db.execute('select * from jobs')];db.close()
    if local:rows=[z for z in rows if '_a00_' in z['node_id']]
    for z in rows:
        assert z['state']=='COMPLETE';p2=mapped(z['artifact']);assert sha(p2)==z['artifact_sha256' if local else 'sha256']
        jobs.append({**z,'mapped_artifact':p2})
assert len(jobs)==300 and sum('_Entropy_' in z['node_id'] for z in jobs)==200
for z in jobs:
    if '_Entropy_' in z['node_id']:
        row=index[index.node_id==z['node_id']].iloc[0];assert AG/row.result_path==z['mapped_artifact']
gate_count=0
for gp in list((AG/'local_a00/runtime/official/round_gates').glob('*_a00_R*.json'))+list((AG/'remote/defect_al_entropy_recovery/outputs/entropy/gates').glob('*.json')):
    g=load(gp);assert g['status']=='PASS' and g['locked_test_accesses']==0;gate_count+=1
assert gate_count==100
table('cell_identity_audit',checks,'Evidence: metadata audit, 644 physical training cells')
table('environment_comparison',env,'Evidence: execution metadata; no hardware equivalence claim')
audit={'status':'PASS','endpoint_grade':'A: same development membership and metric extraction','overall_grade':'B: comparable development analysis conditional on cross-hardware/OS execution; not bitwise equivalence or randomized hardware contrast','physical_cells':644,'historical':440,'entropy':200,'full':4,'queries':100,'gates':100,'common_R0_references':40,'nonpath_arg_differences':0,'train_pool_counts':POOL,'validation_counts':{k:len(v) for k,v in vals.items()},'acquisition_seeds':'a00-a09','learner_seeds':'matched to frozen historical numeric mapping','locked_final_metrics_used':False,'metric_calculation_started':False}
jwrite(P/'data_structure_audit.json',audit)
jwrite(P/'input_hashes.json',list(sources.values()))
if '--compute' not in sys.argv:
    print(json.dumps(audit,ensure_ascii=False));sys.exit(0)
# Numerical analysis starts only after persisted pre-calculation audit.
src=read(H/'confirmatory_v1_1/final_analysis/src/analyze_final_results.py');tree=ast.parse(src)
safe=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'trapezoid_aulc','percentile_ci'}],type_ignores=[])
ns={'np':np,'Any':object};exec(compile(safe,'frozen_numerical_functions','exec'),ns)
physical=[];concept=[]
for row in index.itertuples(index=False):
    df=frame(AG/row.results_path);assert df.epoch.tolist()==list(range(1,101))
    values=df['metrics/mAP50-95(B)'];assert np.isfinite(values).all() and values.between(0,1).all()
    k=values.idxmax();v=float(values.loc[k]);assert abs(v-row.validation_map50_95)<1e-12
    a=yaml((AG/row.results_path).with_name('args.yaml'))
    r={**row._asdict(),'map50_95':v,'best_epoch':int(df.loc[k,'epoch']),'training_seed':int(a['seed']),'results_sha256':sha(AG/row.results_path)}
    physical.append(r)
    if row.method=='Full':continue
    for m in METHODS if row.method=='SHARED' else [row.method]:
        seed=int(row.acquisition_index);ref=records[f'{row.dataset}_a{seed:02d}_SHARED_{row.learner}_R0']
        concept.append({**r,'method':m,'seed':f'a{seed:02d}','round':int(row.round),'acquisition_seed':ref['acquisition_seed'],'shared_R0_reference':row.method=='SHARED'})
table('complete_development_runs',physical,'Evidence: Development validation; 644 physical cells, not independent statistical replicates; Full-data n=2 learners per dataset; Locked-final: not used')
c=table('conceptual_development_runs',concept);assert len(c)==720 and not c.duplicated(['dataset','method','seed','learner','round']).any()
assert c.groupby(['dataset','method','seed','round']).size().eq(2).all()
pair=c.groupby(['dataset','method','seed','round','label_budget'],as_index=False).map50_95.mean()
table('seed_budget_AB_means',pair)
budget=pair.groupby(['dataset','method','round','label_budget']).map50_95.agg(['count','mean','std','median','min','max']).reset_index()
assert budget['count'].eq(10).all();table('method_budget_summary',budget)
curves=[]
for (ds,m,s),g in pair.groupby(['dataset','method','seed']):
    g=g.sort_values('round');assert g['round'].tolist()==list(range(6)) and g.label_budget.tolist()==BUD[ds]
    points=[{'round':int(z['round']),'budget_fraction':z.label_budget/POOL[ds],'metric':z.map50_95} for _,z in g.iterrows()]
    v=ns['trapezoid_aulc'](points,'metric')
    w=np.array([.1,.2,.2,.2,.2,.1]);assert abs(v-np.dot(g.map50_95,w))<1e-12
    curves.append({'dataset':ds,'method':m,'seed':s,'nAULC':v,'nAULC_pct':v*100})
cu=table('seedwise_nAULC',curves);assert len(cu)==60
contrasts=[('DINO','Random'),('Entropy','Random'),('DINO','Entropy')];summary=[];effects=[];budgetdiff=[]
for ds in POOL:
    piv=cu[cu.dataset==ds].pivot(index='seed',columns='method',values='nAULC').sort_index();assert piv.index.tolist()==[f'a{x:02d}' for x in range(10)]
    for aa,bb in contrasts:
        v=(piv[aa]-piv[bb]).to_numpy();lo,hi=ns['percentile_ci'](v.tolist(),BOOT[ds],replicates=100000)
        summary.append({'dataset':ds,'comparison':aa+' - '+bb,'n':10,'mean_difference_pp':v.mean()*100,'median_difference_pp':np.median(v)*100,'ci95_low_pp':lo*100,'ci95_high_pp':hi*100,'positive_seeds':int((v>0).sum()),'negative_seeds':int((v<0).sum()),'zero_seeds':int((v==0).sum()),'bootstrap_draws':100000,'bootstrap_seed':BOOT[ds],'inference':'descriptive development; no confirmatory p-value or equivalence test'})
        effects += [{'dataset':ds,'comparison':aa+' - '+bb,'seed':s,'difference_pp':float(x)*100} for s,x in zip(piv.index,v)]
        for rr in range(6):
            p=pair[(pair.dataset==ds)&(pair['round']==rr)].pivot(index='seed',columns='method',values='map50_95')
            budgetdiff += [{'dataset':ds,'round':rr,'label_budget':BUD[ds][rr],'comparison':aa+' - '+bb,'seed':s,'difference_pp':float(x)*100} for s,x in (p[aa]-p[bb]).items()]
su=table('paired_nAULC_comparison',summary);table('paired_nAULC_seed_effects',effects)
bd=table('budget_paired_differences',budgetdiff)
table('budget_paired_summary',bd.groupby(['dataset','round','label_budget','comparison']).difference_pp.agg(['mean','std','median','min','max']).reset_index())
full=[];r5=[]
ph=pd.DataFrame(physical)
for ds in POOL:
    f=ph[(ph.dataset==ds)&(ph.method=='Full')].set_index('learner');assert set(f.index)=={'A','B'}
    mean=float(f.map50_95.mean());full.append({'dataset':ds,'learner_count':2,'A_map50_95':float(f.loc['A','map50_95']),'B_map50_95':float(f.loc['B','map50_95']),'mean_map50_95':mean,'full_train_images':POOL[ds],'role':'full-label reference; not statistical upper bound'})
    for m in METHODS:
        g=pair[(pair.dataset==ds)&(pair.method==m)&(pair['round']==5)];v=float(g.map50_95.mean())
        r5.append({'dataset':ds,'method':m,'al_seeds':10,'full_learners':2,'R5_map50_95_pct':v*100,'full_map50_95_pct':mean*100,'R5_minus_full_pp':(v-mean)*100,'percent_of_full_reference':100*v/mean,'R5_images':BUD[ds][-1],'full_images':POOL[ds],'label_fraction_pct':100*BUD[ds][-1]/POOL[ds]})
table('full_data_reference_summary',full,'Evidence: Development validation; n=2 learners per dataset, A/B mean; descriptive reference; Locked-final: not used')
table('r5_vs_full_data',r5,EVID+'; Full-data: n=2 learner reference, not n=10')
# Acquisition-only labels: exact allowlist from previously sealed train pool.
labels={};labelrows=[];queryrows=[];composition=[]
for ds in POOL:
    ix=load(AG/f'remote/defect_al_entropy/inputs/manifests/datasets/{ds}.json');labels[ds]={}
    for sid in sorted(pools[ds]):
        p=LOCAL/f'confirmatory_v1_1/full_run/yolo_data_cache/{ds}/labels/{sid}.txt'
        text=read(p);assert sha(p)==ix['labels'][sid]['sha256'],(ds,sid,'label hash')
        labs=sorted({int(float(line.split()[0])) for line in text.splitlines() if line.strip()});assert all(0<=l<len(classes[ds]) for l in labs)
        labels[ds][sid]=set(labs);labelrows.append({'dataset':ds,'image_id':sid,'classes':';'.join(map(str,labs)),'label_sha256':sha(p)})
table('acquisition_pool_class_membership',labelrows,'Evidence: acquisition-pool labels only; descriptive selection composition; no validation/final labels read')
querymap={z['node_id']:z for z in jobs if '_Q_' in z['node_id']}
for ds in POOL:
    for si in range(10):
        seed=f'a{si:02d}'
        for m in METHODS:
            prior=manifests[f'{ds}_{seed}_SHARED_A_R0']
            for rr in range(1,6):
                cur=manifests[f'{ds}_{seed}_{m}_A_R{rr}'];assert cur==manifests[f'{ds}_{seed}_{m}_B_R{rr}']
                selected=cur-prior;candidate=pools[ds]-prior
                assert prior<=cur<=pools[ds] and len(selected)==BUD[ds][rr]-BUD[ds][rr-1]
                if m=='Entropy':
                    qp=querymap[f'{ds}_{seed}_Q_R{rr}']['mapped_artifact'];aq=load(qp)
                    sp=next(mapped(f['path']) for f in aq['files'] if f['path'].endswith('scores.json'))
                    sx=load(sp);assert sha(sp)==next(f['sha256'] for f in aq['files'] if f['path'].endswith('scores.json'))
                    assert set(sx['next_labeled_ids'])==cur and set(sx['selected_ids'])==selected
                    d=pd.DataFrame(sx['score_table']);assert set(d.image_id)==candidate and not d.image_id.duplicated().any()
                    assert set(d.loc[d.selected,'image_id'])==selected
                    rho=float(d.entropy_sum.rank().corr(d.n_detections.rank()))
                    pc=aq['policy_checkpoint'];refid=f'{ds}_{seed}_SHARED_A_R0' if rr==1 else f'{ds}_{seed}_Entropy_A_R{rr-1}'
                    target=records[refid]['best_pt_sha256'] if rr==1 else load(AG/index[index.node_id==refid].iloc[0].result_path)['best_sha256']
                    assert pc['sha256']==target and pc['slot']=='A' and pc['round']==rr-1
                    assert not aq['gt_used_for_selection']
                    if 'inference_config' in aq:
                        assert aq['inference_config']['conf']==.25 and aq['inference_config']['iou']==.7 and aq['inference_config']['max_det']==300
                    else:
                        # Remote compact acquisition records do not duplicate the guarded config.
                        guard=read(AG/'remote/defect_al_entropy_recovery/source_continuation/src/defect_al/entropy.py')
                        assert "'conf': 0.25" in guard and "'iou': 0.70" in guard and "'max_det': 300" in guard
                    queryrows.append({'dataset':ds,'seed':seed,'round':rr,'candidate_count':len(candidate),'selected_count':len(selected),'score_count_spearman':rho,'selected_detection_mean':float(d.loc[d.selected,'n_detections'].mean()),'nonselected_detection_mean':float(d.loc[~d.selected,'n_detections'].mean()),'selected_detection_median':float(d.loc[d.selected,'n_detections'].median()),'nonselected_detection_median':float(d.loc[~d.selected,'n_detections'].median()),'scores_sha256':sha(sp),'lineage':'PASS'})
                for k,name in enumerate(classes[ds]):
                    sc=sum(k in labels[ds][sid] for sid in selected);cc=sum(k in labels[ds][sid] for sid in candidate)
                    spv=sc/len(selected);cpv=cc/len(candidate)
                    composition.append({'dataset':ds,'method':m,'seed':seed,'round':rr,'class_id':k,'class_name':name,'selected_count':sc,'candidate_count':cc,'selected_n':len(selected),'candidate_n':len(candidate),'selected_prevalence':spv,'candidate_prevalence':cpv,'prevalence_difference_pp':100*(spv-cpv),'prevalence_ratio':spv/cpv if cpv else np.nan})
                prior=cur
q=table('entropy_query_diagnostics',queryrows,'Evidence: frozen acquisition score logs; 50 queries per dataset nested within 10 seeds; not 50 independent trials; no new inference')
assert len(q)==100
co=table('selection_composition_per_query',composition,'Evidence: acquisition-pool class labels; image-level multi-label prevalence; candidate-relative baseline; 10 seeds')
cs=co.groupby(['dataset','method','class_id','class_name']).agg(mean_prevalence_difference_pp=('prevalence_difference_pp','mean'),median_prevalence_ratio=('prevalence_ratio','median'),mean_selected_prevalence=('selected_prevalence','mean'),mean_candidate_prevalence=('candidate_prevalence','mean')).reset_index()
table('selection_behavior_summary',cs,'Evidence: selection diagnostics; 5 rounds per seed, 10 seeds; descriptive, not causal')
table('entropy_round_diagnostics',q.groupby(['dataset','round']).agg(mean_rho=('score_count_spearman','mean'),median_rho=('score_count_spearman','median'),selected_detection_mean=('selected_detection_mean','mean'),nonselected_detection_mean=('nonselected_detection_mean','mean')).reset_index(),'Evidence: per-query selection diagnostics averaged over 10 seeds; no new inference')
# Exploratory within-dataset/method associations, seed is the unit; no causal claim.
links=[]
for (ds,m,k,cn),g in co.groupby(['dataset','method','class_id','class_name']):
    x=g.groupby('seed').prevalence_difference_pp.mean();y=cu[(cu.dataset==ds)&(cu.method==m)].set_index('seed').nAULC
    links.append({'dataset':ds,'method':m,'class_id':k,'class_name':cn,'n_seeds':10,'composition_nAULC_spearman':x.rank().corr(y.rank()),'role':'exploratory association; no multiplicity inference; no causal effect'})
table('selection_performance_associations',links)
evidence=[{'analysis':'three-way curves/nAULC','endpoint':'development validation CSV maximum across 100 epochs','unit':'10 acquisition seeds; A/B first averaged','grade':'B overall / A endpoint','limitation':'hardware/OS differs; adaptively reused validation; descriptive CIs'}, {'analysis':'historical DINO-Random primary','endpoint':'historical locked-final','unit':'10 acquisition seeds','grade':'historical confirmatory','limitation':'separate published endpoint; not pooled with new development'}, {'analysis':'full-label reference','endpoint':'development validation','unit':'2 learners per dataset','grade':'descriptive reference','limitation':'not statistical upper bound or n=10'}, {'analysis':'selection diagnostics','endpoint':'acquisition-only scores and labels','unit':'10 seeds; 5 dependent rounds each','grade':'exploratory descriptive','limitation':'not classwise validation AP; association not causation'}, {'analysis':'new 3-way final comparison','endpoint':'not executed','unit':'not yet evaluable','grade':'NOT YET EVALUABLE','limitation':'needs separately frozen scope/statistics and authorization'}]
table('evidence_level_matrix',evidence,'Evidence levels are intentionally segregated by endpoint and analysis role.')
result={'audit':audit,'paired_effects':summary,'nAULC_mean_pct':cu.groupby(['dataset','method']).nAULC_pct.mean().reset_index().to_dict('records'),'full_data':full,'r5_full':r5,'entropy_score_count':q.groupby('dataset').score_count_spearman.agg(['median','min','max']).reset_index().to_dict('records'),'physical_rows':644,'conceptual_rows':720,'seed_budget_rows':360,'seed_nAULC_rows':60,'queries':100,'training_calls':0,'acquisition_calls':0,'inference_calls':0,'locked_test_accesses':0,'runpod_accesses':0}
jwrite(R/'analysis_summary.json',result);jwrite(P/'input_hashes.json',list(sources.values()))
print(json.dumps(result,ensure_ascii=False,indent=2))

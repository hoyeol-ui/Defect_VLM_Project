"""Fixed five-pair design and artifact-list validation; standard library only."""
import csv, json
from pathlib import Path

DATASETS=['GC10-DET','NEU-DET']
METHODS=['Random','DINOv2_S14_F518_FF','Entropy_NCS_MEAN_V2']
BUDGETS=[10,15,20,25,30,35]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def need(ok,msg):
    if not ok:raise RuntimeError('DESIGN_INVALID: '+msg)
def model_id(row):return f"{row['dataset']}_{row['method']}_pair{row['pair']}_R{row['round']}"
def ordered_rows(rows):
    priority={m:i for i,m in enumerate(['SHARED_R0',*METHODS,'Full'])}
    return sorted(rows,key=lambda r:(6 if r['method']=='Full' else r['round'],r['pair'],DATASETS.index(r['dataset']),priority[r['method']]))
def expected_logical(rows):
    lookup={(r['dataset'],r['pair'],r['method'],r['round']):r for r in rows};out=[]
    for ds in DATASETS:
        for pair in range(1,6):
            for method in METHODS:
                for r,percent in enumerate(BUDGETS):
                    row=lookup[ds,pair,'SHARED_R0' if r==0 else method,r]
                    out.append({'dataset':ds,'pair':str(pair),'method':method,'budget_percent':str(percent),'physical_model_id':model_id(row)})
            row=lookup[ds,pair,'Full',None]
            out.append({'dataset':ds,'pair':str(pair),'method':'Full','budget_percent':'100','physical_model_id':model_id(row)})
    return out
def verify_design(root):
    root=Path(root);p=read(root/'config/PROTOCOL.json');rows=read(root/'config/MODEL_MATRIX.json')
    need(p['datasets']==DATASETS and p['methods']==METHODS and p['budgets']==BUDGETS,'Dataset/method/budget drift')
    expected_pairs=[{'pair':i,'acquisition':2026102400+i,'learner':2026102500+i} for i in range(1,6)]
    need(p['seed_pairs']==expected_pairs,'Exactly the first five predeclared paired seeds required')
    need(p['physical_models']==170 and p['logical_metric_rows']==190 and p['precomputed_selection_count']==110,'Design count drift')
    need(p['epochs']==200 and p['FP32'] is True and p['TF32'] is False and p['nbs']==64,'Scientific recipe drift')
    need(p['batch_candidates']==[16,32] and p['automatic_retry'] is False and p['test_in_package'] is False,'Execution policy drift')
    need(len(rows)==170 and len({model_id(r) for r in rows})==170,'Model count or duplicate ID')
    actual={};expected={}
    for ds in DATASETS:
        idx=read(root/'data'/ds/'index.json');n=idx['counts']['train']
        counts=[int(n*b/100+.5) for b in BUDGETS]
        need(counts==p['counts'][ds],'Rounded image budgets differ')
        for pair in expected_pairs:
            for method in ['SHARED_R0',*METHODS,'Full']:
                rounds=[0] if method=='SHARED_R0' else [None] if method=='Full' else range(1,6)
                for r in rounds:
                    expected[ds,pair['pair'],method,r]={'budget_percent':100 if method=='Full' else BUDGETS[r],
                        'train_images':n if method=='Full' else counts[r],'acquisition':pair['acquisition'],'learner':pair['learner']}
    for row in rows:
        key=(row['dataset'],row['pair'],row['method'],row['round']);need(key not in actual,'Duplicate design cell');actual[key]=row
        need(row['epochs']==200 and row['amp'] is False and row['batch']=='FROM_FROZEN_PLAN_16_OR_32','Cell training recipe drift')
    need(set(actual)==set(expected),'Missing or extra design cells')
    for key,value in expected.items():
        need(all(actual[key][k]==v for k,v in value.items()),'Seed, budget or count mismatch: '+str(key))
    ordered=ordered_rows(rows)
    need(all(r['method']=='SHARED_R0' for r in ordered[:10]),'R0 must run first')
    need(all(r['method']=='Full' for r in ordered[-10:]),'Full must run last as separate cold starts')
    physical=read_csv(root/'MODEL_LIST_170.csv')
    need([r['model_id'] for r in physical]==[model_id(r) for r in ordered],'Physical list/order differs')
    for i,(listed,row) in enumerate(zip(physical,ordered),1):
        need(listed['execution_order']==str(i),'Execution order differs')
        need(listed['round']==('' if row['round'] is None else str(row['round'])),'Physical CSV round differs')
        need(listed['FP32']=='True' and listed['batch']=='FROM_FROZEN_PLAN_16_OR_32' and listed['state']=='NOT_STARTED','Physical CSV preparation policy differs')
        for k in ['dataset','pair','acquisition','learner','method','budget_percent','train_images','epochs']:
            need(listed[k]==str(row[k]),'Physical CSV field differs: '+k)
    logical=read_csv(root/'LOGICAL_RESULT_MAP_190.csv')
    need(logical==expected_logical(rows),'Shared R0/result alias mapping differs')
    reporting=read(root/'config/REPORTING_PLAN.json')
    need(reporting['paired_seed_count']==5 and reporting['sample_std_ddof']==1 and reporting['logical_rows']==190,'Reporting count drift')
    need(reporting['two_sided_exact_sign_flip_min_p']==.0625,'Five-pair statistical limitation missing')
    return {'seed_pairs':5,'physical_models':170,'logical_rows':190,'precomputed_selections':110,'entropy_acquisitions':50,
        'Full_models':10,'shared_R0_models':10,'ordered_budgets':BUDGETS+[100],'GPU_work_performed_by_check':False,'test_inference':False}
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

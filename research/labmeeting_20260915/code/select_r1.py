"""GT-free process: reads only feature arrays, opaque IDs and predeclared seeds.
No image paths, XML files, class labels or detector metrics are read here.
"""
from common import OUT,DATASETS,SEEDS,core,np,dump,sha,savecsv
import time

def main():
    results=[]; costs=[]
    for ds in DATASETS:
        p=OUT/'features'/ds; ids=np.load(p/'sample_ids.npy',allow_pickle=False).astype(str)
        initial_n=core.round_half_up_count(len(ids),'0.10'); query_n=core.round_half_up_count(len(ids),'0.05')
        for seed in SEEDS:
            initial=core.initial_manifest(ids,size=initial_n,dataset=ds,acquisition_seed=seed)
            remaining=sorted(set(ids)-set(initial)); base=OUT/'selections'/ds/str(seed); base.mkdir(parents=True,exist_ok=True)
            dump(base/'R0.json',{'ids':list(initial),'membership_sha256':core.manifest_sha256(initial)})
            for mode in ['Random','C224','F224','F518']:
                start=time.perf_counter()
                if mode=='Random':
                    chosen=core.random_query(remaining,size=query_n,dataset=ds,acquisition_seed=seed,round_idx=1)
                    scores=[None]*len(chosen)
                else:
                    x=np.load(p/f'{mode}.npy',allow_pickle=False)
                    c=core.validate_embedding_cache(ids,ids,x)
                    chosen=core.farthest_first_query(remaining,initial,c,size=query_n)
                    lookup=c.as_lookup(); refs=list(initial); scores=[]
                    for sid in chosen:
                        scores.append(1-float(np.max(np.stack([lookup[s] for s in refs])@lookup[sid])))
                        refs.append(sid)
                assert len(chosen)==query_n and len(set(chosen))==query_n and not set(chosen)&set(initial)
                payload={'dataset':ds,'seed':seed,'mode':mode,'R0_membership_sha256':core.manifest_sha256(initial),'R1_query_membership_sha256':core.manifest_sha256(chosen),'R1_cumulative_membership_sha256':core.manifest_sha256([*initial,*chosen]),'order_semantics':'greedy selection order' if mode!='Random' else 'canonical ID order returned by frozen uniform sampler; no priority ranking','selected_ids':list(chosen),'selection_scores':scores}
                dump(base/f'{mode}_R1.json',payload)
                for k,(sid,score) in enumerate(zip(chosen,scores),1):results.append({'dataset':ds,'seed':seed,'mode':mode,'order':k,'sample_id':sid,'selection_score':score})
                cost={'dataset':ds,'seed':seed,'mode':mode,'seconds':time.perf_counter()-start};costs.append(cost)
                print(cost,flush=True)
    savecsv('R1_selection_order_GT_free.csv',results);savecsv('selection_costs.csv',costs)
    dump(OUT/'checks/selection_seal.json',{'GT_join_permitted_after_this_seal':True,'selector_code_sha256':sha(__file__),'core_code_sha256':sha(core.__file__),'files':{str(p.relative_to(OUT)):sha(p) for p in (OUT/'selections').rglob('*.json')}})
    print('ALL R1 SELECTIONS SEALED BEFORE GT JOIN',flush=True)

if __name__=='__main__': main()

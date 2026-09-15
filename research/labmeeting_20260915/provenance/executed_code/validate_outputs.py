from common import *
import importlib.util

def main():
    checks={};independent=[]
    for ds in DATASETS:
        ids,x,p=cache(ds)
        for label in ['A','B']:
            cp=p.parent/f'cache_{label}'
            for f,h in json.loads((cp/'metadata.json').read_text())['file_sha256'].items():assert sha(cp/f)==h
        folder=OUT/'features'/ds
        assert np.array_equal(np.load(folder/'C224.npy'),x)
        assert np.array_equal(np.load(folder/'sample_ids.npy'),ids)
        for mode in ['C224','F224','F518']:
            original=np.load(folder/f'{mode}.npy'); assert original.shape==(len(ids),384) and np.isfinite(original).all()
            # Independent vectorized FP64 recomputation of every greedy step.
            # Use the same FP32 normalization boundary as the reference, then compute dots independently.
            z=(original/np.linalg.norm(original,axis=1,keepdims=True)).astype('float64')
            similarity=z@z.T;ix={s:i for i,s in enumerate(ids)}
            for seed in SEEDS:
                base=OUT/'selections'/ds/str(seed)
                initial=json.loads((base/'R0.json').read_text())['ids'];selected=json.loads((base/f'{mode}_R1.json').read_text())['selected_ids']
                available=np.ones(len(ids),dtype=bool);initial_ix=[ix[s] for s in initial];available[initial_ix]=False
                score=1-similarity[:,initial_ix].max(axis=1);observed=[];margins=[]
                for sid in selected:
                    values=score.copy();values[~available]=-np.inf;best=int(values.argmax())
                    sorted_scores=np.sort(values[available]);margins.append(float(sorted_scores[-1]-sorted_scores[-2]))
                    observed.append(ids[best]); available[best]=False; score=np.minimum(score,1-similarity[:,best])
                same=observed==selected
                independent.append({'dataset':ds,'seed':seed,'mode':mode,'FP64_independent_order_exact_match':same,'smallest_winner_margin':min(margins)})
                assert same,(ds,seed,mode,'independent order differs')
    checks['all_30_farthest_first_orders_independently_reproduced']=True
    checks['old_caches_preserved']=True
    # Verify available materialized YOLO labels against original frozen XML conversion,
    # without creating or changing any labels.
    modulepath=DEV/'confirmatory_v1_1/src/prepare_yolo_benchmark_data.py'
    spec=importlib.util.spec_from_file_location('frozen_label_adapter',modulepath);adapter_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter_module)
    found=0;missing=0;labelrows=[]
    for r in registry().to_dict('records'):
        lp=FROZEN/f"confirmatory_v1_1/yolo_data_cache/{r['dataset']}/labels/{r['sample_id']}.txt"
        if not lp.exists():missing+=1;continue
        expected=adapter_module.yolo_label(r['dataset'],DATA/r['relative_image_path'],DATA/r['relative_annotation_path'])
        assert lp.read_text(encoding='ascii')==expected,lp
        found+=1;labelrows.append({'dataset':r['dataset'],'sample_id':r['sample_id'],'materialized_YOLO_label_match':True})
    checks['materialized_label_files_verified']=found;checks['pool_label_files_not_materialized']=missing
    savecsv('selection_independent_validation.csv',independent);savecsv('materialized_label_validation.csv',labelrows)
    checks['output_csv_files']=len(list((OUT/'tables').glob('*.csv')))
    checks['all_png_images_readable']=True
    for p in (OUT/'figures').glob('*.png'):
        with Image.open(p) as im:im.verify()
    dump(OUT/'checks/final_validation.json',checks);print(checks,flush=True)

if __name__=='__main__':main()

"""Prepare frozen train/dev indexes and inherit only detector-independent selections."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit import *
import shutil

def main():
    src=P/'src';src.mkdir(exist_ok=True)
    copied={}
    for name in ['core.py','strict_loop.py','trainer_adapter.py','amp_runtime.py','amp_compat.py','modern_trainer.py']:
        original=FULL/'src'/name;target=src/name
        data=original.read_text(encoding='utf-8')
        if name=='modern_trainer.py':
            old='np.cos(np.pi*.25)'
            assert data.count(old)==1
            # Same expression for all scientific 100-epoch cells; enables a one-epoch technical run.
            data=data.replace(old,'np.cos(np.pi*25/self.args.epochs)')
        target.write_text(data,encoding='utf-8')
        copied[name]={'source':rel(original),'source_sha':sha(original),'prepared_sha':sha(target)}
    entropy=B/'output/final_main_runpod_execution_v2_20260919/entropy.py'
    shutil.copyfile(entropy,src/'entropy.py')
    copied['entropy.py']={'source':rel(entropy),'source_sha':sha(entropy),'prepared_sha':sha(src/'entropy.py')}
    old=B/'output/final_main_runpod_execution_v2_20260919'
    overlay=B/'output/annotation_patch_v2_0_2_20260919/payload/annotation_overlays/ANNOTATION_PATCH_v2_0_2'
    annotations=read(overlay/'TRAIN_DEV_NORMALIZATION.json')
    lookup={(r['dataset'],r['partition'],r['id']):r for r in annotations['records']}
    prepared={}
    for ds in ['GC10-DET','NEU-DET']:
        groups={}
        for part in ['train','dev']:
            rows=[]
            for row in read(old/'inputs'/ds/(part+'_images.json')):
                key=(ds,part,row['id']);ann=lookup[key]
                image=old/row['image'];label=overlay/'labels'/ds/part/(row['id']+'.txt')
                assert image.exists() and label.exists(),(image,label)
                assert sha(image)==row['image_sha'] and sha(label)==ann['after_label_sha'],key
                # Cross-check recent copied data, including NEU's corrected development label.
                recent=(FULL/'datasets'/ds if ds=='GC10-DET' else B/'research_extensions/modern_yolo_A_20261015_01/datasets'/ds)
                for typ,path in [('images',image),('labels',label)]:
                    other=recent/typ/part/path.name
                    if other.exists():assert sha(path)==sha(other),(key,typ)
                rows.append({'id':row['id'],'image':str(image),'image_sha':row['image_sha'],
                    'label':str(label),'label_sha':ann['after_label_sha']})
            groups[part]=rows
        groups['names']=read(FULL/'datasets/GC10-DET/names.json') if ds=='GC10-DET' else read(B/'research_extensions/modern_yolo_A_20261015_01/datasets/NEU-DET/names.json')
        put(P/'inputs'/ds/'index.json',groups)
        prepared[ds]={'train':len(groups['train']),'dev':len(groups['dev']),'index_sha':sha(P/'inputs'/ds/'index.json')}
    lineage=[]
    selroot=HIST/'state_attempts/ANNOTATION_PATCH_v2_0_2/selections'
    protocol=read(P/'config/protocol.json')
    for ds,counts in protocol['counts'].items():
        pool={r['id'] for r in read(P/'inputs'/ds/'index.json')['train']}
        for pair in protocol['seed_pairs']:
            seed=pair['acquisition']
            shared=read(selroot/ds/str(seed)/'SHARED/R0.json')
            r0=shared['cumulative_ids']
            assert len(r0)==counts[0] and set(r0)<=pool
            for method in ['Random','DINO_F518']:
                prev=r0
                for k,n in enumerate(counts):
                    origin=selroot/ds/str(seed)/('SHARED' if k==0 else method)/f'R{k}.json'
                    m=read(origin)
                    assert len(m['cumulative_ids'])==n and len(set(m['cumulative_ids']))==n
                    assert set(m['cumulative_ids'])<=pool and m['split_sha']==read(B/'output/final_main_runpod_execution_v2_20260919/config/protocol.json')['datasets'][ds]['split_sha']
                    assert m['cumulative_ids'][:len(prev)]==prev
                    prev=m['cumulative_ids']
                    target=P/'selections'/ds/str(seed)/method/f'R{k}.json'
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,target)
                    lineage.append({'source':rel(origin),'target':rel(target),'sha':sha(target)})
    put(P/'evidence/preparation.json',{'status':'PASS','copied_sources':copied,'datasets':prepared,
        'selections':lineage,'annotation_normalization_sha':sha(overlay/'TRAIN_DEV_NORMALIZATION.json'),
        'test_images_or_labels_copied':False,'old_entropy_selections_reused':False})
    print(json.dumps({'status':'PASS','datasets':prepared,'inherited_selection_files':len(lineage)},indent=2))

if __name__=='__main__':main()

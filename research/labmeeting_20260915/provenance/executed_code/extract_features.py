from common import *
import time

def main():
    assert json.loads((OUT/'checks/reproduction_gate.json').read_text())['passed']
    import torch
    from transformers import AutoImageProcessor,AutoModel
    torch.set_num_threads(4); adapter.configure_torch_determinism(torch)
    adapter.verify_snapshot(SNAPSHOT)
    processor=AutoImageProcessor.from_pretrained(SNAPSHOT,local_files_only=True)
    model=AutoModel.from_pretrained(SNAPSHOT,local_files_only=True).eval().to('cuda',dtype=torch.float32)
    rows=registry(); costs=[]
    for ds in DATASETS:
        ids,original,_=cache(ds); lookup=rows[rows.dataset==ds].set_index('sample_id')
        dest=OUT/'features'/ds; dest.mkdir(exist_ok=True)
        np.save(dest/'sample_ids.npy',ids); np.save(dest/'C224.npy',original)
        for mode in ['F224','F518']:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start=time.perf_counter(); zs=[]
            for offset in range(0,len(ids),32):
                ims=[]
                for sid in ids[offset:offset+32]:
                    with Image.open(DATA/lookup.loc[sid,'relative_image_path']) as im: ims.append(render_input(im.convert('RGB'),mode))
                pv=processor(images=ims,do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values']
                assert pv.shape[2:]==(int(mode[1:]),)*2
                with torch.inference_mode(): z=model(pixel_values=pv.cuda()).pooler_output.cpu().numpy()
                assert z.shape==(len(ims),384) and np.isfinite(z).all()
                z=z/np.linalg.norm(z,axis=1,keepdims=True); zs.append(z.astype('float32'))
                for im in ims: im.close()
                if offset%320==0: print(ds,mode,min(offset+32,len(ids)),'/',len(ids),flush=True)
            x=np.concatenate(zs); np.save(dest/f'{mode}.npy',x)
            torch.cuda.synchronize()
            cost={'dataset':ds,'mode':mode,'n_images':len(ids),'batch_size':32,'seconds':time.perf_counter()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'file_sha256':sha(dest/f'{mode}.npy'),'max_norm_error':float(np.max(np.abs(np.linalg.norm(x,axis=1)-1)))}
            costs.append(cost); dump(OUT/'checks/extraction_costs.json',costs); print('COMPLETE',cost,flush=True)
    savecsv('extraction_costs.csv',costs)
    dump(OUT/'checks/features_manifest.json',{str(p.relative_to(OUT)):sha(p) for p in (OUT/'features').rglob('*.npy')})

if __name__=='__main__': main()

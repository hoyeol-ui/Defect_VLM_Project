from common import *
import struct, xml.etree.ElementTree as ET, time, platform, subprocess

def prepare():
    # Freeze rules before inspecting feature distances or selecting examples.
    rows=registry(); plan={'created_before_new_analysis':True,'seeds':SEEDS,
      'scope':'Existing split exploratory analysis; no detector training',
      'fixed_slide_prefixes':PREFIXES,'representative_rule':'First 6 IDs ranked by SHA256(representative-v1|ID); never replace after viewing',
      'random_pair_rule':'One independently uniform nonself partner per anchor, PCG64 seed 20260915, sorted ID order',
      'diagnostic_rule':'Nearest nonsharing-label pair and farthest sharing-label pair, sorted by distance then ID; selected posthoc as counterexamples, not representatives',
      'reproduction_rule':'First 32 sorted IDs plus 32 IDs ranked by SHA256(repro-v1|ID), excluding first 32; batch 32',
      'reproduction_thresholds':{'max_abs_component':1e-4,'max_cosine_distance':1e-6,'processor_pixel_max_abs':1e-6},
      'padding_rgb':[124,116,104],'F_resize':'longest edge target; half-up integer size; odd padding extra on right/bottom; bicubic',
      'extraction':'FP32, deterministic, no autocast/TF32, same weights and pooler, batch32; F518 batch32 initially',
      'coordinate_convention':'XML xyxy continuous edges, xmax-xmin, no +1/-1 (matches frozen labels); out-of-image boxes fail',
      'R1':'Initial10%, query5% of original pool half-up; shared R0; protocol namespace; candidate GT never passed to selector',
      'datasets':{}}
    for ds in DATASETS:
        ids,_,_=cache(ds); initial=list(ids[:32]); rest=rank_ids([s for s in ids if s not in initial],'repro-v1')[:32]
        plan['datasets'][ds]={'count':len(ids),'representative_ids':rank_ids(ids,'representative-v1')[:6], 'reproduction_ids':initial+rest,
          'initial_size':core.round_half_up_count(len(ids),'0.10'),'query_size':core.round_half_up_count(len(ids),'0.05')}
    path=OUT/'checks/analysis_plan_frozen.json'
    if path.exists(): assert json.loads(path.read_text(encoding='utf-8'))==plan
    else: dump(path,plan)
    return rows,plan

def audit_data(rows):
    records=[]; boxes=[]; anomalies=[]
    for k,r in enumerate(rows.to_dict('records')):
        image=DATA/r['relative_image_path']; ann=DATA/r['relative_annotation_path']
        assert sha(image)==r['raw_sha256'], image
        assert sha(ann)==r['annotation_sha256'], ann
        with Image.open(image) as im:
            rgb=im.convert('RGB'); w,h=rgb.size
            identity=hashlib.sha256(b'CONFIRMATORY_RGB_U8_V1\0'+struct.pack('>II',w,h)+rgb.tobytes()).hexdigest()
        assert identity==r['sample_id']==r['canonical_pixel_sha256'], image
        root=ET.parse(ann).getroot()
        xmlw=int(root.findtext('size/width')); xmlh=int(root.findtext('size/height'))
        assert (w,h)==(xmlw,xmlh),(image,(w,h),(xmlw,xmlh))
        labels=[]; seen=set(); duplicate=0
        for ob in root.findall('object'):
            raw=ob.findtext('name').strip()
            if r['dataset']=='GC10-DET':
                prefix=raw.split('_')[0]
                cid=int(prefix) if prefix.isdigit() else int(image.parent.name)
                name=GC_NAMES[cid-1]
                if not prefix.isdigit(): anomalies.append({'sample_id':r['sample_id'],'class_name_fallback':raw,'mapped_to':name,'rule':'existing frozen adapter: image parent directory number'})
            else: name=raw
            b=[float(ob.findtext('bndbox/'+a)) for a in ['xmin','ymin','xmax','ymax']]
            assert 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h,(image,b)
            sig=(name,*b)
            if sig in seen: duplicate+=1; continue
            seen.add(sig); labels.append(name)
            train_b=[min(b[0],w-1),min(b[1],h-1),min(b[2],w-1),min(b[3],h-1)]
            boxes.append({'dataset':r['dataset'],'sample_id':r['sample_id'],'class_name':name,'xmin':b[0],'ymin':b[1],'xmax':b[2],'ymax':b[3],'width':w,'height':h,'legacy_yolo_clamped':train_b!=b,'legacy_yolo_area_ratio':(train_b[2]-train_b[0])*(train_b[3]-train_b[1])/((b[2]-b[0])*(b[3]-b[1]))})
        assert labels,image
        if duplicate: anomalies.append({'sample_id':r['sample_id'],'duplicate_boxes_removed_for_summary':duplicate})
        records.append({**r,'width':w,'height':h,'labels':'|'.join(sorted(set(labels))),'n_boxes':len(labels),'duplicate_boxes':duplicate})
        if (k+1)%400==0: print('identity/coordinates checked',k+1,flush=True)
    savecsv('verified_pool_records.csv',records); savecsv('verified_boxes.csv',boxes)
    dump(OUT/'checks/data_integrity.json',{'images_checked':len(records),'all_raw_pixel_annotation_hashes_match':True,'all_xml_dimensions_and_box_bounds_match':True,'annotation_notes':anomalies,'box_count_unique':len(boxes),'legacy_yolo_clamped_box_count':sum(b['legacy_yolo_clamped'] for b in boxes),'coordinate_clarification':'Original XML continuous edges are used for geometry; legacy YOLO clamps to width-1/height-1. Both use xmax-xmin (no +1). This clarifies the original plan, and does not change IDs, sample rules or feature extraction.'})

def reproduce(rows,plan):
    import torch, transformers
    from transformers import AutoImageProcessor, AutoModel
    adapter.verify_snapshot(SNAPSHOT)
    torch.set_num_threads(4); adapter.configure_torch_determinism(torch)
    processor=AutoImageProcessor.from_pretrained(SNAPSHOT,local_files_only=True)
    print('processor',type(processor).__name__,flush=True)
    # An implementation rename is documented; numerical processor parity remains mandatory.
    model=AutoModel.from_pretrained(SNAPSHOT,local_files_only=True).eval().to('cuda',dtype=torch.float32)
    results=[]; pixelmax=0.; t=time.perf_counter()
    for ds in DATASETS:
        ids,old,p=cache(ds); meta=json.loads((p/'metadata.json').read_text())
        for f,h in meta['file_sha256'].items(): assert sha(p/f)==h
        assert set(ids)==set(rows[rows.dataset==ds].sample_id)
        bpath=p.parent/'cache_B'; assert np.array_equal(ids,np.load(bpath/'sample_ids.npy',allow_pickle=False))
        for f,h in json.loads((bpath/'metadata.json').read_text())['file_sha256'].items(): assert sha(bpath/f)==h
        assert np.allclose(old,np.load(bpath/'embeddings.npy'),rtol=0,atol=1e-6)
        mapping=rows[rows.dataset==ds].set_index('sample_id'); index={sid:i for i,sid in enumerate(ids)}
        chosen=plan['datasets'][ds]['reproduction_ids']; new=[]
        for start in range(0,len(chosen),32):
            images=[Image.open(DATA/mapping.loc[sid,'relative_image_path']).convert('RGB') for sid in chosen[start:start+32]]
            px=processor(images=images,return_tensors='pt')['pixel_values']
            manual=processor(images=[render_input(im,'C224') for im in images],do_resize=False,do_center_crop=False,return_tensors='pt')['pixel_values']
            err=float((px-manual).abs().max()); pixelmax=max(pixelmax,err)
            with torch.inference_mode(): z=model(pixel_values=px.cuda()).pooler_output.cpu().numpy()
            z=z/np.linalg.norm(z,axis=1,keepdims=True); new.append(z)
            for im in images: im.close()
        new=np.concatenate(new); prev=old[[index[s] for s in chosen]]
        cosine=1-np.sum(new.astype('float64')*prev,axis=1)/(np.linalg.norm(new.astype('float64'),axis=1)*np.linalg.norm(prev.astype('float64'),axis=1))
        for sid,a,b,c in zip(chosen,new,prev,cosine): results.append({'dataset':ds,'sample_id':sid,'max_abs_component':float(np.max(np.abs(a-b))),'cosine_distance':float(c)})
        np.save(OUT/'checks'/f'{ds}_C224_reproduced.npy',new)
        print(ds,'reproduction max abs',np.max(np.abs(new-prev)),'cos',cosine.max(),'pixel',pixelmax,flush=True)
    savecsv('C224_reproduction.csv',results)
    gate=all(r['max_abs_component']<=1e-4 and r['cosine_distance']<=1e-6 for r in results) and pixelmax<=1e-6
    dump(OUT/'checks/reproduction_gate.json',{'passed':gate,'n_images':len(results),'processor_class':type(processor).__name__,'expected_legacy_class':'BitImageProcessor','processor':processor.to_dict(),'pixel_max_abs':pixelmax,'max_abs_component':max(r['max_abs_component'] for r in results),'max_cosine_distance':max(r['cosine_distance'] for r in results),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'snapshot_sha256':adapter.verify_snapshot(SNAPSHOT),'seconds':time.perf_counter()-t})
    assert gate,'C224 reproduction failed: do not proceed with alternatives'
    print('PREFLIGHT PASSED',flush=True)

if __name__=='__main__':
    rows,plan=prepare()
    if '--only-reproduce' not in sys.argv: audit_data(rows)
    reproduce(rows,plan)

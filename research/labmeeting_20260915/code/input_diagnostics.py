from common import *
from matplotlib.patches import Rectangle
plt=configure_plot()

def box_transform(b,mode):
    g=geometry(b.width,b.height,mode);sx=g['nw']/b.width;sy=g['nh']/b.height
    xy=np.array([b.xmin*sx+g['left'],b.ymin*sy+g['top'],b.xmax*sx+g['left'],b.ymax*sy+g['top']])
    clipped=np.clip(xy,0,g['size']);iw=max(0,clipped[2]-clipped[0]);ih=max(0,clipped[3]-clipped[1])
    area=(b.xmax-b.xmin)*sx*(b.ymax-b.ymin)*sy
    return g,clipped,dict(retention=iw*ih/area,visible_width_px=iw,visible_height_px=ih,visible_min_side_px=min(iw,ih),visible_sqrt_area_px=np.sqrt(iw*ih),uncropped_sqrt_area_px=np.sqrt(area),scale_x=sx,scale_y=sy)

def main():
    seal=json.loads((OUT/'checks/selection_seal.json').read_text())
    for rel,h in seal['files'].items():assert sha(OUT/rel)==h
    records=pd.read_csv(OUT/'tables/verified_pool_records.csv');boxes=pd.read_csv(OUT/'tables/verified_boxes.csv')
    plan=json.loads((OUT/'checks/analysis_plan_frozen.json').read_text());rows=[]
    for b in boxes.itertuples():
        for mode in ['C224','F224','F518']:
            g,clipped,metrics=box_transform(b,mode)
            rows.append({'dataset':b.dataset,'sample_id':b.sample_id,'class_name':b.class_name,'mode':mode,**metrics})
    geom=pd.DataFrame(rows);savecsv('box_geometry_all_modes.csv',geom)
    imagegeom=geom.groupby(['dataset','sample_id','mode']).agg(min_retention=('retention','min'),median_retention=('retention','median'),mean_retention=('retention','mean'),any_below_half=('retention',lambda a:bool((a<.5).any())),median_visible_min_side_px=('visible_min_side_px','median'),median_visible_sqrt_area_px=('visible_sqrt_area_px','median')).reset_index()
    savecsv('image_geometry_all_modes.csv',imagegeom)
    summaries=[]
    for (ds,mode),g in geom.groupby(['dataset','mode']):
        im=imagegeom[(imagegeom.dataset==ds)&(imagegeom['mode']==mode)]
        summaries.append({'dataset':ds,'mode':mode,'boxes':len(g),'mean_box_retention':g.retention.mean(),'median_box_retention':g.retention.median(),'box_zero_percent':(g.retention<=1e-12).mean()*100,'box_below_half_percent':(g.retention<.5).mean()*100,'box_below_90_percent':(g.retention<.9).mean()*100,'image_any_box_below_half_percent':im.any_below_half.mean()*100,'median_visible_min_side_px':g.visible_min_side_px.median(),'median_visible_sqrt_area_px':g.visible_sqrt_area_px.median(),'median_uncropped_sqrt_area_px':g.uncropped_sqrt_area_px.median()})
    summary=pd.DataFrame(summaries);savecsv('input_geometry_summary.csv',summary)
    classgeometry=geom.groupby(['dataset','mode','class_name']).agg(n_boxes=('retention','count'),mean_retention=('retention','mean'),below_half_percent=('retention',lambda a:(a<.5).mean()*100),median_min_side_px=('visible_min_side_px','median')).reset_index();savecsv('class_geometry_summary.csv',classgeometry)
    shiftrows=[];shiftsummary=[]
    for ds in DATASETS:
        p=OUT/'features'/ds;ids=np.load(p/'sample_ids.npy',allow_pickle=False).astype(str)
        z={m:np.load(p/f'{m}.npy').astype('float64') for m in ['C224','F224','F518']}
        for m in z:z[m]/=np.linalg.norm(z[m],axis=1,keepdims=True)
        nn={}
        for m in z:
            sim=z[m]@z[m].T;np.fill_diagonal(sim,-np.inf);nn[m]=sim.argmax(axis=1)
        retention=imagegeom[(imagegeom.dataset==ds)&(imagegeom['mode']=='C224')].set_index('sample_id').loc[ids,'min_retention'].to_numpy()
        for mode in ['F224','F518']:
            shifts=np.clip(1-np.sum(z['C224']*z[mode],axis=1),0,2)
            for i,sid in enumerate(ids):shiftrows.append({'dataset':ds,'sample_id':sid,'mode':mode,'cosine_shift_from_C224':shifts[i],'C224_min_box_retention':retention[i],'nearest_id_changed':bool(nn['C224'][i]!=nn[mode][i])})
            shiftsummary.append({'dataset':ds,'mode':mode,'mean_cosine_shift':shifts.mean(),'median_cosine_shift':np.median(shifts),'p95_cosine_shift':np.quantile(shifts,.95),'nearest_changed_percent':np.mean(nn['C224']!=nn[mode])*100,'spearman_crop_loss_vs_shift':pd.Series(1-retention).corr(pd.Series(shifts),method='spearman')})
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for mode,col in [('C224','#637485'),('F224','#007D97'),('F518','#B47029')]:
            g=geom[(geom.dataset==ds)&(geom['mode']==mode)]
            axes[0].plot(np.sort(g.retention),np.arange(1,len(g)+1)/len(g),label=mode,color=col)
            axes[1].plot(np.sort(g.visible_min_side_px),np.arange(1,len(g)+1)/len(g),label=mode,color=col)
        for mode,col in [('F224','#007D97'),('F518','#B47029')]:
            g=pd.DataFrame(shiftrows);values=g[(g.dataset==ds)&(g['mode']==mode)].cosine_shift_from_C224
            axes[2].hist(values,bins=35,density=True,histtype='step',lw=2,label=mode,color=col)
        axes[0].set(title='실제 박스 면적 보존',xlabel='남은 박스 면적 / 원래 박스 면적',ylabel='누적 비율')
        axes[1].set(title='입력상 보이는 박스의 짧은 변',xlabel='입력 픽셀',ylabel='누적 비율')
        axes[2].set(title='동일 이미지의 특징 변화',xlabel='C224와의 cosine 거리',ylabel='확률 밀도')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{ds} 입력 구성 진단 · 기존 분할의 탐색 결과',fontsize=14)
        fig.text(.5,-.02,'박스는 결함 픽셀의 대리 지표 · 시야·크기·여백이 함께 달라짐 · 탐지 성능 개선을 측정한 결과가 아님',ha='center',fontsize=9)
        fig.tight_layout();savefig(fig,f'{ds}_입력진단_분포');plt.close(fig)
        lookup=records.set_index('sample_id')
        chosen=plan['datasets'][ds]['representative_ids']
        for page in [0,1]:
            fig,axes=plt.subplots(3,4,figsize=(15,10))
            for row,sid in enumerate(chosen[page*3:page*3+3]):
                r=lookup.loc[sid]
                with Image.open(DATA/r.relative_image_path) as im:original=im.convert('RGB')
                bs=boxes[boxes.sample_id==sid]
                for col,mode in enumerate(['Original','C224','F224','F518']):
                    im=original if mode=='Original' else render_input(original,mode);ax=axes[row,col];ax.imshow(im)
                    vals=[]
                    for b in bs.itertuples():
                        if mode=='Original':xy=[b.xmin,b.ymin,b.xmax,b.ymax]
                        else:
                            _,xy,m=box_transform(b,mode);vals.append(m['retention'])
                        if xy[2]>xy[0] and xy[3]>xy[1]:ax.add_patch(Rectangle(xy[:2],xy[2]-xy[0],xy[3]-xy[1],fill=False,color='#26EF87',lw=1.1))
                    subtitle=f'{sid[:8]} · {im.width}×{im.height}'
                    if vals:subtitle+=f'\n최저 박스 보존 {min(vals)*100:.1f}%'
                    ax.set_title(mode+'\n'+subtitle,fontsize=9);ax.axis('off')
            fig.suptitle(f'{ds} 사전 고정 ID 입력 비교 {page+1}/2\n기존 분할의 탐색 결과 · 초록 박스는 사후 GT',fontsize=14)
            fig.text(.5,.005,'패널은 지면 크기에 맞춰 표시했습니다. 실제 입력 해상도는 각 제목의 픽셀 수를 참고하세요.',ha='center',fontsize=9)
            fig.tight_layout(rect=[0,0,1,.94]);savefig(fig,f'{ds}_실제입력비교_{page+1}');plt.close(fig)
    savecsv('image_feature_shifts.csv',shiftrows);savecsv('feature_shift_summary.csv',shiftsummary)
    # Only now join class/geometry information onto the sealed selections.
    selection=pd.read_csv(OUT/'tables/R1_selection_order_GT_free.csv');joined=selection.merge(records[['dataset','sample_id','labels','n_boxes']],on=['dataset','sample_id'],validate='many_to_one')
    savecsv('R1_selection_posthoc_GT.csv',joined)
    overlap=[];changes=[];classcounts=[];selectedgeom=[]
    for (ds,seed),g in selection.groupby(['dataset','seed']):
        sets={m:set(g[g['mode']==m].sample_id) for m in ['Random','C224','F224','F518']}
        for method in ['Random','F224','F518']:
            a,b=sets['C224'],sets[method];overlap.append({'dataset':ds,'seed':seed,'comparison':method+' vs C224','shared_count':len(a&b),'query_size':len(a),'shared_percent':100*len(a&b)/len(a),'jaccard':len(a&b)/len(a|b)})
            for status,values in [('new_vs_C224',b-a),('removed_vs_C224',a-b)]:
                for sid in sorted(values):changes.append({'dataset':ds,'seed':seed,'method':method,'change':status,'sample_id':sid})
        for method,chosen in sets.items():
            initial=json.loads((OUT/'selections'/ds/str(seed)/'R0.json').read_text())['ids']
            for stage,picks in [('query',chosen),('cumulative',set(initial)|chosen)]:
                p=records[(records.dataset==ds)&records.sample_id.isin(picks)]
                for cls in sorted(set(c for v in records[records.dataset==ds].labels for c in v.split('|'))):
                    count=sum(cls in x.split('|') for x in p.labels)
                    classcounts.append({'dataset':ds,'seed':seed,'mode':method,'stage':stage,'class_name':cls,'n_images_with_class':count,'n_selected_images':len(p),'percent_images_with_class':count/len(p)*100})
            for inputmode in ['C224','F224','F518']:
                p=imagegeom[(imagegeom.dataset==ds)&(imagegeom['mode']==inputmode)&imagegeom.sample_id.isin(chosen)]
                selectedgeom.append({'dataset':ds,'seed':seed,'selection_method':method,'measured_input':inputmode,'selected_image_count':len(p),'mean_min_retention':p.min_retention.mean(),'any_box_below_half_percent':p.any_below_half.mean()*100,'median_min_side_px':p.median_visible_min_side_px.median()})
    savecsv('R1_overlap_per_seed.csv',overlap);savecsv('R1_changed_ids.csv',changes);savecsv('R1_class_composition.csv',classcounts);savecsv('R1_selected_geometry.csv',selectedgeom)
    ov=pd.DataFrame(overlap);ovsummary=ov.groupby(['dataset','comparison']).agg(shared_percent_mean=('shared_percent','mean'),shared_percent_min=('shared_percent','min'),shared_percent_max=('shared_percent','max'),jaccard_mean=('jaccard','mean')).reset_index();savecsv('R1_overlap_summary.csv',ovsummary)
    for ds in DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for i,method in enumerate(['Random','F224','F518']):
            g=ov[(ov.dataset==ds)&(ov.comparison==method+' vs C224')];axes[0].scatter(np.full(5,i)+np.linspace(-.1,.1,5),g.shared_percent,s=45);axes[0].hlines(g.shared_percent.mean(),i-.2,i+.2,color='black')
        axes[0].set_xticks(range(3),['Random','F224','F518']);axes[0].set(title='C224와 겹치는 R1 추가 선택',ylabel='공통 선택 수 / 추가 예산 (%)',ylim=(0,100));axes[0].grid(axis='y',alpha=.2)
        cc=pd.DataFrame(classcounts);cc=cc[(cc.dataset==ds)&(cc.stage=='query')]
        pivot=cc.groupby(['mode','class_name']).percent_images_with_class.mean().unstack().reindex(['Random','C224','F224','F518'])
        im=axes[1].imshow(pivot,cmap='Blues',vmin=0,vmax=max(50,pivot.to_numpy().max()),aspect='auto')
        axes[1].set_xticks(range(len(pivot.columns)),pivot.columns,rotation=40,ha='right',fontsize=8);axes[1].set_yticks(range(4),pivot.index);axes[1].set_title('선택 후 결합한 클래스별 이미지 비율 (%)')
        for i in range(4):
            for j in range(len(pivot.columns)):axes[1].text(j,i,f'{pivot.iloc[i,j]:.0f}',ha='center',va='center',fontsize=8)
        fig.colorbar(im,ax=axes[1],shrink=.8)
        fig.suptitle(f'{ds} 5개 공통 초기집합의 R1 선택 · 기존 분할의 탐색 결과')
        fig.text(.5,-.03,'왼쪽 점 1개 = 선택 시드 1개 · 오른쪽 = 5개 시드 평균, 다중 라벨 합계는 100%를 넘을 수 있음 · 학습 성능 아님',ha='center',fontsize=9)
        fig.tight_layout();savefig(fig,f'{ds}_R1선택비교');plt.close(fig)
    dump(OUT/'checks/diagnostics_completed.json',{'geometry_summary':summaries,'feature_shift_summary':shiftsummary,'R1_overlap_summary':ovsummary.to_dict('records'),'GT_join_after_all_selection_seals':True,'no_detector_training':True})
    print(summary.to_string(index=False),flush=True);print(ovsummary.to_string(index=False),flush=True)

if __name__=='__main__':main()

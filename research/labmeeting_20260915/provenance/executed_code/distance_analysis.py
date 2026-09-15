from common import *
plt=configure_plot()
from matplotlib.patches import Rectangle

def show_image(ax,sid,records,boxes,title):
    r=records.loc[sid]
    with Image.open(DATA/r.relative_image_path) as im: ax.imshow(im.convert('RGB'))
    for b in boxes[boxes.sample_id==sid].itertuples():ax.add_patch(Rectangle((b.xmin,b.ymin),b.xmax-b.xmin,b.ymax-b.ymin,fill=False,color='#25ED85',lw=1.2))
    ax.set_title(title+'\n'+sid[:8]+' · '+str(r.labels).replace('|',' / '),fontsize=9,wrap=True);ax.axis('off')

def main():
    assert json.loads((OUT/'checks/reproduction_gate.json').read_text())['passed']
    records=pd.read_csv(OUT/'tables/verified_pool_records.csv').set_index('sample_id')
    boxes=pd.read_csv(OUT/'tables/verified_boxes.csv');plan=json.loads((OUT/'checks/analysis_plan_frozen.json').read_text())
    summaries=[];pairrows=[];examples=[];classrows=[]
    for ds in DATASETS:
        ids,x,_=cache(ds); z=x.astype('float64');z/=np.linalg.norm(z,axis=1,keepdims=True)
        d=np.clip(1-z@z.T,0,2);np.fill_diagonal(d,np.inf)
        labs=[set(records.loc[sid,'labels'].split('|')) for sid in ids]
        classes=sorted(set.union(*labs));membership=np.array([[c in ls for c in classes] for ls in labs],dtype=np.int32)
        common=(membership@membership.T)>0
        exact=np.array([[a==b for b in labs] for a in labs])
        nn=d.argmin(axis=1);ar=np.arange(len(ids)); rng=np.random.Generator(np.random.PCG64(20260915))
        rnd=rng.integers(0,len(ids)-1,size=len(ids));rnd=rnd+(rnd>=ar)
        a,b=np.triu_indices(len(ids),1);v=d[a,b];rel=common[a,b];eq=exact[a,b]
        np.savez_compressed(OUT/'tables'/f'{ds}_all_unordered_pairs.npz',sample_ids=ids,anchor_index=a.astype('int32'),partner_index=b.astype('int32'),cosine_distance=v.astype('float32'),share_any_label=rel,identical_label_set=eq)
        for kind,partners in [('nearest',nn),('random',rnd)]:
            for i,j in enumerate(partners):pairrows.append({'dataset':ds,'pair_kind':kind,'anchor_id':ids[i],'partner_id':ids[j],'cosine_distance':d[i,j],'share_any_label':bool(common[i,j]),'identical_label_set':bool(exact[i,j])})
        summary={'dataset':ds,'n_images':len(ids),'n_unordered_pairs':len(v),'nearest_share_percent':float(common[ar,nn].mean()*100),'random_one_partner_share_percent':float(common[ar,rnd].mean()*100),'all_nonself_pairs_share_percent':float(rel.mean()*100),'nearest_median_distance':float(np.median(d[ar,nn])),'random_median_distance':float(np.median(d[ar,rnd])),'shared_allpairs_median_distance':float(np.median(v[rel])),'nonshared_allpairs_median_distance':float(np.median(v[~rel])),'exact_labelset_allpairs_percent':float(eq.mean()*100)}
        summaries.append(summary)
        for c in classes:
            ix=np.array([c in ls for ls in labs]);classrows.append({'dataset':ds,'class_name':c,'anchor_images':int(ix.sum()),'nearest_share_any_label_percent':float(common[ar,nn][ix].mean()*100),'nearest_contains_anchor_class_percent':float(np.array([c in labs[j] for j in nn])[ix].mean()*100),'nearest_median_distance':float(np.median(d[ar,nn][ix]))})
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        bins=np.linspace(0,max(.8,float(v.max()))+1e-8,45)
        axes[0].hist(d[ar,nn],bins=bins,density=True,alpha=.65,label='자기 제외 최근접');axes[0].hist(d[ar,rnd],bins=bins,density=True,alpha=.45,label='앵커별 무작위 짝')
        axes[0].set_title('전체 앵커: 최근접 / 무작위')
        for mask,label in [(rel,'공통 라벨 있음'),(~rel,'공통 라벨 없음')]:axes[1].hist(v[mask],bins=bins,density=True,histtype='step',lw=2,label=f'{label} (n={mask.sum():,})')
        axes[1].set_title('모든 비중복 쌍: 공통 라벨')
        for mask,label in [(eq,'라벨 집합 완전 동일'),(rel&~eq,'일부만 공유'),(~rel,'공유 없음')]:
            if mask.any():axes[2].hist(v[mask],bins=bins,density=True,histtype='step',lw=1.8,label=f'{label} (n={mask.sum():,})')
        axes[2].set_title('다중 라벨 관계 구분')
        for ax in axes:ax.set(xlabel='cosine 거리 (작을수록 가까움)',ylabel='확률 밀도');ax.legend(fontsize=8);ax.grid(alpha=.15)
        fig.suptitle(f'{ds} 기존 동결 특징의 사후 거리 분석 · 기존 분할의 탐색 결과',fontsize=13)
        fig.text(.5,-.025,'GT는 관계 해석에만 사용 · 최근접은 최소 거리를 고른 결과 · 쌍들은 이미지를 공유하므로 독립 표본이 아님',ha='center',fontsize=9)
        fig.tight_layout();savefig(fig,f'{ds}_거리분포');plt.close(fig)
        frozen=plan['datasets'][ds]['representative_ids'];index={s:i for i,s in enumerate(ids)}
        fig,axes=plt.subplots(6,2,figsize=(12,16))
        for row,sid in enumerate(frozen):
            i=index[sid];j=int(nn[i]);dist=float(d[i,j]);share=bool(common[i,j])
            show_image(axes[row,0],sid,records,boxes,'사전 고정 앵커');show_image(axes[row,1],ids[j],records,boxes,f'최근접 · d={dist:.4f} · 공통 라벨 {"있음" if share else "없음"}')
            examples.append({'dataset':ds,'kind':'predeclared_representative','anchor_id':sid,'partner_id':ids[j],'distance':dist,'share_any_label':share})
        fig.suptitle(f'{ds} 사전 고정 ID 6개와 최근접 이웃\n기존 분할의 탐색 결과 · 초록 박스는 사후 GT · 결과를 보고 교체하지 않음',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.96]);savefig(fig,f'{ds}_사전고정_실제사례');plt.close(fig)
        # Posthoc diagnostic examples deliberately expose counterexamples; not representative sampling.
        non=np.flatnonzero(~rel);shared=np.flatnonzero(rel)
        chosen=[('가까운데 공통 라벨 없음',non[np.argmin(v[non])]),('멀지만 공통 라벨 있음',shared[np.argmax(v[shared])])]
        fig,axes=plt.subplots(2,2,figsize=(12,6.7))
        for row,(title,k) in enumerate(chosen):
            i,j=int(a[k]),int(b[k]);show_image(axes[row,0],ids[i],records,boxes,title);show_image(axes[row,1],ids[j],records,boxes,f'd={v[k]:.4f}')
            examples.append({'dataset':ds,'kind':'posthoc_extreme_counterexample','rule':title,'anchor_id':ids[i],'partner_id':ids[j],'distance':v[k],'share_any_label':bool(rel[k])})
        fig.suptitle(f'{ds} 사후 진단 반례 · 극단값 규칙으로 선정\n대표 비율을 뜻하지 않음 · 기존 분할의 탐색 결과',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.92]);savefig(fig,f'{ds}_진단반례');plt.close(fig)
        if ds=='GC10-DET':
            selected=[]
            for prefix in PREFIXES:
                matching=[sid for sid in ids if sid.startswith(prefix)];assert len(matching)==1;selected.append(matching[0])
            ix=[index[s] for s in selected];matrix=d[np.ix_(ix,ix)].copy();np.fill_diagonal(matrix,0)
            pd.DataFrame(matrix,index=selected,columns=selected).to_csv(OUT/'tables/slide_four_image_distance.csv',encoding='utf-8-sig')
            fig=plt.figure(figsize=(12,9));gs=fig.add_gridspec(2,4,height_ratios=[1,2])
            for k,sid in enumerate(selected):show_image(fig.add_subplot(gs[0,k]),sid,records,boxes,f'발표 이미지 {k+1}')
            ax=fig.add_subplot(gs[1,:]);im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(.1,matrix.max()))
            tick=[f'{i+1}: {s[:8]}' for i,s in enumerate(selected)];ax.set_xticks(range(4),tick);ax.set_yticks(range(4),tick)
            for i in range(4):
                for j in range(4):ax.text(j,i,f'{matrix[i,j]:.4f}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.65 else 'black')
            fig.colorbar(im,ax=ax,label='cosine 거리',shrink=.8);ax.set_title('동일한 네 이미지의 모든 교차 거리')
            fig.suptitle('발표 사례의 교차 비교 · 기존 분할의 탐색 결과',fontsize=15)
            fig.tight_layout();savefig(fig,'발표_네이미지_교차거리');plt.close(fig)
        print(ds,summary,flush=True)
    savecsv('nearest_and_random_pairs.csv',pairrows);savecsv('distance_summary.csv',summaries);savecsv('fixed_and_diagnostic_example_ids.csv',examples);savecsv('nearest_classwise.csv',classrows)
    dump(OUT/'checks/distance_analysis.json',{'passed':True,'all_acquisition_images_included':True,'summary':summaries,'GT_use':'posthoc relationships and boxes only; not passed to R1 selector','random_label_shuffle_vs_random_partner':'random partner is a new fixed draw; do not call it the historical label-shuffle baseline'})

if __name__=='__main__':main()

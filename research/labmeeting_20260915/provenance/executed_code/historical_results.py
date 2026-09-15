from common import *
plt=configure_plot()
SOURCE=DEV/'results/original_split_exploratory_completion_20260913'

def main():
    historical=pd.read_csv(SOURCE/'historical_original_final_reuse_440.csv')
    entropy=pd.read_csv(SOURCE/'entropy_original_final_results.csv')
    full=pd.read_csv(SOURCE/'full_data_original_final_results.csv')
    observed_full=full[full.record_type=='OBSERVED_LEARNER']
    assert len(historical)==440 and len(entropy)==200 and len(observed_full)==4
    allcells=pd.concat([historical,entropy,observed_full],ignore_index=True)
    assert allcells.cell_id.nunique()==644
    source_checks=[]
    for r in historical.to_dict('records'):
        p=Path(r['source_path']); actual=sha(p); assert actual==r['source_sha256']
        source_checks.append({'cell_id':r['cell_id'],'source_sha256_matches':True})
    active=pd.concat([historical[historical['method']!='SHARED'],entropy],ignore_index=True)
    shared=historical[historical['method']=='SHARED']
    expanded=pd.concat([active,*[shared.assign(method=m) for m in ['Random','DINO','Entropy']]],ignore_index=True)
    keys=['dataset','method','seed','acquisition_seed','round','labeled_count']
    means=expanded.groupby(keys,as_index=False)[['map50','map50_95']].mean()
    assert len(means)==360
    original=pd.read_csv(SOURCE/'original_final_round_AB_means_360.csv')
    verify=means.merge(original,on=keys,suffixes=('_new','_source'),validate='one_to_one')
    discrepancy=max(float((verify[m+'_new']-verify[m+'_source']).abs().max()) for m in ['map50','map50_95'])
    assert discrepancy<1e-12
    for ds in DATASETS:
        means.loc[means.dataset==ds,'budget_fraction']=means.loc[means.dataset==ds,'labeled_count']/len(cache(ds)[0])
    means['analysis_scope']='기존 분할의 탐색 결과'
    savecsv('historical_AB_means_360.csv',means)
    summary=means.groupby(['dataset','method','round','labeled_count']).agg(map50_mean=('map50','mean'),map50_sd=('map50','std'),map50_95_mean=('map50_95','mean'),map50_95_sd=('map50_95','std'),n_seeds=('seed','count')).reset_index()
    savecsv('historical_absolute_curves.csv',summary)
    effects=[]; aucs=[]
    for (ds,method,seed),g in means.groupby(['dataset','method','seed']):
        g=g.sort_values('round'); assert g['round'].tolist()==list(range(6))
        x=g.budget_fraction.to_numpy()
        for m in ['map50','map50_95']:
            aucs.append({'dataset':ds,'method':method,'seed':seed,'metric':m,'nAULC':np.trapezoid(g[m],x)/(x[-1]-x[0]),'R5':g[m].iloc[-1]})
    auc=pd.DataFrame(aucs);savecsv('historical_naulc_both_metrics.csv',auc)
    for (ds,m),g in auc.groupby(['dataset','metric']):
        for endpoint in ['nAULC','R5']:
            piv=g.pivot(index='seed',columns='method',values=endpoint)
            for other in ['Random','Entropy']:
                for seed,value in (piv.DINO-piv[other]).items():effects.append({'dataset':ds,'metric':m,'endpoint':endpoint,'contrast':'DINO - '+other,'seed':seed,'delta_pp':value*100,'analysis_scope':'기존 분할의 탐색 결과'})
    effects=pd.DataFrame(effects);savecsv('appendix_paired_effects.csv',effects)
    effectsummary=effects.groupby(['dataset','metric','endpoint','contrast']).agg(mean_delta_pp=('delta_pp','mean'),sd_delta_pp=('delta_pp','std'),positive_seeds=('delta_pp',lambda a:int((a>0).sum())),n=('delta_pp','count')).reset_index()
    savecsv('appendix_paired_summary.csv',effectsummary)
    colors={'Random':'#687586','DINO':'#007D97','Entropy':'#BB6B16'}
    for ds in DATASETS:
        fig,axes=plt.subplots(1,2,figsize=(12,4.4))
        for ax,m,title in zip(axes,['map50','map50_95'],['mAP50','mAP50–95']):
            for method in colors:
                g=summary[(summary.dataset==ds)&(summary.method==method)]
                x=g.labeled_count.to_numpy();y=g[m+'_mean'].to_numpy()*100;sd=g[m+'_sd'].to_numpy()*100
                ax.plot(x,y,'o-',color=colors[method],label=method,lw=2,ms=4);ax.fill_between(x,y-sd,y+sd,color=colors[method],alpha=.12)
            ref=observed_full[observed_full.dataset==ds][m].mean()*100
            ax.axhline(ref,color='#824F85',ls='--',label=f'Full-data 100% 참조: {ref:.2f}')
            ax.set(title=title,xlabel='누적 라벨링 이미지 수',ylabel='성능 (%)',ylim=(0,100 if m=='map50' else 60))
            ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower right')
        fig.suptitle(f'{ds} 절대 성능 · 기존 분할의 탐색 결과',fontsize=14)
        fig.text(.5,-.015,'선: 선택 시드 10개의 A/B 평균을 다시 평균 · 음영: 시드 간 ±1 SD (신뢰구간 아님) · 정비된 분할에서 재검증 예정',ha='center',fontsize=9)
        fig.tight_layout();savefig(fig,f'본문_{ds}_절대성능');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,metric in zip(axes,['map50_95','map50']):
            for x,contrast in enumerate(['DINO - Random','DINO - Entropy']):
                g=effects[(effects.dataset==ds)&(effects.metric==metric)&(effects.endpoint=='nAULC')&(effects.contrast==contrast)].sort_values('seed')
                jitter=np.linspace(-.13,.13,len(g));ax.scatter(x+jitter,g.delta_pp,s=35,alpha=.8)
                ax.hlines(g.delta_pp.mean(),x-.22,x+.22,color='black',lw=2)
                ax.text(x,.96,f'평균 {g.delta_pp.mean():+.3f}%p',ha='center',va='top',fontsize=9,transform=ax.get_xaxis_transform())
            values=effects[(effects.dataset==ds)&(effects.metric==metric)&(effects.endpoint=='nAULC')].delta_pp
            low,high=values.min(),values.max();span=max(.1,high-low);ax.set_ylim(low-.08*span,high+.22*span)
            ax.axhline(0,color='gray',ls=':');ax.set_xticks([0,1],['DINO - Random','DINO - Entropy'])
            ax.set(title=('mAP50–95' if metric=='map50_95' else 'mAP50')+' nAULC',ylabel='동일 선택 시드의 차이 (%p)');ax.grid(axis='y',alpha=.2)
        fig.suptitle(f'{ds} 부록: paired 비교 · 기존 분할의 탐색 결과')
        fig.text(.5,-.02,'점 1개 = 선택 시드 1개 (A/B 평균 후 차이) · 가로선 = 10개 차이 평균 · 새로운 확정 검정 아님',ha='center',fontsize=9)
        fig.tight_layout();savefig(fig,f'부록_{ds}_paired');plt.close(fig)
    final=[]
    for (ds,method),g in means[means['round']==5].groupby(['dataset','method']):final.append({'dataset':ds,'method':method+' R5','map50_percent':g.map50.mean()*100,'map50_95_percent':g.map50_95.mean()*100,'analysis_scope':'기존 분할의 탐색 결과'})
    for ds,g in observed_full.groupby('dataset'):final.append({'dataset':ds,'method':'Full-data','map50_percent':g.map50.mean()*100,'map50_95_percent':g.map50_95.mean()*100,'analysis_scope':'기존 분할의 탐색 결과'})
    savecsv('historical_final_both_metrics.csv',final)
    dump(OUT/'checks/historical_audit.json',{'unique_training_cells_represented':644,'shared_R0':len(shared),'Random_R1_R5':200,'DINO_R1_R5':200,'Entropy_R1_R5':len(entropy),'Full_data':len(observed_full),'historical_raw_json_hashes_checked':len(source_checks),'AB360_max_reaggregation_error':discrepancy,'source_files':{str(p.relative_to(DEV)):sha(p) for p in [SOURCE/'historical_original_final_reuse_440.csv',SOURCE/'entropy_original_final_results.csv',SOURCE/'full_data_original_final_results.csv',SOURCE/'original_final_round_AB_means_360.csv']},'completed_epochs_note':'644 existing metric cells verified; no claim to have independently audited every epoch log in this run.'})
    print('Historical metrics verified and figures written',flush=True)

if __name__=='__main__': main()

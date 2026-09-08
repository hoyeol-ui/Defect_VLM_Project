"""Publish the verified, development-only descriptive artifacts."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    result=json.loads((ROOT/'interim_descriptive_results.json').read_text())
    audit=json.loads((ROOT/'post_run_audit.json').read_text())
    runtime=json.loads((ROOT/'runtime_and_remaining_work.json').read_text())
    after=json.loads((ROOT/'after_analysis_integrity.json').read_text())
    assert after['status']=='PASS' and after['registry_bytes_unchanged']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
    colors={'Entropy':'#2468b4','Random':'#7a818b','DINO':'#cb7623'}
    for ax,ds in zip(axes,['GC10-DET','NEU-DET']):
        for method in ['Random','DINO','Entropy']:
            rows=[r for r in result['curves'] if r['dataset']==ds and r['method']==method]
            assert [r['round'] for r in rows]==list(range(6))
            ax.plot([r['budget_fraction']*100 for r in rows],[r['AB_mean_map50_95'] for r in rows],
                color=colors[method],marker='o',markersize=5,linewidth=2 if method=='Entropy' else 1.5,label=method)
        ax.set_title(f'{ds} · a00');ax.set_xlabel('Labeled acquisition pool (%)')
        ax.set_xticks([10,15,20,25,30,35]);ax.set_ylim(0,1);ax.grid(alpha=.17);ax.legend(loc='upper left',frameon=False)
    axes[0].set_ylabel('Development mAP50–95 · A/B mean')
    fig.suptitle('INTERIM DESCRIPTIVE ONLY',fontsize=16,fontweight='bold',y=.99)
    fig.text(.5,.915,'NOT FINAL — 1/10 acquisition trajectories',ha='center',fontsize=12,color='#7b3030')
    fig.text(.5,.025,'Fixed protocol · Shared historical R0 · No confidence intervals or significance tests',ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=[0,.055,1,.90])
    fig.savefig(ROOT/'interim_development_curves.png',dpi=180)
    fig.savefig(ROOT/'interim_development_curves.svg');plt.close(fig)
    lines=[f'# POST-RUN AUDIT: {audit["status"]}', '', '**INTERIM DESCRIPTIVE ONLY**',
        '', '**NOT FINAL — 1/10 acquisition trajectories**', '',
        '완료된 official GC10 a00 / NEU a00 결과만 감사했다. 추가 학습, 통계적 유의성 검정, CI 계산, 최종 우위 주장, protocol 변경은 수행하지 않았다.', '',
        '| 감사 항목 | 결과 |','|---|---|',
        '| Training cells | 20/20 COMPLETE, 각각 attempt 1 |',
        '| Acquisition jobs | 10/10 COMPLETE, 각각 attempt 1 |',
        '| Round gates | 10/10 PASS |',
        f'| Historical 보호 | {audit["historical_protected_hashes_unchanged"]:,}개 보호 hash 동일; 440개 셀의 best/last/manifest 재검증 |',
        f'| Historical analysis 및 reference 보호 | {audit["historical_analysis_metadata_unchanged"]:,}개 파일 size/mtime 동일; locked-test 결과 내용은 열지 않음 |',
        '| Locked-test | 입력·동결 코드·authorization·실행 기록 기준 접근 0회. OS syscall 추적은 수행하지 않음 |',
        '| A/B manifests | 같은 round의 A/B 모두 동일 |',
        '| Policy dependency | 10개 acquisition 모두 이전 round A best.pt 사용. B policy edge 0 |',
        '| 중복·재선택·누수 | 모두 0 |',
        '| Freeze / authorization | 모든 결과에서 일치 |',
        '| Integration 재사용 | 없음 |',
        '| 분석 중 변경 | official registry byte hash 및 historical 보호 hash 유지 |', '',
        '경고는 Ultralytics의 기존 `half` 인자 deprecation뿐이다. FP32 scorer 설정은 변경하지 않았다. Foreground 종료 코드 15는 GC10 R1 A/B seal 후 사용자가 요청한 독립 runner 전환 기록이며 학습 실패가 아니다.', '',
        f'- Official freeze SHA-256: `{audit["official_freeze_sha256"]}`',
        f'- Authorization SHA-256: `{audit["authorization_sha256"]}`', '',
        '## GC10 interim curve', '', 'A/B 평균 development mAP50–95, 0–1 척도. Random/DINO도 같은 a00의 고정 development 평가값이다.', '']
    for ds in ['GC10-DET','NEU-DET']:
        if ds=='NEU-DET':lines+=['## NEU interim curve','']
        lines+=['| Round | Labeled n | Entropy | Random | DINO |','|---|---:|---:|---:|---:|']
        for rr in range(6):
            rows={r['method']:r for r in result['curves'] if r['dataset']==ds and r['round']==rr}
            lines.append(f'| R{rr} | {rows["Entropy"]["labeled_n"]} | '+ ' | '.join(f'{rows[m]["AB_mean_map50_95"]:.6f}' for m in ['Entropy','Random','DINO'])+' |')
        lines+=['']
    lines+=['![Interim development curves](interim_development_curves.png)', '', '## Provisional nAULC', '',
        'R0–R5의 실제 labeled fraction을 x축으로 사다리꼴 적분한 뒤 `(x_R5 − x_R0)`로 나눴다. A/B learner별 nAULC의 평균과 일치함을 독립 계산으로 확인했다.', '',
        '| Dataset | Entropy | Random (reference) | DINO (reference) |','|---|---:|---:|---:|']
    for ds in ['GC10-DET','NEU-DET']:
        values={r['method']:r['provisional_nAULC'] for r in result['nAULC'] if r['dataset']==ds}
        lines.append(f'| {ds} | '+' | '.join(f'{values[m]:.6f}' for m in ['Entropy','Random','DINO'])+' |')
    lines+=['', '이 표와 곡선은 동일 acquisition index 하나의 참고 기술통계다. 방법 간 우위 또는 일반화에 대한 결론을 내리지 않는다.', '',
        '## 평가값 출처', '',
        'Entropy R1–R5는 완료된 official validation-best checkpoint를 원래와 동일한 고정 development 이미지·라벨의 독립 복사본에서 재검증했다. imgsz=640, batch=8, conf=0.001, iou=0.7, max_det=300, split=val을 사용했다. Acquisition의 conf=0.25와 별개의 기존 validation 설정이다. R0 및 Random/DINO는 동일 development membership을 확인한 archived post-training best.pt 평가값이다. GC10 229장, NEU 180장이다.', '',
        '## 남은 official 작업과 시간', '',
        '- 남은 official training: **180 cells**. 남은 acquisition: **90 jobs**. 새 실행은 시작하지 않았다.',
        f'- 현재 RTX 3070 기준 잔여 중앙 계획치: **{runtime["remaining_hours_central"]:.2f}시간**.',
        f'- 운영 여유 범위: **{runtime["planning_hours_range"][0]:.1f}–{runtime["planning_hours_range"][1]:.1f}시간**. 통계적 CI가 아니다.',
        f'- 이번 원래 시작부터 완료까지: {runtime["total_elapsed_wall_seconds_including_handoff"]/3600:.3f}시간. Handoff 대기시간을 제외한 합산 active 실행 추정: {runtime["estimated_active_seconds"]/3600:.3f}시간.',
        f'- 관측 peak device VRAM: {runtime["sampled_peak_vram_mib"]:,} MiB. GPU 전체 메모리 표본값이다.', '',
        '각 dataset a00의 round별 실측 시간에 남은 trajectory 9개씩을 곱했다. GC10 R1 B의 process overhead와 수동 R1 gate 시간은 scheduler 인계로 별도 runtime 기록이 없어 다른 실행에서 관측한 median으로 보완했다. 학습·validation 자체 시간은 모두 실측이다. 이미지 구성, 시스템 부하, 온도에 따라 달라질 수 있다.', '',
        '| Dataset / round | Train + validation (min) | Acquisition (min) | Total active estimate (min) |',
        '|---|---:|---:|---:|']
    for row in runtime['rounds']:
        lines.append(f'| {row["dataset"]} R{row["round"]} | {row["train_and_validation_seconds"]/60:.2f} | {row["acquisition_seconds"]/60:.2f} | {row["estimated_active_wall_seconds"]/60:.2f} |')
    lines+=['', '## 재현 자료', '',
        '- `post_run_audit.json`: 각 셀·round gate와 전체 감사 근거.',
        '- `interim_raw_development.json`: 72개 learner 단위 development 값과 checkpoint/source hash.',
        '- `interim_descriptive_results.json`: 36개 평균 curve point, provisional nAULC 및 계산 정의.',
        '- `runtime_and_remaining_work.json`: round별 시간과 잔여 시간 산출 근거.',
        '- `after_analysis_integrity.json`: 분석 후 official registry 및 historical 보호 무결성.', '']
    (ROOT/'POST_RUN_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'report':str(ROOT/'POST_RUN_REPORT.md'),'figure':str(ROOT/'interim_development_curves.png')},ensure_ascii=False))


if __name__=='__main__':main()

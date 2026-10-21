"""Campaign order and elapsed-time estimates; no model or selector code."""
from pathlib import Path
import csv,json,datetime,statistics
P=Path(__file__).resolve().parent
KST=datetime.timezone(datetime.timedelta(hours=9))

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def kst():return datetime.datetime.now(KST).isoformat(timespec='seconds')
def cell_id(row):
    method='Random' if row['method']=='SHARED_R0' else row['method']
    return f"{row['dataset']}_{method}_pair{row['pair']}_R{row['round'] or 0}"
def load_matrix():
    return list(csv.DictReader((P/'execution_matrix.csv').open(encoding='utf-8-sig')))
def ordered_new(rows):
    lookup={(r['dataset'],int(r['pair']),r['method'],int(r['round'] or 0)):r for r in rows}
    result=[lookup['NEU-DET',1,'FullySupervised',0]]
    for pair in [1,2,3]:
        for ds in ['GC10-DET','NEU-DET']:result.append(lookup[ds,pair,'SHARED_R0',0])
    for pair in [2,3]:
        for ds in ['GC10-DET','NEU-DET']:result.append(lookup[ds,pair,'FullySupervised',0])
    for round_ in range(1,6):
        for pair in [1,2,3]:
            for ds in ['GC10-DET','NEU-DET']:
                for method in ['Random','DINO_F518','Entropy']:result.append(lookup[ds,pair,method,round_])
    assert len(result)==101 and len({cell_id(r) for r in result})==101
    assert {cell_id(r) for r in result}=={cell_id(r) for r in rows if r['status']=='NEW_TRAINING_REQUIRED'}
    return result
def prior_seconds(row):
    return (240.0468532600205+3.314152522531159*int(row['train_n']))*(1.033398184112206 if row['dataset']=='NEU-DET' else 1.)
def eta(ledger,current=None,elapsed_training=None,epoch=0):
    multipliers={}
    for ds in ['GC10-DET','NEU-DET']:
        complete=[r for r in ledger if r['dataset']==ds and r['state']=='SUCCEEDED']
        ratios=[float(r['training_seconds'])/prior_seconds(r) for r in complete if r['training_seconds']]
        multipliers[ds]=statistics.median(ratios) if ratios else 1.
    seconds=0
    for r in ledger:
        if r['state'] not in ['PENDING','RUNNING']:continue
        estimate=prior_seconds(r)*multipliers[r['dataset']]
        if current==r['cell_id'] and epoch>=5 and elapsed_training is not None:
            estimate=max(0.,elapsed_training/epoch*(100-epoch))
        seconds+=estimate
    return {'remaining_training_hours':seconds/3600,'dataset_measured_multipliers':multipliers,
        'measured_complete_cells':sum(r['state']=='SUCCEEDED' for r in ledger),
        'basis':'Completed-cell training seconds / prior model, median by dataset; active cell uses elapsed training per epoch after epoch5. Acquisitions and final test are separate.'}

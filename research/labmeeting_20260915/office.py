"""Portable office reader and explicit reanalysis launcher; never trains a detector."""
from pathlib import Path
import argparse,csv,json,hashlib,os,shutil,subprocess,sys
BASE=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    s.add_parser('summary');s.add_parser('verify')
    r=s.add_parser('run');r.add_argument('task',choices=['metrics','select','distance','diagnostics']);r.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='summary':
        for filename in ['historical_final_both_metrics.csv','input_geometry_summary.csv','R1_overlap_summary.csv']:
            print('\n'+filename)
            with (BASE/'tables'/filename).open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):print(' | '.join(f'{k}: {v}' for k,v in row.items()))
        print('\nExploratory results on the original split. No new detector training. See reports/03_거리와_입력진단_결과.md.')
    elif a.command=='verify':
        manifest=json.loads((BASE/'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
        failed=[name for name,h in manifest['files'].items() if not (BASE/name).is_file() or digest(BASE/name)!=h]
        if failed:raise SystemExit('FAILED: '+', '.join(failed[:10]))
        print('PASS:',len(manifest['files']),'files match the published manifest.')
    else:
        out=a.output.resolve()
        if out==BASE or BASE in out.parents:raise SystemExit('Use an output folder outside the published package.')
        if out.exists() and any(out.iterdir()):raise SystemExit('Output folder must be new or empty.')
        repo=BASE.parents[1]
        private=Path(os.environ.get('VLM_INPUT_ROOT',str(repo.parent/'private_inputs')))
        if a.task in ['distance','diagnostics'] and not (private/'data').is_dir():
            raise SystemExit('Raw images required. Extract the full backup or set VLM_INPUT_ROOT to private_inputs.')
        out.mkdir(parents=True,exist_ok=True)
        # Only copy sealed inputs/results; all regenerated files go to the new folder.
        for sub in ['features','tables','checks','selections']:
            shutil.copytree(BASE/sub,out/sub,dirs_exist_ok=True)
        env=dict(os.environ,VLM_OUTPUT_ROOT=str(out),VLM_SOURCE_ROOT=str(repo),VLM_INPUT_ROOT=str(private))
        script={'metrics':'historical_results.py','select':'select_r1.py','distance':'distance_analysis.py','diagnostics':'input_diagnostics.py'}[a.task]
        subprocess.run([sys.executable,str(BASE/'code'/script)],env=env,check=True)
        print('New output:',out)

if __name__=='__main__':main()

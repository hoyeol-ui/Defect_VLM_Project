from common import *
import re,datetime,subprocess

def main():
    broken=[];link_count=0
    for path in [OUT/'00_먼저읽기.md',*(OUT/'reports').glob('*.md')]:
        text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:')):continue
            p=Path(target);p=p if p.is_absolute() else path.parent/p
            link_count+=1
            if not p.exists():broken.append({'document':str(path),'target':target})
    assert not broken,broken
    for ds in DATASETS:
        _,_,base=cache(ds)
        for label in ['A','B']:
            p=base.parent/f'cache_{label}'
            for name,h in json.loads((p/'metadata.json').read_text())['file_sha256'].items():assert sha(p/name)==h
    for p in (OUT/'figures').glob('*.png'):
        with Image.open(p) as im:im.verify()
    dump(OUT/'checks/implementation_notes.json',{
      'resolved_before_alternative_extraction':['GC10 nonnumeric class name uses the frozen folder-number fallback, logged explicitly','Manual PIL resizing differed slightly; changed to installed BitImageProcessor TorchvisionBackend uint8 bicubic+antialias. C224 pixel and feature parity then both exact'],
      'geometry_clarification':'Original XML continuous coordinates used in geometric diagnostics; legacy YOLO boundary clamp is separately counted. Original frozen plan preserved, not silently rewritten.',
      'random_order_clarification':'Frozen Random API returns sorted membership; raw RNG draw order supplied separately without modifying selection seals.',
      'figure_QA':'Checked absolute and paired plots, distance plots, fixed examples, input images and selection plots. Corrected mean-label placement, minus glyph fallback and histogram full-range coverage.',
      'executed_scope':'Existing metrics, frozen-cache analysis, C224 subset reproduction, F224/F518 feature extraction and R1 selection. No detector training.'})
    dump(OUT/'checks/delivery_inventory.json',{'completed_at_local':datetime.datetime.now().astimezone().isoformat(),'meeting_date':'2026-09-15','markdown_links_checked':link_count,'broken_links':broken,'png_figures':len(list((OUT/'figures').glob('*.png'))),'svg_figures':len(list((OUT/'figures').glob('*.svg'))),'csv_tables':len(list((OUT/'tables').glob('*.csv'))),'selection_json':len(list((OUT/'selections').rglob('*.json'))),'original_caches_preserved':True,'base_git_status':subprocess.check_output(['git','-C',str(DEV),'status','--short'],text=True)})
    files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='ARTIFACT_SHA256.json'}
    dump(OUT/'ARTIFACT_SHA256.json',files)
    print('DELIVERY COMPLETE:',len(files),'files;',link_count,'valid document links;',len(list((OUT/'figures').glob('*.png'))),'figures')

if __name__=='__main__':main()

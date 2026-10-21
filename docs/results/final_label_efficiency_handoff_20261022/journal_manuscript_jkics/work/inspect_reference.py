from pathlib import Path
import zipfile,xml.etree.ElementTree as ET,json,hashlib,sys
sys.stdout.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parent;reference=Path('C:/Users/user/Downloads/논문투고양식.docx')
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(reference) as z:
    parts={n:{'size':len(z.read(n)),'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist()}
    root=ET.fromstring(z.read('word/document.xml'))
    paragraphs=[]
    for i,p in enumerate(root.findall('.//w:body/w:p',ns)):
        paragraphs.append({'index':i,'text':''.join(n.text or '' for n in p.findall('.//w:t',ns)),
            'pPr':ET.tostring(p.find('w:pPr',ns),encoding='unicode') if p.find('w:pPr',ns) is not None else None,
            'runs':[{'text':''.join(n.text or '' for n in r.findall('.//w:t',ns)),
                     'rPr':ET.tostring(r.find('w:rPr',ns),encoding='unicode') if r.find('w:rPr',ns) is not None else None} for r in p.findall('w:r',ns)]})
    alltext=[]
    for name in z.namelist():
        if name.startswith('word/') and name.endswith('.xml'):
            try:rr=ET.fromstring(z.read(name))
            except ET.ParseError:continue
            tt=[''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in rr.findall('.//w:p',ns)]
            if tt:alltext.append({'part':name,'paragraphs':tt})
    report={'reference':str(reference),'sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'parts':parts,
        'paragraphs':paragraphs,'sections':[ET.tostring(s,encoding='unicode') for s in root.findall('.//w:sectPr',ns)],
        'tables':[ET.tostring(t,encoding='unicode') for t in root.findall('.//w:body/w:tbl',ns)],'all_part_text':alltext}
    (P/'template_structure.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (P/'template_document.xml').write_bytes(z.read('word/document.xml'))
    (P/'template_styles.xml').write_bytes(z.read('word/styles.xml'))
    print('REFERENCE',report['reference'],reference.stat().st_size,report['sha256'])
    print('SECTIONS',len(report['sections']),'TABLES',len(report['tables']),'PARAGRAPHS',len(paragraphs))
    for x in alltext:
        if x['part'] not in ['word/document.xml','word/styles.xml']:
            print('PART',x['part'],json.dumps(x['paragraphs'],ensure_ascii=False)[:1500])
    for r in paragraphs:print(r['index'],r['text'][:800])
    print('SECTION XML',report['sections'])

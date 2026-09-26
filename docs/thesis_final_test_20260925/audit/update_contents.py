import sys,json,re
from pathlib import Path
from pypdf import PdfReader
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from copy import deepcopy
W=Path(__file__).parent; root=W.parent.parent
p=root/'심사용논문_최종test반영본_20260925.docx'
pdf=PdfReader(next((W/sys.argv[1]).glob('*.pdf')))
entries=json.loads((W/'contents_entries.json').read_text(encoding='utf-8'))
def norm(s):return re.sub(r'\s+','',s).replace('−','-').replace('–','-')
texts=[x.extract_text() for x in pdf.pages]
# Body starts where the first page label is Arabic 1 and the introduction follows.
start=next(i for i,t in enumerate(texts) if re.match(r'^1\s',t) and 'Ⅰ' in t)
pages={}
for e in entries:
    key=norm(e['title'])
    found=[i for i,t in enumerate(texts) if i>=start and key in norm(t)]
    if not found:raise ValueError('Missing rendered target '+e['title'])
    i=found[0]; label=str(i-start+1);pages[e['anchor']]={'title':e['title'],'pdf_page':i+1,'page_label':label}
d=Document(p)
toc=[x for x in d.paragraphs if x._p.get(qn('w:rsidR'))=='00F1A100']
assert len(toc)==len(entries)
for para,e in zip(toc,entries):
    rpr=deepcopy(para.runs[0]._r.rPr) if para.runs and para.runs[0]._r.rPr is not None else None
    para.clear()
    h=OxmlElement('w:hyperlink');h.set(qn('w:anchor'),e['anchor']);h.set(qn('w:history'),'1')
    r=OxmlElement('w:r')
    if rpr is not None:r.append(deepcopy(rpr))
    t=OxmlElement('w:t');t.text=e['title'];r.append(t);h.append(r)
    rr=OxmlElement('w:r')
    if rpr is not None:rr.append(deepcopy(rpr))
    rr.append(OxmlElement('w:tab'));tt=OxmlElement('w:t');tt.text=pages[e['anchor']]['page_label'];rr.append(tt);h.append(rr);para._p.append(h)
d.save(p)
(W/'page_map.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated',len(pages),'linked entries; body starts PDF page',start+1)

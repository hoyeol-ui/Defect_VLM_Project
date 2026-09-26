from pathlib import Path
from copy import deepcopy
import csv, json, hashlib, statistics, re, sys
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'C:\Users\user\Desktop\vlm')
WORK=ROOT/'output/thesis_test_sync_20260925'
OUT=ROOT/'심사용논문_최종test반영본_20260925.docx'
SRC=next(p for p in ROOT.glob('*.docx') if '20260924' in p.name)
DATA=ROOT/'final_test_evaluation_20260925'
S=DATA/'statistics'
WORK.mkdir(parents=True,exist_ok=True)
def read(name):
    return list(csv.DictReader((S/name).open(encoding='utf-8-sig')))
budget=read('budget_summary.csv'); seeds=read('seed_nAULC.csv')
pairs=read('paired_comparisons.csv'); classes=read('R5_class_AP_by_seed.csv')
classdiff=read('R5_class_AP_difference.csv'); full=read('R5_vs_full_data.csv')
worst=read('worst_class_comparisons.csv'); devtest=read('development_vs_test.csv')
methods=['Random','DINO_F518','Entropy']; datasets=['GC10-DET','NEU-DET']
labels={'Random':'Random','DINO_F518':'DINOv2-FF','Entropy':'Entropy'}
def find(rows,**kw): return next(x for x in rows if all(x[k]==str(v) for k,v in kw.items()))
def f(x): return f'{float(x):.2f}'
def pc(x): return f(float(x)*100)
def b(ds,m,r): return find(budget,dataset=ds,method=m,round=r)
def bm(ds,m,r): return pc(b(ds,m,r)['mean_mAP50_95'])
def pr(ds,other='Random'): return find(pairs,dataset=ds,method_a='DINO_F518',method_b=other)
def diff(ds,other='Random'): return f(pr(ds,other)['paired_mean_difference_pp'])
def cl(ds,c): return find(classdiff,dataset=ds,class_name=c)
def cd(ds,c): return f(cl(ds,c)['dino_minus_random_mean_ap_pp'])
def fullval(ds): return pc(find(full,dataset=ds,method='DINO_F518')['full_data_mAP50_95'])
def gap(ds,m='DINO_F518'): return f(-float(find(full,dataset=ds,method=m)['r5_minus_full_pp']))
def na(ds,m):
    v=[float(x['nAULC']) for x in seeds if x['dataset']==ds and x['method']==m]
    return statistics.mean(v),statistics.stdev(v)
def nmean(ds,m): return pc(na(ds,m)[0])
G,N=datasets
d=Document(SRC); P=list(d.paragraphs); T=list(d.tables)
changes=[]
def settext(p,text):
    # Preserve paragraph properties and bookmarks; replace text runs only.
    rp=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    for child in list(p._p):
        if child.tag not in [qn('w:pPr'),qn('w:bookmarkStart'),qn('w:bookmarkEnd')]: p._p.remove(child)
    r=p.add_run(text)
    if rp is not None: r._r.insert(0,rp)
def edit(i,text):
    changes.append({'source_paragraph':i,'before':P[i].text,'after':text});settext(P[i],text)
def insert_before(anchor,text,template=140,style=None):
    el=deepcopy(P[template]._p)
    for x in el.xpath('.//w:bookmarkStart|.//w:bookmarkEnd'): x.getparent().remove(x)
    anchor.addprevious(el);p=Paragraph(el,d._body);settext(p,text)
    if style: p.style=style
    return p
def cell(t,i,j,text):
    c=t.cell(i,j);settext(c.paragraphs[0],str(text))
    for p in list(c.paragraphs[1:]): p._p.getparent().remove(p._p)
def newtable(anchor,headers,rows,template=5):
    el=deepcopy(T[template]._tbl)
    anchor.addprevious(el)
    from docx.table import Table
    t=Table(el,d._body)
    while len(t.rows)>1:t._tbl.remove(t.rows[-1]._tr)
    assert len(t.columns)==len(headers)
    for j,h in enumerate(headers):cell(t,0,j,h)
    proto=deepcopy(T[template].rows[1]._tr)
    for row in rows:
        t._tbl.append(deepcopy(proto))
        for j,v in enumerate(row):cell(t,len(t.rows)-1,j,v)
    return t

edit(1,'석사학위논문 심사용 수정본')
edit(10,'실험에는 금속 표면 결함 데이터셋인 GC10-DET와 NEU-DET를 사용하였다. 사전학습된 DINOv2-small의 이미지 특징과 코사인 거리를 이용하여 현재 라벨 집합에서 가장 먼 이미지를 순차적으로 추가하는 방법(DINOv2-FF)을 무작위 선택(Random) 및 탐지 결과의 클래스 점수 엔트로피를 이용한 선택(Entropy)과 비교하였다. 라벨 이미지는 학습셋의 약 10%에서 35%까지 다섯 단계에 걸쳐 늘렸으며, 각 단계에서 YOLOv8n을 동일한 사전학습 가중치에서 새로 학습하였다. 데이터셋별로 5개의 선택 시드와 하나의 고정 학습 시드를 사용하였다. 개발셋으로 선택한 총 162개 모델을 최종평가셋에서 평가하고, mAP50-95 학습곡선, 정규화 학습곡선 아래 면적(nAULC), 클래스별 AP 및 선택된 이미지의 클래스 구성을 분석하였다.')
edit(11,f'GC10-DET에서 DINOv2-FF의 평균 nAULC는 Random보다 {diff(G)}%p, Entropy보다 {diff(G,"Entropy")}%p 높았으며, 두 비교 모두 다섯 선택 시드에서 차이가 양수였다. NEU-DET에서는 Random 대비 평균 차이가 {diff(N)}%p로 작고 시드별 방향이 달랐으며, Entropy 대비 {diff(N,"Entropy")}%p의 양의 평균 차이는 다섯 시드에서 모두 관찰되었다. 네 주 비교 모두 Holm 보정 후 통계적 유의성은 확인되지 않았다. R5의 클래스별 AP 차이는 GC10-DET의 welding_line에서 +{cd(G,"welding_line")}%p, NEU-DET의 inclusion에서 {cd(N,"inclusion")}%p였으며, NEU-DET의 최저 클래스 nAULC는 DINOv2-FF가 Random보다 1.64%p 낮았다.')
edit(12,'개발셋에서 관찰된 네 주 비교의 평균 차이 방향은 최종평가셋에서도 유지되었다. 그러나 특징 기반 다양성 선택의 평균적 이점은 데이터셋에 따라 달랐고, 모든 결함 클래스와 최저 클래스 성능의 향상으로 이어지지는 않았다.')

# Methods: acquisition, model selection, and statistical design remain fixed.
edit(140,'학습 과정에서 저장할 모델은 개발셋의 성능을 기준으로 결정하였다. 최종평가에서는 이렇게 선택한 모델과 평가 조건 및 분석 방법을 고정한 뒤, 별도로 분리해 둔 최종평가셋(test)에서 성능을 측정하였다. 본 논문의 주 성능 결과와 통계 비교는 최종평가셋에 근거하며, 개발셋 결과는 관찰된 차이의 방향을 비교하는 보조 자료로 제시한다. 최종평가셋은 표본 선택, 학습 및 저장 모델 선택에 사용하지 않았다.')
edit(147,'학습셋은 표본 선택과 탐지 모델 학습에, 개발셋은 저장 모델 선택과 개발 과정의 성능 확인에 사용하였다. 최종평가셋은 모델과 분석 절차를 고정한 뒤 최종 성능 평가에 사용하였다. 평가 전 이미지 식별자, 파일 및 RGB 해시와 연결요소를 기준으로 분할 간 중복이 없음을 다시 확인하였다.')
edit(186,'학습 중에는 개발셋의 성능을 기준으로 저장 모델을 선택하였다. 선택된 모델의 개발셋 재평가와 최종평가셋 평가는 표 3-3의 동일한 평가 조건으로 수행하였다. Entropy 선택용 추론의 점수 기준(0.25)은 성능 평가의 점수 기준(0.001)과 별도로 설정하였다. 최종평가 전에 기존 모델 한 개로 개발셋 평가값의 일치를 확인하였으며, 최종평가 환경과 검증 범위는 부록 A에 제시하였다.')
cell(T[3],8,0,'개발셋 재평가 및 최종평가')
edit(193,'라벨 예산에 따른 성능 변화는 최종평가셋의 학습곡선으로 나타내었다. 가로축은 각 단계에서 사용한 실제 누적 라벨 이미지 수 또는 학습셋 대비 비율이며 세로축은 mAP50-95이다. 같은 방법과 라운드에서 다섯 선택 시드의 평균을 사용하고 표본 표준편차를 함께 제시하였다.')
insert_before(P[208]._p,'이상의 지표, 네 주 비교, 부트스트랩 횟수 및 검정 절차는 최종평가셋의 결과를 확인하기 전에 고정하였다. 부트스트랩 구간은 고정된 학습 시드와 데이터 분할에서 선택 시드에 따른 변동을 나타내며, 새로운 학습 시드나 평가 이미지 표본에 따른 불확실성을 모두 포함하지는 않는다.')

edit(210,'두 데이터셋에서 학습한 탐지 모델은 공통 R0 모델 10개, Random 50개, DINOv2-FF 50개, Entropy 50개, 전체 학습자료 참고 모델 2개로 총 162개이다. 최종평가에서는 기록된 SHA-256과 일치하는 162개 모델을 각각 한 번 평가하였다. 공통 R0는 세 방법이 같은 평가값을 공유하며, 기술 점검용 모델은 분석 대상에서 제외하였다.')
edit(211,'본 장에서는 GC10-DET 344장과 NEU-DET 270장으로 구성된 최종평가셋의 결과를 제시한다. 세 표본 선택 방법은 같은 선택 시드에서 동일한 R0로 시작하였으며, 이후 R1부터 R5까지 서로 다른 이미지를 추가하였다. 성능은 라벨 예산별 mAP50-95, nAULC, 클래스별 AP와 선택된 이미지의 클래스 구성을 중심으로 비교하였다. 개발셋과의 비교는 4.7절에서 별도로 제시한다.')
edit(213,f'GC10-DET의 R0 평균 mAP50-95는 {bm(G,"Random",0)}%였다. 이후 R1부터 R5까지 DINOv2-FF의 평균 성능은 모든 라벨 예산에서 Random과 Entropy보다 높게 나타났다.')
edit(214,f'R1에서 Random과 DINOv2-FF의 평균 성능은 각각 {bm(G,"Random",1)}%와 {bm(G,"DINO_F518",1)}%였으며, R2에서는 {bm(G,"Random",2)}%와 {bm(G,"DINO_F518",2)}%였다. R5에서는 Random {bm(G,"Random",5)}%, DINOv2-FF {bm(G,"DINO_F518",5)}%, Entropy {bm(G,"Entropy",5)}%로 나타났다. R5에서 DINOv2-FF와 Random의 평균 차이는 {f((float(b(G,"DINO_F518",5)["mean_mAP50_95"])-float(b(G,"Random",5)["mean_mAP50_95"]))*100)}%p였다.')
edit(216,'그림 4-1. 라벨 이미지 비율에 따른 최종평가셋 학습곡선. 왼쪽은 GC10-DET, 오른쪽은 NEU-DET이다. 값은 선택 시드 5개의 평균 ± 표본 표준편차이며, 점선은 전체 학습자료 참고 모델 1개의 성능이다.')
edit(218,f'NEU-DET에서는 세 방법의 평균 성능 차이가 GC10-DET보다 작게 나타났다. R0의 평균 mAP50-95는 {bm(N,"Random",0)}%였으며, R1에서는 Random {bm(N,"Random",1)}%, DINOv2-FF {bm(N,"DINO_F518",1)}%, Entropy {bm(N,"Entropy",1)}%였다.')
edit(219,f'DINOv2-FF는 R1, R2, R3 및 R5에서 세 방법 중 가장 높은 평균 성능을 보였다. R4에서는 Random {bm(N,"Random",4)}%, DINOv2-FF {bm(N,"DINO_F518",4)}%로 Random의 평균이 높았다. R5의 성능은 Random {bm(N,"Random",5)}%, DINOv2-FF {bm(N,"DINO_F518",5)}%, Entropy {bm(N,"Entropy",5)}%였다.')
edit(220,'따라서 NEU-DET에서는 DINOv2-FF가 여러 라벨 예산에서 높은 평균값을 보였지만, 모든 예산에서 일관되게 가장 높지는 않았다. 시드별 차이와 전체 예산 구간의 결과는 nAULC의 대응 비교로 확인하였다.')
edit(221,'표 4-1. 라벨 예산별 최종평가셋 mAP50-95')
edit(225,f'GC10-DET에서 Random의 평균 nAULC는 {nmean(G,"Random")}%, DINOv2-FF는 {nmean(G,"DINO_F518")}%, Entropy는 {nmean(G,"Entropy")}%였다. DINOv2-FF는 Random보다 평균 {diff(G)}%p, Entropy보다 {diff(G,"Entropy")}%p 높게 나타났다.')
edit(226,f'NEU-DET에서는 Random {nmean(N,"Random")}%, DINOv2-FF {nmean(N,"DINO_F518")}%, Entropy {nmean(N,"Entropy")}%였다. DINOv2-FF와 Random의 평균 차이는 {diff(N)}%p, Entropy와의 차이는 {diff(N,"Entropy")}%p였다.')
edit(227,'표 4-2. 최종평가셋 nAULC')
edit(232,f'NEU-DET에서 DINOv2-FF와 Random의 차이는 다섯 시드 중 세 개에서 양수였으며, 95% 구간은 0을 포함하였다. Entropy 대비 차이는 다섯 시드 모두에서 양수였고 95% 구간은 0보다 높았으나, 정확 양측 p값은 0.0625, Holm 보정 후 p값은 0.25였다. 네 주 비교 모두 유의수준 0.05에서 통계적 유의성은 확인되지 않았다.')
erG=find(pairs,dataset=G,method_a='Entropy',method_b='Random');erN=find(pairs,dataset=N,method_a='Entropy',method_b='Random')
insert_before(P[233]._p,f'보조 비교인 Entropy와 Random의 nAULC 차이는 GC10-DET에서 {f(erG["paired_mean_difference_pp"])}%p(95% 구간 {f(erG["bootstrap_ci_lower_pp"])}~{f(erG["bootstrap_ci_upper_pp"])}, 양수 시드 1/5, 정확 p=0.1250), NEU-DET에서 {f(erN["paired_mean_difference_pp"])}%p(95% 구간 {f(erN["bootstrap_ci_lower_pp"])}~{f(erN["bootstrap_ci_upper_pp"])}, 양수 시드 2/5, 정확 p=0.6250)였다. 이 보조 비교에는 주 비교의 Holm 보정을 적용하지 않았다.')
edit(234,f'학습셋 전체를 사용한 참고 모델의 최종평가셋 mAP50-95는 GC10-DET {fullval(G)}%, NEU-DET {fullval(N)}%였다.')
edit(235,f'약 35%의 라벨 이미지를 사용한 R5에서 GC10-DET의 평균 성능은 Random {bm(G,"Random",5)}%, DINOv2-FF {bm(G,"DINO_F518",5)}%, Entropy {bm(G,"Entropy",5)}%였다. NEU-DET에서는 각각 {bm(N,"Random",5)}%, {bm(N,"DINO_F518",5)}%, {bm(N,"Entropy",5)}%로 나타났다.')
edit(237,'최종평가셋 mAP50-95 (%). R5는 다섯 선택 시드의 평균이며 전체 자료 모델은 데이터셋별 1회 결과이다. 본문의 절대 차이는 반올림 전 수치에 근거한다.')
edit(238,f'GC10-DET에서 전체 학습자료 모델과 DINOv2-FF의 차이는 {gap(G)}%p였고 Random과의 차이는 {gap(G,"Random")}%p였다. NEU-DET에서 전체 학습자료 모델과 DINOv2-FF의 차이는 {gap(N)}%p였다. 전체 학습자료 결과는 단일 모델의 참고값이므로, 이를 성능의 상한이나 약 35% 라벨 사용의 동등성 근거로 해석하지 않았다.')
edit(241,'그림 4-2. 최종평가셋 R5에서 DINOv2-FF와 Random의 클래스별 AP50-95 차이. GC10-DET 10개와 NEU-DET 6개 클래스에 대한 선택 시드 5개의 평균 차이(%p)이다.')
edit(243,f'GC10-DET에서는 클래스에 따라 차이의 크기와 방향이 달랐다. DINOv2-FF와 Random의 평균 차이는 welding_line에서 +{cd(G,"welding_line")}%p로 가장 컸으며, punching_hole +{cd(G,"punching_hole")}%p, oil_spot +{cd(G,"oil_spot")}%p, silk_spot +{cd(G,"silk_spot")}%p 등이 양의 차이를 보였다. 반면 rolled_pit은 {cd(G,"rolled_pit")}%p, waist_folding은 {cd(G,"waist_folding")}%p, inclusion은 {cd(G,"inclusion")}%p였다(표 4-5).')
edit(245,f'NEU-DET에서는 scratches +{cd(N,"scratches")}%p, patches +{cd(N,"patches")}%p, pitted_surface +{cd(N,"pitted_surface")}%p로 DINOv2-FF의 평균 AP가 Random보다 높았다. 반면 crazing은 {cd(N,"crazing")}%p, inclusion은 {cd(N,"inclusion")}%p, rolled-in_scale은 {cd(N,"rolled-in_scale")}%p로 낮았다.')
wN=find(worst,dataset=N,method_a='DINO_F518',method_b='Random');wG=find(worst,dataset=G,method_a='DINO_F518',method_b='Random')
wNE=find(worst,dataset=N,method_a='DINO_F518',method_b='Entropy');wGE=find(worst,dataset=G,method_a='DINO_F518',method_b='Entropy')
edit(252,f'NEU-DET의 전체 클래스 결과와 선택 구성은 표 4-7, 표 4-8과 같다. 라운드별 최저 AP 클래스로 구성한 학습곡선의 nAULC는 NEU-DET에서 Random {pc(wN["mean_b"])}%, DINOv2-FF {pc(wN["mean_a"])}%, Entropy {pc(wNE["mean_b"])}%였다. DINOv2-FF와 Random의 대응 차이는 {f(wN["paired_mean_difference_pp"])}%p였으며 다섯 시드 중 네 개에서 음수였다. GC10-DET에서는 각각 {pc(wG["mean_b"])}%, {pc(wG["mean_a"])}%, {pc(wGE["mean_b"])}%였고, DINOv2-FF와 Random의 차이는 {f(wG["paired_mean_difference_pp"])}%p였다. 이는 각 모델·라운드에서 가장 낮은 클래스 AP를 취한 보조 지표로, 특정 클래스 하나의 고정된 성능 곡선과는 구별된다.')
for idx in [248,255]:edit(idx,'최종평가셋 AP50-95의 평균 ± 표본 표준편차(%), n=5. 마지막 열은 DINOv2-FF와 Random의 평균 차이(%p)이다. 차이는 반올림 전 값으로 계산하였다.')

# Update existing numeric tables directly from the machine-readable ledger.
for di,ds in enumerate(datasets):
    for r in range(6):
        for mi,m in enumerate(methods):
            row=b(ds,m,r);cell(T[4],1+di*6+r,3+mi,f'{pc(row["mean_mAP50_95"])} ± {pc(row["sample_sd_mAP50_95"])}')
    for mi,m in enumerate(methods):
        mu,sd=na(ds,m);cell(T[5],di+1,mi+1,f'{pc(mu)} ± {pc(sd)}')
    cell(T[7],di+1,1,fullval(ds))
    for mi,m in enumerate(methods):cell(T[7],di+1,mi+2,bm(ds,m,5))
    for oi,other in enumerate(['Random','Entropy']):
        row=pr(ds,other);ri=1+di*2+oi
        vals=[f(row['paired_mean_difference_pp']),f'{f(row["bootstrap_ci_lower_pp"])} ~ {f(row["bootstrap_ci_upper_pp"])}',f'{row["positive_seeds"]}/5',f'{float(row["exact_two_sided_sign_flip_p"]):.4f}',f'{float(row["holm_adjusted_p"]):.4f}']
        for j,v in enumerate(vals,1):cell(T[6],ri,j,v)
    table=T[8 if di==0 else 10]
    for ri in range(1,len(table.rows)):
        cname=table.cell(ri,0).text
        for mi,m in enumerate(methods):
            vals=[float(x['R5_AP50_95']) for x in classes if x['dataset']==ds and x['method']==m and x['class_name']==cname]
            assert len(vals)==5
            cell(table,ri,mi+1,f'{pc(statistics.mean(vals))} ± {pc(statistics.stdev(vals))}')
        cell(table,ri,4,cd(ds,cname))

# Short development/test comparison: four prespecified primary contrasts only.
comparison_heading=insert_before(P[259]._p,'4.7 개발셋과 최종평가셋의 비교',template=209,style='Heading 2')
comparison_heading.paragraph_format.page_break_before=True
insert_before(P[259]._p,'표 4-9. 개발셋과 최종평가셋의 대응 nAULC 차이',template=227)
dtrows=[]
for ds in datasets:
    for other in ['Random','Entropy']:
        row=find(devtest,dataset=ds,comparison=f'DINOv2-FF - {other}')
        dtrows.append([ds.replace('-DET',''),f'DINOv2-FF−{other}',f(row['dev_paired_mean_difference_pp']),f(row['test_paired_mean_difference_pp'])])
comparison_table=newtable(P[259]._p,['자료','비교','개발셋','최종평가셋'],dtrows)
comparison_table.autofit=False
for col,width in zip(comparison_table.columns,[.75,2.85,1.4,1.4]):col.width=Inches(width)
for row in comparison_table.rows:
    for c,width in zip(row.cells,[.75,2.85,1.4,1.4]):c.width=Inches(width)
insert_before(P[259]._p,'단위는 %p이다. 같은 학습 모델을 두 평가셋에서 평가한 대응 차이의 평균이며, 별도의 재학습 반복실험을 의미하지 않는다.',template=228)
insert_before(P[259]._p,'네 주 비교에서 개발셋의 양의 평균 차이 방향은 최종평가셋에서도 유지되었다. GC10-DET에서는 DINOv2-FF가 Random과 Entropy보다 높은 평균 nAULC를 보였고, NEU-DET에서는 Random 대비 차이가 작았다. NEU-DET의 Entropy 대비 차이는 최종평가셋에서 다섯 선택 시드 모두 양수였으나, 네 주 비교 모두 Holm 보정 후 유의하지 않았다.')
insert_before(P[259]._p,f'평균 차이의 방향이 같더라도 세부 결과가 모두 일치한 것은 아니다. NEU-DET에서 DINOv2-FF가 가장 높은 평균 성능을 보인 라운드는 개발셋의 R2·R3에서 최종평가셋의 R1·R2·R3·R5로 달라졌다. GC10-DET의 rolled_pit과 silk_spot, NEU-DET의 inclusion에서는 DINOv2-FF와 Random의 R5 AP 차이 방향이 바뀌었다. 따라서 개발셋의 경향이 최종평가셋에서 유지되었다는 해석은 주 비교의 평균 차이와 클래스별 결과를 구분하여 적용해야 한다.')

edit(261,'최종평가셋에서도 DINOv2-FF의 평균적 이점은 데이터셋에 따라 달랐다. GC10-DET에서는 R1부터 R5까지 모든 라벨 예산에서 DINOv2-FF의 평균이 가장 높았으며, 개발셋에서 관찰된 방향이 유지되었다. NEU-DET에서는 Random 대비 평균 nAULC 차이가 작고 시드별 방향이 달랐다.')
edit(262,'이러한 결과는 특징 기반 다양성 선택의 효과를 데이터셋의 특성과 함께 해석할 필요가 있음을 보여준다. DINOv2-FF는 현재 선택된 이미지와 특징 공간에서 멀리 떨어진 표본을 추가한다. 결함의 형태와 크기, 배경, 같은 클래스 안의 시각적 변화 등이 선택 결과에 영향을 줄 수 있으나, 본 실험은 이들 요인의 개별 효과를 분리하여 검증하지는 않았다.')
edit(264,f'NEU-DET에서는 DINOv2-FF가 R1·R2·R3·R5에서 가장 높은 평균값을 보였지만 R4에서는 Random이 높았다. 전체 구간에서 Random 대비 평균 차이는 {diff(N)}%p이고 양수 시드는 3/5였으므로 일관된 우위를 주장하기 어렵다. Entropy 대비 평균 차이는 {diff(N,"Entropy")}%p이며 다섯 시드에서 모두 양수였다. 다만 이 비교 역시 Holm 보정 후 유의하지 않아, 차이의 크기와 방향을 중심으로 해석하였다.')
edit(266,f'전체 평균 성능의 차이는 모든 결함 클래스에서 같은 방향으로 나타나지 않았다. GC10-DET에서 DINOv2-FF는 Random보다 welding_line의 AP50-95가 {cd(G,"welding_line")}%p 높았지만 rolled_pit은 0.96%p 낮았다. NEU-DET에서는 scratches가 {cd(N,"scratches")}%p 높았던 반면 inclusion은 1.37%p, crazing은 1.94%p 낮았다.')
edit(269,f'일부 클래스에서는 선택된 이미지 수와 AP의 변화 방향이 일치하였다. welding_line과 scratches는 DINOv2-FF에서 선택 수와 AP가 모두 증가하였고, rolled-in_scale과 waist_folding은 모두 감소하였다. 반면 NEU-DET의 inclusion은 선택 수가 92.8장에서 116.2장으로 증가했으나 AP는 1.37%p 낮았다. GC10-DET의 silk_spot은 선택 수가 180.2장에서 98.0장으로 감소했으나 AP는 {cd(G,"silk_spot")}%p 높았다. 따라서 선택 수만으로 AP 변화를 설명하기 어렵다. 선택된 이미지의 형태와 난이도 등이 함께 작용했을 가능성은 있으나, 이를 인과적으로 확인한 결과는 아니다.')
insert_before(P[270]._p,'평균 성능과 취약 클래스 성능의 차이도 관찰되었다. NEU-DET에서 DINOv2-FF의 평균 nAULC는 두 비교 방법보다 높았지만, 최저 클래스 nAULC는 Random보다 1.64%p 낮았고 다섯 시드 중 네 개에서 같은 방향이었다. GC10-DET에서도 최저 클래스 nAULC는 Entropy의 평균이 가장 높았다. 이는 전체 평균의 이점을 각 클래스의 균등한 향상으로 해석할 수 없음을 보여준다.')
edit(271,f'본 연구의 Entropy 선택은 두 데이터셋에서 Random보다 낮은 평균 nAULC를 보였다. GC10-DET에서 Entropy는 {nmean(G,"Entropy")}%, Random은 {nmean(G,"Random")}%였으며, NEU-DET에서는 각각 {nmean(N,"Entropy")}%, {nmean(N,"Random")}%였다. 보조 비교의 정확 검정에서 두 차이는 유의하지 않았으므로, 이 결과만으로 Entropy 선택이 일반적으로 열등하다고 해석하기는 어렵다.')
edit(278,'GC10-DET에서 관찰된 양의 평균 차이는 고정된 실험 조건에서 특징 기반 다양성 선택이 유용할 수 있음을 보여준다. 반면 NEU-DET의 Random 대비 차이는 작고 클래스별 이점과 감소가 함께 나타났다. 특징 공간에서 다양한 이미지를 확보하는 것이 각 결함 클래스에 필요한 학습 정보를 균등하게 확보하는 것과 같지는 않았다.')
edit(284,'첫째, 최종평가셋은 모델 선택과 분리하여 사용하였으나 같은 공개 데이터셋 안에서 구성한 분할이다. 이미지와 연결요소의 분할 간 중복은 제거했지만, 다른 생산 설비나 공정에서 수집한 외부 데이터에 대한 일반화까지 검증한 것은 아니다.')
edit(289,'후속 연구에서는 학습 시드를 확대하여 모델 학습 과정의 변동을 함께 측정하고, DINOv2 입력 방식과 특징 표현을 달리한 비교를 통해 특징 공간의 영향을 분석할 수 있다. 실제 산업 적용을 고려한다면 외부 생산 환경의 데이터와 이미지 수뿐 아니라 경계 상자 작성 시간 및 검수 시간까지 반영하여 라벨 비용과 탐지 성능의 관계를 평가할 필요가 있다.')
edit(293,f'최종평가셋에서 GC10-DET의 DINOv2-FF 평균 nAULC는 Random보다 {diff(G)}%p, Entropy보다 {diff(G,"Entropy")}%p 높았으며 두 비교 모두 다섯 선택 시드에서 차이가 양수였다. NEU-DET에서는 Random 대비 평균 차이가 {diff(N)}%p로 작고 시드별 방향이 달랐으며, Entropy 대비 {diff(N,"Entropy")}%p의 양의 평균 차이는 다섯 시드에서 모두 관찰되었다. 네 주 비교의 평균 차이 방향은 개발셋에서도 같았으나, 최종평가의 네 주 비교 모두 Holm 보정 후 통계적 유의성은 확인되지 않았다.')
edit(294,'클래스별 결과에서는 이점과 감소가 함께 나타났다. GC10-DET의 welding_line과 NEU-DET의 scratches에서는 DINOv2-FF의 평균 AP가 Random보다 높았지만, GC10-DET의 rolled_pit과 NEU-DET의 inclusion·crazing·rolled-in_scale에서는 낮았다. 선택된 이미지의 클래스 구성도 방법에 따라 달랐으며, 선택 수의 변화와 AP 변화의 방향은 모든 클래스에서 일치하지 않았다.')
edit(295,'RQ1에 대해 DINOv2-FF는 GC10-DET에서 R1부터 R5까지 Random과 Entropy보다 높은 평균 성능을 보였다. NEU-DET에서는 Random 대비 이점이 작고 시드별 방향이 달랐으며, Entropy 대비로는 작은 양의 차이가 일관되게 관찰되었다. RQ2에 대해서는 평균 성능의 차이가 결함 클래스에 따라 다른 방향으로 나타났고, NEU-DET의 최저 클래스 nAULC는 DINOv2-FF가 두 비교 방법보다 낮았다. 따라서 특징 기반 다양성 선택의 평균적 이점을 모든 클래스와 최저 클래스 성능의 향상으로 일반화할 수 없다.')
edit(296,'이 결과는 고정된 학습 시드와 두 공개 데이터셋의 최종평가셋에서 얻은 것이다. 향후에는 학습 시드와 특징 표현을 달리한 반복실험, 외부 생산 환경에서의 평가, 실제 주석 시간을 반영한 비용 분석으로 적용 범위를 확인할 필요가 있다.')

# English abstract mirrors Korean claims and numerical scope.
edit(320,'Training a metal surface defect detector requires annotations of defect categories and bounding boxes, which are costly to obtain in industrial settings. Active learning selects images for annotation to train models with limited labeled data. This study examined whether sample selection using pretrained DINOv2 image features improves metal surface defect detection under limited labeling.')
edit(321,'On GC10-DET and NEU-DET, greedy farthest-first selection using cosine distances between frozen DINOv2-small features (DINOv2-FF) was compared with random selection (Random) and detection-score entropy selection (Entropy). The labeled set increased from approximately 10% to 35% of the training set over five acquisition rounds. YOLOv8n was retrained from the same pretrained weights at each round. Five acquisition seeds and one fixed training seed were used per dataset. All 162 models selected using the development set were evaluated on the held-out test sets. Performance was analyzed using mAP50-95 learning curves, normalized area under the learning curve (nAULC), class-wise AP, and the class composition of selected images.')
edit(322,f'On GC10-DET, DINOv2-FF had a mean nAULC {diff(G)} percentage points higher than Random and {diff(G,"Entropy")} points higher than Entropy; both differences were positive for all five acquisition seeds. On NEU-DET, the mean difference from Random was small ({diff(N)} points), with mixed directions across seeds. The mean difference from Entropy was {diff(N,"Entropy")} points and was positive for all five seeds. None of the four primary comparisons was statistically significant after Holm correction.')
edit(323,f'At R5, the DINOv2-FF minus Random AP difference was +{cd(G,"welding_line")} points for welding_line on GC10-DET and {cd(N,"inclusion")} points for inclusion on NEU-DET. On NEU-DET, the nAULC formed from the lowest class AP at each round was 1.64 points lower for DINOv2-FF than for Random.')
edit(324,'The direction of the mean difference in each of the four primary comparisons observed on the development sets was maintained on the held-out test sets. However, the average benefit of feature-based diversity selection depended on the dataset and did not translate into improvements for every defect class or for worst-class performance.')

# Appendix additions keep training and evaluation environments separate.
anchor=P[332]._p
insert_before(anchor,'최종평가는 2026년 9월 25일에 수행하였다. 학습 환경은 표 A-2와 같으며, 최종평가 환경은 표 A-3에 별도로 제시하였다. 두 환경에서 Ultralytics 버전과 평가 조건을 동일하게 유지하였다.')
insert_before(anchor,'표 A-3. 최종평가 환경과 실행 조건',template=331)
envrows=[['GPU','NVIDIA GeForce RTX 3070, VRAM 8 GB'],['Python 및 CUDA','Python 3.11.9, PyTorch CUDA 빌드 12.6'],['주요 패키지','PyTorch 2.12.1+cu126, torchvision 0.27.1+cu126, Ultralytics 8.4.90'],['성능 평가','imgsz 640, batch 8, conf 0.001, NMS IoU 0.70, max_det 300'],['추가 평가 설정','workers 4, rect False, seed 2026091900, deterministic True']]
newtable(anchor,['항목','값 또는 규칙'],envrows,template=13)
insert_before(anchor,'평가 전 기록된 SHA-256과 162개 과학 실험 모델의 가중치가 모두 일치함을 확인하였다. 데이터셋·선택 시드·방법·라운드의 연결을 검증하였으며 기술 점검용 TECHNICAL_NEU-DET 모델은 제외하였다. 공통 R0는 한 번 평가한 값을 세 방법이 공유하였다.')
insert_before(anchor,'환경 확인용 개발셋 실행은 GC10-DET의 선택 시드 2026091901, DINOv2-FF R5 모델 하나를 사용하였다. 저장된 개발셋 mAP50-95, mAP50 및 모든 클래스 AP50-95와의 최대 절대 차이는 0이었다. 이는 해당 모델과 개발셋에 대한 평가 일치 확인이며, 모든 모델에서 소프트웨어·GPU 차이의 영향이 없음을 증명한 것은 아니다.')
insert_before(anchor,'최종평가 결과는 final_test_evaluation_20260925 디렉터리에 보존하였다. raw/results.csv와 results.json에는 모델별 결과를, raw/class_ap.csv에는 클래스별 AP를 기록하였다. raw/runs에는 모델별 결과 JSON, 실행 설정과 로그를 보존하였으며, statistics에는 seed_nAULC.csv, paired_comparisons.csv, 클래스별 결과 및 최저 클래스 분석 파일을 저장하였다. 사전 검증과 고정 프로토콜의 기록은 preflight에 보존하였다.')

# Scientific plots from the same CSV ledger, preserving the two-panel figure layout.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#78818a'})
colors={'Random':'#465664','DINO_F518':'#b56021','Entropy':'#17877d'}
fig,axs=plt.subplots(1,2,figsize=(9,4.0),sharey=True)
for ax,ds in zip(axs,datasets):
    for m,marker in zip(methods,['o','^','s']):
        rows=sorted([r for r in budget if r['dataset']==ds and r['method']==m],key=lambda x:int(x['round']))
        x=[100*int(r['labeled_images'])/(1604 if ds==G else 1258) for r in rows]
        y=[100*float(r['mean_mAP50_95']) for r in rows];sd=[100*float(r['sample_sd_mAP50_95']) for r in rows]
        ax.plot(x,y,marker=marker,label=labels[m],color=colors[m],linewidth=1.7,markersize=4)
        ax.fill_between(x,[a-bb for a,bb in zip(y,sd)],[a+bb for a,bb in zip(y,sd)],color=colors[m],alpha=.09)
    fv=float(find(full,dataset=ds,method='DINO_F518')['full_data_mAP50_95'])*100
    ax.axhline(fv,ls='--',lw=1,color='#7c8187');ax.text(35.7,fv+.4,f'Full-data: {fv:.2f}',ha='right',fontsize=8,color='#5a626b')
    ax.set_title(('(a) ' if ds==G else '(b) ')+ds,loc='left');ax.set_xticks([10,15,20,25,30,35]);ax.set_xlim(9,36.5);ax.set_ylim(14,40);ax.grid(axis='y',alpha=.2);ax.set_xlabel('Labeled images (%)')
axs[0].set_ylabel('Test mAP50-95 (%)');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,1.03));fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(WORK/'learning_curve.png',dpi=300,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9,4.4),gridspec_kw={'width_ratios':[1,1]})
for ax,ds in zip(axs,datasets):
    rows=[r for r in classdiff if r['dataset']==ds];y=list(range(len(rows)));v=[float(r['dino_minus_random_mean_ap_pp']) for r in rows]
    ax.barh(y,v,color=['#b56021' if x>=0 else '#465664' for x in v],height=.64);ax.set_yticks(y,[r['class_name'] for r in rows]);ax.invert_yaxis();ax.axvline(0,color='#444',lw=.7);ax.set_title(('(a) ' if ds==G else '(b) ')+ds,loc='left');ax.grid(axis='x',alpha=.15);ax.set_xlabel('Test AP50-95 difference (pp)');ax.set_xlim(-5,13.5);ax.set_xticks([-5,0,5,10])
    for i,vv in enumerate(v):ax.text(vv+(.14 if vv>=0 else -.14),i,f'{vv:+.2f}',ha='left' if vv>=0 else 'right',va='center',fontsize=8)
fig.tight_layout(w_pad=3);fig.savefig(WORK/'class_difference.png',dpi=300,bbox_inches='tight');plt.close(fig)
for idx,file in [(215,'learning_curve.png'),(240,'class_difference.png')]:
    P[idx].clear();P[idx].add_run().add_picture(str(WORK/file),width=Inches(6.4));P[idx].alignment=WD_ALIGN_PARAGRAPH.CENTER

# Fresh linked static contents, populated from final render page mapping in a later pass.
for p in P[17:60]:p._p.getparent().remove(p._p)
for p in P[62:78]:p._p.getparent().remove(p._p)
entries=[];bookmark_id=8000
def bookmark(p,name):
    global bookmark_id
    bookmark_id+=1
    st=OxmlElement('w:bookmarkStart');st.set(qn('w:id'),str(bookmark_id));st.set(qn('w:name'),name)
    en=OxmlElement('w:bookmarkEnd');en.set(qn('w:id'),str(bookmark_id))
    p._p.insert(1 if p._p.pPr is not None else 0,st);p._p.append(en)
def entry(anchor,target,style,name):
    bookmark(target,name)
    title=target.text
    if re.match(r'^그림 \d+-\d+\.',title):title='. '.join(title.split('. ')[:2])
    p=insert_before(anchor,title+'\t0',template=17,style=style)
    p.paragraph_format.keep_with_next=False
    entries.append({'anchor':name,'title':title,'target_text':target.text,'style':style})
    p._p.set(qn('w:rsidR'),'00F1A100')
    # Bookmarked title mapped to TOC through the sidecar, no stale live TOC field.
headings=[p for p in d.paragraphs if p.style.name.startswith('Heading') and p.text.strip()]
for i,p in enumerate(headings):entry(P[60]._p,p,'toc '+p.style.name.split()[-1],f'final_heading_{i}')
captions=[p for p in d.paragraphs if re.match(r'^(표|그림) [234A]-\d+\.',p.text)]
for i,p in enumerate(captions):entry(P[78]._p,p,'Normal',f'final_caption_{i}')
# Avoid large indentation inherited from body text in the list of figures/tables.
for p in d.paragraphs:
    if p._p.get(qn('w:rsidR'))=='00F1A100':
        p.paragraph_format.first_line_indent=Pt(0)
        p.paragraph_format.space_after=Pt(0)
        p.paragraph_format.line_spacing=1.0
        for r in p.runs:r.font.size=Pt(10)
        if p.style.name=='Normal':
            for r in p.runs:r.font.size=Pt(9)

# Repeat table headers and prevent individual row splits without changing table design.
for idx in [245,246]:P[252]._p.addprevious(P[idx]._p)
# Keep the English abstract and keywords within the existing single-page layout.
for idx in range(320,326):
    P[idx].paragraph_format.line_spacing=Pt(15)
    P[idx].paragraph_format.space_after=Pt(4)
P[208].paragraph_format.page_break_before=True
P[217].paragraph_format.page_break_before=True
P[233].paragraph_format.page_break_before=True
P[259].paragraph_format.page_break_before=True
P[327].paragraph_format.page_break_before=True
for table in d.tables:
    for ri,row in enumerate(table.rows):
        trpr=row._tr.get_or_add_trPr()
        if trpr.find(qn('w:cantSplit')) is None:trpr.append(OxmlElement('w:cantSplit'))
        if ri==0 and trpr.find(qn('w:tblHeader')) is None:trpr.append(OxmlElement('w:tblHeader'))
    # Tables already carry explicit widths/borders from the supplied draft.
d.core_properties.title='소량 라벨 환경에서 특징 기반 능동학습을 활용한 금속 표면 결함 객체 탐지'
d.core_properties.subject='최종평가셋 결과를 반영한 심사용 수정본'
d.save(OUT)
(WORK/'contents_entries.json').write_text(json.dumps(entries,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'paragraph_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
(WORK/'source_manifest.json').write_text(json.dumps({'source_docx':str(SRC),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'output':str(OUT),'ledger_files':{str(x.relative_to(DATA)):hashlib.sha256(x.read_bytes()).hexdigest() for x in list(S.glob('*.csv'))+[DATA/'raw/results.csv',DATA/'raw/results.json',DATA/'raw/class_ap.csv']},'changed_source_paragraphs':len(changes)},ensure_ascii=False,indent=2),encoding='utf-8')
print(str(OUT));print('Changed paragraphs',len(changes),'tables',len(d.tables),'contents',len(entries))

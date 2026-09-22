"""Independent invariants for the derived tables and rendered figures."""
import csv
import json
import math
import subprocess
from pathlib import Path

from PIL import Image

BASE=Path(__file__).resolve().parents[1]


def read_csv(name):
    with (BASE/"02_tables"/name).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def main():
    raw=read_csv("RAW_SEED_METRICS.csv")
    areas=read_csv("RAW_SEED_NAULC.csv")
    numeric=read_csv("02_FINAL_NUMERIC_RESULTS.csv")
    assert len(list((BASE/"07_reproducibility/train_labels/GC10-DET").glob("*.txt")))==1604
    assert len(list((BASE/"07_reproducibility/train_labels/NEU-DET").glob("*.txt")))==1258
    assert len(raw)==180 and len(areas)==30 and len(numeric)==38
    for dataset in ("GC10-DET","NEU-DET"):
        for seed in range(2026091901,2026091906):
            r0=[r for r in raw if r["dataset"]==dataset and int(r["acquisition_seed"])==seed and r["round"]=="R0"]
            assert len(r0)==3 and len({r["run_id"] for r in r0})==1
            for method in ("Random","DINO_F518","Entropy"):
                curve=sorted((r for r in raw if r["dataset"]==dataset and int(r["acquisition_seed"])==seed and r["method"]==method),key=lambda r:int(r["round"][1:]))
                assert len(curve)==6
                n=[int(r["labeled_images"]) for r in curve]
                y=[float(r["map5095_raw"]) for r in curve]
                independently=sum((n[i]-n[i-1])*(y[i]+y[i-1])/2 for i in range(1,6))/(n[-1]-n[0])
                a=next(r for r in areas if (r["dataset"],int(r["acquisition_seed"]),r["method"])==(dataset,seed,method))
                assert math.isclose(independently,float(a["naulc_raw"]),abs_tol=1e-12)
        for method in ("Random","DINO_F518","Entropy"):
            for rnd in range(6):
                vals=[float(r["map5095_pct"]) for r in raw if (r["dataset"],r["method"],r["round"])==(dataset,method,f"R{rnd}")]
                summary=next(r for r in numeric if (r["dataset"],r["method"],r["round"])==(dataset,method,f"R{rnd}"))
                assert len(vals)==5 and math.isclose(sum(vals)/5,float(summary["map5095_mean_pct"]),abs_tol=1e-11)
    stems=list((BASE/"03_figures_main").glob("*.pdf"))+list((BASE/"04_figures_appendix").glob("*.pdf"))
    assert len(stems)==7
    for pdf in stems:
        info=subprocess.run(["pdfinfo",str(pdf)],capture_output=True,text=True,check=True).stdout
        assert "Pages:           1" in info
        png=pdf.with_suffix(".png");svg=pdf.with_suffix(".svg")
        assert png.is_file() and svg.is_file() and "<svg" in svg.read_text(encoding="utf8")[:500]
        with Image.open(png) as im:
            dpi=im.info["dpi"]
            assert all(abs(x-300)<1 for x in dpi)
            assert min(im.size)>700
    print("PASS: 2,862 active overlay train labels; 180 raw seed-round rows; 30 nAULC rows; 38 summary rows; 7 figure triplets; shared R0; area and means independently agree")


if __name__=="__main__":main()

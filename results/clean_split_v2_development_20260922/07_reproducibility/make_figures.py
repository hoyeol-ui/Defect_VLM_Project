"""Thesis figures from audited DEVELOPMENT CSVs. Python 3.14 + Matplotlib/NumPy."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
TABLES = BASE / "02_tables"
MAIN = BASE / "03_figures_main"
APP = BASE / "04_figures_appendix"
METHODS = ("Random", "DINO_F518", "Entropy")
DISPLAY = {"Random":"Random", "DINO_F518":"DINO-F518", "Entropy":"Entropy"}
STYLE = {
    "Random":dict(color="#40566d",marker="o",linestyle="-"),
    "DINO_F518":dict(color="#b45f18",marker="^",linestyle="--"),
    "Entropy":dict(color="#418e83",marker="s",linestyle="-.")
}
plt.rcParams.update({
    "figure.facecolor":"white", "axes.facecolor":"white", "savefig.facecolor":"white",
    "font.family":"DejaVu Sans", "font.size":9, "axes.labelsize":10, "axes.titlesize":11,
    "legend.fontsize":8, "xtick.labelsize":8.5, "ytick.labelsize":8.5,
    "axes.linewidth":0.8, "pdf.fonttype":42, "svg.fonttype":"none"
})


def rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


NUM = rows(TABLES/"02_FINAL_NUMERIC_RESULTS.csv")
NAULC = rows(TABLES/"03_FINAL_NAULC_RESULTS.csv")
RAW_NAULC = rows(TABLES/"RAW_SEED_NAULC.csv")
ROUND = rows(TABLES/"04_ROUNDWISE_RESULTS.csv")
CLASS = rows(TABLES/"05_CLASSWISE_RESULTS.csv")
PAIRED = rows(BASE/"05_statistics/PAIRED_NAULC_RAW.csv")


def save(fig, stem, folder):
    fig.savefig(folder/(stem+".png"),dpi=300,bbox_inches="tight",pad_inches=0.08)
    fig.savefig(folder/(stem+".pdf"),bbox_inches="tight",pad_inches=0.08)
    fig.savefig(folder/(stem+".svg"),bbox_inches="tight",pad_inches=0.08)
    plt.close(fig)


def tidy(ax, ymax=None):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#dddddd",linewidth=0.55,zorder=0)
    if ymax is not None:ax.set_ylim(0,ymax)
    ax.tick_params(length=3)


def performance(dataset, short):
    fig,ax=plt.subplots(figsize=(6.35,4.1),layout="constrained")
    for method in METHODS:
        sub=sorted((r for r in NUM if r["dataset"]==dataset and r["method"]==method),key=lambda r:int(r["round"][1:]))
        x=np.array([float(r["labeled_pct"]) for r in sub]);y=np.array([float(r["map5095_mean_pct"]) for r in sub]);sd=np.array([float(r["map5095_sd_pp"]) for r in sub])
        s=STYLE[method]
        ax.fill_between(x,np.maximum(0,y-sd),y+sd,color=s["color"],alpha=.13,linewidth=0,zorder=1)
        ax.plot(x,y,label=DISPLAY[method],color=s["color"],marker=s["marker"],linestyle=s["linestyle"],linewidth=1.65,markersize=5.1,zorder=3)
    full=next(r for r in NUM if r["dataset"]==dataset and r["round"]=="Full")
    ax.axhline(float(full["map5095_mean_pct"]),color="#242424",linestyle=(0,(5,3)),linewidth=1.25,label="Full-data reference (1 run)",zorder=2)
    ax.set_xlim(8.8,36.3);ax.set_xticks([10,15,20,25,30,35]);ax.set_ylim(0,50)
    ax.set_xlabel("Labeled images (% of training pool)");ax.set_ylabel("mAP@0.50:0.95 (%)")
    ax.set_title(f"{dataset}: development performance",loc="left")
    tidy(ax);ax.legend(frameon=False,loc="upper left",ncol=2,handlelength=2.6)
    save(fig,f"FIGURE_{1 if short=='GC10' else 2}_{short}_AL_CURVE",MAIN)


def naulc_figure():
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.0),sharey=True,layout="constrained")
    for ax,(dataset,label) in zip(axes,(("GC10-DET","GC10"),("NEU-DET","NEU"))):
        seeds=sorted({int(r["acquisition_seed"]) for r in RAW_NAULC if r["dataset"]==dataset})
        for seed in seeds:
            ys=[float(next(r["naulc_pct"] for r in RAW_NAULC if (r["dataset"],r["method"],int(r["acquisition_seed"]))==(dataset,m,seed))) for m in METHODS]
            ax.plot(range(3),ys,color="#c7c7c7",linewidth=.8,alpha=.75,zorder=1)
        for i,m in enumerate(METHODS):
            s=next(r for r in NAULC if (r["dataset"],r["method"])==(dataset,m))
            vals=[float(r["naulc_pct"]) for r in RAW_NAULC if (r["dataset"],r["method"])==(dataset,m)]
            st=STYLE[m]
            ax.scatter([i]*len(vals),vals,s=16,facecolors="white",edgecolors=st["color"],linewidths=.85,zorder=2)
            ax.errorbar(i,float(s["naulc_mean_pct"]),yerr=float(s["naulc_sd_pp"]),fmt=st["marker"],color=st["color"],markersize=7,capsize=3,elinewidth=1.5,zorder=4)
        ax.set_xlim(-.55,2.55);ax.set_ylim(0,40);ax.set_xticks(range(3),[DISPLAY[m] for m in METHODS]);ax.set_title(label,loc="left")
        tidy(ax)
    axes[0].set_ylabel("nAULC of mAP@0.50:0.95 (%)")
    fig.suptitle("Development nAULC across five acquisition seeds",x=.05,ha="left",fontsize=11)
    save(fig,"FIGURE_3_NAULC_COMPARISON",MAIN)


def class_figure(dataset,short):
    clses=list(dict.fromkeys(r["class_name"] for r in CLASS if r["dataset"]==dataset))
    height=5.4 if dataset=="GC10-DET" else 4.25
    fig,ax=plt.subplots(figsize=(7.5,height),layout="constrained")
    offsets={"Random":-.21,"DINO_F518":0,"Entropy":.21}
    for method in METHODS:
        subset=[next(r for r in CLASS if (r["dataset"],r["method"],r["class_name"])==(dataset,method,cls)) for cls in clses]
        y=np.arange(len(clses))+offsets[method]
        st=STYLE[method]
        ax.errorbar([float(r["r5_ap_mean_pct"]) for r in subset],y,xerr=[float(r["r5_ap_sd_pp"]) for r in subset],fmt=st["marker"],color=st["color"],markersize=4.5,linewidth=1.0,capsize=2.2,label=DISPLAY[method],zorder=3)
    ax.set_yticks(range(len(clses)),[c.replace("_"," ") for c in clses]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel("R5 class AP@0.50:0.95 (%)");ax.set_title(f"{dataset}: class-wise development AP",loc="left")
    ax.spines[["top","right"]].set_visible(False);ax.grid(axis="x",color="#ddd",linewidth=.55)
    ax.legend(frameon=False,ncol=3,loc="lower right")
    save(fig,f"FIGURE_4_{short}_R5_CLASS_AP",MAIN)


def paired_figure(dataset,short):
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.2,3.85),layout="constrained",gridspec_kw={"width_ratios":[1.65,1]})
    contrasts=[("DINO_F518 - Random","DINO−Random","DINO_F518"),("DINO_F518 - Entropy","DINO−Entropy","Entropy"),("Entropy - Random","Entropy−Random","Random")]
    for contrast,label,stylename in contrasts:
        s=STYLE[stylename]
        sub=[]
        for rnd in range(6):
            sub.append(next(r for r in ROUND if r["dataset"]==dataset and r["contrast"]==contrast and r["round"]==f"R{rnd}"))
        x=np.array([float(r["labeled_pct"]) for r in sub]);y=np.array([float(r["mean_delta_pp"]) for r in sub]);sd=np.array([float(r["sd_delta_pp"]) for r in sub])
        ax.fill_between(x,y-sd,y+sd,color=s["color"],alpha=.10,linewidth=0)
        ax.plot(x,y,color=s["color"],marker=s["marker"],linestyle=s["linestyle"],linewidth=1.35,markersize=4,label=label)
    ax.axhline(0,color="#333",linewidth=.8)
    ax.set_xlim(8.8,36.2);ax.set_xticks([10,15,20,25,30,35]);ax.set_xlabel("Labeled images (% of training pool)");ax.set_ylabel("Paired mAP difference (pp)")
    ax.set_title("Round-wise paired differences",loc="left");tidy(ax)
    ax.legend(frameon=False,fontsize=7,loc="best")
    for i,(contrast,label,stylename) in enumerate(contrasts):
        vals=[float(r["delta_naulc_pp"]) for r in PAIRED if r["dataset"]==dataset and r["contrast"]==contrast]
        s=STYLE[stylename]
        bx.scatter([i]*len(vals),vals,facecolors="white",edgecolors=s["color"],marker=s["marker"],s=25,zorder=3)
        bx.plot(i,np.mean(vals),marker="D",color=s["color"],markersize=6,zorder=4)
        bx.vlines(i,np.mean(vals)-np.std(vals,ddof=1),np.mean(vals)+np.std(vals,ddof=1),color=s["color"],linewidth=1.2,zorder=2)
    bx.axhline(0,color="#333",linewidth=.8)
    bx.set_xticks(range(3),["DINO−\nRandom","DINO−\nEntropy","Entropy−\nRandom"]);bx.set_ylabel("Paired nAULC difference (pp)")
    bx.set_title("nAULC by seed",loc="left");tidy(bx)
    fig.suptitle(f"{dataset}: development paired effects (descriptive)",x=.01,ha="left",fontsize=11)
    save(fig,f"APPENDIX_PAIRED_DELTA_{short}",APP)


def main():
    performance("GC10-DET","GC10");performance("NEU-DET","NEU")
    naulc_figure()
    class_figure("GC10-DET","GC10");class_figure("NEU-DET","NEU")
    paired_figure("GC10-DET","GC10");paired_figure("NEU-DET","NEU")
    print("Wrote 7 figures × PNG/PDF/SVG")


if __name__=="__main__":main()

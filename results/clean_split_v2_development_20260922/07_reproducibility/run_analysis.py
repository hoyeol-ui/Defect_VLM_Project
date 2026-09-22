"""Recompute clean-split v2 DEVELOPMENT results from sealed per-run artifacts.

No test files are read. Uses only the active ANNOTATION_PATCH_v2_0_2 attempt.
Run with Python 3.14 + NumPy after preflight.py passes.
"""
from __future__ import annotations

import collections
import csv
import itertools
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
EVIDENCE = BASE.parents[0] / "evidence/final_main_runpod_execution_v2_20260919"
STATE = EVIDENCE / "state_attempts/ANNOTATION_PATCH_v2_0_2"
CFG = json.loads((EVIDENCE / "config/protocol.json").read_text(encoding="utf8"))
CELLS = json.loads((EVIDENCE / "config/cells.json").read_text(encoding="utf8"))
METHODS = ("Random", "DINO_F518", "Entropy")
COMPARISONS = (("DINO_F518", "Random"), ("DINO_F518", "Entropy"), ("Entropy", "Random"))
T_95_DF4 = 2.7764451051977987


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def describe(values):
    vals = list(values)
    assert len(vals) == 5 and all(math.isfinite(x) for x in vals)
    mean, sd = st.mean(vals), st.stdev(vals)
    half = T_95_DF4 * sd / math.sqrt(5)
    return dict(mean=mean, sd=sd, ci95_low=mean-half, ci95_high=mean+half, median=st.median(vals), minimum=min(vals), maximum=max(vals))


def naulc(values, counts):
    assert len(values) == len(counts) == 6
    return sum((counts[i]-counts[i-1])*(values[i]+values[i-1])/2 for i in range(1,6))/(counts[-1]-counts[0])


def bootstrap_ci(diffs, dataset):
    rng = np.random.default_rng(2026091912 if dataset == "GC10-DET" else 2026091913)
    arr = np.asarray(diffs, dtype=np.float64)
    indices = rng.integers(0, len(arr), size=(100000, len(arr)))
    means = arr[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def sign_flip_p(diffs):
    arr = np.asarray(diffs, dtype=np.float64)
    observed = abs(float(arr.mean()))
    values = [abs(float(np.mean(arr*np.asarray(signs)))) for signs in itertools.product((-1,1), repeat=5)]
    return sum(v >= observed-1e-12 for v in values)/32


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adjusted = [None]*len(pvals)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(pvals)-rank)*pvals[i]))
        adjusted[i] = previous
    return adjusted


def source_path(run_id):
    return f"state_attempts/ANNOTATION_PATCH_v2_0_2/runs/{run_id}/sealed_dev_metrics.json"


def main():
    assert read(BASE / "00_preflight/preflight_machine.json")["status"] == "PASS"
    cell_by_key = {(c["dataset"], c["seed"], c["method"], c["round"]): c for c in CELLS}
    metric = {c["run_id"]: read(STATE / "runs" / c["run_id"] / "sealed_dev_metrics.json") for c in CELLS}
    raw, numerical, curve, naulc_raw, naulc_summary, paired_raw, pair_summary, roundwise = [], [], {}, [], [], [], [], []
    class_raw, class_summary, worst_raw, selection_rows = [], [], [], []
    full_ref = {}
    for dataset, spec in CFG["datasets"].items():
        counts = spec["counts"]
        full = next(c for c in CELLS if c["dataset"] == dataset and c["method"] == "Full_data")
        fm = metric[full["run_id"]]["overall"]
        full_ref[dataset] = {"run_id": full["run_id"], "map5095_pct": 100*fm["metrics/mAP50-95(B)"], "map50_pct": 100*fm["metrics/mAP50(B)"]}
        numerical.append(dict(dataset=dataset, method="Full-data reference (1 run)", round="Full", labeled_images=spec["train"], labeled_pct=100.0, n_seeds=1, map5095_mean_pct=full_ref[dataset]["map5095_pct"], map5095_sd_pp="", map5095_ci95_low_pct="", map5095_ci95_high_pct="", map50_mean_pct=full_ref[dataset]["map50_pct"], source_run_ids=full["run_id"], partition="development"))
        for seed in CFG["seeds"]:
            base = cell_by_key[(dataset, seed, "SHARED", 0)]
            for method in METHODS:
                values=[]
                for rnd in range(6):
                    c = base if rnd == 0 else cell_by_key[(dataset, seed, method, rnd)]
                    m = metric[c["run_id"]]
                    ap = m["overall"]["metrics/mAP50-95(B)"]
                    ap50 = m["overall"]["metrics/mAP50(B)"]
                    values.append(ap)
                    raw.append(dict(dataset=dataset, method=method, acquisition_seed=seed, learner_seed=CFG["learner_seed"], round=f"R{rnd}", labeled_images=counts[rnd], labeled_pct=100*counts[rnd]/spec["train"], map5095_raw=ap, map5095_pct=100*ap, map50_raw=ap50, map50_pct=100*ap50, run_id=c["run_id"], source_relpath=source_path(c["run_id"]), metric_key="overall.metrics/mAP50-95(B)", attempt="ANNOTATION_PATCH_v2_0_2", partition="development"))
                    for cls, cls_ap in m["per_class_AP50_95"].items():
                        class_raw.append(dict(dataset=dataset, method=method, acquisition_seed=seed, round=f"R{rnd}", class_name=cls, ap5095_raw=cls_ap, ap5095_pct=100*cls_ap, run_id=c["run_id"], source_relpath=source_path(c["run_id"])))
                    low_class, low_value=min(m["per_class_AP50_95"].items(),key=lambda kv:(kv[1],kv[0]))
                    worst_raw.append(dict(dataset=dataset, method=method, acquisition_seed=seed, round=f"R{rnd}", worst_class=low_class, worst_ap_pct=100*low_value, run_id=c["run_id"]))
                curve[(dataset, seed, method)]=values
                area = naulc(values, counts)
                naulc_raw.append(dict(dataset=dataset, method=method, acquisition_seed=seed, learner_seed=CFG["learner_seed"], naulc_raw=area, naulc_pct=100*area, r0_run_id=base["run_id"], r1_to_r5_run_ids=";".join(cell_by_key[(dataset,seed,method,r)]["run_id"] for r in range(1,6)), partition="development"))
        for method in METHODS:
            areas=[next(r["naulc_raw"] for r in naulc_raw if (r["dataset"],r["method"],r["acquisition_seed"])==(dataset,method,seed)) for seed in CFG["seeds"]]
            d=describe([100*x for x in areas])
            naulc_summary.append(dict(dataset=dataset,method=method,n_seeds=5,naulc_mean_pct=d["mean"],naulc_sd_pp=d["sd"],naulc_ci95_low_pct=d["ci95_low"],naulc_ci95_high_pct=d["ci95_high"],naulc_median_pct=d["median"],naulc_min_pct=d["minimum"],naulc_max_pct=d["maximum"],partition="development"))
            for rnd in range(6):
                subset=[r for r in raw if r["dataset"]==dataset and r["method"]==method and r["round"]==f"R{rnd}"]
                ap=describe([r["map5095_pct"] for r in subset]);ap50=describe([r["map50_pct"] for r in subset])
                numerical.append(dict(dataset=dataset,method=method,round=f"R{rnd}",labeled_images=counts[rnd],labeled_pct=100*counts[rnd]/spec["train"],n_seeds=5,map5095_mean_pct=ap["mean"],map5095_sd_pp=ap["sd"],map5095_ci95_low_pct=ap["ci95_low"],map5095_ci95_high_pct=ap["ci95_high"],map50_mean_pct=ap50["mean"],map50_sd_pp=ap50["sd"],map50_ci95_low_pct=ap50["ci95_low"],map50_ci95_high_pct=ap50["ci95_high"],source_run_ids=";".join(r["run_id"] for r in subset),partition="development"))
        for left,right in COMPARISONS:
            diffs=[100*(naulc(curve[(dataset,s,left)],counts)-naulc(curve[(dataset,s,right)],counts)) for s in CFG["seeds"]]
            for seed,difference in zip(CFG["seeds"],diffs):
                paired_raw.append(dict(dataset=dataset,contrast=f"{left} - {right}",acquisition_seed=seed,delta_naulc_pp=difference))
            d=describe(diffs); ci=bootstrap_ci(diffs,dataset); p=sign_flip_p(diffs)
            pair_summary.append(dict(dataset=dataset,contrast=f"{left} - {right}",role="SAP primary contrast on future test; exploratory on development" if (left,right)!=("Entropy","Random") else "secondary descriptive",n_pairs=5,mean_delta_pp=d["mean"],sd_delta_pp=d["sd"],median_delta_pp=d["median"],min_delta_pp=d["minimum"],max_delta_pp=d["maximum"],wins=sum(x>0 for x in diffs),losses=sum(x<0 for x in diffs),ties=sum(x==0 for x in diffs),bootstrap95_low_pp=ci[0],bootstrap95_high_pp=ci[1],exact_signflip_p=p,holm_p=""))
            for rnd in range(6):
                round_diffs=[100*(curve[(dataset,s,left)][rnd]-curve[(dataset,s,right)][rnd]) for s in CFG["seeds"]]
                desc=describe(round_diffs)
                for seed,delta in zip(CFG["seeds"],round_diffs):
                    roundwise.append(dict(dataset=dataset,contrast=f"{left} - {right}",round=f"R{rnd}",labeled_images=counts[rnd],labeled_pct=100*counts[rnd]/spec["train"],acquisition_seed=seed,paired_delta_pp=delta,mean_delta_pp=desc["mean"],sd_delta_pp=desc["sd"],ci95_low_pp=desc["ci95_low"],ci95_high_pp=desc["ci95_high"]))
        # Classwise R5 rows include per-seed paired differences and full R5 class coverage.
        names=list(metric[full["run_id"]]["per_class_AP50_95"])
        for cls in names:
            by_method={m:[next(r["ap5095_pct"] for r in class_raw if (r["dataset"],r["method"],r["acquisition_seed"],r["round"],r["class_name"])==(dataset,m,s,"R5",cls)) for s in CFG["seeds"]] for m in METHODS}
            for method in METHODS:
                d=describe(by_method[method])
                class_summary.append(dict(dataset=dataset,class_name=cls,method=method,r5_ap_mean_pct=d["mean"],r5_ap_sd_pp=d["sd"],r5_ap_ci95_low_pct=d["ci95_low"],r5_ap_ci95_high_pct=d["ci95_high"],full_reference_ap_pct=100*metric[full["run_id"]]["per_class_AP50_95"][cls],delta_vs_random_mean_pp=st.mean(a-b for a,b in zip(by_method[method],by_method["Random"])),n_seeds=5))
    # Family adjustment only for the four SAP-defined comparisons, never the secondary one.
    primary=[p for p in pair_summary if p["contrast"] in ("DINO_F518 - Random","DINO_F518 - Entropy")]
    adjusted=holm([r["exact_signflip_p"] for r in primary])
    for row,p in zip(primary,adjusted):row["holm_p"]=p
    # Train-only annotation composition; missing YOLO label file means zero objects.
    for dataset, spec in CFG["datasets"].items():
        with (EVIDENCE / "inputs" / dataset / "SPLIT_MEMBERSHIP.csv").open(encoding="utf8",newline="") as f:
            components={r["sample_id"]:r["component_id"] for r in csv.DictReader(f) if r["split"]=="train"}
        label_dir=BASE / "07_reproducibility/train_labels" / dataset
        assert all((label_dir/(sid+".txt")).exists() for sid in components),f"missing active patched train label: {dataset}"
        labels={sid:[int(line.split()[0]) for line in (label_dir/(sid+".txt")).read_text().splitlines()] for sid in components}
        class_names=list(metric[next(c["run_id"] for c in CELLS if c["dataset"]==dataset and c["method"]=="Full_data")]["per_class_AP50_95"])
        full_run=next(c["run_id"] for c in CELLS if c["dataset"]==dataset and c["method"]=="Full_data")
        yaml_names=[line.split(":",1)[1].strip() for line in (STATE/"runs"/full_run/"data.yaml").read_text(encoding="utf8").splitlines() if line.startswith("  ") and line.strip().split(":",1)[0].isdigit()]
        assert yaml_names==class_names, f"class index mismatch: {dataset}"
        pool_img=collections.Counter(k for classes in labels.values() for k in set(classes));pool_box=collections.Counter(k for classes in labels.values() for k in classes)
        for seed in CFG["seeds"]:
            for method in METHODS:
                for rnd in range(6):
                    cell=cell_by_key[(dataset,seed,"SHARED",0)] if rnd==0 else cell_by_key[(dataset,seed,method,rnd)]
                    selection=read(STATE/"runs"/cell["run_id"]/"selection_manifest.json")
                    new_ids=selection["selected_ids"];all_ids=selection["cumulative_ids"]
                    for cls_i,cls in enumerate(class_names):
                        new_img=sum(cls_i in labels[sid] for sid in new_ids);all_img=sum(cls_i in labels[sid] for sid in all_ids)
                        new_box=sum(labels[sid].count(cls_i) for sid in new_ids);all_box=sum(labels[sid].count(cls_i) for sid in all_ids)
                        selection_rows.append(dict(dataset=dataset,method=method,acquisition_seed=seed,round=f"R{rnd}",class_name=cls,new_images_with_class=new_img,new_boxes=new_box,cumulative_images_with_class=all_img,cumulative_boxes=all_box,pool_images_with_class=pool_img[cls_i],pool_boxes=pool_box[cls_i],cumulative_class_image_pool_pct=100*all_img/pool_img[cls_i] if pool_img[cls_i] else "",cumulative_components=len({components[sid] for sid in all_ids}),run_id=cell["run_id"]))
    # Seed-paired class effects, selection counts, and changing worst-class curves.
    class_pairs=[];composition_summary=[];worst_naulc=[]
    for dataset,spec in CFG["datasets"].items():
        names=list(dict.fromkeys(r["class_name"] for r in class_summary if r["dataset"]==dataset))
        for cls in names:
            for left,right in COMPARISONS:
                for seed in CFG["seeds"]:
                    ap_left=next(r["ap5095_pct"] for r in class_raw if (r["dataset"],r["class_name"],r["method"],r["acquisition_seed"],r["round"])==(dataset,cls,left,seed,"R5"))
                    ap_right=next(r["ap5095_pct"] for r in class_raw if (r["dataset"],r["class_name"],r["method"],r["acquisition_seed"],r["round"])==(dataset,cls,right,seed,"R5"))
                    class_pairs.append(dict(dataset=dataset,class_name=cls,contrast=f"{left} - {right}",acquisition_seed=seed,paired_r5_ap_delta_pp=ap_left-ap_right))
            for method in METHODS:
                subset=[r for r in selection_rows if (r["dataset"],r["class_name"],r["method"],r["round"])==(dataset,cls,method,"R5")]
                composition_summary.append(dict(dataset=dataset,class_name=cls,method=method,n_seeds=5,mean_cumulative_images_with_class=st.mean(int(r["cumulative_images_with_class"]) for r in subset),mean_cumulative_boxes=st.mean(int(r["cumulative_boxes"]) for r in subset),pool_images_with_class=int(subset[0]["pool_images_with_class"]),pool_boxes=int(subset[0]["pool_boxes"]),mean_new_r5_images_with_class=st.mean(int(r["new_images_with_class"]) for r in subset)))
        for method in METHODS:
            for seed in CFG["seeds"]:
                subset=[next(r for r in worst_raw if (r["dataset"],r["method"],r["acquisition_seed"],r["round"])==(dataset,method,seed,f"R{rnd}")) for rnd in range(6)]
                worst_naulc.append(dict(dataset=dataset,method=method,acquisition_seed=seed,worst_class_curve_naulc_pct=100*naulc([r["worst_ap_pct"]/100 for r in subset],spec["counts"]),r5_worst_class=subset[-1]["worst_class"],r5_worst_ap_pct=subset[-1]["worst_ap_pct"]))
    tables=BASE/"02_tables";stats=BASE/"05_statistics"
    write_csv(tables/"02_FINAL_NUMERIC_RESULTS.csv",numerical)
    write_csv(tables/"03_FINAL_NAULC_RESULTS.csv",naulc_summary)
    write_csv(tables/"04_ROUNDWISE_RESULTS.csv",roundwise)
    write_csv(tables/"05_CLASSWISE_RESULTS.csv",class_summary)
    write_csv(tables/"RAW_SEED_METRICS.csv",raw)
    write_csv(tables/"RAW_SEED_NAULC.csv",naulc_raw)
    write_csv(tables/"RAW_CLASSWISE_ALL_ROUNDS.csv",class_raw)
    write_csv(tables/"SELECTION_COMPOSITION.csv",selection_rows)
    write_csv(tables/"SELECTION_COMPOSITION_R5_SUMMARY.csv",composition_summary)
    write_csv(tables/"R5_PAIRED_CLASS_DIFFERENCES.csv",class_pairs)
    write_csv(tables/"WORST_CLASS_ALL_ROUNDS.csv",worst_raw)
    write_csv(tables/"WORST_CLASS_NAULC.csv",worst_naulc)
    write_csv(stats/"PAIRED_NAULC_RAW.csv",paired_raw)
    write_csv(stats/"PAIRED_NAULC_SUMMARY.csv",pair_summary)
    summary={"full_data_reference":full_ref,"naulc":naulc_summary,"paired":pair_summary,"r5":[r for r in numerical if r["round"]=="R5"],"n_rows":{"raw_seed_metrics":len(raw),"class_all_rounds":len(class_raw),"selection_composition":len(selection_rows),"class_pairs":len(class_pairs)}}
    (stats/"computed_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":main()

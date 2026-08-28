# Datasets

## Release rule

This repository publishes code, split/sample identifiers, hashes, derived numerical tables, and figures. It does **not** newly publish dataset image bytes, masks, XML annotations, dataset archives, or trained weights. “Publicly downloadable” does not automatically mean “redistributable.” When the original owner did not state clear terms, this document says `VERIFY REQUIRED` or `REDISTRIBUTION UNKNOWN` instead of borrowing a mirror's license.

The authoritative counts below distinguish an upstream dataset description from the exact snapshot curated in this project.

## Controlled datasets

### GC10-DET

| Field | Record |
|---|---|
| Canonical name | GC10-DET Metallic Surface Defect Dataset |
| Original work | Xiaoming Lv, Fajie Duan, Jiajia Jiang, Xiao Fu, and Lin Gan, “Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network,” *Sensors* 20(6):1562, 2020 |
| DOI | `10.3390/s20061562` |
| Official/project page | https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Datasets |
| Task / annotation | metallic-surface object detection; 10 defect classes; bounding boxes |
| Upstream size | Literature and mirrors disagree across versions/augmentations. A commonly described detection release is about 2,300 images. **Do not substitute that count for the audited local snapshot.** |
| Exact project snapshot | 2,312 raw registry rows inspected; 2,292 curated pixel-identity records after 6 missing-XML exclusions, 2 invalid-GT exclusions, and 12 equivalent duplicate collapses |
| Controlled split | acquisition 1,605; development 229; locked final 458 |
| Controlled budgets | 161, 241, 321, 401, 481, 561 images |
| Historical role | rare discovery/taxonomy, detector translation, DINO/hybrid/Random audits |
| Controlled role | Random-PCG64-v1 versus FrozenDINO-FF-v1 downstream detector learning |
| Preprocessing | pixel-identity deduplication; deterministic rare-label-first 70/10/20 split; YOLO conversion; images processed as RGB/640 during YOLO training |
| License / terms | Original repository invites research use and citation but exposes no explicit dataset license in its README/repository metadata. `VERIFY REQUIRED` |
| Redistribution | `REDISTRIBUTION UNKNOWN — RAW DATA NOT INCLUDED` |
| Raw data in this branch | No newly added raw images/XML. Sample-ID/hash/manifests and derived tables only. |

The project count differs from popular summaries because this research audited a specific local snapshot and grouped by canonical pixel identity. That difference is documented rather than silently forced to 2,300.

### NEU-DET

| Field | Record |
|---|---|
| Canonical name | NEU Surface Defect Database / NEU-DET bounding-box release |
| Original attribution | Kechen Song and Yunhui Yan, Northeastern University, commonly cited to 2013. Exact canonical publication/DOI for the detection release: `VERIFY REQUIRED` |
| Original/project page | Northeastern University surface-defect database page; stable official download metadata should be rechecked before manuscript submission |
| Task / annotation | hot-rolled steel surface object detection; six classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches; bounding boxes |
| Upstream size | 1,800 grayscale 200×200 images, 300 per class |
| Exact project snapshot | 1,800 raw records inspected; 1,798 curated pixel identities after excluding two identical-pixel/conflicting-GT identities |
| Controlled split | acquisition 1,259; development 180; locked final 359 |
| Controlled budgets | 126, 189, 252, 315, 378, 441 images |
| Historical role | fixed-set learner stability, repeated acquisition confirmation, detector-aware and scale audits |
| Controlled role | Random-PCG64-v1 versus FrozenDINO-FF-v1 downstream detector learning |
| Preprocessing | pixel-identity conflict exclusion; deterministic multilabel 70/10/20 split; grayscale decoded to RGB for DINO/YOLO pipelines |
| License / terms | No explicit original dataset redistribution license was verified. `VERIFY REQUIRED` |
| Redistribution | `REDISTRIBUTION UNKNOWN — RAW DATA NOT INCLUDED` |
| Raw data in this branch | No newly added images/XML. Sample-ID/hash/manifests and derived tables only. |

The locked-final annotation audit found one exact duplicated bounding box in `crazing_120.xml`. The source was not edited; Ultralytics removed the duplicate consistently. See `confirmatory_v1_1/final_analysis/FINAL_GT_DUPLICATE_ANNOTATION_AUDIT.md`.

## Historical evidence datasets

### MPDD

| Field | Record |
|---|---|
| Canonical name | Metal Parts Defect Detection Dataset (MPDD) |
| Original work | Štěpán Ježek, Martin Jonák, Radim Burget, Pavel Dvořák, and Miloš Škóták, “Deep learning-based defect detection of metal parts: evaluating current methods in complex conditions,” ICUMT 2021, pp. 66–71 |
| DOI | `10.1109/ICUMT54235.2021.9631567` |
| Official page/download | https://github.com/stepanje/MPDD (owner-linked download) |
| Task / annotation | industrial anomaly detection/localization; pixel-precise masks; six product categories |
| Upstream size | Official page states more than 1,000 images. Frequently reported exact counts should be verified against the downloaded release before quoting them. |
| Exact project use | A normal-centered heterogeneous acquisition pool; initial 20/query 20; 200 paired acquisition perturbations. The historical endpoint was anomaly-image discovery plus product-category/source composition, not detector AULC. |
| Result role | anomaly images `+6.245` versus Random; product-category coverage `−0.325`; safety gate failed; source-origin confounding audited |
| License | Official repository `LICENSE`: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0) |
| Redistribution | Permitted only under license conditions, but **raw data is not included here** to avoid duplicating owner distribution and attribution obligations. Commercial use is not permitted by the license. |

### VisA

| Field | Record |
|---|---|
| Canonical name | Visual Anomaly (VisA) Dataset |
| Original work | Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer, “SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation,” ECCV 2022 / arXiv:2207.14315 |
| Official page/download | https://github.com/amazon-science/spot-diff and its owner-hosted S3 link |
| Task / annotation | image-level anomaly labels and pixel-level masks; 12 object subsets |
| Upstream size | 10,821 images: 9,621 normal and 1,200 anomalous |
| Exact project use | Protocol-audited acquisition pool of 8,650 images including 960 anomalies; initial 20/query 20; 200 paired selection perturbations. Selection-only discovery/composition analysis. |
| Result role | anomaly images `+14.480`; category coverage `−4.110`; concentration/category-collapse safety failure |
| License | CC BY 4.0 for dataset; cite and attribute the owners |
| Redistribution | License allows redistribution with attribution, but **raw data is not included here**; use the official download. |

### KolektorSDD2 / KSDD2

| Field | Record |
|---|---|
| Canonical name | Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2) |
| Original work | Jakob Božič, Domen Tabernik, and Danijel Skočaj, “Mixed supervision for surface-defect detection: from weakly to fully supervised learning,” *Computers in Industry*, 2021 |
| DOI | `10.1016/j.compind.2021.103459` |
| Official page/download | https://www.vicos.si/resources/kolektorsdd2/ |
| Task / annotation | industrial surface-defect classification/segmentation; image labels and masks |
| Upstream size | Official page: 356 defective + 2,979 non-defective = 3,335 images; official train 246 positive + 2,085 negative = 2,331; official test 110 positive + 894 negative = 1,004 |
| Exact project use | 2,331 unique official-train images after archive-copy exclusion; target-blind development 466; acquisition 1,865; official test 1,004 remained locked. PatchCore initial/query sets used a 200-image labeled-set budget (100 initial + 100 query) across 10 acquisition seeds. The bounded U-Net stage ran 2 strategies × 10 acquisitions × 3 learner seeds = 60 development runs and evaluated 53 positive development images. |
| Result role | prospective selection validity/discovery and bounded learner gate; positive Dice estimate `+0.037565` below frozen `+0.05`, so no PASS claim |
| License | CC BY-NC-SA 4.0; commercial use requires owner contact |
| Redistribution | Noncommercial/share-alike conditions apply; **raw data not included**; obtain from ViCoS. |

### DeepPCB

| Field | Record |
|---|---|
| Canonical name | DeepPCB |
| Original work | Sanli Tang, Fan He, Xiaolin Huang, and Jie Yang, “Online PCB Defect Detector On A New PCB Defect Dataset,” arXiv:1902.06197, 2019 |
| Official page/download | https://github.com/tangsanli5201/DeepPCB |
| Task / annotation | paired defect-free template and tested PCB images; bounding boxes for open, short, mousebite, spur, pin hole, and spurious copper |
| Upstream size | 1,500 aligned image pairs, approximately 640×640; repository states 1,000 training images and the remainder test |
| Exact project use | prospective paired-reference selection gate and read-only mechanism/development audit. The branch stopped at primary total-box enrichment `1.107693 < 1.20`; detector training/inference and official/final-test use were zero. |
| License / terms | Repository code is marked MIT, while the dataset README separately states “research purpose” use only. Treat dataset bytes as research-only; do not infer MIT coverage for the dataset. |
| Redistribution | `RESEARCH-ONLY / REDISTRIBUTION NOT ASSUMED — RAW DATA NOT INCLUDED` |

## Other early inputs

No additional dataset has been promoted to a canonical thesis evidence layer by this release audit. Some legacy scripts may contain ad-hoc/local image inputs. They are `SCRATCH` unless linked to a named evidence ledger in `CODE_INDEX.md`; they must not be cited as an additional benchmark without a separate provenance and license audit.

## Reconstructing the controlled data locally

1. Download GC10-DET and NEU-DET from owner/authorized sources and retain their original layout.
2. Do not copy raw bytes into this repository.
3. Point the split builder to a local source root and verify that the audited raw/curated counts and SHA-based identity grouping match.
4. Require exact hashes for `split_assignment.csv`, blind universes, seed manifest, and protocol before using the released manifest seals.
5. A different dataset snapshot is a new replication, not a bitwise reproduction; report the difference explicitly.

## Citation and legal note

`CITATIONS_DATASETS.bib` contains only citations with sufficient verification for this branch and flags the incomplete NEU entry in comments rather than inventing fields. This document is a research release audit, not legal advice; downstream users remain responsible for the upstream terms.

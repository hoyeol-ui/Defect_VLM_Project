# V7 DINO Full Learning Curve

Development full-curve result only. Final test was not used.

## Config

```json
{
  "experiment_id": "v7_full_curve_20260711_211848",
  "PROJECT_ROOT": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project",
  "save_dir": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\runs\\active_learning_ablation_v7_full_curve\\v7_full_curve_20260711_211848",
  "dataset_root": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\datasets\\active_learning_ablation_v7_full_curve\\v7_full_curve_20260711_211848",
  "resumed": false,
  "priority_csv": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\outputs\\priority_sensitivity_20260706_152020\\penalty_0\\priority_scores_pseudo.csv",
  "priority_csv_sha256": "111e9d4d862a844d398bc23f94837873317615c84342724e111f08675a983c3d",
  "eval_protocol_dir": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\runs\\evaluation_protocol_v7\\eval_protocol_20260711_173723",
  "development_eval_path": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\runs\\evaluation_protocol_v7\\eval_protocol_20260711_173723\\development_eval_v7.csv",
  "development_eval_sha256": "0f850cfbaafe5a1bf3002d3dcea0d25a3610ef80d03b66af59579fc7ee035376",
  "final_test_used": false,
  "embedding_dir": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\outputs\\visual_embeddings_v7\\dinov2_20260711_193147",
  "DINO_manifest_sha256": "0afd9dc5a301625c81e8acad2bc2e17970c394052dc65b830a71ddda63c09988",
  "DINO_config": {
    "PROJECT_ROOT": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project",
    "priority_csv": "C:\\Users\\user\\Desktop\\vlm\\Defect_VLM_Project\\outputs\\priority_sensitivity_20260706_152020\\penalty_0\\priority_scores_pseudo.csv",
    "backend": "dinov2",
    "uses_gt_labels": false,
    "uses_class_hint": false,
    "uses_xml_bbox": false,
    "paper_facing_warning": "handcrafted backend is for smoke tests unless explicitly justified.",
    "allow_model_download": true,
    "reuse_existing_embedding_cache": true,
    "force_rebuild_embeddings": false,
    "num_manifest_images": 99,
    "model_id": "facebook/dinov2-small",
    "model_source": "huggingface_transformers",
    "device": "cuda",
    "amp": true,
    "batch_size": 32,
    "local_files_only": false,
    "embedding_dim": 384,
    "status": "success",
    "num_embeddings": 99,
    "runtime_sec": 20.981826800001727
  },
  "strategies": [
    "GTFreeRandom",
    "GTFreeDatasetBalancedConsistency",
    "GTFreeDatasetBalancedVisualDiversity"
  ],
  "acquisition_seeds": [
    42,
    43,
    44,
    45,
    46
  ],
  "training_seed_rule": "training_seed = 1000 + acquisition_seed",
  "initial_seed_size": 15,
  "rounds": 4,
  "query_size": 5,
  "budgets": [
    15,
    20,
    25,
    30,
    35
  ],
  "model": "yolov8n.pt",
  "epochs": 100,
  "patience": 100,
  "batch": 8,
  "workers": 4,
  "cache": "ram",
  "dry_run": false,
  "selection_only": false,
  "git_commit": "50a74b47f75e8c15e435547eac2922a7fca76dca",
  "git_dirty": true,
  "runner_sha256": "f43c78ef2bc19df13ff762ca872a1cccbc0f922d6ca9a43abe1e4346c354acce",
  "gtfree_prohibited_columns": [
    "actual_bbox_count",
    "actual_xml_class",
    "class_hint",
    "map50",
    "map5095",
    "num_xml_instances",
    "primary_xml_class",
    "xml_mapped_classes"
  ],
  "primary_metric": "normalized_aulc_map5095"
}
```

## Run status

- Successful or skipped/dry-run rows: 75
- Failed rows: 0

## Strategy aggregate summary

| strategy                             |   num_acquisition_seeds |   final_map50_mean |   final_map50_std |   final_map50_ci95_low |   final_map50_ci95_high |   final_map5095_mean |   final_map5095_std |   final_map5095_ci95_low |   final_map5095_ci95_high |   best_map50_mean |   best_map50_std |   best_map50_ci95_low |   best_map50_ci95_high |   best_map5095_mean |   best_map5095_std |   best_map5095_ci95_low |   best_map5095_ci95_high |   normalized_aulc_map50_mean |   normalized_aulc_map50_std |   normalized_aulc_map50_ci95_low |   normalized_aulc_map50_ci95_high |   normalized_aulc_map5095_mean |   normalized_aulc_map5095_std |   normalized_aulc_map5095_ci95_low |   normalized_aulc_map5095_ci95_high |
|:-------------------------------------|------------------------:|-------------------:|------------------:|-----------------------:|------------------------:|---------------------:|--------------------:|-------------------------:|--------------------------:|------------------:|-----------------:|----------------------:|-----------------------:|--------------------:|-------------------:|------------------------:|-------------------------:|-----------------------------:|----------------------------:|---------------------------------:|----------------------------------:|-------------------------------:|------------------------------:|-----------------------------------:|------------------------------------:|
| GTFreeDatasetBalancedConsistency     |                       5 |           0.334138 |         0.0325366 |               0.308252 |                0.35768  |             0.147344 |          0.0197532  |                 0.131755 |                  0.16136  |          0.347521 |        0.0286374 |              0.323744 |               0.366134 |            0.152659 |         0.0151178  |                0.14014  |                 0.163236 |                     0.271759 |                   0.028975  |                         0.250323 |                          0.294161 |                       0.117527 |                     0.0156231 |                           0.105991 |                            0.130751 |
| GTFreeDatasetBalancedVisualDiversity |                       5 |           0.369815 |         0.0179673 |               0.357024 |                0.3853   |             0.161657 |          0.00913234 |                 0.154239 |                  0.168008 |          0.370844 |        0.0170954 |              0.359083 |               0.38564  |            0.163841 |         0.0105698  |                0.155625 |                 0.171794 |                     0.283657 |                   0.034924  |                         0.257732 |                          0.310519 |                       0.123027 |                     0.0156625 |                           0.11199  |                            0.135317 |
| GTFreeRandom                         |                       5 |           0.37785  |         0.022668  |               0.357406 |                0.389562 |             0.168967 |          0.0158801  |                 0.154695 |                  0.178143 |          0.387352 |        0.0027814 |              0.385246 |               0.389577 |            0.174608 |         0.00446162 |                0.171074 |                 0.178143 |                     0.285344 |                   0.0160234 |                         0.272407 |                          0.296854 |                       0.125729 |                     0.010078  |                           0.117979 |                            0.133542 |

## Paired acquisition-seed comparisons

| treatment                            | baseline                         | metric                  |   num_pairs |   mean_paired_difference |   std_paired_difference |   wins |   ties |   losses |   bootstrap_ci95_low |   bootstrap_ci95_high |   exact_sign_flip_pvalue |   relative_improvement_percent_mean |
|:-------------------------------------|:---------------------------------|:------------------------|------------:|-------------------------:|------------------------:|-------:|-------:|---------:|---------------------:|----------------------:|-------------------------:|------------------------------------:|
| GTFreeDatasetBalancedVisualDiversity | GTFreeRandom                     | normalized_aulc_map5095 |           5 |              -0.00270165 |              0.0186927  |      3 |      0 |        2 |          -0.0185478  |            0.0100499  |                   0.875  |                           -1.65841  |
| GTFreeDatasetBalancedVisualDiversity | GTFreeRandom                     | normalized_aulc_map50   |           5 |              -0.00168712 |              0.0375159  |      3 |      0 |        2 |          -0.0325316  |            0.0259943  |                   1      |                           -0.374068 |
| GTFreeDatasetBalancedVisualDiversity | GTFreeRandom                     | final_map5095           |           5 |              -0.0073104  |              0.0201608  |      1 |      0 |        4 |          -0.0219044  |            0.0091526  |                   0.4375 |                           -3.45272  |
| GTFreeDatasetBalancedVisualDiversity | GTFreeRandom                     | final_map50             |           5 |              -0.008035   |              0.0200334  |      2 |      0 |        3 |          -0.0234888  |            0.0075518  |                   0.4375 |                           -1.9498   |
| GTFreeDatasetBalancedVisualDiversity | GTFreeDatasetBalancedConsistency | normalized_aulc_map5095 |           5 |               0.00550023 |              0.00723976 |      4 |      0 |        1 |           0.00076435 |            0.0116401  |                   0.1875 |                            4.8856   |
| GTFreeDatasetBalancedVisualDiversity | GTFreeDatasetBalancedConsistency | normalized_aulc_map50   |           5 |               0.0118982  |              0.0186161  |      3 |      0 |        2 |          -0.00269282 |            0.0264891  |                   0.3125 |                            4.39985  |
| GTFreeDatasetBalancedVisualDiversity | GTFreeDatasetBalancedConsistency | final_map5095           |           5 |               0.0143126  |              0.0136021  |      5 |      0 |        0 |           0.0039246  |            0.0247006  |                   0.0625 |                           10.8256   |
| GTFreeDatasetBalancedVisualDiversity | GTFreeDatasetBalancedConsistency | final_map50             |           5 |               0.0356768  |              0.0350041  |      4 |      0 |        1 |           0.0070604  |            0.0615242  |                   0.1875 |                           11.4447   |
| GTFreeDatasetBalancedConsistency     | GTFreeRandom                     | normalized_aulc_map5095 |           5 |              -0.00820187 |              0.0196526  |      2 |      0 |        3 |          -0.0233496  |            0.00609197 |                   0.4375 |                           -5.94284  |
| GTFreeDatasetBalancedConsistency     | GTFreeRandom                     | normalized_aulc_map50   |           5 |              -0.0135853  |              0.0382796  |      2 |      0 |        3 |          -0.0449172  |            0.0152863  |                   0.5    |                           -4.33167  |
| GTFreeDatasetBalancedConsistency     | GTFreeRandom                     | final_map5095           |           5 |              -0.021623   |              0.0228303  |      0 |      0 |        5 |          -0.0419594  |           -0.006695   |                   0.0625 |                          -12.3146   |
| GTFreeDatasetBalancedConsistency     | GTFreeRandom                     | final_map50             |           5 |              -0.0437118  |              0.0281519  |      0 |      0 |        5 |          -0.0679138  |           -0.023085   |                   0.0625 |                          -11.5284   |

## Budget-to-target

|   acquisition_seed | strategy                             | metric   |   random_round4_target |   budget_to_reach_target | reached   | uses_cumulative_max_envelope   |
|-------------------:|:-------------------------------------|:---------|-----------------------:|-------------------------:|:----------|:-------------------------------|
|                 42 | GTFreeDatasetBalancedConsistency     | map50    |               0.337541 |                       30 | True      | True                           |
|                 42 | GTFreeDatasetBalancedVisualDiversity | map50    |               0.337541 |                       30 | True      | True                           |
|                 42 | GTFreeRandom                         | map50    |               0.337541 |                       25 | True      | True                           |
|                 42 | GTFreeDatasetBalancedConsistency     | map5095  |               0.141215 |                       30 | True      | True                           |
|                 42 | GTFreeDatasetBalancedVisualDiversity | map5095  |               0.141215 |                       30 | True      | True                           |
|                 42 | GTFreeRandom                         | map5095  |               0.141215 |                       25 | True      | True                           |
|                 43 | GTFreeDatasetBalancedConsistency     | map50    |               0.391134 |                      nan | False     | True                           |
|                 43 | GTFreeDatasetBalancedVisualDiversity | map50    |               0.391134 |                       35 | True      | True                           |
|                 43 | GTFreeRandom                         | map50    |               0.391134 |                       35 | True      | True                           |
|                 43 | GTFreeDatasetBalancedConsistency     | map5095  |               0.170621 |                      nan | False     | True                           |
|                 43 | GTFreeDatasetBalancedVisualDiversity | map5095  |               0.170621 |                       30 | True      | True                           |
|                 43 | GTFreeRandom                         | map5095  |               0.170621 |                       35 | True      | True                           |
|                 44 | GTFreeDatasetBalancedConsistency     | map50    |               0.384979 |                      nan | False     | True                           |
|                 44 | GTFreeDatasetBalancedVisualDiversity | map50    |               0.384979 |                      nan | False     | True                           |
|                 44 | GTFreeRandom                         | map50    |               0.384979 |                       35 | True      | True                           |
|                 44 | GTFreeDatasetBalancedConsistency     | map5095  |               0.178507 |                      nan | False     | True                           |
|                 44 | GTFreeDatasetBalancedVisualDiversity | map5095  |               0.178507 |                      nan | False     | True                           |
|                 44 | GTFreeRandom                         | map5095  |               0.178507 |                       35 | True      | True                           |
|                 45 | GTFreeDatasetBalancedConsistency     | map50    |               0.386166 |                      nan | False     | True                           |
|                 45 | GTFreeDatasetBalancedVisualDiversity | map50    |               0.386166 |                      nan | False     | True                           |
|                 45 | GTFreeRandom                         | map50    |               0.386166 |                       35 | True      | True                           |
|                 45 | GTFreeDatasetBalancedConsistency     | map5095  |               0.179207 |                      nan | False     | True                           |
|                 45 | GTFreeDatasetBalancedVisualDiversity | map5095  |               0.179207 |                      nan | False     | True                           |
|                 45 | GTFreeRandom                         | map5095  |               0.179207 |                       35 | True      | True                           |
|                 46 | GTFreeDatasetBalancedConsistency     | map50    |               0.38943  |                      nan | False     | True                           |
|                 46 | GTFreeDatasetBalancedVisualDiversity | map50    |               0.38943  |                      nan | False     | True                           |
|                 46 | GTFreeRandom                         | map50    |               0.38943  |                       35 | True      | True                           |
|                 46 | GTFreeDatasetBalancedConsistency     | map5095  |               0.175286 |                      nan | False     | True                           |
|                 46 | GTFreeDatasetBalancedVisualDiversity | map5095  |               0.175286 |                      nan | False     | True                           |
|                 46 | GTFreeRandom                         | map5095  |               0.175286 |                       35 | True      | True                           |

## Method-lock notes

- Visual-only normalized AULC mAP50-95 mean: 0.123027

Do not run final_test_v7 from this runner. Lock decision should be made from development full-curve criteria first.
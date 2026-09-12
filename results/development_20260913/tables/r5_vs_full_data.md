# r5_vs_full_data

Evidence: Development validation; Statistical unit: acquisition seed, n=10; A/B averaged within seed; Locked-final: not used; Full-data: n=2 learner reference, not n=10

| dataset | method | al_seeds | full_learners | R5_map50_95_pct | full_map50_95_pct | R5_minus_full_pp | percent_of_full_reference | R5_images | full_images | label_fraction_pct |
|---|---|---|---|---|---|---|---|---|---|---|
| GC10-DET | Random | 10 | 2 | 29.7565 | 34.717 | -4.9605 | 85.7116 | 561 | 1605 | 34.9533 |
| GC10-DET | DINO | 10 | 2 | 30.3322 | 34.717 | -4.3848 | 87.3699 | 561 | 1605 | 34.9533 |
| GC10-DET | Entropy | 10 | 2 | 30.34 | 34.717 | -4.37695 | 87.3925 | 561 | 1605 | 34.9533 |
| NEU-DET | Random | 10 | 2 | 38.508 | 42.8955 | -4.3875 | 89.7717 | 441 | 1259 | 35.0278 |
| NEU-DET | DINO | 10 | 2 | 38.8113 | 42.8955 | -4.0842 | 90.4787 | 441 | 1259 | 35.0278 |
| NEU-DET | Entropy | 10 | 2 | 39 | 42.8955 | -3.8955 | 90.9186 | 441 | 1259 | 35.0278 |

# Summer-based Objective 1 results

Completed job records: 126. Source: original summer repository commit `792ef355e8ba94da638c6b14b103aea0dabb0e07` with matched, documented O1 changes.

Scores below average seeds within each subject, then subjects. Fixed 33-class macro-F1. The three subjects were inspected previously; this study is exploratory.

| Condition | Jobs | Ideal F1 | Self F1 | Induced F1 |
|---|---:|---:|---:|---:|
| dynamic | 9/9 | 93.10% | 65.27% | 11.10% |
| dynamicwhar | 9/9 | 92.90% | 77.47% | 11.99% |
| global | 9/9 | 93.14% | 64.87% | 10.96% |
| none | 9/9 | 92.88% | 61.35% | 13.63% |
| physical | 9/9 | 92.12% | 63.83% | 10.79% |
| physical_drop | 9/9 | 92.48% | 66.89% | 11.20% |
| physical_gae | 9/9 | 92.55% | 66.70% | 11.13% |
| physical_rotate | 9/9 | 64.86% | 63.99% | 60.67% |
| residual | 9/9 | 92.70% | 65.18% | 11.31% |
| residual_balanced | 9/9 | 92.33% | 65.26% | 11.57% |
| residual_drop | 9/9 | 92.73% | 68.00% | 12.30% |
| residual_gae | 9/9 | 92.90% | 68.33% | 12.23% |
| residual_rotate | 9/9 | 68.56% | 66.66% | 66.55% |
| static_sum | 9/9 | 92.11% | 60.76% | 9.40% |

## Matched residual differences

- Residual minus static_sum: +1.91 points across 9 matched runs; positive in 7. Subjects, not overlapping windows, are generalization units.
- Residual minus physical: +0.52 points across 9 matched runs; positive in 6. Subjects, not overlapping windows, are generalization units.
- Residual minus none: -2.32 points across 9 matched runs; positive in 2. Subjects, not overlapping windows, are generalization units.

## Interpretation and next checks

A higher score alone is insufficient: inspect subject/seed variation, augmentation-only controls, ideal-placement cost and the recorded routing interventions in each result.json. The branch-balanced variant is an explicit alternative, not the authoritative raw mixture.

All models use the restored summer GIN-attention and gated temporal pipeline except the external DynamicWHAR comparator. The principal loss is cross-entropy by user selection. GAE ablations add masked-target reconstruction. The six-axis common input, corrected data exclusions, training-only normalization and source-only checkpoint selection differ from historical summer results; do not claim numerical reproduction of the historical score.

Exclude all-zero subject6_self, subject13_self and subject15_mutual4. Subject 15 therefore has induced recordings 5-7. Per-recording fixed/supported-class metrics and predictions are inside each job. No model choice for O2/O3 should be made solely from this aggregate table.

# Summer-based Objective 2 results

Exploratory REALDISP study: held-out placement subjects 2, 5 and 15 had been inspected before this design. Scores are fixed 33-class macro-F1, averaged over seeds within subject and then over subjects.

| Condition | Ideal | Self | Induced |
|---|---:|---:|---:|
| raw_physical | 92.12% | 63.83% | 10.79% |
| raw_residual | 92.70% | 65.18% | 11.31% |
| raw_none | 92.88% | 61.35% | 13.63% |
| raw_aug_physical | 64.86% | 63.99% | 60.67% |
| raw_aug_residual | 68.56% | 66.66% | 66.55% |
| mag_dot_none | 93.72% | 92.56% | 87.12% |
| mag_dot_physical | 92.88% | 92.23% | 85.99% |
| mag_dot_residual | 93.41% | 92.36% | 86.55% |
| magnitude_none | 93.20% | 92.27% | 85.84% |
| magnitude_physical | 93.47% | 91.40% | 83.91% |
| magnitude_residual | 93.34% | 91.51% | 84.77% |
| raw_aug_none | 80.05% | 78.86% | 74.78% |

## Paired recorded-displacement differences

- raw_residual minus raw_physical: +0.52 points; 6/9 positive pairs.
- raw_residual minus raw_none: -2.32 points; 2/9 positive pairs.
- raw_aug_residual minus raw_aug_physical: +5.87 points; 9/9 positive pairs.
- raw_aug_residual minus raw_aug_none: -8.23 points; 0/9 positive pairs.
- magnitude_residual minus magnitude_physical: +0.86 points; 7/9 positive pairs.
- magnitude_residual minus magnitude_none: -1.07 points; 2/9 positive pairs.
- mag_dot_residual minus mag_dot_physical: +0.56 points; 7/9 positive pairs.
- mag_dot_residual minus mag_dot_none: -0.57 points; 1/9 positive pairs.
- magnitude_physical minus raw_physical: +73.12 points; 9/9 positive pairs.
- mag_dot_physical minus raw_physical: +75.20 points; 9/9 positive pairs.
- magnitude_residual minus raw_residual: +73.46 points; 9/9 positive pairs.
- mag_dot_residual minus raw_residual: +75.24 points; 9/9 positive pairs.
- magnitude_none minus raw_none: +72.22 points; 9/9 positive pairs.
- mag_dot_none minus raw_none: +73.49 points; 9/9 positive pairs.

## Interpretation limits

Inspect per-recording fixed and supported-class metrics, prediction files, synthetic-rotation scores, routing interventions and latency before selecting a model. Activity coverage differs by recording; subject, not overlapping windows, is the generalization unit. Static and no-graph controls remain controls. A feature gain alone does not establish useful dynamic routing. No independent recorded placement cohort was used for confirmation.

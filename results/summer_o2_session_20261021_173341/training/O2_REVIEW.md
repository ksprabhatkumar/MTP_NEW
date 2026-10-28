# O2 result audit

All 63 new jobs have 31-epoch histories, both checkpoints, and ideal/self/induced and three rotated prediction files.
The outer supervisor and final analysis completed successfully.

## Source-only O3 representation choice

Magnitude residual: 93.54%; magnitude+dot residual: 94.18%.
The predeclared source-validation rule selects **mag_dot**. Held-out placement scores were not used to choose it.

## Recorded placement

| Condition | Ideal | Self | Induced | Synthetic rotated ideal | Batch-one median |
|---|---:|---:|---:|---:|---:|
| magnitude_physical | 93.47% | 91.40% | 83.91% | 93.47% | 2.03 ms |
| magnitude_residual | 93.34% | 91.51% | 84.77% | 93.34% | 2.31 ms |
| magnitude_none | 93.20% | 92.27% | 85.84% | 93.20% | 2.31 ms |
| mag_dot_physical | 92.88% | 92.23% | 85.99% | 92.88% | 2.12 ms |
| mag_dot_residual | 93.41% | 92.36% | 86.55% | 93.41% | 2.19 ms |
| mag_dot_none | 93.72% | 92.56% | 87.12% | 93.72% | 2.15 ms |
| raw_aug_none | 80.05% | 78.86% | 74.78% | 85.74% | 1.97 ms |

Paired induced differences, averaging seeds within subject and subjects equally:
- magnitude_residual_minus_magnitude_physical: +0.86 points; 7/9 positive pairs; S2 +1.11 points, S5 +1.37 points, S15 +0.09 points.
- magnitude_residual_minus_magnitude_none: -1.07 points; 2/9 positive pairs; S2 -0.39 points, S5 -1.53 points, S15 -1.31 points.
- mag_dot_residual_minus_mag_dot_physical: +0.56 points; 7/9 positive pairs; S2 -0.11 points, S5 +0.26 points, S15 +1.53 points.
- mag_dot_residual_minus_mag_dot_none: -0.57 points; 1/9 positive pairs; S2 -0.15 points, S5 -0.60 points, S15 -0.94 points.

## Dynamic routing evidence

- magnitude_residual: induced dynamic branch mass 5.66%; adjacency element SD 1.27e-09; uniform/shuffled intervention prediction flip rates 0.00%/0.00%.
- mag_dot_residual: induced dynamic branch mass 5.64%; adjacency element SD 8.47e-09; uniform/shuffled intervention prediction flip rates 0.00%/0.00%.

The invariant representation yields the large placement improvement. Residual routing has a small advantage over the physical graph but trails no graph for both feature families. The near-zero affinity variation and prediction interventions do not support a useful sample-conditioned routing claim. The previously inspected placement subjects make this exploratory, not confirmatory evidence.

Per-recording fixed-33-class results and subject breakdowns are in O2_REVIEW.json. A future placement claim needs untouched recorded data. O3 may test federation of the selected representation with residual, physical, and no-graph controls; it must not call the residual graph superior on these O2 results.

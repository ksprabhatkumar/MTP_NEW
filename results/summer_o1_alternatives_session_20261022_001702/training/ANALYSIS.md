# O1 alternative graph full analysis

The summer backbone and magnitude+dot input are fixed across all arms. Checkpoints use source ideal validation only. Recorded displacement subjects were previously inspected, so full placement results are exploratory.

| Arm | Ideal | Self | Induced | Latency | Dynamic mass |
|---|---:|---:|---:|---:|---:|
| physical | 92.88% | 92.23% | 85.99% | 2.47 ms | 0.00% |
| residual | 93.41% | 92.36% | 86.55% | 2.15 ms | 5.64% |
| none | 93.72% | 92.56% | 87.12% | 1.91 ms | 0.00% |
| masked_balanced | 93.42% | 92.06% | 86.58% | 2.31 ms | 13.37% |
| masked_floor | 93.42% | 92.37% | 86.47% | 2.60 ms | 34.95% |
| pairwise_balanced | 93.40% | 92.03% | 86.58% | 2.21 ms | 13.37% |
| pairwise_floor | 93.36% | 92.42% | 86.46% | 2.35 ms | 34.96% |
| gated_masked | 93.68% | 92.66% | 86.76% | 2.49 ms | 4.05% |
| gated_pairwise | 93.68% | 92.79% | 86.95% | 2.45 ms | 3.30% |

## Paired induced differences

- masked_balanced_minus_physical: +0.59 points; 8/9 positive pairs.
- masked_balanced_minus_residual: +0.03 points; 4/9 positive pairs.
- masked_balanced_minus_none: -0.54 points; 3/9 positive pairs.
- masked_floor_minus_physical: +0.47 points; 7/9 positive pairs.
- masked_floor_minus_residual: -0.08 points; 3/9 positive pairs.
- masked_floor_minus_none: -0.65 points; 1/9 positive pairs.
- pairwise_balanced_minus_physical: +0.59 points; 8/9 positive pairs.
- pairwise_balanced_minus_residual: +0.03 points; 4/9 positive pairs.
- pairwise_balanced_minus_none: -0.54 points; 3/9 positive pairs.
- pairwise_floor_minus_physical: +0.47 points; 7/9 positive pairs.
- pairwise_floor_minus_residual: -0.09 points; 4/9 positive pairs.
- pairwise_floor_minus_none: -0.65 points; 1/9 positive pairs.
- gated_masked_minus_physical: +0.77 points; 8/9 positive pairs.
- gated_masked_minus_residual: +0.21 points; 8/9 positive pairs.
- gated_masked_minus_none: -0.35 points; 2/9 positive pairs.
- gated_pairwise_minus_physical: +0.95 points; 7/9 positive pairs.
- gated_pairwise_minus_residual: +0.40 points; 8/9 positive pairs.
- gated_pairwise_minus_none: -0.17 points; 4/9 positive pairs.

Use the saved job-level routing interventions and per-recording metrics before making any graph claim. A larger adjacency variation by itself does not establish useful sample-conditioned routing.

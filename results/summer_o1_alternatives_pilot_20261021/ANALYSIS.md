# O1 alternative graph pilot analysis

The summer backbone and magnitude+dot input are fixed across all arms. Checkpoints use source ideal validation only. Recorded displacement subjects were previously inspected, so full placement results are exploratory.

| Arm | Source ideal | Source self | Adjacency SD | Uniform flip | Shuffled flip |
|---|---:|---:|---:|---:|---:|
| physical | 92.02% | 89.77% | 0 | 0.00% | 0.00% |
| residual | 92.63% | 90.82% | 1.55e-06 | 0.00% | 0.00% |
| none | 93.40% | 92.44% | 0 | 0.00% | 0.00% |
| masked_balanced | 92.09% | 90.41% | 3.3e-06 | 0.00% | 0.00% |
| masked_floor | 92.42% | 90.90% | 1.92e-05 | 0.00% | 0.00% |
| pairwise_balanced | 92.09% | 90.35% | 5.06e-05 | 0.00% | 0.00% |
| pairwise_floor | 92.38% | 90.77% | 0.00297 | 0.59% | 0.10% |

A pilot establishes learning and routing activity, not placement superiority. Review these source-only results before starting full training.

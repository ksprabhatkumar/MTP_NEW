# Source-only query/key decay experiment — 31 epochs

Eight matched jobs completed on 21 October 2026: four `mag_dot` arms × held-out target subjects 2 and 5, seed 13. Checkpoint selection used distinct source ideal validation only; no held-out target placement scores were read. The 8-epoch screen is in `../summer_o2_affinity_repair_20261021/`.

| Arm | Mean best source ideal macro-F1 | Mean source self macro-F1 at best checkpoint | Best-checkpoint query norm | Mean donor-affinity centered-logit RMS change |
|---|---:|---:|---:|---:|
| Residual, standard Adam decay | 94.40% | 92.51% | 0.0221 | Approximately zero |
| Residual, zero query/key decay | 94.37% | 92.46% | 2.6337 | 0.000077 |
| Physical, standard Adam decay | **94.50%** | 91.94% | 0.0001 (unused affinity) | Zero |
| No graph, standard Adam decay | 93.74% | **94.36%** | 2.4865 (unused affinity) | Zero |

Per split, source ideal F1: S2 standard residual 93.49%, repaired residual 93.45%, physical 93.33%, no graph 92.74%; S5 standard/repaired residual 95.30% each, physical 95.66%, no graph 94.73%. The repaired query norm at the last epoch remained about 2.63 in both splits, while standard residual shrank effectively to zero. In the repaired best checkpoints, mean between-window adjacency standard deviation was only 2.26e-7, and donor-affinity effects on logits were tiny. A forced substantially different final adjacency changed centered logits by about 1.1 RMS, confirming the message path was capable of affecting the classifier.

**Conclusion:** coupled Adam weight decay causes much of the query/key shrinkage, but removing it alone does not create useful sample-conditioned routing or improve source performance here. The source ideal/self ordering is mixed, and two validation splits with one seed cannot settle recorded-placement robustness. Keep the original 63-job O2 matrix as an exploratory representation and graph-ablation study, including no graph and raw-augmentation/no-graph. The repaired arm is **not** silently substituted into that matrix. A later O1 method study would need a separate graph mechanism change and matched controls.

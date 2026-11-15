# T08 source-only candidate decision

Promotion criteria passed: False.
Candidate: gated_neighbor. Do not promote as a validated improvement; retain as a provisional mechanism experiment for workstation review.

No recorded-displacement test was evaluated. O1 superiority remains unresolved.

| Arm | Source ideal F1 % | Source self F1 % |
|---|---:|---:|
| gated_neighbor | 93.8066 | 92.5689 |
| local_control | 93.8509 | 92.8400 |
| none | 93.8024 | 92.3278 |

All twelve jobs, checkpoint recovery, prediction metrics and plot regeneration are audited. Selection is based on the frozen source criteria, not a test score. Four fold/seed pairs are not four independent subjects. Compare graph interventions in selection_decision.json before attributing any benefit to neighbor information. Historical eight-epoch corrections are not a matched comparator to this sixteen-epoch experiment.

Distant-window donor mean effects (percentage points): {'val_ideal': -0.921144233788293, 'val_self': -0.42662282859334266}. Receiver local features and adjacency were retained; donors were offset by half the complete validation set. This demonstrates some useful message dependence, not superiority over a separately trained no-graph model.

Next research decision: test whether small initial gates slow neighbor learning, or whether matched source-side sensor perturbations are needed to teach rejection. Freeze that experiment before training; preserve identical perturbation and optimization exposure in controls.

Gate initialization is an optimization variable: delta=g W_o phi(m), with W_o=0. Under plain SGD the first effective correction update scales approximately as g squared; Adam changes that scaling through moment normalization. Consult config.json for this run's gate bias and the frozen experiment specification before proposing another ablation.

# T08 source-only candidate decision

Promotion criteria passed: False.
Candidate: gated_neighbor. Do not promote as a validated improvement; retain as a provisional mechanism experiment for workstation review.

No recorded-displacement test was evaluated. O1 superiority remains unresolved.

| Arm | Source ideal F1 % | Source self F1 % |
|---|---:|---:|
| gated_neighbor | 93.8185 | 92.2214 |
| local_control | 94.0051 | 92.5451 |
| none | 93.8024 | 92.3278 |

All twelve jobs, checkpoint recovery, prediction metrics and plot regeneration are audited. Selection is based on the frozen source criteria, not a test score. Four fold/seed pairs are not four independent subjects. Compare graph interventions in selection_decision.json before attributing any benefit to neighbor information. Historical eight-epoch corrections are not a matched comparator to this sixteen-epoch experiment.

Distant-window donor check: mean F1 effect -0.1879 pp ideal and -0.1708 pp self. Donors were offset by half the entire source-validation set (all labels differ in these data); receiver local features/adjacency were preserved. This demonstrates some useful neighbor dependence, not superiority over a separately trained no-graph model.

T08 decision: do not promote this operator as the winning graph. Retain it as the provisional graph arm for matched temporal/mechanism experiments; O1 superiority remains OPEN. No held-out test scores were consulted. Before committing large GPU time, the next research decision should isolate whether the initially small gate slows neighbor learning and whether source-side sensor perturbation is needed to teach rejection. Keep the same perturbations and optimization exposure in no-graph/local controls. Do not assume either change will win.

At initialization delta=g W_o phi(m), W_o=0 and g about0.119. Under plain SGD the first effective correction update scales approximately with g squared; Adam changes that scaling through its moment normalization. Thus the small gate is a plausible optimization confound, not a proven explanation. A future open-gate ablation should be specified and smoke-tested separately, without overwriting this run.

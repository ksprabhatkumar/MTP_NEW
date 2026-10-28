# Summer O2 preparatory gate report — 21 October 2026

The requested preparatory checks are complete. **The 63-job full O2 run has not started.** Execution readiness and scientific promise are separate: the existing session JSON passes its schema check, but the original residual graph did not beat no graph in two short source-only pilots.

| Gate | Status | Evidence |
|---|---|---|
| O1 analysis-only repair | Passed | 126/126 O1 jobs verified; final `placement.png` generated; outer state `complete`. Original failure preserved in `../summer_o1_session_20261017_232945/status_before_analysis_repair.json`; artifact hashes/command in `analysis_repair.json`. No model retraining. |
| Saved-checkpoint graph-path audit | Completed with failure mechanism | `../summer_o1_routing_audit_20261021.json` (S2/S5, seed 13) and `../summer_o1_routing_audit_s15_seed13_20261021.json`. FP32 forward/adjacency reconstruction matched the saved model. Dynamic affinity interventions barely changed centered logits; forcing a different consumed adjacency did. |
| CPU invariant features | Passed | `../summer_o2_cpu_recheck_20261021.json`, all six feature/graph cases. |
| Real-fold data/rotation | Passed | `../summer_o2_data_rotation_audit_20261021.json`; 76,268 retained windows, corrected exclusions, disjoint source/validation/test subjects, proper rotations, physical-before-normalization feature path and first-lag check. Maximum measured feature difference 5.72e-6 and logit difference 7.60e-7 in the tested batch. |
| Real 8-epoch source-only pilots | Completed | `../summer_o2_source_pilot_20261021/`: 14/14 jobs, seven arms across held-out S2/S5 with source validation only, seed 13, full training epochs, best checkpoints and timing. No held-out placement scores were used. |
| O2 GPU preflight | Passed | `preflight/passed.json` and fold manifest. |
| Seven-arm two-batch smoke | Passed | `smoke/passed.json` and seven short job records. This is an execution test, not a performance pilot. |
| Raw/physical bridge | Passed exactly | `bridge/bridge_review.json`: same subject/seed/split/31 epochs/best epoch; maximum weight and validation-F1 difference both 0.0. Bridge job was validation-only. |
| Existing supervisor schema | Passed | `run_long_training.py --plan summer_o2_session.json --check` reports ready, no schema blockers. This command does not test the new scientific gates. |

## Routing diagnosis

In eight source-validation checkpoint audits across S2/S5, seed 13, residual query norms were 0.00558 and 0.00125, with affinity score magnitudes around 1e-9 and 1e-11. Across all nine completed residual best checkpoints, the query-norm median was approximately 0.00558 (range 0.00005–1.418). The high-norm S15/seed13 checkpoint still had only 8.51e-6 mean between-window adjacency standard deviation and a donor-affinity centered-logit RMS effect of 4.97e-5 on the tested source windows. In contrast, a forced uniform final adjacency produced roughly 0.77–0.80 centered-logit RMS change in two sampled residual models. The graph computation reaches the classifier; the learned *sample dependence* is nearly absent in these checks. Adam with coupled weight decay is a plausible cause of query/key shrinkage, but this audit does not isolate that cause. A no-decay query/key experiment requires a matched source-side test.

## Source-only pilot scores

Best source-ideal validation fixed-33-class macro-F1, mean of the two held-out-target folds' distinct source-validation subjects, one seed and eight epochs. These are screening values, not converged full-run or placement results.

| Condition | Mean best source ideal F1 | Mean source self F1 at best checkpoint | Mean wall time/job |
|---|---:|---:|---:|
| magnitude + physical | 88.61% | 86.46% | 31.6 s |
| magnitude + residual | 89.27% | 86.47% | 29.8 s |
| magnitude + no graph | **90.99%** | **89.40%** | 24.9 s |
| magnitude+dot + physical | 92.02% | 89.77% | 30.4 s |
| magnitude+dot + residual | 92.63% | 90.82% | 31.1 s |
| magnitude+dot + no graph | **93.40%** | **92.44%** | 25.3 s |
| raw axes + rotation augmentation + no graph | 53.46% | 51.39% | 28.3 s |

Magnitude+dot residual exceeded physical on each pilot split (S2: 92.02% versus 91.43%; S5: 93.23% versus 92.62%), but magnitude+dot no graph was higher on both (92.19% and 94.60%). The raw-augmentation arm learns more slowly; its eight-epoch result cannot be compared as a converged 31-epoch outcome. Residual query norms in the two eight-epoch magnitude+dot best checkpoints averaged about 1.09, while long-run raw residual checkpoints often shrank much further. Pilot peak PyTorch allocated memory was at most about 0.88 GiB; that does not include all driver/process memory.

## Time calibration and next decision

The raw/physical 31-epoch bridge took 118.9 seconds in its result record. Historical O1 mean full-job times were 151.9 seconds for physical, 157.9 for residual, and 119.5 for no graph across nine jobs each; fold and feature width matter. These measurements keep **2.5–3.5 hours for the 63-job full O2 matrix** a reasonable provisional range, excluding preparation and final analysis. Recalibrate after any optimizer/model repair because it changes the source fingerprint and may change runtime.

The strict scientific go decision for an improved dynamic-routing claim is **not met** by the current residual pilot. Before launching 63 jobs as a putative graph improvement, test one small, matched routing repair on source validation: first remove coupled weight decay from query/key/affinity parameters while retaining all other recipe settings; measure whether affinity variation and intervention effects return, and compare source F1 to fixed/no-graph controls. If the repair fails, the original 63-job matrix can still serve as a documented representation/ablation study, but it should not be presented as evidence that dynamic routing is better. No target-placement result should select the repair.

Open application decisions: tolerable ideal-F1 drop, batch-one latency limit, and availability of new untouched placement recordings. Existing subjects 2/5/15 have been inspected before, so later placement evaluation remains exploratory.

## Follow-up repair gate, 21 October

The isolated query/key zero-decay repair was tested for eight and then 31 full epochs with matched source-only controls. Its 31-epoch result is in `../summer_o2_affinity_repair_31e_20261021/REPAIR_REPORT.md`. It preserved query/key norms but did not produce a meaningful donor-affinity logit effect or improve source F1 over standard residual. The proposed simple repair is therefore **not** promoted. The original 63-job O2 matrix remains justified as a matched representation/placement comparison, not as a prevalidated graph-superiority run. Its results must retain the no-graph baseline and any negative finding.

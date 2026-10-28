# Entire project plan - summer pipeline and residual dynamic graph

Updated: 17 October 2026. This is the authoritative execution checklist for the
new work requested in this session. Older run completion is evidence of executed
experiments, not completion of the corrected summer-based O1 objective.

**Research target:** retain the summer-colloquium pipeline and evaluate whether
replacing its static topology with residual dynamic routing improves recorded
placement robustness. The intended model remains graph-based. No graph remains a
required ablation, not the chosen replacement project. Superiority is a hypothesis
to establish with matched evidence, not an outcome to impose on the results.

**User-confirmed sources:** `resources/colloquium_report.pdf` and
`resources/objectives_methodology.tex`. Other reports and `context.text` provide
supporting evidence. Conflicting recipes must be resolved explicitly.

Legend: [x] verified complete; [ ] not complete. BLOCKED means a specified input is
missing, not that the experiment failed. `current_step_update.txt` records the
active step and its evidence. `update.txt` remains the historical status record.

## P0. Recover the intended baseline and freeze the protocol

- [x] P0.1 Reread summer architecture, loss, protocol and hyperparameter tables;
  reread the authoritative O1 adjacency equations and decision rule.
- [x] P0.2 Audit current code against the summer pipeline. Confirmed differences:
  GIN multiplicative attention removed; decoder absent in principal runs;
  different loss/optimizer/capacity; affinity on raw rather than encoded inputs.
  See `SUMMER_O1_AUDIT.md`.
- [x] P0.3 Identify partial summer source in `context.text`, including GIN
  aggregation/attention, GIN decoder, temporal block and pooling/head. It omits
  batching helpers and contains historical training variants.
- [x] P0.4 Recover original source from the user-provided local repository
  `C:\Users\General\Desktop\prabhat\mamba_GT_WHAR\Mamba_WHAR_Phase1`, commit
  `792ef355e8ba94da638c6b14b103aea0dabb0e07`; archive its hashes and source.
- [x] P0.5 Resolve recipe using actual recovered runtime source, not conflicting
  PDF tables: hidden 16, dropout .5, Adam 1e-4, weight decay .0005, batch 64,
  OneCycle. User explicitly selected cross-entropy main with GAE ablation.
  Requested worse-accuracy ordering is not identifiable from the two tables;
  no ranking is invented and all controls use identical settings.
- [x] P0.6 Freeze `summer_o1_session.json`: distinct source ideal validation,
  14 conditions x 3 subjects x 3 seeds = 126 jobs; runtime-calibrated common
  20-50 epochs targeting 5.5 hours. Report paired differences, clean accuracy,
  cost and subject/seed variation; no automatic superiority flag is assigned.
  Scientific acceptance remains part of P4.7, not a guaranteed positive result.

## P1. Existing environment, data and historical evidence

- [x] P1.1 Create/use `prabhat` CUDA environment; GPU and dependency checks passed.
- [ ] P1.2 Remove unused old `pr` environment. Prior automatic sandbox approval
  rejected deletion. This cleanup is not a training dependency.
- [x] P1.3 Audit 46 REALDISP recordings; cache 81,666 six-axis windows and record
  subject/scenario/source identity. Training-only normalization is implemented.
- [x] P1.4 Identify and exclude the three wholly zero-sensor files in the corrected
  data view, retaining 76,268 windows. Preserve original artifacts.
- [x] P1.5 Execute historical workstation O1 (118/118 jobs) and analyze its negative
  result. This used the modified pipeline, not a certified summer reproduction.
- [x] P1.6 Execute all 72 primary O2 feature jobs on the modified pipeline. These
  show a representation benefit, not established dynamic-routing superiority.
- [x] P1.7 Verify existing O2 complete: 128/128 jobs, finished 17 October at
  23:13:50 IST; two PNG figures exist. It remains modified-pipeline evidence.

## P2. Restore the summer baseline faithfully

- [x] P2.1 Archive original source or label reconstructed source honestly; save
  hashes and provenance. Keep prior O1/O2 code and outputs intact.
- [x] P2.2 Restore the summer encoder's GIN aggregation multiplied by its parallel
  tanh GIN attention, learned self coefficients, auxiliary GIN decoder, existing
  gated temporal block, flattening, global average pooling, LayerNorm and head.
- [x] P2.3 Implement dense weighted GIN equivalent to the original static
  edge-list sum. Check output and gradient equivalence on a small graph; retain
  original static-sum and row-normalized-static controls separately.
- [x] P2.4 Retain recovered capacity/optimizer/scheduler and user-selected CE
  main/GAE-ablation policy. Explicitly correct train-only normalization and
  source validation; the old test-based checkpoint choice is not reproduced.
- [x] P2.5 Verify dimensions, reconstruction targets, parameter counts, causality,
  finite loss and backward pass. The local temporal source has no selective scan;
  do not silently replace it with a different temporal backbone.
- [x] P2.6 Run six-epoch source-only pilots: 91.02% static-sum, 91.12%
  normalized-static, 91.11% residual best validation F1. These are readiness
  checks, not reproduced historical or held-out-placement scores.

## P3. Implement Objective 1 as a controlled spatial change

- [x] P3.1 Build per-IMU encoded features H for query/key affinity rather than
  computing affinity directly from six raw channels. Record the encoder and
  retain it consistently in matched conditions.
- [x] P3.2 Implement the authoritative graph contract:
  C = row_softmax((Hhat Wq)(Hhat Wk)^T / (tau sqrt(d_a)));
  S_ij = m_j (1 - indicator(i=j)) [alpha A_phys + beta A_global + gamma C]_ij;
  A_ij = S_ij / max(epsilon, sum_k S_ik).
  Use nonnegative branches/gates, gates summing to one and initially favoring
  anatomy. Preserve a single explicit self contribution in message passing.
  The primary formula removes self-edges; do not use the conflicting older
  report's extra identity-loop formulation without a separately labelled study.
- [x] P3.3 Connect weighted adjacency to the restored summer spatial encoder and
  decoder. Keep temporal block/head/loss/training recipe matched. If the exact
  methodology message operator replaces GIN, include a static bridge control so
  a GIN-to-message-layer change is not confused with dynamic topology.
- [x] P3.4 Test sender and receiver masking, empty neighborhoods, all-missing
  abstention, zero unavailable input, finite gradients and sensitivity to input.
- [x] P3.5 Implement effective branch mass logging after final normalization. Diagnose
  degree/scale imbalance; any branch-wise normalization alternative is a named
  ablation, not a silent rewrite of the requested formula.
- [x] P3.6 Add uniform/shuffled-dynamic inference interventions, attention
  variability and gradient measurements. Heatmaps alone do not show usefulness.

## P4. Matched O1 pilots and long GPU training

- [x] P4.1 Prepare one sequential Python entry point with an explicit config,
  smoke mode, validation pilot, speed calibration, frozen full schedule, atomic
  checkpoints, optimizer/scheduler/RNG resume, source hashes and OS run lock.
- [x] P4.2 Include restored static-sum, normalized-static, global-only,
  dynamic-only, residual-mixture and no-graph controls; use identical temporal
  architecture, capacity recipe, data and training budget. Keep residual dynamic
  routing as the target method. Include matched augmentation and GAE ablations
  before attributing a benefit to topology. Keep DynamicWHAR as external control.
- [x] P4.3 Validate every scheduled fold's class coverage before launching any job, rather
  than discovering an invalid validation subject hours into training.
- [x] P4.4 Pass unit tests, short real-data GPU smoke tests, output/resume checks
  and source-validation pilots. Inspect learning and routing diagnostics.
- [ ] P4.5 Freeze configuration from source-side evidence; target about 5-6 hours
  by measured calibration. Run sequentially in a separate background process
  only after source/recipe questions and smoke failures are resolved.
- [ ] P4.6 Save all epoch metrics, clean/self/induced predictions, per-recording
  class coverage, reconstruction loss, branch diagnostics, batch-one cost,
  telemetry and handoff/resume commands in an isolated result directory.
- [ ] P4.7 Confirm all scheduled jobs completed and write matched subject/seed
  comparisons. Claim a graph benefit only if it beats static, augmentation and
  no-graph controls beyond observed variation at acceptable clean accuracy/cost.

## P5. Objective 2 on the confirmed summer-based pipeline

- [ ] P5.1 Reassess earlier O2 evidence after P4. It used the modified pipeline and
  only no-graph/residual fusion, so it does not complete the intended fixed-versus-
  dynamic integration study on the restored summer model.
- [ ] P5.2 Cross raw, magnitudes, magnitude/dot features and raw-axis augmentation
  with the locked static and residual graph choices, keeping training matched.
- [ ] P5.3 Verify feature, adjacency and prediction invariance. If reconstruction
  is retained, use invariant targets or a separately declared orientation path;
  do not require exact raw axes from an invariant code alone.
- [ ] P5.4 Report recorded displacement separately from synthetic rotation,
  missing sensors, class coverage, clean accuracy and cost. Lock final model
  based on the declared criteria, retaining negative results.

## P6. Objective 3 and final thesis evidence

- [ ] P6.1 After O1/O2 are resolved, implement subject-client federation with
  centralized, local-only, FedAvg and client-local-normalization controls.
- [ ] P6.2 Separate unseen-user performance from known-client personalization;
  log client variation, rounds, bytes, local training time and inference cost.
- [ ] P6.3 Smoke-test, train and analyze the federated study with the same data
  exclusions and provenance. No on-device or privacy guarantee is implied.
- [ ] P6.4 Produce final plots/tables/report with corrected historical claims,
  implementation details, limitations and independent-placement validation needs.

## Current execution

**P4.5 is next:** the original source, loss policy, source-backed runtime recipe,
model parity, formula tests, all-condition GPU smoke and source pilots are ready.
Launch the versioned sequential session, calibrate the common epoch budget, and
record the background PID/output path. Keep P4.5 unchecked until all training
jobs have finished; P4.6/P4.7 require final artifact/analysis verification.

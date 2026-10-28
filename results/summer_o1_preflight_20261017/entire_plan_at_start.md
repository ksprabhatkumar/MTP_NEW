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
- [ ] P0.4 Resolve original-source provenance: user to provide original source,
  or explicitly choose reconstruction from PDF + context excerpts. Question sent.
- [ ] P0.5 Resolve summer REALDISP recipe conflict: Table 5.3 versus 5.4, or a
  source-validation pilot comparing both. Also resolve summer focal+MSE versus
  the methodology's explicit principal cross-entropy/GAE-ablation instruction.
  Questions sent. Do not silently select.
- [ ] P0.6 Freeze a versioned experiment configuration, acceptance tolerance,
  source-validation rule, graph variants, seeds and training budget. Record every
  difference between historical reproduction and the controlled O1 study.

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
- [ ] P1.7 Verify final completion of the existing O2 supplementary run and plots.
  It was still running when this plan was created. Do not alter its source files
  or compete with it for the GPU. Do not advance to O3 based on its scores alone.

## P2. Restore the summer baseline faithfully

- [ ] P2.1 Archive original source or label reconstructed source honestly; save
  hashes and provenance. Keep prior O1/O2 code and outputs intact.
- [ ] P2.2 Restore the summer encoder's GIN aggregation multiplied by its parallel
  tanh GIN attention, learned self coefficients, auxiliary GIN decoder, existing
  gated temporal block, flattening, global average pooling, LayerNorm and head.
- [ ] P2.3 Implement dense weighted GIN equivalent to the original static
  edge-list sum. Check output and gradient equivalence on a small graph; retain
  original static-sum and row-normalized-static controls separately.
- [ ] P2.4 Restore the selected summer recipe for the baseline audit; use the
  user-selected loss policy consistently across the main matched graph variants.
  Separate an exact historical recipe from corrections
  needed for leakage-free normalization and source validation.
- [ ] P2.5 Verify dimensions, reconstruction targets, parameter counts, causality,
  finite loss and backward pass. The local temporal source has no selective scan;
  do not silently replace it with a different temporal backbone.
- [ ] P2.6 Run an ideal-placement baseline pilot and log the comparison with the
  summer report. Differences in channels/splits/normalization must be explicit.

## P3. Implement Objective 1 as a controlled spatial change

- [ ] P3.1 Build per-IMU encoded features H for query/key affinity rather than
  computing affinity directly from six raw channels. Record the encoder and
  retain it consistently in matched conditions.
- [ ] P3.2 Implement the authoritative graph contract:
  C = row_softmax((Hhat Wq)(Hhat Wk)^T / (tau sqrt(d_a)));
  S_ij = m_j (1 - indicator(i=j)) [alpha A_phys + beta A_global + gamma C]_ij;
  A_ij = S_ij / max(epsilon, sum_k S_ik).
  Use nonnegative branches/gates, gates summing to one and initially favoring
  anatomy. Preserve a single explicit self contribution in message passing.
  The primary formula removes self-edges; do not use the conflicting older
  report's extra identity-loop formulation without a separately labelled study.
- [ ] P3.3 Connect weighted adjacency to the restored summer spatial encoder and
  decoder. Keep temporal block/head/loss/training recipe matched. If the exact
  methodology message operator replaces GIN, include a static bridge control so
  a GIN-to-message-layer change is not confused with dynamic topology.
- [ ] P3.4 Test sender and receiver masking, empty neighborhoods, all-missing
  abstention, zero unavailable input, finite gradients and sensitivity to input.
- [ ] P3.5 Report effective branch mass after final normalization. Diagnose
  degree/scale imbalance; any branch-wise normalization alternative is a named
  ablation, not a silent rewrite of the requested formula.
- [ ] P3.6 Add uniform/frozen/shuffled-dynamic inference interventions, attention
  variability and gradient measurements. Heatmaps alone do not show usefulness.

## P4. Matched O1 pilots and long GPU training

- [ ] P4.1 Prepare one sequential Python entry point with an explicit config,
  smoke mode, validation pilot, speed calibration, frozen full schedule, atomic
  checkpoints, optimizer/scheduler/RNG resume, source hashes and OS run lock.
- [ ] P4.2 Include restored static-sum, normalized-static, global-only,
  dynamic-only, residual-mixture and no-graph controls; use identical temporal
  architecture, capacity recipe, data and training budget. Keep residual dynamic
  routing as the target method. Include matched augmentation and GAE ablations
  before attributing a benefit to topology. Keep DynamicWHAR as external control.
- [ ] P4.3 Validate every fold's class coverage before launching any job, rather
  than discovering an invalid validation subject hours into training.
- [ ] P4.4 Pass unit tests, short real-data GPU smoke tests, output/resume checks
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

**P0.4-P0.6 are active:** original source/reconstruction and summer recipe choices
are awaiting user clarification. Audit, planning and runner preparation can proceed;
an unverified baseline must not be called the summer reproduction or launched as
the final O1 experiment. The existing O2 job is finishing separately.

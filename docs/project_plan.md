# Project plan

Updated: 27 October 2026 (Asia/Calcutta). Consolidated human-readable roadmap. Completed evidence is in `project_status.md`; operational cautions are in `project_handoff.md`. Existing executable configurations and historical result snapshots remain unchanged.

## Research direction

Retain the summer-colloquium research lineage and pursue a graph model that improves placement-robust activity recognition over fair no-graph controls. The three objectives remain O1 graph routing, O2 rotation-stable representation and O3 subject-client federation. A graph advantage is the target hypothesis, not a result to assume.

Authoritative research sources: `resources/objectives_methodology.tex` and `resources/colloquium_report.pdf`. Keep original resources and `context.text` as source material. The PDF's selective-SSM description differs from the recovered temporal implementation; distinguish them explicitly.

## Completed foundation

- [x] Audit data and exclude wholly zero-signal recordings in the corrected view.
- [x] Restore the summer encoder and verify historical forward/gradient parity.
- [x] Correct train-only normalization and source-validation checkpoint selection.
- [x] Complete restored O1: 126 jobs and analysis.
- [x] Complete restored O2: 63 jobs and analysis; select `mag_dot` from source validation.
- [x] Complete O1 alternatives: 81 jobs and analysis on 22 October.
- [x] Implement O3 framework and run preparatory checks/smoke/centralized bridge.
- [ ] Establish beneficial sample-conditioned routing beyond no graph.
- [ ] Implement and evaluate the report's genuine selective-SSM temporal proposal.
- [ ] Complete the full O3 study and final reporting.

## Recommended next sequence (proposed, not launched)

### A. Diagnose the graph and recording failure

- [ ] Extend source-only audits to the latest alternatives: actual consumed messages, constant gates, frozen source-average graphs, true cross-window substitutions, centered logits and loss.
- [ ] Report relative branch shares separately from total gated neighbor magnitude.
- [ ] Inspect subject 15 mutual7 signals, labels/timestamps and class confusions. Use the test failure descriptively; select changes using source-side evidence.
- [ ] Measure harmful mixing versus redundancy; do not infer causation from heatmaps or zero label flips alone.

### B. Small message-operator experiment on the existing temporal backbone

- [ ] Preserve the current no-neighbor local encoder and add a separately learned neighbor correction in latent feature space.
- [ ] Compare no graph, a capacity control with extra local processing, and the new message operator under equal training budgets.
- [ ] If justified, add receiver-sender message gates that can reject all neighbors, then richer relational message values. Current pairwise alternatives changed scores, not the message-value function.
- [ ] Keep `mag_dot`, splits, validation rule and temporal model fixed during the first comparison. CE remains main loss; GAE stays separate.
- [ ] Use source-only pilot evidence to choose a small full comparison; preserve negative outcomes. Do not require the graph to win before moving to temporal validation.

### C. Genuine Mamba/SSM validation before freezing O3

The restored summer pipeline is already in use. Its temporal implementation lacks selective state recurrence. To fulfill the report's selective-SSM proposal, add a separately named implementation rather than relabeling old checkpoints or silently replacing their code.

- [ ] Select and document a genuine causal selective-SSM implementation and dependency/platform requirements. Check feasibility on the current Windows/GPU environment before promising a kernel or speed.
- [ ] Verify state-update behavior, causality, finite gradients, shapes and numerical/reference agreement as applicable. Calling a linear projection `ssm_proj` is not sufficient.
- [ ] Compare a small factorial matrix: existing temporal block versus genuine Mamba, each with no graph, physical graph and the nominated graph candidate. Six conditions isolate graph, temporal and interaction effects; do not change both without controls.
- [ ] Match input/splits/selection/training exposure and report parameter, memory and batch-one latency differences. Include local capacity controls where needed.
- [ ] If warm-starting from a no-graph checkpoint, give controls equivalent pretraining and additional updates; preserve total compute accounting.
- [ ] Freeze the temporal choice using source validation and declared accuracy/cost criteria. A graph win only against an older temporal baseline is not proof of a graph contribution.

This is the recommended point for the report's genuine Mamba experiment: after the focused graph diagnostic/pilot, before full O3. It is not necessary to wait for O1 to become positive. Implementation and training remain future work, not part of this documentation cleanup.

### D. Federation and thesis evidence

- [ ] Reconcile selected model/representation with O3 configs; rerun architecture bridge and smoke after any temporal change.
- [ ] Fix stale workstation paths and calibrate memory/runtime without editing historical snapshots.
- [ ] Execute centralized, local-only, FedAvg and client-local normalization comparisons. Current configuration totals 144 bundles; freeze any revised scope before launch.
- [ ] Separate unseen users from known-client personalization. Record client variation, communication and runtime; federation alone provides no formal privacy guarantee.
- [ ] Report macro-F1, accuracy, ideal/self/induced outcomes, subject variation, worst cases and measured cost. Repeated seeds are not independent subjects.
- [ ] Seek untouched recorded-placement data for confirmation; existing test subjects support exploratory comparisons.
- [ ] Reconcile thesis/slides with the implemented temporal architecture, losses, channels and actual completed evidence. Do not compare old summer scores as if generated by the corrected protocol.

## Documentation maintenance

Use these three working documents: `project_plan.md` (future work), `project_status.md` (current evidence), `project_handoff.md` (resume context). Keep detailed historical prose in the dated archive and immutable run folders. Do not create another `next_*` or `update_*` working document. Some old root filenames remain code dependencies; consult `documentation_cleanup.md` before removing anything.

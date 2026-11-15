# Project status

## T08 execution handoff - 14 November 2026

Astra finalized receiver-blind predictive repair revision1 and its code. READY_FOR_LUNA_VALIDATION; runtime checks not executed. Read docs/agent_exchange/Astra_to_Luna.md. The exact four-job smoke followed by sixteen-job full source-validation pilot is authorized only through the passed-smoke gate. T08 remains OPEN until candidate review. This supersedes earlier statements that the next method/commands are unspecified. No terminal or training was run by Astra.


## T08 status correction - 14 November 2026

T08 is OPEN / INCOMPLETE: no graph candidate has passed the frozen source promotion criteria for displacement confirmation. Completed training and reviews do not constitute completion of candidate selection. The earlier REVIEW_COMPLETE_NO_PROMOTION records describe finished reviews, not a completed T08 milestone. The original three studies and later repair pilot remain completed evidence; do not rerun them automatically. Next: Astra specifies a justified revision, Luna executes the explicitly authorized smoke/pilot, and Astra selects a candidate only if the unchanged acceptance criteria pass. No new training is authorized by this correction.


## Current reviewed state - 12 November 2026

Reviewed 12 November 2026: the graph-repair queue COMPLETED all 4 smoke and 8 full jobs on 29 October at 08:25:19 IST. Audits passed. Self validation accuracy: graph 93.904%, robust no graph 93.907%, local repair 94.368%. Decision NO_PROMOTION_GRAPH_REPAIR. Read results/research_review_20261112/RESEARCH_REVIEW.md for historical raw/mag+dot placement comparisons and plots. Current handoff: docs/agent_exchange/current.json. The new direction is placement diagnostics before conditional experts; its plan is not an authorized training queue. Earlier execution notes below are historical.


## T08 closed — 29 October 2026

T08 review and handoff are COMPLETE. Decision: REVIEW_COMPLETE_NO_PROMOTION. Read results/o1_t01_t08_20261028/T08_FINAL_REVIEW.md and its accompanying T08_FINAL_DECISION.json. All three source studies completed (36 full jobs, 9 smoke jobs). The final training finished28 October at22:44:19 IST; no new training was launched by the final review.

Original correction source-self F1:92.7117%; matched no graph:92.2249%; local control:93.0090%. The graph gains0.4867pp over no graph but trails local control by0.2973pp. It is retained as a research reference, not promoted as a qualified winner. None of the candidates meets the frozen full promotion rule. Held-out induced displacement has NOT been evaluated; the user's graph-superiority objective remains OPEN.

Next: research chat freezes the next graph-specific mechanism/objective and controls. CLI must not rerun completed pilots or launch an unspecified final study. T09 genuine-Mamba implementation can proceed independently from docs/experiments/mamba_temporal_spec_20261028.md. Workstation setup remains to be verified. All Git actions remain user-only.



Latest update28 October2026: T01-T08 execution/review completed; details in results/o1_t01_t08_20261028/execution_report.md. New gated-neighbor source pilot completed12/12 jobs after3/3 smoke jobs; all audits passed. Graph ideal/self F1=93.8185/92.2214%, none=93.8024/92.3278%, local=94.0051/92.5451%. Candidate NOT PROMOTED; no recorded-displacement evaluation. Genuine Mamba specification written, implementation pending. Current graph-superiority goal remains unresolved. The earlier27 October record below is historical context.


Reviewed: 27 October 2026 (Asia/Calcutta). This is the current human-readable status, not a live process monitor. Read `project_plan.md` for future work and `project_handoff.md` when resuming. Paths below are relative to the project root.

Local follow-up completed on 27 October at 14:54:34 IST: `results/local_message_pilot_20261027`, six jobs at eight epochs in one visible terminal. Isolated `Prabhat`: Python 3.11.15, torch 2.5.1+cu121, CUDA 12.1, RTX 2050. GPU preflight and checkpoint recovery passed with zero parameter/loss discrepancy. CPU staging kept peak allocated GPU memory below 69 MB (allocator measure, not total process/device usage). All six checkpoint/history/prediction sets and source fingerprints were verified.

Source-validation means (ideal/self macro-F1): no graph 93.29/92.62%; local-capacity control 93.46/92.67%; message correction 93.42/92.88%. Graph improves over no graph by approximately 0.13/0.26 points, but trails the local control on ideal validation. This is a two-fold, one-seed pilot, not a held-out placement win. The temporal block remains the existing summer gated convolution. Full comparisons and genuine-Mamba validation remain pending. See `PILOT_REVIEW.md` and `completion_audit.json` inside the run.

## Objectives and evidence

| Objective | Status | Scientific conclusion |
|---|---|---|
| O1: placement-aware physical/global/dynamic graph | Implemented; core and alternative suites completed | Useful sample-dependent routing and superiority over no graph remain unproven. |
| O2: rotation-stable per-IMU representation | Implemented; matched summer suite completed | Magnitude/dot features give a large exploratory placement benefit. `mag_dot` was selected for O3 using source validation. |
| O3: subject-client federated learning | Implemented; preparatory checks, smoke and centralized bridge available | Full federated comparison remains pending; no federated performance claim. |

## Completed major suites

| Result directory under `results/` | Jobs | Role |
|---|---:|---|
| `o1_full_20261016_220301` | 118 | Earlier modified-pipeline O1; historical evidence |
| `o2_full_20261017_075315` | 128 | Earlier modified-pipeline O2; historical evidence |
| `summer_o1_session_20261017_232945/training` | 126 | Restored summer O1, 14 conditions, 31 epochs |
| `summer_o2_session_20261021_173341/training` | 63 | Restored summer O2, 7 conditions, 31 epochs |
| `summer_o1_alternatives_session_20261022_001702/training` | 81 | Nine graph alternatives/controls, 31 epochs |

Latest alternative session and analysis completed on 22 October 2026 at 03:18:23 IST. Older root plans saying this session is active or unstarted are historical. The original summer O1 analysis failure was repaired; it is not a reason to repeat training.

## Current matched results

Fixed-33-class macro-F1 percentages, seeds averaged within each of test subjects 2, 5, 15, then subjects equally. All rows below use magnitude-plus-dot features and the restored summer backbone.

| Graph | Ideal | Self | Induced |
|---|---:|---:|---:|
| Physical | 92.88 | 92.23 | 85.99 |
| Original residual | 93.41 | 92.36 | 86.55 |
| No graph | 93.72 | 92.56 | 87.12 |
| Gated pairwise | 93.68 | 92.79 | 86.95 |

Sources: latest alternatives `training/ANALYSIS.md`; O2 `training/O2_REVIEW.md` and `.json`; individual saved jobs. Gated pairwise improves over physical/original residual but does not establish a no-graph advantage. These subjects have been inspected repeatedly, so the results are exploratory.

## Protocol and implementation

- REALDISP: nine IMUs, acceleration/gyroscope channels, 50-sample windows, stride 25. Corrected view: 76,268 windows.
- Excluded all-zero recordings: `subject6_self.log`, `subject13_self.log`, `subject15_mutual4.log`. Preserve original data and audit artifacts.
- Main test subjects: 2, 5, 15; separate validation subjects: 3, 8, 16; seeds: 13, 29, 47. Train ideal source recordings only; fit normalization on training subjects; source ideal validation selects checkpoints in the restored summer suites.
- Restored summer: GIN aggregation multiplied by tanh GIN attention, temporal block, mean pooling, LayerNorm and classifier. CE main; GAE separate ablation. Historical source parity is documented in `SUMMER_O1_AUDIT.md`.
- The current temporal block is a gated causal convolution/projection block. Its name contains Mamba, but it has no selective SSM scan. Existing results do not validate the report's selective-Mamba claim.
- No graph removes neighbor aggregation but retains later cross-sensor mixing in flattened projections.

## Open issues

1. Preserve local information while testing useful learned neighbor messages; recent adjacency-only repairs have not established superiority.
2. Extend the deeper source-only routing audit to new alternatives. Routine shuffled affinity rolls flattened time/batch entries; the deeper audit has true window donors. Gated branch masses are absolute attenuated contributions, unlike original relative fractions.
3. Investigate `subject15_mutual7.log`, which performs poorly across leading models. Do not exclude it merely because performance is poor.
4. Validate a genuine selective SSM implementation in a separate matched temporal experiment if implementing the PDF proposal (see plan).
5. Reconcile report wording, prior-art claims and historical metrics with code and current evidence.
6. Repair workstation paths and calibrate on this machine before training. Current reviewed GPU: RTX 2050, 4 GB; later saved experiments used RTX 2000 Ada, 16 GB. Whole-fold GPU residency may need adjustment.

O3 currently specifies four model arms and four methods across three subjects and three seeds: 144 bundles, not the older 108. The selected feature is `mag_dot`. Its config contains stale paths and a stale blocked-note string; readiness fields alone do not authorize or establish a completed run.

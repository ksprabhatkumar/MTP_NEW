# Project status

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

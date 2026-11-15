# Experiment coverage and next rounds — 15 November 2026

Reuse requires matching architecture, data/exclusions, features, normalization, optimization exposure, precision, seeds/order and checkpoint-selection rule. Historical evidence is not automatically a substitute for a new-backbone matched control. Preserve all studies. This ledger distinguishes specified jobs from downstream matrices still awaiting T11/T18 decisions; those unspecified jobs cannot truthfully be enumerated yet.

| Planned experiment family | Existing evidence | Coverage decision / pending work |
|---|---|---|
| Data/rotation audit | results/summer_o2_data_rotation_audit_20261021.json; dataset/workstation_o1/audit.json | Reuse; reopen only on concrete defect. |
| Restored summer O1, 126 jobs | results/summer_o1_session_20261017_232945/training | Completed historical family per project status; verify completed manifest before job-level reuse. No blind rerun. |
| Restored summer O2, 63 jobs | results/summer_o2_session_20261021_173341/training | Reuse representation evidence on original backbone; new SSM feature bridge pending. |
| Graph alternatives, 81 jobs | results/summer_o1_alternatives_session_20261022_001702/training | Historical negative/ablation evidence; no repeat sweep. |
| Local correction, 6 jobs | results/local_message_pilot_20261027 | Reuse source pilot only. |
| Original T08 selection/gate/reference studies | results/o1_t01_t08_20261028; o1_gate_ablation_20261028; o1_reference_selection_20261028 | Completed prior reviews; not a promoted candidate. |
| Graph repair pilot | results/o1_graph_repair_20261029/full | Prior review no promotion; reuse historical evidence. |
| Predictive repair | results/t08_predictive_20261114_r1/full | Completed source study; supplies frozen baseline checkpoints, not final induced proof. |
| Donor/temporal donor interventions | results/t08_temporal_donor_audit_20261114_r1 | Reuse checkpoint-specific diagnostics; not transferable to new weights. |
| Numerical diagnostics | results/t08_utility_diagnostics_r4 and r5 | Reuse explanation of freeze-state mismatch; no training required. |
| Utility gate expanded smoke/full | results/t08_utility_gate_r6/{smoke,full} | Verified passed, 8 full jobs; 32 full rows; 16 best/latest each. Review complete, NO PROMOTION. |
| Failed utility/donor outputs | earlier r1/fix1 folders | Failure provenance only; never counted as complete experiments. |
| T09 genuine selective SSM | models/selective_mamba_temporal.py newly prepared | Pending independent parity, gradients, causality, reset, recovery and real-data profiling. |
| T10 temporal factorial | none compatible with new SSM/order contract | Pending old/new x none/physical/nominated graph, plus local control where needed. T08 candidate slot unresolved; do not silently omit it. |
| T12 final O1/O2 | historical baseline evidence only | Pending T11 exact matrix; all required features, raw/rotation controls, graph controls, GAE/augmentation ablations, comparator, seeds/folds and scenario evaluations retained. |
| T13/T22 interventions and efficiency | existing audits apply only to their models | Final-model mechanism, latency/memory, prediction audit and reproducibility pending. |
| T16/T17 federation engineering | results/summer_o3_gates_20261021; o3 CPU checks/manifests | Existing framework/bridge only; final architecture laptop validation pending. |
| T19 federation | no completed full suite identified in project status | 144 planned bundles explicitly enumerated in o3_pending_jobs.json, 4 arms x 4 methods x 3 subjects x 3 seeds, 31 epochs/rounds. No scope reduction. Reconcile stale paths and final architecture before launch. |
| Independent confirmation | none identified | Cannot claim independent confirmation from previously inspected subjects; new cohort only if separately obtained/approved. |

Next rounds in order: R1 T09 correctness + bounded real-data timing (no convergence training); R2 T10 full source temporal factorial after candidate/protocol decision; R3 complete locked T12 centralized O1/O2 matrix; R4 T17 federation calibration; R5 complete T19 approved federation queue; R6 audits/reproduction. R2/R3 exact exposure is frozen after R1 measurements, not invented now. Old 144 O3 bundles remain pending, not silently deleted. Report wall-time estimates from new measured workloads with 25% contingency.

# Binding scope clarification — 15 November 2026

User confirms laptop may run as long as needed. Research-grade completion takes priority over the provisional accelerated dates. Retain ALL required experiments previously planned for the workstation; relocate them to the laptop. Do not reduce controls, folds, seeds, epochs, rounds, model arms or evaluation coverage merely because hardware is smaller. The ten-day allocation below is superseded as a completion estimate. No deadline has been imposed.

Preserve original T10 matrix: two temporal blocks x no-graph/physical/nominated-graph, plus local-capacity control on the new backbone where needed for attribution. Preserve all approved final O1/O2 and O3 comparisons, external comparators and ablations. Historical O3 144-bundle scope must be reconciled against its actual approved config; it cannot be reduced for laptop convenience. Any scientific scope amendment (including graph objective) needs explicit agreement and is independent of hardware migration. All tasks T01–T24 remain in scope.

Statements below suggesting a reduced temporal matrix or minimal federation scope are superseded. Completed exact-compatible runs may satisfy planned jobs; this is evidence reuse, not skipping. Create a coverage ledger mapping every planned condition to a valid existing artifact or a pending laptop job. New conditions receive all required seeds/folds and smoke checks. Preserve trained results and immutable sources. Do not launch from a planning document alone.

Execution stages: (1) artifact review and complete matrix/coverage ledger; (2) genuine SSM implementation and full temporal pilot; (3) frozen complete centralized O1/O2 suite and analysis; (4) complete approved federation suite and analysis; (5) reproducibility, thesis, slides and defense. One GPU training job at a time; CPU-staged batches/client states to fit 4 GB. Changes to batch/microbatch/precision are frozen before execution and matched across controls; verify correctness and expose any unavoidable protocol impact.

Time budget: calibrate representative jobs for EACH architecture and federation method. Remaining GPU time = sum(pending jobs x measured per-job duration) + audit/evaluation time + 25% contingency. Convert to calendar time using actual uptime. Report a range after calibration; do not promise ten days or extrapolate all tasks from the 32-minute utility pilot. Implement and write in parallel with training where CPU/RAM permit. Queue only scientifically approved jobs, stop on failure, checkpoint/resume safely. No unnecessary waiting to fill a training window.

---

# Laptop-only completion plan — 15 November 2026

This section supersedes older machine assignments and calendar dates below; historical evidence is retained. Original task IDs run T01–T24, not 23. Planning only: no new training authorized by this schedule. All implementation/training now uses the RTX 2050 4 GB laptop. Astra handles decisions/files/code without terminal execution; Luna/Terra handles execution. Git remains user-only. Visible terminal is optional under the latest user preference.

## Immediate decisions and scope

1. Finish revision 6 artifact/paired review and record it in Luna_to_Astra.md. Smoke and full audit passed (32 rows each); full runtime 1928.8 seconds (~32.1 min), smoke 547.4 seconds (~9.1 min). Full graph-gated self accuracy/F1 equals frozen no-graph mean; no demonstrated self improvement. No induced evaluation in this pilot. Do not repeat completed runs.
2. T08 stays OPEN under its existing graph-promotion criterion. Freeze with user/supervisor whether robustness rather than a mandatory graph win becomes the final research objective. A no-graph final candidate is a proposed scope amendment, not an already approved completion of T08. Review existing raw/invariant evidence before commissioning another representation search.
3. No additional graph architecture search is scheduled. Keep existing graph results as controls/negative evidence. Anomaly detection, new datasets and extra expert branches are deferred unless explicitly required for approved objectives.
4. Preserve three-objective scope including genuine SSM and federation pending supervisor scope decision. Do not call the existing gated convolution a selective SSM. If laptop calibration makes the remaining scope infeasible, request an explicit scope reduction rather than silently omitting an objective.

## Task ledger and dependencies

| ID | Type | Status and next action |
|---|---|---|
| T01 | Research | Foundation DONE; amend final claims, acceptance rules and compute budget. |
| T02 | Implementation | Laptop GPU/checkpoint foundation DONE; verify lifecycle and capacity only for new runners. No workstation setup. |
| T03 | Analysis | Existing donor/equivalence diagnostics DONE; summarize the revision 6 saved evidence, no repeat audits. |
| T04 | Research | Existing operator specifications DONE; final candidate scope decision remains. |
| T05 | Research | Existing Mamba specification available; review laptop feasibility and freeze small supported implementation. |
| T06 | Implementation | Current utility-gate implementation DONE. No further graph changes scheduled. |
| T07 | Training | Revision 6 smoke/full DONE; reuse all completed compatible pilots. |
| T08 | Research | OPEN: review artifacts/paired results; choose candidate only under agreed criteria or explicitly amend scope. |
| T09 | Implementation | Implement/validate genuine SSM on laptop; reference, gradients, causality, recovery and memory checks. |
| T10 | Training | Bounded source temporal pilot; compare old/new backbone with matched no-graph and essential graph control. Exact reduced matrix needs T01/T08 decision. |
| T11 | Research | Freeze final O1/O2 matrix, features and held-out evaluation rules after T10. |
| T12 | Training | Sequential final comparisons; reuse exact compatible checkpoints; recorded displacement evaluation only after protocol lock. |
| T13 | Analysis | Validate artifacts, paired metrics, per-recording failures and plots; follows T12. |
| T14 | Research | Close O1/O2 claims and freeze final model for O3; follows T13. |
| T15 | Research | Freeze minimal defensible federation matrix; final model first, not old 144-bundle Cartesian sweep. |
| T16 | Implementation | Adapt existing federation simulator; CPU-resident client states, one GPU client at a time. |
| T17 | Training | Federation smoke, recovery and time-per-round calibration. |
| T18 | Research | Approve rounds/steps/seeds using T17 timing; no invented runtime guarantee. |
| T19 | Training | Required federation queue; central/local/FedAvg/local-normalization comparisons with exposure accounted for. |
| T20 | Analysis | Audit client coverage, communication, worst-client scores and saved evidence. |
| T21 | Research | Final synthesis, limitations, exploratory versus independent evidence. |
| T22 | Implementation | Reproduction, inference efficiency and artifact packaging after T13/T20. |
| T23 | Research/writing | Draft existing methods/results immediately; finalize claims after T21. |
| T24 | Deliverables | Compile, figures, slides, demo, supervisor corrections and defense; follows T22/T23. |

Critical order: T01/T02 + completed T03/T04/T06/T07 -> T08; T05 -> T09; T08+T09 -> T10 -> T11 -> T12 -> T13 -> T14 -> T15 -> T16 -> T17 -> T18 -> T19 -> T20 -> T21 -> final T23. T22 follows T13/T20; T24 follows T22/T23. Draft writing and federation code inspection can proceed while GPU runs; keep one GPU job at a time.

## Provisional accelerated timetable

This is a 10-working-day allocation, NOT a verified completion promise. Deadline and daily powered-on hours have been requested. If day 1 is 15 November, day 10 is 24 November. Reforecast after T09 and T17; no minimum-duration training.

- Day 1: finish T08 review/scope decision, T01 amendment, T05 laptop spec; prepare T09; start thesis methods/results consolidation.
- Day 2: T09 GPU checks and timing; bounded T10 pilot if checks pass. Freeze exact queue before execution.
- Day 3: T10 review/T11; launch required T12 jobs; draft negative graph results honestly.
- Days 4–5: T12/T13/T14, source-only model selection and clearly labelled displacement evaluation. Inspect existing federation code in parallel.
- Day 6: T15/T16/T17; calibrate federation and issue T18 budget decision.
- Days 7–8: T19/T20; prepare final figures and tables while runs execute.
- Day 9: T21/T22/T23; reproducibility and thesis conclusions.
- Day 10: T24, supervisor corrections, slides/demo and delivery buffer.

## Laptop execution policy

- Reuse Prabhat; one GPU writer, data on CPU with minibatches on GPU; monitor RAM/disk and preserve checkpoints.
- Preserve matched batches/precision/optimization for scientific comparisons. Select a fitting configuration during smoke; never silently alter it mid-run.
- Estimate a queue from measured new-model epoch/round time × planned exposure + load/evaluation/checkpoint overhead and 25% margin. Existing 32-minute utility pilot is NOT an estimate for SSM or federation.
- Each new materially different configuration must pass smoke, resume, finite-value and artifact checks before full. Cover every checkpoint/freeze regime relevant to the prior numerical failure.
- Save automatic audit/results and launch receipt so a closed chat is not needed to finish a queue. Record failures and stop; do not retry by weakening checks.
- All final claims depend on evidence, not the desired graph win. Previously inspected displacement subjects support exploratory comparisons, not new independent confirmation.

---

# Current constraints and 15-day schedule (28 October 2026 revision)

## T08 execution handoff - 14 November 2026

Astra finalized receiver-blind predictive repair revision1 and its code. READY_FOR_LUNA_VALIDATION; runtime checks not executed. Read docs/agent_exchange/Astra_to_Luna.md. The exact four-job smoke followed by sixteen-job full source-validation pilot is authorized only through the passed-smoke gate. T08 remains OPEN until candidate review. This supersedes earlier statements that the next method/commands are unspecified. No terminal or training was run by Astra.


## T08 status correction - 14 November 2026

T08 is OPEN / INCOMPLETE: no graph candidate has passed the frozen source promotion criteria for displacement confirmation. Completed training and reviews do not constitute completion of candidate selection. The earlier REVIEW_COMPLETE_NO_PROMOTION records describe finished reviews, not a completed T08 milestone. The original three studies and later repair pilot remain completed evidence; do not rerun them automatically. Next: Astra specifies a justified revision, Luna executes the explicitly authorized smoke/pilot, and Astra selects a candidate only if the unchanged acceptance criteria pass. No new training is authorized by this correction.


## Current review override - 12 November 2026

The 29 October graph-repair run is complete and not promoted. Read results/research_review_20261112/RESEARCH_REVIEW.md, docs/experiments/placement_diagnostics_plan_20261112.md and docs/agent_exchange/current.json. The next decision is placement diagnostics and label/evaluation protocol before conditional experts. Earlier 15-day dates and active-run statements below are historical; no new training is authorized by this review.


## T08 closed � 29 October 2026

T08 review and handoff are COMPLETE. Decision: REVIEW_COMPLETE_NO_PROMOTION. Read results/o1_t01_t08_20261028/T08_FINAL_REVIEW.md and its accompanying T08_FINAL_DECISION.json. All three source studies completed (36 full jobs, 9 smoke jobs). The final training finished28 October at22:44:19 IST; no new training was launched by the final review.

Original correction source-self F1:92.7117%; matched no graph:92.2249%; local control:93.0090%. The graph gains0.4867pp over no graph but trails local control by0.2973pp. It is retained as a research reference, not promoted as a qualified winner. None of the candidates meets the frozen full promotion rule. Held-out induced displacement has NOT been evaluated; the user's graph-superiority objective remains OPEN.

Next: research chat freezes the next graph-specific mechanism/objective and controls. CLI must not rerun completed pilots or launch an unspecified final study. T09 genuine-Mamba implementation can proceed independently from docs/experiments/mamba_temporal_spec_20261028.md. Workstation setup remains to be verified. All Git actions remain user-only.



Execution update28 October15:20:58 IST: today's smoke and12-job source pilot are COMPLETE. T01-T07 laptop work/specifications and T08 review were performed; T08 promotion FAILED. Read results/o1_t01_t08_20261028/execution_report.md and full/T08_DECISION.md. A winning graph has not been selected; prioritize the next bounded source-justified graph revision before final confirmation. Workstation-only verification remains tomorrow. Calendar below is a planning allocation, not a claim of future completion.


This section supersedes conflicting scheduling/ownership text below. Detailed task IDs remain stable.

- Deadline: 15 calendar days, 28 October through 11 November 2026 inclusive; laptop available 12 hours today; workstation available continuously from 29 October.
- User-required O1 outcome: graph must outperform matched no graph on recorded sensor displacement. This is an outstanding success criterion, not an established result or a guaranteed outcome.
- Primary comparison: fixed-33 macro-F1 on induced displacement; report self placement separately, alongside accuracy, ideal scores, degradation and every subject/seed effect. Keep existing folds (test 2/5/15, validation 3/8/16), seeds 13/29/47, source-only checkpoint selection and train-only normalization. Do not weaken no graph or select checkpoints on test results.
- Proposed practical target to freeze before new runs: >=0.5 percentage-point mean induced F1 improvement, positive effects on at least two of three test subjects, ideal F1 loss <=0.5 points. These numerical margins are new planning defaults, not decisions found in the resources. Show seed variation and subject-level uncertainty; three repeatedly inspected subjects cannot support a strong independent-confirmation claim. A positive mean alone does not establish reliability.
- T01 substantially DONE: objectives, mathematical foundations, literature trail, dataset/protocol, controls and metrics exist in resources/objectives_methodology.tex and project_status.md. Only new numerical acceptance margins, exact final matrix and unresolved comparator/claim details need freezing. The newer three-objective document supersedes roadmap_review.tex's older optional-FL framing.
- Data audit DONE: reuse dataset/workstation_o1/audit.json, results/summer_o2_data_rotation_audit_20261021.json and corrected manifests. T03 is new-operator mechanism diagnosis, not another dataset audit.
- Same folder structure on both machines. User alone synchronizes via Git. Agents do not execute Git commands or change Git settings. Public repository HTTPS accessibility verified HTTP 200 on 28 October: https://github.com/ksprabhatkumar/MTP_NEW . This is not verification of remote/local file parity.
- Research: GPT-6 Astra Medium on laptop. Implementation, smoke and full-run launch: GPT-6 Sol Medium in Codex CLI, as requested. Smoke success is an engineering gate, not evidence of research superiority.
- Full run follows a successful end-to-end smoke for the exact frozen configuration. CLI then closes; one independent visible training terminal remains open. Verify process independence first. A closed CLI cannot repair new failures.
- See execution_handoff.md for the shared chat/CLI communication contract, plotting-data schema and ready/blocked gates. No training is launched merely by editing this plan.

## Calendar and dependency schedule

1. Day 1 / 28 Oct / laptop / T01-T08 target, 12-hour allocation:
   - Hours 0-1 [RD]: reconcile existing decisions and freeze remaining O1 criteria; save an executable experiment specification (T01).
   - Hours 1-2.5 [I]: runner/logging/checkpoint checks and CLI-close survival procedure (T02 laptop part); do not repeat passed unchanged GPU/data checks.
   - Hours 2.5-4 [I/RD]: diagnose current correction versus no graph and local control, choose one justified operator revision (T03-T04).
   - Hours 4-5 [RD]: genuine Mamba interface/recurrence specification only (T05); workstation implementation tomorrow.
   - Hours 5-7 [I]: implement approved O1 change and artifact contract (T06).
   - Hours 7-8 [I/T]: complete smoke, reload checkpoint, regenerate metrics/plot from saved data; repair before full launch.
   - Hours 8-10.5 [T]: full approved laptop pilot (T07), existing three arms, matched protocol; exact additional folds/seeds/epochs frozen by T04. Calibrate ETA first. Do not call an eight-epoch smoke a final study.
   - Hours 10.5-12 [RD]: compare paired source results and message interventions; select candidate and write workstation handoff (T08). If checks fail, record the precise incomplete task; do not label T08 complete automatically.
2. Day 2 / 29 Oct [I/T]: workstation environment verification, Mamba integration and smoke (T09); start frozen graph/backbone pilot (T10).
3. Day 3 / 30 Oct [T/RD]: finish T10; select backbone and lock centralized matrix (T11). First documented displacement evaluation identifies whether the priority graph target is on track.
4. Day 4 / 31 Oct [T/I]: matched full O1 runs and paired analysis (T12-T13 initial batch).
5. Day 5 / 1 Nov [RD/I/T]: reserved graph-revision slot if needed; source-justified revision, fresh smoke and matched confirmation. Record repeated test inspection as exploratory. If no revision needed, advance O2.
6. Day 6 / 2 Nov [T/I]: O2 bridge, essential raw/invariant/augmentation and graph controls; required external comparator where feasible (T12-T13).
7. Day 7 / 3 Nov [I/RD]: finalize centralized ablations/efficiency, interpret O1/O2 and freeze O3 (T13-T15). Graph target unresolved remains explicitly open, not relabeled successful.
8. Day 8 / 4 Nov [I/T/RD]: adapt federation, smoke/resume/centralized bridge, timing calibration and final O3 budget (T16-T18).
9. Day 9 / 5 Nov [T]: federated main comparisons (T19).
10. Day 10 / 6 Nov [T/I]: finish required federation seeds/controls, audit per-client evidence (T19-T20).
11. Day 11 / 7 Nov [I/T]: buffer for failed required jobs, necessary confirmation and efficiency/reproduction (T20/T22). No indiscriminate architecture search.
12. Day 12 / 8 Nov [RD/I]: final scientific synthesis, regenerate figures/tables from saved data (T21-T22).
13. Day 13 / 9 Nov [RD]: finalize report/thesis and presentation claims (T23); methods/literature drafting occurs throughout days 2-11.
14. Day 14 / 10 Nov [I/RD]: compile, inspect figures/citations, supervisor review and corrections, defense practice (T24).
15. Day 15 / 11 Nov [I]: final reproduction/package/demo, backups and user submission (T24).

These are planning allocations, not measured Mamba/federation runtimes. Calibrate first; split required queues into 5-7-hour blocks without padding. Continuous workstation availability allows successive blocks. If mandatory jobs exceed capacity, surface the projected overrun and a concrete scope decision; do not silently drop controls. No extra dataset or broad architecture sweep is scheduled within these 15 days.

---

# MTP completion plan: laptop research and workstation execution

Updated: 28 October 2026, Asia/Calcutta. Planning only; no new training launched. This replaces the earlier working roadmap in the same file. Historical results and archived plans remain unchanged. No Git operation is authorized: the user alone handles repository synchronization.

## 0. Labels and ownership

- RD = research decision, mathematical specification, interpretation or scientific writing. Owner: laptop / ChatGPT Work / GPT-6 Astra Medium.
- I = implementation, deterministic analysis, testing, packaging or technical documentation. Owner: Codex CLI / GPT-6 Sol Medium; laptop for lightweight work, workstation for GPU/platform integration.
- T = training or GPU benchmark execution. Owner: ordinary scripts on the selected machine, prepared by CLI; no model needs to remain running during training.
- DONE denotes completed work, not proof that its research hypothesis succeeded. OPEN denotes work still required. Conditional extensions are explicit and are not all mandatory.
- The order below is a valid topological order: every prerequisite precedes its dependent task. Independent branches may overlap across the two machines, but one GPU training writer per machine is the default.
- Interpret the user's 5-7 duration as HOURS per workstation session. It is a scheduling window, not a minimum workload or an epoch stopping rule.

## 1. Completed foundation

- DONE D01 [RD] Define three objectives: O1 placement-aware routing, O2 rotation-stable representation, O3 subject-client federation; assemble literature/resources. Final novelty/citation review remains open.
- DONE D02 [I] Correct REALDISP preprocessing, exclusion of three all-zero recordings, subject separation and train-only normalization; corrected view 76,268 windows.
- DONE D03 [I] Restore summer GIN aggregation times attention; record source parity. Recover the temporal block honestly as gated causal convolution, not selective SSM.
- DONE D04 [T/I] Historical modified O1/O2 suites (118/128 jobs); restored summer O1/O2 suites (126/63 jobs); save analyses and artifacts.
- DONE D05 [T/I] Nine-arm graph follow-up, 81 jobs and analysis; no established graph superiority over the no-graph control.
- DONE D06 [RD] Select mag_dot for subsequent comparison from source validation; preserve physical and no-graph controls.
- DONE D07 [I] O3 framework and preliminary client/preflight/smoke/centralized bridge; full federation remains open.
- DONE D08 [I] Laptop Prabhat Python 3.11.15 + torch 2.5.1+cu121; real GPU checks; exact checkpoint-restoration check. CPU-staged batches fit RTX 2050 4 GB.
- DONE D09 [I/T] Implement local-preserving latent neighbor correction, no-graph and local-capacity controls; six eight-epoch source-only jobs on 27 October. Ideal/self F1: 93.29/92.62 no graph, 93.46/92.67 local control, 93.42/92.88 graph. Pilot only, one seed/two folds.
- DONE D10 [I] Consolidate working plan/status/handoff; archive prior planning documents; preserve all originals/results.

## 2. Remaining tasks in topological order

### T01 [RD, laptop] Freeze the scientific completion criteria

- Prerequisites: D01-D09.
- Reconcile summer PDF, objectives and current implementation. Define the contribution and claims that can be tested; do not assume graph superiority or SOTA.
- Freeze primary fixed-33 macro-F1, supporting accuracy, clean-performance tolerance, practical improvement threshold, cost limits, paired subject aggregation and uncertainty reporting BEFORE further tests.
- Confirm required subject coverage: three recorded-displacement subjects; decide whether final ideal evaluation covers all 17 subjects and the seed budget. Report subject 7's thin class coverage and supported-class sensitivity.
- Decide feasibility of untouched recorded-placement data now. If unavailable, explicitly bound the final study as exploratory; do not manufacture a new independent cohort by re-splitting inspected recordings.
- Verify key novelty and comparator citations from primary sources. New datasets, GAE extensions and exhaustive architecture searches are optional unless a specific claim requires them.
- Deliverable: frozen protocol and claim checklist in an experiment specification; decisions/rationale, not private internal reasoning.

### T02 [I, laptop + workstation] Make execution portable and independent of the CLI lifetime

- Prerequisite: D08. Can proceed alongside T01.
- Replace stale machine-specific paths in FUTURE launch configurations with project-relative paths or explicit local configuration. Do not alter immutable run snapshots.
- Create a separate environment on each machine; recreate from a lock file rather than copying a venv. Verify GPU, RAM, disk, dependency versions and data/cache identity on the workstation.
- Run one visible terminal containing the ordinary training supervisor; no hidden trainer. Its lifetime must be independent of the chat/CLI process. The existing foreground pilot proves execution/checkpointing, NOT survival after closing the CLI.
- Before any long session, perform a short close-CLI survival test: verify a heartbeat/checkpoint advances after CLI exit; reopen later and verify recovery. Never claim this is solved merely because Start-Process returned a PID.
- Add/preflight atomic checkpoints, source/config/data hashes, run lock, finite-value checks, disk/memory checks, heartbeat, clear status, exception traceback and artifact validation. Preserve best/latest weights, optimizer, scheduler, RNG, normalization and completed epoch.
- On unexpected error: save what can safely be saved, mark failed/interrupted and stop. No silent change to batch size, precision, splits, architecture or epoch budget. Do not promise a closed CLI can repair failures.
- Budget scheduler: estimate jobs from measured epoch times plus loading/evaluation/checkpoint overhead; reserve margin. Stop before the NEXT job if the session budget is exhausted; use a configured safe epoch checkpoint if needed and resume the same job later. Scientific epoch budgets stay fixed.
- Output an exact visible-terminal launch/resume command, environment identity and shutdown behavior. Keep the machine awake/powered and the training terminal open; closing the CLI is different from closing the terminal or shutting down the machine.

### T03 [I, laptop] Run new-message diagnostics only

- Prerequisites: T01, T02.
- Extend true cross-window donor, frozen source-average graph, zero-message and constant-gate interventions to current candidates; inspect centered logits/loss as well as label flips.
- Separate relative branch fractions from total gate attenuation; log local/message/correction norms and gradient flow.
- Data auditing is DONE and is not repeated. Retain the documented hard recordings. Reopen a data issue only if a new concrete implementation defect appears.
- Deliverable: audit JSON and concise report with verified defects distinguished from hypotheses. No retraining if analysis alone resolves an issue.

### T04 [RD, laptop] Decide the next O1 operator experiment

- Prerequisite: T03.
- Interpret existing six-job pilot against BOTH no graph and local-capacity controls. Current evidence is too small to establish superiority.
- Choose the smallest informative follow-up: source-only mechanistic audit and/or additional source seed; if warranted, preserve the existing correction candidate for matched temporal evaluation.
- Conditional candidates, in priority order: receiver-sender rejection gates; sender-quality weighting; relational message values; short temporal messages; sparse/null-neighbor selection; separate branch messages. Do not implement/train all options automatically.
- Write equations, tensor dimensions, masking/normalization, initialization, invariance conditions, parameter matching, objectives and falsifiable acceptance criteria. Declare a bounded revision budget and retain negative outcomes.
- Any source-justified revision is a new version with its own forward dependency chain, not an unbounded loop chasing test scores.

### T05 [RD, laptop] Specify genuine causal Mamba/SSM

- Prerequisite: T01. Independent of T03/T04; prepare while diagnostics run.
- Select a genuine selective-state recurrence/reference, define causal temporal behavior, state dimensions, precision and supported platform/kernel requirements. Check workstation OS feasibility before choosing an implementation.
- Define input/output contracts to preserve the summer spatial encoder and classification head. Distinguish this extension from restoring the original code.
- Specify numerical/reference and gradient checks and a matched old-backbone comparison. Existing reports must not call the gated convolution a selective SSM.

### T06 [I, laptop/CLI] Implement only the approved O1 changes

- Prerequisites: T02, T04.
- Reuse the existing latent-correction code when unchanged. Add tests for shape, masks, invariance where claimed, gradients and checkpoint resume only where new behavior requires them.
- Keep no graph and local-capacity controls, comparable optimization and feature representation. Preserve old code/results or isolate revisions in new files/versions.

### T07 [T, laptop first] Run the specified small source-only O1 pilot

- Prerequisite: T06.
- Reuse completed D09 jobs where source/config/protocol match exactly; run only additional conditions/seeds required by T04.
- Run full epochs, distinguish smoke from pilot, select checkpoints by source ideal validation, retain source self as a declared supplementary diagnostic.
- Produce paired fold/seed effects, learning curves, message sensitivity, memory/runtime and a machine-readable summary. No automatic promotion based on held-out scores.

### T08 [RD, laptop] Select the O1 candidate for displacement confirmation

- Prerequisite: T07.
- Select using the frozen source criteria. A worthwhile candidate need not already prove a test win.
- If evidence is negative, record O1 as unresolved and use the reserved graph-revision time. A negative outcome does not satisfy the user's requested graph-superiority milestone. Preserve failed results and keep other independent work moving.
- Nominate graph and capacity controls for the temporal comparison; keep an honest conclusion if extra capacity explains the gain.

### T09 [I, workstation/CLI] Implement and validate Mamba

- Prerequisites: T02, T05. May execute before T07/T08 finish because the interface is independent.
- Verify recurrence/causality, reference outputs where available, gradients, checkpoint round-trip and GPU compatibility.
- Smoke-test old/new temporal blocks across matched graph modes. Measure memory and batch-one latency including declared preprocessing; GPU throughput is not edge-device latency.

### T10 [T, workstation] Run the source-only graph-by-temporal pilot

- Prerequisites: T08, T09.
- Minimal matrix: two temporal blocks x no graph/physical/nominated graph = six conditions. Include the local-capacity control on the new backbone if needed to attribute the correction gain.
- Fix features, folds, source validation and training exposure. Record parameter differences. Reuse compatible existing runs; changed data order/precision requires fresh matched controls.
- This is a bounded pilot, not a full Cartesian search over all prior graph ideas.

### T11 [RD, laptop] Freeze the final centralized study

- Prerequisite: T10.
- Lock architecture, required arms, seeds, epochs, data view, augmentation/GAE controls, efficiency procedure and exclusion rules before held-out evaluation.
- O2 bridge: recheck magnitude/mag_dot versus raw/rotation-augmented controls on the final backbone where necessary; do not claim historical O2 findings automatically transfer to changed architecture.
- Decide which external comparisons are feasible and essential (e.g. adapted DynamicWHAR, a matched invariant comparator). Reuse valid results, label adaptations and avoid unmatched published-score rankings.
- Publish exact expected job/artifact counts and session partitions. No editing the queue based on interim test winners.

### T12 [T, workstation] Execute final O1/O2 centralized comparisons

- Prerequisite: T11.
- Sequential frozen jobs with repeated seeds and chosen ideal/recorded-displacement coverage; evaluate ideal, self, induced, synthetic rotation and required dropout tests separately.
- Produce best/latest checkpoints, predictions, recording/class metrics, normalization and split manifest, timing/memory and source snapshots. Queue postprocessing without an LLM.

### T13 [I, workstation then laptop] Verify artifacts and assemble the analysis bundle

- Prerequisite: T12.
- Check expected counts and histories, readable checkpoints, prediction/label/index alignment and hashes. Recompute summary metrics from predictions.
- Save paired subject/seed/recording effects, per-class/confusion tables, worst-subject analysis, learning curves, mechanism interventions and hardware-qualified cost.
- Build compact summary.json, metrics.csv, analysis.md, figures, manifest.json and error/status files. Retain large checkpoints/raw predictions on workstation plus user-managed backup; make required subsets available locally for laptop analysis.
- The user manually transfers/synchronizes. No agent Git commands, staging, commits, pushes, pulls, hooks or repository setting changes.

### T14 [RD, laptop] Close the centralized O1/O2 conclusions

- Prerequisite: T13.
- Assess practical effect and subject/seed variation; seeds are not independent subjects. Separate representation, message operator, dynamic routing and temporal contributions.
- Decide whether routing interventions establish useful sample dependence. State what is exploratory and which claims remain unsupported.
- Freeze the architecture/representation entering O3. Additional data collection/confirmation, if feasible under T01, is a separately frozen branch before final claims.

### T15 [RD, laptop] Freeze O3 protocol and completion criteria

- Prerequisite: T14.
- Specify subject clients, participation, rounds/local steps, optimizer-state policy, normalization and aggregation. Keep known-client disjoint-recording evaluation separate from unseen users with source-only statistics.
- Select centralized, local-only, FedAvg and client-local-normalization controls. Specify mean/worst-client metrics, communication bytes and runtime.
- Current old config has 4 arms x 4 methods x 3 folds x 3 seeds = 144 bundles; re-evaluate required scope for the FINAL model. This is a reference count, not a mandatory rerun of 144 unchanged bundles.
- Match or explicitly account for central versus federated optimizer exposure. Federation is not a formal privacy or on-device demonstration.

### T16 [I, workstation/CLI] Adapt existing O3 framework to the frozen model

- Prerequisites: T02, T15.
- Reuse existing simulator; bridge any new encoder/SSM and check aggregation, local normalization, client boundaries, per-client optimizer and RNG recovery.
- Fix path/config inconsistencies in future runs; preserve old snapshots. Add deterministic result and communication accounting.

### T17 [T, workstation] O3 smoke/pilot and calibration

- Prerequisite: T16.
- Verify real-client round execution, centralized architecture bridge, interrupted round recovery, hardware fit and method-specific runtime.
- Pilot results do not replace the full federated study.

### T18 [RD, laptop] Approve the frozen O3 run specification

- Prerequisite: T17.
- Review source-side behavior and feasibility; freeze rounds/steps, seeds, arms and session partitions. Resolve failures before launch.

### T19 [T, workstation] Run required O3 comparisons

- Prerequisite: T18.
- Execute fixed sequential batches, checkpoint safely, log communication and client metrics, stop on failure. No target-statistics tuning for unseen users.

### T20 [I, workstation then laptop] Validate O3 results and export compact evidence

- Prerequisite: T19.
- Verify rounds/client coverage/checkpoints; recompute mean and worst-client placement metrics; compare robustness retention against centralized and local-only controls.
- Package raw supporting artifacts and compact summaries with manifest/hash identities for user-managed transfer.

### T21 [RD, laptop] Final scientific synthesis and any independent confirmation

- Prerequisites: T14, T20; independent cohort results if promised in T01.
- Resolve the final claims, negative results, uncertainty, novelty boundaries and limitations across O1/O2/O3.
- Do not add unplanned model searches merely to improve a table. External confirmation must use a frozen model/protocol.

### T22 [I, laptop/workstation] Reproducibility and efficiency closeout

- Prerequisites: T13, T20.
- Verify a clean environment can load the final model and reproduce an inference/metric example. Check manifests, configurations, prediction metrics and one recovery path.
- Finish measured parameter/memory/batch-one latency and communication tables. Preserve raw evidence and produce a reproducible analysis command; never delete older results.

### T23 [RD, laptop] Write and critically review the thesis and presentation

- Prerequisite: T21. Literature/method drafting may begin after T01 while training runs; final result claims wait until T21.
- Align objectives, equations, implementation, experiments and conclusions. Correct Mamba terminology, historical comparisons, citation mismatches and privacy/generalization claims.
- Include limitations, failure cases, negative findings and scoped future work. Distinguish completed experiments from supported hypotheses.

### T24 [I, laptop] Final deliverables and defense preparation

- Prerequisites: T22, T23.
- Compile/check thesis, inspect figures/tables/citations, finalize presentation and a reproducible demo. Rehearse answers about graph controls, invariance, data leakage, Mamba fidelity, federation and efficiency.
- Obtain supervisor feedback, incorporate corrections, and prepare the final submission package. Submission itself remains a user action unless separately requested.

## 3. Parallelism and handoff cadence

- T01 and T02 can proceed independently.
- After T01, prepare T05 while T03/T04 investigate O1.
- T09 can overlap T06/T07 on separate machines; never alter sources already snapshotted for an active run.
- Use laptop training only for small pilots that fit comfortably. Use workstation for required multi-seed suites, SSM integration and federation; large GPU does not automatically imply shorter runtime for tiny models.
- While a workstation queue runs, the laptop may study literature, draft methods or analyze COMPLETED immutable runs. Do not alter that queue's scientific decisions.
- Compact cycle: RD specification -> user manual synchronization -> CLI implementation/tests -> visible independent training -> automatic analysis bundle -> user manual synchronization -> laptop RD review.
- A failed check returns a failure report to the next chat. No AI remains available to diagnose after the CLI is closed, and no automatic agent scheduling is being created here.

## 4. Required specification before closing each research chat

Create only one detailed specification per approved experiment: `docs/experiments/<experiment_id>.md`, with a matching machine-readable run configuration in the existing configuration layout. This directory/pattern is a planned convention, not additional files created by this planning turn. Freeze a copy under the run's source snapshot when executed. Reuse the three working documents rather than adding repeated next/update/handoff files.

Each specification must contain:

1. Experiment ID, prerequisite IDs, objective, factual evidence, competing hypotheses and concise mathematical justification.
2. Equations, tensor shapes, units, node order, masks, invariance assumptions, self/neighbor paths, initialization, normalization, losses and gradients that should be nonzero.
3. Exact code files/interfaces allowed to change and protected historical artifacts; reference implementation/parity expectations.
4. Dataset/cache identity and exclusions, train/validation/test manifests, metrics, checkpoint selection and prohibited test-dependent decisions.
5. Exact candidate/control matrix, seeds, epoch/round/local-step budgets, precision, batch size and parameter/capacity accounting.
6. Mandatory tests, smoke/pilot/full gates and quantitative pass/fail criteria; explicit scientific choices that CLI must return rather than invent.
7. Environment requirements, relative/configured paths, memory expectations and calibration procedure.
8. Entry/resume commands, visible-terminal/CLI-close survival requirement, source/data hashes, expected outputs/counts, failure behavior and session budget.
9. Analysis equations and report schema, uncertainty unit, paired comparisons, mechanism interventions and interpretation boundaries.
10. Decision state: draft / ready for implementation / ready for execution / complete / failed, plus the next RD review question. Do not claim unspecified choices are already frozen.

Before the CLI closes, save a launch receipt: run ID, absolute output path on that machine, environment, source/config/data hashes, current PID/heartbeat, evidence of an advancing checkpoint, expected jobs, measured ETA and exact resume command. A safe launch does not mean the experiment completed.

## 5. Session duration and efficiency

- Five to seven HOURS is the maximum preferred session grouping when useful, not a requirement to occupy the GPU that long.
- Calibrate separately on each machine and each materially changed architecture. Estimate T = sum(job epochs x measured epoch time + evaluation/checkpoint/loading time) + safety margin.
- The latest laptop pilot consumed about 255 seconds summed across six eight-epoch jobs. At unchanged workload, a three-arm/three-fold/three-seed 31-epoch study is roughly 75 minutes of job work by linear extrapolation, plus overhead and uncertainty; measure rather than promise. Mamba/federation need fresh calibration.
- Group only preapproved independent jobs to fill a session. Split a larger fixed matrix across sessions without reducing scientific training budgets or changing checkpoints based on interim test scores.
- Prefer reusable artifact checks and ordinary reporting scripts over an LLM watching epochs. Once chats are closed, training consumes compute/storage but requires no further model interaction unless a model-calling feature was explicitly added (none is planned).

## 6. Model/token policy

- Keep GPT-6 Astra Medium for ambiguous scientific decisions, mathematical contracts, contradictory evidence and final claims. Request a concise evidence-backed decision record and implementation specification, not verbose internal reasoning transcripts.
- Keep GPT-6 Sol Medium for new architecture code, checkpoint/aggregation bugs and CUDA integration.
- Use GPT-6 Luna Low/Medium, if available, for a frozen launch checklist, copying known configuration values, deterministic report extraction and routine resume triage. Escalate unexpected bugs to Sol; scientific redesign returns to Astra.
- No model is needed simply to execute a validated Python/PowerShell command or run a deterministic queue. Manual launching can save the entire CLI interaction when nothing needs changing.
- Load project_status.md and the current experiment specification first; retrieve only relevant code/artifacts. Avoid re-reading resources and all historical logs every session.
- Keep full raw logs locally but feed the model compact metrics, paired differences, curves, failure excerpts and hashes. Ask for one decision per review and avoid speculative implementations.
- Model recommendations reflect official guidance reviewed on 28 October 2026: https://learn.chatgpt.com/docs/models and https://learn.chatgpt.com/docs/model-selection . Exact availability/usage depend on the account; no fixed savings percentage is promised.

## 7. Completion condition

The MTP is ready for final submission when required O1/O2/O3 experiments and controls are completed, the implementation matches the claimed method, results are reproducible, positive/negative conclusions are supported, and thesis/presentation/limitations are coherent. An O1 graph win is the research target; its absence does not justify hiding evidence or endless unplanned experiments. Agree any change of thesis scope with the supervisor.


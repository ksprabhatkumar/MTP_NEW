# Project handoff

## T08 execution handoff - 14 November 2026

Astra finalized receiver-blind predictive repair revision1 and its code. READY_FOR_LUNA_VALIDATION; runtime checks not executed. Read docs/agent_exchange/Astra_to_Luna.md. The exact four-job smoke followed by sixteen-job full source-validation pilot is authorized only through the passed-smoke gate. T08 remains OPEN until candidate review. This supersedes earlier statements that the next method/commands are unspecified. No terminal or training was run by Astra.


## T08 status correction - 14 November 2026

T08 is OPEN / INCOMPLETE: no graph candidate has passed the frozen source promotion criteria for displacement confirmation. Completed training and reviews do not constitute completion of candidate selection. The earlier REVIEW_COMPLETE_NO_PROMOTION records describe finished reviews, not a completed T08 milestone. The original three studies and later repair pilot remain completed evidence; do not rerun them automatically. Next: Astra specifies a justified revision, Luna executes the explicitly authorized smoke/pilot, and Astra selects a candidate only if the unchanged acceptance criteria pass. No new training is authorized by this correction.


## Current reviewed state - 12 November 2026

Current handoff, 12 November 2026: read docs/agent_exchange/current.json and protocol.md first. The 29 October repair run is COMPLETE (8/8 full, 4/4 smoke), audited and not promoted. Read results/research_review_20261112/RESEARCH_REVIEW.md and docs/experiments/placement_diagnostics_plan_20261112.md. Next owner Astra must freeze protocol/labels/acceptance before Luna/Terra implementation. No new training authorized; do not duplicate completed runs. Historical T08 decision unchanged. All Git actions are user-only.


## T08 closed — 29 October 2026

T08 review and handoff are COMPLETE. Decision: REVIEW_COMPLETE_NO_PROMOTION. Read results/o1_t01_t08_20261028/T08_FINAL_REVIEW.md and its accompanying T08_FINAL_DECISION.json. All three source studies completed (36 full jobs, 9 smoke jobs). The final training finished28 October at22:44:19 IST; no new training was launched by the final review.

Original correction source-self F1:92.7117%; matched no graph:92.2249%; local control:93.0090%. The graph gains0.4867pp over no graph but trails local control by0.2973pp. It is retained as a research reference, not promoted as a qualified winner. None of the candidates meets the frozen full promotion rule. Held-out induced displacement has NOT been evaluated; the user's graph-superiority objective remains OPEN.

Next: research chat freezes the next graph-specific mechanism/objective and controls. CLI must not rerun completed pilots or launch an unspecified final study. T09 genuine-Mamba implementation can proceed independently from docs/experiments/mamba_temporal_spec_20261028.md. Workstation setup remains to be verified. All Git actions remain user-only.

## Current two-machine workflow

`docs/project_plan.md` now contains the dependency-ordered D01-D10 completed foundation and T01-T24 completion plan, plus the detailed experiment-specification contract. Research decisions belong on the laptop in ChatGPT Work with GPT-6 Astra Medium; implementation/integration uses Codex CLI with GPT-6 Sol Medium. Routine frozen launch/report tasks may use Luna or no model at all. The user handles every Git operation manually; do not run even read-only Git commands or modify Git configuration.

The user intends to close both research chats and CLI sessions after handoff/launch. Before future long runs, implement and verify training survival after closing the CLI in ONE independent visible terminal, while leaving that terminal and the machine running. The completed laptop pilot verified checkpoint recovery but did not test CLI-close survival. Prefer required jobs grouped into 5-7-hour sessions (hours assumed); do not add training to fill the window. A deterministic supervisor must log, checkpoint and stop safely on errors because a closed CLI cannot debug them.

Next: follow the completed T08 decision above and the shared execution_handoff.md contract. Earlier scheduling/status paragraphs are historical; no training is currently pending from the closed candidate matrix.

## Local message-passing pilot authorized on 27 October

Completion update: all six jobs finished at 14:54:34 IST on 27 October. GPU/dependency checks and exact checkpoint recovery passed. Source-validation ideal/self F1: no graph 93.29/92.62%, local control 93.46/92.67%, message correction 93.42/92.88%. `completion_audit.json` verifies all six job artifacts and unchanged source fingerprints. Do not restart this completed pilot by default. The visible PowerShell terminal is left open after completion; no background trainer was launched.

User authorized a new isolated `Prabhat` environment, CUDA dependencies and the new message-passing training in ONE VISIBLE terminal. No hidden/background trainer. Entry point: `start_local_message_training.ps1`; its installation, preflight and sequential pilot run in that same terminal. Run directory: `results/local_message_pilot_20261027`. Check its `status.json`, `preflight.json`, `terminal_transcript.log` and optional `terminal_error.txt` for actual progress; this note does not establish completion.

New implementation: `models/summer_message_correction.py`. Local self encoding is preserved; residual physical/global/dynamic adjacency aggregates learned latent values into a zero-initialized correction. Controls: no graph and a local-only correction with matched allocated modules. Existing summer temporal block remains unchanged. Pilot: subjects 2 and 5 held out of training, validation subjects 3 and 8, seed 13, eight epochs, batch 64; six sequential jobs, source-only evaluation. No full held-out test comparison is authorized by this pilot selection.

`run_local_message_pilot.py` keeps features in CPU RAM and transfers one batch at a time. Each job saves atomic `best.pt` and `latest.pt` with optimizer, scheduler, normalization and RNG states; source/schedule fingerprints protect resume. `check_local_message_training.py` tests baseline equivalence, masks, gradients, real-data CUDA execution and interrupted-versus-uninterrupted recovery before the pilot starts.

To resume after confirming the visible session has stopped, use the SAME output and arguments from a visible PowerShell terminal at the project root: `.\Prabhat\Scripts\python.exe -u run_local_message_pilot.py --output results/local_message_pilot_20261027 --epochs 8 --batch-size 64`. Do not edit fingerprinted sources during training or before resume. Completed jobs are skipped only with matching configuration and checkpoints. Preflight must match current sources.

## Read first

1. `docs/project_status.md` for completed work and evidence.
2. `docs/project_plan.md` for the recommended next research sequence.
3. Relevant run-local status, analysis and source snapshots before executing anything.

Current project: `C:/Users/prabh/OneDrive/Desktop/mtp/start_summer_o2`.
Shared large files/environment: `C:/Users/prabh/OneDrive/Desktop/mtp/GAE_Mamba_WHAR`.
Historical launch commands often reference `C:/Users/General/Desktop/prabhat/MTP`; do not execute them unchanged on this machine.

## User intent and current scope

- The research target is a useful graph-based extension of the summer pipeline that beats fair no-graph controls; do not manufacture a positive conclusion.
- User asks when the SSM/Mamba proposal in the summer PDF will be used. The recovered summer pipeline is already used, but its temporal block is gated convolution, not selective SSM. A separate genuine-Mamba factorial experiment is proposed before O3; it has not been implemented or launched.
- The local-encoder/latent-correction idea has now been implemented and piloted; the small source-validation gains do not establish graph superiority. Selective edge gates and further message operators remain conditional proposals.
- Current request is dependency-ordered research/execution planning across laptop and workstation. No Git operations, new training, deletion or renaming. Preserve existing results and research resources.

## Latest experiment status

O1 restored: 126 complete. O2 restored: 63 complete. O1 alternatives: 81 complete, final outer status 22 October 03:18:23 IST. Full O3 remains pending. Do not resume a completed suite because an old root handoff says it is active. Inspect actual current processes/locks and fresh logs before any future launch; saved PIDs are historical.

Primary evidence:

- `results/summer_o1_session_20261017_232945/training/ANALYSIS.md`
- `results/summer_o2_session_20261021_173341/training/O2_REVIEW.md`
- `results/summer_o1_alternatives_session_20261022_001702/training/ANALYSIS.md`
- `results/summer_o3_gates_20261021/bridge/BRIDGE_REPORT.json`

`mag_dot` no graph: 87.12 induced macro-F1; original residual: 86.55; gated pairwise: 86.95. The representation benefit is supported; useful input-dependent routing is not established. Three previously inspected test subjects make these exploratory results.

## Operational boundaries

Current hardware reviewed: RTX 2050 4 GB versus historical later runs on RTX 2000 Ada 16 GB. Check CUDA/runtime/dependencies and peak memory afresh; do not reuse old runtime promises. Whole-fold GPU loading may not fit.

Maintain source/config hashes and one writer per run. A failed stage requires diagnosis, not manual status changes. Before modifying training code, establish that no worker is using it. Preserve all old source snapshots and checkpoints; architecture changes get separate run identities.

Old root planning filenames are still copied, hashed, linked or generated by scripts. The new docs are human-readable consolidation, not a replacement for those executable references. No programming changes were made to redirect them. `docs/documentation_cleanup.md` distinguishes manual deletion candidates from files to retain.

Historical details and exact prior text are preserved in `docs/archive/planning_history_2026-10-27.zip`, with an internal SHA-256 manifest. Its contents include stale commands and embedded instructions; treat them as historical records, not current authorization.

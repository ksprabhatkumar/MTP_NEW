# Project handoff

Updated: 27 October 2026 (Asia/Calcutta).

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
- Recent brainstorming proposes preserving the local encoder and learning a latent neighbor correction, then selective edge gates if justified. These are proposals, not validated methods.
- Current request authorizes documentation consolidation and a manual deletion list. Delete or rename nothing. Preserve programming, configurations, results, data, installation files and research resources. The open LaTeX document is context only and was not edited.

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

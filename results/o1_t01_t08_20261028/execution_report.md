# T01-T08 execution report — 28 October 2026

Final closeout29 October: T08 review COMPLETE, no candidate promoted. The complete three-study decision superseding the first-pilot-only conclusions below is [T08_FINAL_REVIEW.md](T08_FINAL_REVIEW.md). 36 full jobs and9 smoke jobs finished; last training28 October22:44:19 IST. Graph superiority on held-out displacement remains unresolved.


## Outcome

- T01 DONE (research): inherited protocol/resources retained; remaining numerical criteria frozen in docs/experiments/o1_neighbor_selection_20261028.md.
- T02 DONE for laptop (implementation): CUDA preflight, exact recovery, source/data fingerprints, run locking, atomic checkpoints, heartbeats, separate visible terminal and artifact audit. Original launcher exited while trainer continued. Actual closure of this desktop app was not tested. Workstation environment verification remains tomorrow's step.
- T03 DONE (diagnostics): old correction interventions plus new distant-window donor audit; no repeated dataset audit.
- T04 DONE (research): neighbor-only value correction with receiver gate, matched local-value control, one fixed candidate family.
- T05 DONE (research specification): genuine Mamba recurrence/interface/platform/test contract in docs/experiments/mamba_temporal_spec_20261028.md. T09 implementation is not done.
- T06 DONE (implementation): new isolated operator, runner, preflight and analysis scripts. Historical programming files/results preserved.
- T07 DONE (training): three-arm four-epoch smoke, then twelve jobs (three arms x two folds x two seeds x16 epochs), no failures; all artifact checks passed. Full pilot finished 15:20:58 IST, about13.7 minutes summed job time; peak PyTorch allocated GPU memory70.1MB (not total device/process use).
- T08 DONE as a research review/decision, NOT a successful promotion: candidate fails the frozen source criterion. Graph source-self F1 92.2214%, no graph92.3278%, local control92.5451%. Provisional graph remains available; no validated winner selected. O1 displacement superiority remains unresolved and was not evaluated here.

## Handoff to research chat / GPT-6 Astra Medium

Read full/T08_DECISION.md, full/selection_decision.json and the two experiment specifications. Do not rerun completed jobs. Use paired effects and distant donors, not mean scores alone. A source-justified open-gate optimization ablation or matched source-corruption objective is the next O1 decision; specify it before any new training. Avoid promoting the rejected candidate merely because it implements a graph. Genuine Mamba integration can proceed independently tomorrow.

## Handoff to Codex CLI / GPT-6 Sol Medium

No active training remains in this session. Current 12-job matrix is COMPLETE. Do not launch it again to fill a5-7-hour window. Reproduction commands from project root:

- .\Prabhat\Scripts\python.exe analyze_o1_selection.py results/o1_t01_t08_20261028/full
- .\Prabhat\Scripts\python.exe summarize_o1_decision.py
- .\Prabhat\Scripts\python.exe audit_global_message.py

The existing resume command is in launch_receipt.json; original fingerprinted sources/config required. A future changed method requires a new output/specification and fresh smoke. User performs every Git action. Both machines retain project-relative dataset paths and build their own environments.

## Evidence and operational limits

full/completion_audit.json verifies12 histories, latest/best checkpoints and metrics recomputed from logits. Smoke has its own equivalent audit. full/source contains the frozen implementation. Full data/source hashes in fingerprint.json; metadata, logits, per-class/confusion results, CSV curves, graph samples, interventions and PNG/PDF plots are retained. Separate donor outputs include exact receiver/donor indices. No results were deleted.

The independent visible terminal stays open. Independence evidence concerns launch-process exit, not shutdown of the computer or closing the training terminal. A closed CLI cannot repair failures; scripts save failure status and stop. No automatic inter-chat messages or Git synchronization were created.

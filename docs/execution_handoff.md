# Research chat <-> Codex CLI handoff

## Current reviewed state - 12 November 2026

Current override, 12 November 2026: use docs/agent_exchange/protocol.md and current.json for Astra <-> Luna/Terra exchange. Graph repair is COMPLETE and reviewed, not running. Full report: results/research_review_20261112/RESEARCH_REVIEW.md; latest run audit: results/o1_graph_repair_20261029/full/reproducibility_audit.json. Future diagnostics/experts plan needs a frozen executable revision before implementation/training. Older run instructions below are historical. No Git actions.


## 1. Research-chat responsibilities (GPT-6 Astra Medium)

- Read project_status.md, the current dated override in project_plan.md, and the latest execution report. Reuse completed resource decisions and audits.
- Write one docs/experiments/<experiment_id>.md containing: current state, hypothesis, equations/shapes, exact permitted code changes, matched controls, data/split hashes, seeds, epochs, batch/precision, selection metric, acceptance thresholds, tests, expected artifact count, full/resume commands and measured-time budget when available.
- Distinguish source-side candidate selection from recorded-displacement confirmation. No graph superiority claim from smoke/source pilot alone. Preserve strong matched no-graph and local-capacity controls.
- State READY_FOR_IMPLEMENTATION only when scientific choices are explicit. Missing scientific choices return to research chat; CLI must not invent them.
- Next immediate scientific task: T03/T04 mechanism review of completed local_message_pilot_20261027, then the exact next O1 specification. The existing pilot is complete and should not be blindly rerun.

## 2. CLI responsibilities (GPT-6 Sol Medium)

- Read this contract, current experiment specification and relevant code only. Verify synchronized source/config hashes and local dataset/cache identity without Git. Never copy a venv between machines.
- Implement the frozen specification and logging below. Save execution_report.md under the experiment output, including changes, checks, exact commands, unresolved questions and state.
- Run an end-to-end smoke in one visible terminal. Cover every materially different arm, real CUDA forward/backward, finite values, evaluation, checkpoint save/load/resume, source hashes, prediction alignment, all expected artifacts and regeneration of a metric plus plot from saved data. A full epoch on representative data must finish. Use a separate smoke output.
- Full-run gate: smoke exit code zero AND required artifact checks passed for current source/config/environment. Changed source/config invalidates the gate. Never use a stale preflight merely because its file exists.
- Launch the exact preapproved full queue after the gate passes; no additional permission is needed for that specified run. If a scientific choice is missing or the smoke fails, write NEEDS_RESEARCH_DECISION or FAILED; do not launch.
- The current run_local_message_pilot.py is a fixed six-job SOURCE-ONLY pilot, not the final displacement runner. Do not relabel it or fabricate a full-run command before implementing the required matrix/evaluation.
- Before CLI exit: save launch_receipt.json with run ID, source/config/data hashes, executable/environment, PID, output path, advancing heartbeat/checkpoint, expected jobs, ETA, exact resume command and terminal-survival verification evidence.
- Verify trainer lifetime is independent of CLI. A first close-CLI test needs a short run plus later heartbeat/checkpoint verification; merely opening a window is insufficient. Leave the independent training terminal open and machine powered/awake.
- Queue automatically audits artifacts and generates summaries after completion. On error save traceback/status and a recoverable checkpoint where possible, then stop. Do not change hyperparameters automatically. CLI cannot repair errors while closed.

## 3. Required plot-ready evidence (implementation requirement)

- manifest.json: experiment/job/arm IDs, timestamp, source/config/data hashes, exact command, software/hardware, split and exclusion manifests, sensor/channel order, feature/normalization parameters, class mapping, seed and checkpoint-selection rule.
- history.csv or equivalent lossless JSON: job ID, arm, fold, seed, epoch/round, train/validation loss, accuracy, fixed-33 macro-F1, learning rate, elapsed seconds, epoch duration, peak allocated/reserved GPU memory. Include sample counts and metric definitions. Flush incrementally.
- predictions.npz per evaluation split/checkpoint: true class, predicted class, probabilities or logits, stable original window indices, subject, recording, scenario and window timestamp/start index. Save index metadata separately if necessary. Explicitly align every row. Preserve per-run raw values rather than only rounded means.
- metrics.csv/JSON: scenario/subject/recording/class metrics, confusion counts and class support; paired graph-minus-no-graph and graph-minus-local-control effects, ideal-to-displaced degradation. Keep each seed before aggregation; distinguish percentage points from fractions.
- graph diagnostics: local/message/correction norms, branch weights, edge entropy/sparsity, relevant gradient norms and zero/uniform/shuffled-neighbor effects. Save deterministic representative adjacency samples with window IDs/labels/checkpoint IDs. Do not save all training activations; fixed sampling controls disk cost.
- efficiency.json/CSV: parameters, warmup/repetition protocol, batch-one latency distribution, throughput, memory, hardware, precision and included preprocessing.
- federation: per-round/per-client loss and metrics, participation and sample counts, aggregation weights, local steps, uploaded/downloaded bytes, client/server timing, normalization policy and global/local checkpoints.
- best.pt and latest.pt: model, optimizer, scheduler, RNG, epoch/round, normalization and resume metadata. Save atomically. status.json and heartbeat distinguish running/failed/interrupted/complete; never mark complete before final artifact audit.
- analysis scripts plus plotting configuration, CSV inputs and figures in PNG and vector PDF/SVG. Recompute a selected plot and its metrics without rerunning training before declaring logging complete.

## 4. Return to research chat

- CLI/scripts leave execution_report.md, summary.json, metrics.csv, plots, completion_audit.json and failure excerpts where applicable in the same run directory. Full checkpoints remain backed up and accessible; user decides which large artifacts to synchronize.
- State sequence: DRAFT -> READY_FOR_IMPLEMENTATION -> SMOKE_PASSED -> RUNNING -> COMPLETE/FAILED -> RESEARCH_REVIEWED. Use NEEDS_RESEARCH_DECISION for unresolved science. T08 means candidate selected, not O1 superiority achieved.
- Research chat reads the compact results, checks paired effects and uncertainties, records the decision and writes the next specification. Never overwrite a completed run or silently modify an active experiment.

## T08 closed — 29 October 2026

T08 review and handoff are COMPLETE. Decision: REVIEW_COMPLETE_NO_PROMOTION. Read results/o1_t01_t08_20261028/T08_FINAL_REVIEW.md and its accompanying T08_FINAL_DECISION.json. All three source studies completed (36 full jobs, 9 smoke jobs). The final training finished28 October at22:44:19 IST; no new training was launched by the final review.

Original correction source-self F1:92.7117%; matched no graph:92.2249%; local control:93.0090%. The graph gains0.4867pp over no graph but trails local control by0.2973pp. It is retained as a research reference, not promoted as a qualified winner. None of the candidates meets the frozen full promotion rule. Held-out induced displacement has NOT been evaluated; the user's graph-superiority objective remains OPEN.

Next: research chat freezes the next graph-specific mechanism/objective and controls. CLI must not rerun completed pilots or launch an unspecified final study. T09 genuine-Mamba implementation can proceed independently from docs/experiments/mamba_temporal_spec_20261028.md. Workstation setup remains to be verified. All Git actions remain user-only.


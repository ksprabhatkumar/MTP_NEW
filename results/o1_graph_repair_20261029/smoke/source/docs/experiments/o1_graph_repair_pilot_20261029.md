# O1 reliability-aware graph repair: frozen pilot specification

Date: 2026-10-29. Status: implementation and validation pilot; not an established 4–5 percentage-point improvement.

## Purpose and controls

Determine whether receiver-conditioned repair can improve natural self-placement validation over an equally trained no-graph model. Preserve the previous T08 no-promotion decision and all historical results.

Four arms allocate identical modules and initialize the shared encoder, temporal block, and classifier identically:
- `none_clean`: existing no-graph path, clean classification only.
- `none_robust`: no graph, paired clean/corrupted training with all auxiliary losses.
- `local_repair`: same repair capacity, but each receiver uses its own feature as the sender. No other sensor feature or quality enters its correction.
- `graph_repair`: actual other-sensor messages, learned edge scores and sender reliability.

The existing temporal block/head is preserved; it already mixes sensor channels. Its name includes Mamba but this experiment does not introduce a genuine selective SSM.

## Repair mechanism

Per-node temporal mean/std predicts reliability q. Receiver-conditioned signed values are phi(h_i,h_j,e_ij)-phi(h_i,0,e_ij), with learned pair embeddings. Edge logits combine learned compatibility, a physical/global positive prior, and log(q_j). Self edges and unavailable senders are masked. Softmax includes a learned null neighbor; real weights are never renormalized after excluding it. Thus low reliability can reduce the total received signal. A receiver gate initially sigmoid(2) and zero-initialized output projection preserve the local path at initialization. No corruption mask is given to the classifier.

## Training recipe

Train only valid ideal recordings, excluding the held-out subject and source validation subject. Existing folds exclude subjects 2 and 5 and validate on subjects 3 and 8 respectively. Normalization and corruption scales use training recordings only.

Two clean warmup epochs. Thereafter the three robust arms use:
L = 0.5*(CE_clean+CE_corrupt) + ramp*(0.5*KL_T2 + 0.1*cosine_repair + 0.1*quality_BCE).
Ramp is min(1,(epoch-2)/3). KL includes T squared. Teacher is an EMA (0.99) of student parameters and floating buffers, evaluated on clean inputs. Integer buffers are copied. Latent repair targets the clean teacher's local encoder and is averaged only over perturbed sensors/time. Reliability target is exp(-relative squared clean/corrupted teacher-local discrepancy), clipped to [0.01,1]. It is a training proxy, not calibrated real-world confidence. Evaluation uses the student.

Synthetic stress: 75% of windows select 1–3 available sensors; each acc/gyro modality has gain U(0.8,1.2), plus four-knot smooth drift scaled by 0.15 times train-only modality standard deviation. Draws use independent CPU RNG streams, separate from shuffling and model dropout. The same robust arms receive identical draws and sample ordering.

The local REALDISP manual confirms 50 Hz but does not establish acc/gyro units. Consequently this is unit-free synthetic stress, NOT a validated physical displacement simulator. No assumed SI lever-arm formula is used. Pure rotation is ineffective augmentation for these rotation-invariant magnitude/dot features. Validation recordings are untouched natural ideal/self-placement recordings.

Adam lr 1e-4, weight decay 5e-4, existing scheduler, gradient clipping 1, hidden 16, batch 64, BF16 when supported otherwise FP32. Checkpoint selection uses ideal validation fixed-33 macro-F1 only. Log both validation accuracies/F1 each epoch; do not select epochs using self-placement validation.

## Execution and acceptance

1. Run check_graph_repair.py in one visible terminal: graph invariants, controls, nonfinite checks, teacher and exact state recovery.
2. Smoke: four arms, excluded subject 2, seed 13, 4 epochs (includes augmented training).
3. Only if smoke and artifact audit pass, full pilot: four arms, excluded subjects 2/5, seed 13, 16 epochs each (8 jobs).
4. Report per-fold and mean ideal/self validation accuracy and macro-F1. Require a gain over robust no-graph and local capacity controls before proposing broader confirmation. This one-seed pilot cannot establish a 4–5 pp held-out displacement improvement.
5. No held-out induced evaluation or hyperparameter changes during the frozen matrix. Failures stop the launcher. Code changes after a partial frozen run require a new output directory and fresh smoke.

## Saved evidence and handoff

Run `powershell -NoExit -ExecutionPolicy Bypass -File .\start_graph_repair.ps1` from the project. It uses Prabhat and does not install packages. The visible terminal remains open independently of the chat/CLI. Never start a duplicate runner. Resume using the identical command/output only when the original process has stopped; a process lock and source/data/config fingerprint reject incompatible recovery.

Outputs: results/o1_graph_repair_20261029/{smoke,full}. Every epoch commits latest.pt with student, teacher, optimizer, scheduler, global RNG, independent RNG streams, history and normalization; best.pt stores the selected student. Save original logits/labels, corpus indices, subject/recording/scenario/window metadata, graph samples, per-class metrics, graph message interventions, losses, order hashes, memory, durations, plots and CSV tables. Source snapshots and hashes plus environment lock permit workstation reproduction. All changes are transferred by the user; no Git operations are performed.

Chat handoff: inspect full/status.json, audit.json, summary.json and REPORT.md before claiming completion. CLI handoff: launch once visibly, wait for smoke/audit success, then let the same launcher run the full matrix. If failed, preserve error.txt/terminal_error.txt and checkpoints; fix the cause before a fresh incompatible run. A live chat is not required for checkpointing or progression.

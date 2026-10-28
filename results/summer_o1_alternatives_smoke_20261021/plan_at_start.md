# Objective 1 graph follow-up: prepared training design

Status: implementation and smoke validation in progress, 21 October 2026. This is a new exploratory follow-up to the completed O1 and O2 sessions, not a replacement for their results.

## Question and locked setting

The original summer O1 residual graph reached 11.31% induced fixed-33-class macro-F1 versus 13.63% for no graph. With O2's magnitude+dot input, residual reached 86.55% versus 87.12% for no graph. In both runs, learned adjacency was almost constant and changing dynamic affinities flipped no sampled predictions. The follow-up asks whether alternative **inter-sensor routing** makes the graph useful after the representation is fixed. Use the summer GIN-attention encoder and temporal/head path, magnitude+dot input, cross-entropy, 50/25 windows, source-only checkpoint selection, subjects 2/5/15, seeds 13/29/47, batch 64 and 31 epochs. Those held-out placement subjects have been inspected, so their results remain exploratory.

## Seven matched arms

| Arm | Graph operation | Purpose |
|---|---|---|
| physical | Existing normalized physical adjacency | Fixed graph reference |
| residual | Existing physical/global/QK residual mixture | Original dynamic reference |
| none | Existing no-neighbor operation | Required contribution control; residual remains the target |
| masked_balanced | Mask unavailable/self senders before QK softmax, normalize each branch, renormalize mixture over nonempty branches | O1-F routing repair |
| masked_floor | Same, with at least 25% dynamic mixture weight where that branch has senders | Test branch dilution separately |
| pairwise_balanced | Replace QK scores with a learned directed pairwise descriptor MLP; use masked balanced mixing | DecomposeWHAR-inspired affinity adaptation, not its full model |
| pairwise_floor | Pairwise scores with the same dynamic floor | Test pairwise scoring and branch strength together |

The pairwise arms are project adaptations and are not a DynamicWHAR or DecomposeWHAR reproduction. The previously completed six-axis DynamicWHAR adaptation stays an external historical comparator. No published score is imported into this study. Only the graph construction changes between new arms; the summer encoder and downstream backbone are preserved. Pairwise arms have extra active parameters, so report parameter count and cost and interpret capacity as a limit.

## Gates and run order

1. Verify CPU masking, missing-sensor rows, output shapes, finite gradients, and sample sensitivity. Validate the prepared schedule and run seven short GPU smoke jobs, two epochs/two batches each, on source validation only.
2. Run 14 source-only pilot jobs: seven arms on each of two source folds, seed 13, eight complete epochs. Record source ideal/self scores and source-window adjacency variation/intervention effects. Inspect these results before deciding whether the 63-job matrix is justified. Do not choose arms from held-out displacement subjects.
3. If the pilot is sound, freeze all source/config hashes and run 63 sequential 31-epoch jobs in one resumable background session. Each job writes epoch history, latest/best checkpoint, predictions, per-recording metrics, routing diagnostics, rotation check, latency, and memory. The three original controls are rerun on the same representation and budget.
4. Compare each new graph with physical, original residual and no graph by subject/seed, and assess source clean score, recorded self/induced score, routing intervention sensitivity and latency. A graph benefit requires a meaningful paired gain over controls and selective harm when sample dependence is destroyed. Report negative or mixed outcomes.

The 63-job run is estimated at roughly 2.5-4 hours based on the previous 63-job O2 run (about 2 hours 9 minutes of full-job work); pairwise routing and result checks may add time. Recalibrate from the pilot before launch. Keep GPU jobs serial, with no O3 worker running concurrently. `run_summer_o1_alternatives.py` is the single sequential entry point. Each phase uses its own result directory, and full mode requires the completed pilot path.

## Current step and expected output

Current step: finish CPU checks and the seven-arm short smoke. Expected output: a validated new O1 graph runner and a smoke directory with seven completed source-only jobs. The 14-job source pilot and 63-job long run are not started yet.

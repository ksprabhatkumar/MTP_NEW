# Objective 1 graph follow-up: prepared training design

Status: the first seven-arm CPU/smoke and 14-job source pilot completed on 21 October 2026. None of the new arms beat no graph on the two source folds. A targeted gated-message extension is now being checked before the long-run decision. This is a new exploratory follow-up to the completed O1 and O2 sessions, not a replacement for their results.

## Question and locked setting

The original summer O1 residual graph reached 11.31% induced fixed-33-class macro-F1 versus 13.63% for no graph. With O2's magnitude+dot input, residual reached 86.55% versus 87.12% for no graph. In both runs, learned adjacency was almost constant and changing dynamic affinities flipped no sampled predictions. The follow-up asks whether alternative **inter-sensor routing** makes the graph useful after the representation is fixed. Use the summer GIN-attention encoder and temporal/head path, magnitude+dot input, cross-entropy, 50/25 windows, source-only checkpoint selection, subjects 2/5/15, seeds 13/29/47, batch 64 and 31 epochs. Those held-out placement subjects have been inspected, so their results remain exploratory.

## Nine matched arms

| Arm | Graph operation | Purpose |
|---|---|---|
| physical | Existing normalized physical adjacency | Fixed graph reference |
| residual | Existing physical/global/QK residual mixture | Original dynamic reference |
| none | Existing no-neighbor operation | Required contribution control; residual remains the target |
| masked_balanced | Mask unavailable/self senders before QK softmax, normalize each branch, renormalize mixture over nonempty branches | O1-F routing repair |
| masked_floor | Same, with at least 25% dynamic mixture weight where that branch has senders | Test branch dilution separately |
| pairwise_balanced | Replace QK scores with a learned directed pairwise descriptor MLP; use masked balanced mixing | DecomposeWHAR-inspired affinity adaptation, not its full model |
| pairwise_floor | Pairwise scores with the same dynamic floor | Test pairwise scoring and branch strength together |
| gated_masked | Masked balanced QK graph with an input-dependent receiver gate on neighbor strength | Preserve the GIN self path while allowing unreliable neighbors to contribute less |
| gated_pairwise | Pairwise balanced graph with the same receiver gate | Test pairwise routing with adaptive neighbor strength |

The pairwise arms are project adaptations and are not a DynamicWHAR or DecomposeWHAR reproduction. The previously completed six-axis DynamicWHAR adaptation stays an external historical comparator. No published score is imported into this study. Only the graph construction changes between new arms; the summer encoder and downstream backbone are preserved. Pairwise arms have extra active parameters, so report parameter count and cost and interpret capacity as a limit.

## Gates and run order

1. Verify CPU masking, missing-sensor rows, output shapes and finite gradients. The first seven arms passed short GPU smoke. Smoke-test the two gated arms separately.
2. The first 14 source-only pilot jobs are complete. The best two-fold source ideal F1 was 93.40% for no graph; the strongest proposed arm, masked floor, reached 92.42%, and pairwise floor reached 92.38%. Pairwise floor caused a small fraction of source prediction changes under affinity interventions, but did not improve source accuracy. Test the two gated arms on the same two folds/seed for eight epochs and inspect source ideal/self scores and routing effects. Do not choose arms from held-out displacement subjects.
3. If the gated pilot justifies the cost, freeze all source/config hashes and run 81 sequential 31-epoch jobs in one resumable background session. Each job writes epoch history, latest/best checkpoint, predictions, per-recording metrics, routing diagnostics, rotation check, latency, and memory. The three original controls are rerun on the same representation and budget.
4. Compare each new graph with physical, original residual and no graph by subject/seed, and assess source clean score, recorded self/induced score, routing intervention sensitivity and latency. A graph benefit requires a meaningful paired gain over controls and selective harm when sample dependence is destroyed. Report negative or mixed outcomes.

The 81-job run is preliminarily estimated at roughly 3.5-5 hours based on the previous 63-job O2 run (about 2 hours 9 minutes of full-job work); pairwise routing and result checks may add time. Recalibrate from the pilots before launch. Keep GPU jobs serial, with no O3 worker running concurrently. `run_summer_o1_alternatives.py` is the single sequential entry point. Each phase uses its own result directory, and full mode requires both completed pilot paths.

## Current step and expected output

Current step: short smoke and two-fold source pilot for the gated-message arms. Expected output: source validation learning curves and routing intervention diagnostics that show whether adaptive neighbor strength can recover clean performance while retaining sample dependence. CPU checks are in `results/summer_o1_alternatives_cpu_gate_20261021.json`; the first 14-job report is `results/summer_o1_alternatives_pilot_20261021/ANALYSIS.md`. The 81-job long run has not started and remains blocked pending the gated pilot review.

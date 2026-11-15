# T08 receiver-blind predictive repair — revision 1

Frozen by Astra on 14 November 2026. Experiment ID: t08_predictive_20261114_r1.
Status: READY_FOR_LUNA_VALIDATION. Code reviewed as text; NOT executed by Astra. Luna owns syntax, GPU, recovery and end-to-end smoke checks. Full training is conditionally authorized only after all smoke gates pass. T08 remains incomplete until Astra reviews the results and nominates a qualified candidate.

## Reason for this experiment
The previous graph repair matched robust no graph and trailed local capacity. Its receiver-conditioned values did not require neighbors to predict anything useful. This revision asks whether a separate, receiver-blind predictor learns useful cross-sensor information. It broadens the synthetic corruption count from 1–3 to 1–7. This is a new matched recipe, not an isolated comparison against the old run. No 4–5 pp improvement is promised.

## Model and explicit choices
Input remains normalized mag+dot [B,50,9,5], hidden16. The original summer local encoder, temporal block and classifier are unchanged. No genuine Mamba, federation, raw expert or real-placement detector is added.

Allocate identical modules and initial shared weights in all four arms:
- none_clean: clean CE baseline, no correction.
- none_robust: robust paired training plus auxiliary LOCAL predictor, no correction reaching classifier.
- local_repair: identical robust recipe and local predictor, correction enabled.
- graph_repair: identical robust recipe, OTHER-SENSOR predictor, correction enabled.

For the auxiliary predictor, z_j = pointwise Linear/LayerNorm/SiLU(x_j). It has no BatchNorm or cross-sensor preprocessing. Pair embeddings e_ij and a score network use donor descriptor mean(z_j) and sensor identities ONLY, never the receiver signal. Physical/global prior, predicted donor quality and a learned null slot determine weights; self edges and missing senders are masked. Do not renormalize after dropping the null slot.

v_ij = phi(z_j,e_ij)-phi(0,e_ij).
pred_i = W_pred sum(j != i) a_ij v_ij.
correction_i = sigmoid(gate([mean(local_i),mean(pred_i)])) W_out pred_i.
latent_i = local_i + correction_i.

W_out is initialized to zero, gate bias2. W_pred uses ordinary initialization so the auxiliary prediction loss trains the neighbor pathway immediately. The graph prediction must remain unchanged if receiver i's input is changed, in both train and eval modes. Only its final correction gate and original local/classification path may depend on i. Existing classification BatchNorm is preserved; strict receiver exclusion is a property of the separate predictor, not of the whole classifier. Local controls use receiver i as the donor for all virtual pair slots with the same parameter capacity; no other sensor enters their predictor. Missing receiver masks suppress corrections. Inference never receives synthetic corruption labels.

## Training
Train valid ideal recordings only. Existing excluded subjects2/5 use distinct validation subjects3/8. No held-out induced evaluation. Repeatedly inspected validation is exploratory candidate selection, not final confirmation. Keep original split audits/exclusions and train-only normalization.

For each robust-arm training window, corruption probability0.75, request 1–7 uniformly chosen sensor identities, restricted to available sensors. Gain U(0.6,1.4) per sensor/modality applied equally to its xyz axes; four-knot drift with amplitude0.30 times training-only modality std. These are unit-free stress proxies, not a physical displacement simulation. Random draws are identical across robust arms and independent of shuffle/dropout streams.

Epochs1–2 clean CE only. Thereafter:
L = 0.5(CE_clean+CE_corrupt) + ramp*(0.5 KL_T2 + 0.1 prediction_loss + 0.1 quality_BCE).
ramp=min(1,(epoch-2)/3); temperature2, KL multiplied by4. Teacher EMA0.99, eval mode, floating buffers EMA and integer buffers copied. Prediction target is stopped-gradient CLEAN teacher local embedding. Per sensor/time prediction error = mean_hidden squared error / max(mean_hidden target squared,0.01), averaged over synthetically selected sensors/time only. Predicting a constant/collapsed target is a failure to investigate; the classification-trained teacher local encoder is the anchor.

Quality BCE averages clean target1 and corrupted target0.05 on synthetically selected sensors, otherwise1, over available sensors. This predicts synthetic stress exposure, not clinically or physically calibrated placement correctness. No quality-head or alarm claim is authorized from these labels alone.

Adam lr1e-4, weight decay5e-4, existing epoch scheduler, gradient clipping1, batch64, hidden16, BF16 if supported otherwise FP32. All four arms initialize from scratch. CPU-staged batches. Do not weaken no graph or change its temporal/head architecture. No automatic hyperparameter search.

## Queue and checks
Run from start_summer_o2 using Prabhat/Scripts/python.exe and module commands. New files are under experiments/t08_predictive_20261114; existing training files/results are preserved.

Smoke: 4 arms x excluded subject2 x seed13 x4 epochs =4 jobs. Full: 4 arms x excluded subjects2,5 x seeds13,29 x16 epochs =16 jobs. Full output: results/t08_predictive_20261114_r1/full; smoke is the sibling directory. Queue is limited to these20 jobs. Prior eight-job repair took32 minutes; this architecture's runtime must be measured from smoke, not assumed. Do not fill a5–7-hour window unnecessarily.

The PowerShell launcher executes syntax checks, executable invariants, optimizer/teacher/scheduler/RNG recovery, actual CUDA smoke, prediction reconstruction, source/data hashes and matched RNG/prediction-row audits before full. It owns a queue lock, and runner owns a stage lock. Failures halt progression. All run commands are in Astra_to_Luna.md; launcher Stage=All performs the exact sequence. Resume unchanged sources/output only after the previous process has exited. Never bypass a fingerprint mismatch.

Meaningful preflight includes initial no-graph equivalence, zero-message cancellation, unavailable sensor masking, null mass, low-quality suppression, local isolation, strict receiver exclusion in train/eval, donor sensitivity, finite losses/gradients and state recovery within1e-6 FP32. Real smoke exercises BF16 when supported. A syntax pass alone is insufficient.

If Luna fixes a code bug after any result/fingerprint exists, preserve that attempt and use a fresh suffixed output base consistently in launcher/spec, record the patch and source hashes, then redo smoke. No changes to method, split, optimizer, loss weights, batch or epochs without Astra. For OOM/nonfinite failures, preserve traceback and ask Astra if a non-design-changing fix is unavailable. No automatic package installation or machine migration.

## Selection and logging
Best epoch: source ideal fixed-33 macro-F1 ONLY. Log ideal/self accuracy, fixed-33 F1 and losses every epoch; compare self only after the fixed matrix finishes.

Unchanged T08 source promotion screen: graph self F1 > matched robust no graph; graph self F1 >= local repair; graph ideal F1 deficit relative to robust no graph <=0.5 pp. Clean no graph is an additional anchor. Report fold/seed effects, not just means. Passing produces CANDIDATE_REQUIRES_ASTRA_REVIEW, not T08 complete. Failing produces NO_QUALIFYING_CANDIDATE. Smoke produces SMOKE_ONLY. No metric or acceptance changes after observing outcomes.

Graph interventions at selected checkpoints: zero messages, zero correction, and entire-window shuffled donor branches (donor features, quality and availability shuffled together). One donor roll per evaluation batch is a limited diagnostic, not independent causal proof. Report effects even if adverse. Receiver and donor roles are explicit in saved code.

Save latest.pt (student, teacher, optimizer, scheduler, global/independent RNG, epoch, best epoch, history, normalization), best.pt, per-epoch JSONL, logits/labels/indices and recording/subject/scenario/window metadata, per-class metrics, confusion matrices, paired error counts, graph prediction/quality/null/gate samples, norms, interventions, timing/memory, source/config/data hashes and snapshots, requirements lock, CSV and plots, status/heartbeat/traceback, candidate_decision.json and final audits. Extra prediction samples contain only first16 validation-self windows; do not generalize their statistics to the whole corpus.

## Scientific scope
This is a graph-candidate development pilot using ideal training and source validation, not supervised training on induced labels. Per-sensor natural placement annotations are unnecessary for this bounded synthetic objective. The broader anomaly/routing plan remains subsequent work. Synthetic reliability does not identify exact physical sensor position. Confirmed induced graph superiority and final evaluation provenance remain outstanding.

Conceptual inspiration (not an implementation or novelty claim): GraphMAE https://arxiv.org/abs/2205.10803 ; EMA teacher consistency https://arxiv.org/abs/1703.01780 . The receiver-blind architecture and controls above are the specific hypothesis to test here.

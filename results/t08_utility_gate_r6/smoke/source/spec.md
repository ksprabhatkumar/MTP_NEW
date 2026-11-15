# Revision 2 corrective smoke override

Task t08_utility_gate_20261114_r1_fix1. Smoke only for this handoff. See fix1_notes.md for IEEE FP32 verification and endpoint normalization. Existing tolerance values and training design below remain unchanged; runtime outcome unverified. Full execution is not authorized in this handoff.

# Utility-gated frozen-backbone pilot — 14 November 2026

Task t08_utility_gate_20261114_r1 revision1. READY_FOR_LUNA_SMOKE; Astra static preparation only. T08 OPEN. New experiment; prior training and donor audit results retained. No guarantee of a4–5 percentage-point gain.

## Frozen design
Start from each robust-no-graph best checkpoint in results/t08_predictive_20261114_r1/full/jobs/none_robust. Source checkpoint hashes are recorded and checked. Copy ONLY encoder, mamba, classifier and norm_final parameters/buffers into a new PredictiveRepair. Freeze these and use eval mode throughout (including BatchNorm/dropout). The temporal module remains the existing module; this experiment does not introduce genuine Mamba or change the report pipeline. Initialize correction/utility modules from identical seed+70000 for local and graph branches. Keep the receiver-blind graph predictor and virtual-slot local predictor capacity-matched. Original donor quality and sensor repair gate are correction-stage parameters.

Outputs: p0=f(local); p1=f(local+correction); pg=f(local+sigmoid(utility_logit)*correction). Utility gate is35->16->1, SiLU, final bias-2. Its35 detached input features are mean local16, mean(prediction-local)16, mean quality1, mean null mass1 and available fraction1. No natural displacement annotation, activity label or corruption label enters inference. Zero utility restores p0 exactly. Three endpoint passes cost more than a single classifier pass; report this scope rather than claim optimized inference speed.

## Stages and objectives
Correction stage: frozen backbone, train correction modules; utility gate frozen. Use existing raw synthetic corruption probability0.75,1–7 sensors, gains0.6–1.4 and four-knot drift0.30 train-scale. Clean teacher target for reconstruction is frozen backbone local embedding, no EMA. Loss=0.5*(CEclean(p1)+CEcorrupt(p1))+0.1*masked prediction error+0.1*quality BCE+0.5*mean(relu(CEclean(p1)-stopgrad CEclean(p0))). Prediction error and quality targets follow the earlier predictor recipe (clean1, synthetically affected0.05). No warmup/ramp/consistency loss in this finalized experiment.

Gate stage: freeze all correction/backbone parameters; train utility gate only. For clean and synthetic examples compute detached endpoint improvement CE(p0,y)-CE(p1,y). Target1 iff improvement>0.02 nats, else0. Loss=0.5*(BCEclean+BCEcorrupt). No class balancing, validation fitting, threshold search or hard decision threshold. Soft gate at inference. BCE predicts endpoint utility; final pg classification must be judged separately.

Both stages Adam lr1e-4,weight_decay5e-4,constant LR,clip norm1,batch64. BF16 if existing precision_context supports it, else its FP32 fallback. All available training windows each epoch. Fixed CPU order and corruption generators; identical streams across local/graph branches. Stage gate seeds have +50000 offset. Input remains mag+dot [50,9,5]. Train source ideal only using existing folds excluded2/5,validation3/8; train-only normalization must match source checkpoint exactly. No induced or excluded-subject evaluation.

## Budget, controls and selection
Smoke: one fold/seed (2,13), local+graph,2 correction epochs then2 gate epochs each. Smoke must execute both phases and recovery checks.
Full: folds2,5 x seeds13,29 x local/graph =8 training jobs, each8 correction +4 gate epochs. Four evaluated arms: frozen no graph; gated local; ungated graph; gated graph. Ungated and gated graph SHARE the same selected correction checkpoint, so gating is isolated. This intentionally replaces the draft's12 separately trained correction arms with8 shared-stage jobs. Frozen baseline has no new training and must replay archived predictions.

Select earliest maximum source ideal macro-F1 checkpoint separately for correction (p1) and gate (pg). Gate phase starts from selected correction. Do not optimize selection on self placement. Gated/ungated graph share correction weights; report gate parameter overhead. Self validation remains exploratory after many inspections.

## Validation and persistence
Before smoke training: frozen-backbone equality; forced gate0 fallback; receiver blindness or local isolation; missing-sensor invariance; finite gradients; exact optimizer/model/global and dedicated RNG continuation within1e-6 FP32 for BOTH stages. Keep inference/evaluation states and gates explicit. Frozen state checks each epoch; gate phase checks all non-gate state unchanged. Saved baseline predictions must match original archive at1e-5 logits and data row indices. Stage checkpoints contain optimizer,RNG,history,best selection,phase and configuration; normalization is saved per fold. Constant LR has no scheduler state.

Latest and best checkpoints per stage, learning curves, p0/p1/pg, gates, donor-quality/null/repair-gate diagnostics, correction norms, per-window provenance, per-class metrics, confusion matrices, source/data/backbone hashes and code snapshots, heartbeat/status/error and CSV/PNG/PDF are written under results/t08_utility_gate_20261114_r1/{smoke,full}. Stage output uses lock. Resume unchanged source/config only after original process exits, using same launcher. No output deletion. Changed code requires fresh suffixed output and smoke rerun.

Full is gated by passed smoke audit, both phase recovery checks and matching source/upstream fingerprints. Final report audits saved predictions, selections, matched streams and completeness. Full has32 metrics rows (4 arms x4 fold/seed pairs x2 splits),16 stage latest and16 best checkpoints across8 training jobs. T08 never completes automatically.

## Review after execution
Report original numerical screen for gated graph versus frozen baseline and gated local (self macro-F1 win, no worse than local, ideal deficit<=0.5pp), but distinguish changed frozen-backbone experiment from previous joint training. Compare gated vs ungated graph on every fold/seed; if gate hurts, record it. Evidence of gate benefit does not itself prove donor correspondence; a donor audit of these NEW checkpoints is subsequent. No new subject/generalization or induced claim from this pilot. Runtime is measured from smoke, not guaranteed or padded to a5–7-hour allocation.

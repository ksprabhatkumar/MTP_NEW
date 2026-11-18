# T11 core confirmation r1 — frozen design
Date:17 November 2026. T08 OPEN; T10 source pilot COMPLETE. Planning/implementation specification, not a claim that a runner exists or has passed runtime checks.

## Question and scope
Does the modest SSM source advantage persist under the longer main training budget, and does graph repair improve recorded placement beyond robust no graph and capacity-matched local repair? Keep all five arms and both backbones. This90-job core does not replace the remaining representation/external-comparator/GAE/augmentation/O3 requirements. No new alternative-method tuning in this round.

## Exact matrix and optimizer
Authoritative design: t11_core_confirmation_r1_matrix.json,90unique jobs. Excluded subjects2,5,15; source validation3,8,16; seeds13,29,47. All other valid ideal subjects train. No self/induced fitting. Validation16 has33classes in both saved metadata scenarios, but verify actual indices and masks in preflight. Reuse known invalid-file exclusions. Existing build_fold hardcodes2/5 and must NOT be called for15 without a separately implemented generalized loader. Fit normalization and raw_scale on training data only. CPU-stage all windows for4GB VRAM.

Both temporal choices old/ssm, arms none_clean,none_robust,physical_robust,local_repair,graph_repair. All use current mag_dot fivefeatures and identical hidden/classifier architecture. New namespace experiments/t11_core_confirmation; preserve all T10 snapshots/results.

31complete epochs from scratch, batch64, FP32 including validation and saved prediction passes. Disable TF32 and cudnn benchmark. Current inherited evaluate may enable BF16 on capable GPUs; use an explicit FP32 evaluator in the new namespace, not global monkeypatching. Adam lr1e-4, betas(.9,.999), eps1e-8, fused=False,foreach=False; decay5e-4 except A_log/D _no_weight_decay; clip1. OneCycleLR max_lr1e-4,epochs31,steps_per_epoch1,pct_start.3,div_factor1000,final_div_factor100,step after epoch. EMA.99; two clean warmup epochs; identical current gain/drift, source-only losses and ramp: .5*(CEclean+CEcorrupt)+min(1,(epoch-2)/3)*(.5*KL_T2+.1*repair+.1*quality). none_clean remains clean CE. No GAE reconstruction in this core. No performance early stopping. Do not extend old16-epoch OneCycle checkpoints into31epochs: their LR trajectories differ. This is a deliberate final-budget study; old results remain the pilot.

Same common-module initialization within each seed; independent order(seed+10000),corruption(seed+20000),dropout(seed+30000). Save hashes of actual order/stress. Use sensor availability masks. All5spatial arms of each temporal/fold/seed are retained. Full runs do not start from smoke weights.

## Gates and smoke
Luna implements the new runner/check/audit before trying commands. Do not invoke a guessed nonexistent module. Produce a durable launcher with an exclusive device/queue lock; acquire lock BEFORE opening shared console logs, track acquisition before unlocking, and do not use an empty restricted process list as proof no trainer exists. If process visibility is denied, request the required system-level access; never bypass locks.

Preflight: source/reference/data hashes, unique90IDs; subject/recording exclusions for all3folds; class/availability coverage; normalization train-only; all10arm/backbone backward and gradient paths; cross-temporal shared initialization; receiver blindness/local isolation/missing mask and appropriate physical adjacency checks; order and actual corruption matching; existing SSM reference checks; finite predictions. Preserve prior tolerances.

Checkpoint test: serialize student/teacher/optimizer/scheduler/normalization/all global+independent RNG and configuration to disk; compare next stochastic train step including optimizer/teacher/scheduler. Add a fresh-interpreter continuation check as stronger evidence, preserving atol1e-6,rtol0; record observed differences. No tolerance change or silent retry.

Smoke: full source training data,4epochs, independent4-epoch schedule; all10combinations at excluded2/validation3/seed13 AND all10 at excluded15/validation16/seed13 =20jobs. Verify fold5 split construction in preflight. This deliberately covers the newly implemented subject15 fold. Smoke must write best/latest, metrics/logits, metadata and pass independent audit. Save actual per-job times to replace estimates.

Current handoff authorizes implementation and smoke only. Return code/gate/artifact status before a separate full-launch instruction. Full round is scientifically frozen; no unresolved hyperparameter search is delegated to Luna. A failed check stops the attempt, preserves outputs, and returns to Astra; routine fixes require a new attempt and rerun relevant gates.

## Main evaluation contract
Checkpoint selection: source ideal fixed33macroF1 only, strict improvement earliest tie. Record source self diagnostics but never select epoch on them. After all training choices are frozen and training complete, evaluate selected checkpoints on excluded subject ideal/self/induced using TRAIN normalization. All90jobs produce270 primary scenario metric rows if all3scenarios feasible; missing/zero-signal coverage must be explicitly recorded, not invented. There are180source-validation metric rows separately. Preserve recording/window IDs, support, logits, labels, confusions and per-class metrics. No induced result steers training order/settings or checkpoint selection.

Synthetic rotation diagnostic: five fixed rotation seeds9001–9005 per held-out ideal corpus, independent proper per-IMU rotations constant across each window and shared across compared models; transform raw before features/normalization. Report numerical feature/prediction invariance and paired score consistency separately from real displacement. Single-sensor dropout9positions is a separate required diagnostic with matching masks, not replacement evidence for displacement. All-missing should abstain, not receive a normal accuracy claim.

Main contrasts: SSM-old by arm; graph-none_robust and graph-local within temporal. Average seeds within subject then equal subjects; show all paired deltas and worst-subject performance, ideal deficit and runtime. Retain fixed33metric even when coverage differs; supplemental common-class metric labelled. Only3induced subjects, already inspected historically: no untouched test claim, no seed/window pseudoreplication. Future independent data are still needed for a stronger generalization claim.

No graph promotion from smoke or aggregate tie-breaking. Old graph fails pilot source screen; SSM passes aggregate screen only and remains exploratory. Final review must weigh subject consistency, effect sizes, local capacity and donor-mechanism evidence. Do not reinterpret tiny mean improvements as the requested4–5pp. No new post-hoc threshold chosen from final induced scores.

## Saving and estimates
Save best/latest everyepoch, teacher/optimizer/scheduler/RNG, indices/normalization, source/data/reference/config fingerprints, dependency/GPU environment, per-epoch loss components/gradientnorm/LR/validation/peakmemory/time, actual stress/order hashes, predictions/metadata, receipt/status/heartbeat/traceback, CSV/paired results and plots. Durable supervisor stops on failure; no duplicate launch. No Git operations; all existing artifacts preserved.

Measured pilot40jobs cumulative epochs11.765h. Linear projection90/40*31/16=4.359375 times =>51.29h core epoch compute. Add25–40% reserve =>64–72h before unknown final diagnostics and implementation effort. These are estimates: fold15size, logging/transfers and longer learning schedule may differ.20-job smoke roughly1–2h estimated; use measured smoke before approving full. Laptop can do all90sequentially. No workstation/cloud dependency.

After core: freeze any missing representation/external-comparator/GAE/augmentation matrices explicitly, perform final artifact/mechanism reviews, then O3; none silently omitted. Alternative physics/multi-lag proposals remain separate and need their own controls/units validation. This document is not an executable launcher.

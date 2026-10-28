# Project audit and workstation execution plan

The research has three stages: (O1) placement-aware physical/global/dynamic
sensor routing, (O2) rotation-stable within-IMU features, and (O3) subject-client
federated learning. The latest `resources/additional_chat.txt` and `steps.md`
establish O1 as the current implemented scope. This run completes the O1
experiment pipeline; it does not implement or validate O2/O3.

The earlier colloquium studied graph temporal HAR and simulated missing sensors.
Those results use a different protocol. The current temporal implementation is
a gated causal convolution/projection block, NOT selective-scan Mamba.
Keep that backbone fixed for the controlled O1 comparison.

Previous O1 results contain only subjects 2,5,15, with one training subject per
split. Six variants ran for 50 epochs and three seeds. Preserve their negative
finding: no-graph scored about 16.03% induced macro-F1 versus 12.74% for residual.
These are provisional, not 17-subject LOSO. The latest handoff supersedes older
novelty/performance claims in the notes.

## Execution order

1. Create `prabhat` with local Python 3.11 and CUDA PyTorch; verify GPU kernels.
   If network installation is unavailable, reuse the compatible local installed
   distributions, validate imports/dependencies, and record this explicitly.
2. Rebuild REALDISP from all 17 subjects: windows 50, stride 25, nine IMUs,
   acc/gyro only. Audit raw files, finite values, scenario/class coverage and
   manifest alignment. Keep new tensors separate from the old processed files.
3. Fix DynamicWHAR input layout, fit per-sensor/channel normalization on training
   subjects only, and validate graph masks and disjoint subject/recording splits.
4. Run unit tests and real-data GPU smoke tests for every condition, including
   backward passes, GAE, missing sensors, checkpoints and result writing.
5. Calibrate actual epoch speed before freezing a common epoch budget. Target
   approximately 5.5 hours for the suite, without padding or test-score tuning.
6. Run conditions strictly sequentially. Primary displacement study: test
   subjects 2,5,15; seeds 13,29,47; all other ideal subjects supply train/validation.
   Conditions: none, physical, global, dynamic, residual, physical+dropout,
   residual+dropout, residual+GAE+dropout, residual+dropout without sender masking,
   and adapted DynamicWHAR.
7. Supplement ideal LOSO with remaining subjects, seed 13, for none and residual.
   This completes 17 ideal folds for that seed, not three-seed full LOSO.
8. Save epoch logs, best/latest checkpoints, split indices, normalization,
   predictions, confusion matrices, per-class scores, routing diagnostics,
   latency/memory, CSV summaries and plots in a timestamped `results/` directory.
   Aggregate seeds within subjects before reporting subject means.

## Evaluation rules and limits

Select checkpoints only by validation macro-F1. Test induced placement only for
subjects with recordings (2,5,15). Report common-class ideal/induced degradation
separately from unrestricted scores; recordings are not time-aligned pairs.
Self-placement and synthetic sensor dropout are separate evaluations.
Preserve failures and negative results.

A complete multi-objective thesis and exhaustive three-seed LOSO for every
condition cannot be promised in 5–6 hours. This is the current O1 suite with a
measured compute budget. DynamicWHAR is an adapted external comparator, not an
exact reproduction of its original protocol. The old environment deletion was
blocked by automatic approval policy and is left unused. All execution uses
`prabhat/Scripts/python.exe`.

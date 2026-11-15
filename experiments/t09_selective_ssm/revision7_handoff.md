# Astra to Luna — revision 7 / T09 implementation and validation

Current date: 15 November 2026. Laptop only. T08 review complete, NO PROMOTION; T08 milestone OPEN. No more utility training. Read results/research_review_20261115/T08_REVIEW.md and EXPERIMENT_COVERAGE.md. All 144 specified O3 bundles retained in o3_pending_jobs.json; do not launch them now.

## Authorized work

Astra added models/selective_mamba_temporal.py with class SelectiveMambaTemporal. It is NOT runtime-validated. Implement the validation harness and integrate an isolated test wrapper as specified below; this is implementation work, not permission to choose new research settings. Do not touch historical/fingerprinted training sources or Git. Preserve every result. Existing Prabhat/CUDA; no environment replacement. One GPU writer. User permits non-visible execution.

Use docs/experiments/mamba_temporal_spec_20261028.md as the numerical contract. The new module implements [B,T,144], inner288, state16, dt_rank9, causal kernel4, dropout.5, reset per window, FP32 recurrence. Explicit reference backend only. Optimizer grouping must honor A_log/D no decay. Inspect implementation and fix ordinary correctness bugs only; report any required scientific change to Astra.

## Required deliverables and checks

Create experiments/t09_selective_ssm/check.py and run it as:

    .\Prabhat\Scripts\python.exe -u -m experiments.t09_selective_ssm.check --output results/t09_selective_ssm_r1

Fresh output; refuse to overwrite. Add __init__.py. Harness must save environment/device identity, config/source/reference hashes, detailed check errors, timing/memory, status and traceback. Audit passed only if every requirement below passes. Save console output too. No full training triggered by this harness.

1. Obtain the author reference implementations at the two URLs already in mamba_temporal_spec. Pin actual commit/content hashes and preserve the inspected reference. Do not install native extensions. Compare tiny FP32 recurrence to independently implemented author reference scan (do not compare module to itself): output atol1e-5 rtol1e-4; input/parameter gradients atol1e-4 rtol1e-3. Verify actual module dt/B/C/A/D/gate projection wiring against that independent computation. If reference cannot be obtained or checked, mark blocked; do not label parity passed.
2. Causality in eval mode (suffix perturbation leaves prefix unchanged), reset/repeated-window determinism, sequence vs independent step recurrence, finite/nonzero gradients through dt/B/C/A, checkpoint round-trip, interrupted vs uninterrupted optimizer/global/dedicated-order RNG continuation. Keep separate model-init and order generators. No weakening tolerances on failure.
3. Isolated wrapper: instantiate SummerDynamicWHAR with in_features5, hidden16, reconstruction false for graph_variant none and physical; replace ONLY mamba with SelectiveMambaTemporal. Compare with original temporal module in the same wrapper. These four combinations validate interfaces; they do not replace the T10 nominated graph/local-control arms. Never transplant old temporal weights into the SSM.
4. Real-data validation using O2Corpus and build_fold excluded subject2 (validation3), train-only statistics. mag_dot features, batch64, same train-window IDs/order across four combinations. CPU-resident fold. Adam lr1e-4, weight decay5e-4 except marked SSM A_log/D; no scheduler, clip1; FP32 for correctness/calibration across all four. Perform two warmup optimizer steps and five timed optimizer steps per combination; CUDA synchronize around timers. This is a disposable engineering benchmark, not a scientific trained model. Save only diagnostic checkpoints for recovery. Use dropout .5 in train mode and deterministic eval for causality. Profile peak memory, batch-one eval latency (10 warmup/30 repetitions) and per-step timing. Report preprocessing separately. If batch64 does not fit, stop and report; no silent batch adjustment.
5. Save validation.json and calibration.json including every check and all four combinations. Estimate projected epoch time from actual train sample count and measured batch time, label it an estimate excluding full validation/checkpoint costs. Report reference-backend limitation. No 31-epoch/federated runs authorized yet.

On failure save traceback and diagnostics, stop, and update Luna_to_Astra.md. On successful completion append command/device/runtime, exact checks, parity errors, memory/timing, any implementation changes and source hashes. Read back the handoff to verify the write succeeded. Do not end merely because a write tool call had a syntax error; repair the tool call and verify the file. T08 remains OPEN. Return to Astra to freeze T10 matrix/exposure and candidate scope using actual laptop timing.

## Further rounds (not yet executable)

T10 old/new temporal x none/physical/nominated graph plus local control; then complete T11/T12 final O1/O2; then all approved O3 bundles with final bridge. No experiment dropped because of laptop capacity. Historical runs reused only with proven protocol compatibility. These depend on research decisions and must not be fabricated by Luna.

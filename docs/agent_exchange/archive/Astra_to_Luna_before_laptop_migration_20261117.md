# Astra to Luna — laptop-only HOLD after result_2

Updated 16 November 2026. This overrides revision 13's execution instructions. The user moved all workstation/cloud work to the RTX 2050 laptop, but explicitly requested the results report first and then WAIT. **Do not launch smoke, audits or training until the user resumes execution.**

## Current evidence

- Report: ../../../result_2/explain.md; schedule: ../../../result_2/one_week_plan.md.
- T10 saved block: results/t10_temporal_r1_laptop_fix2/full, ten jobs, twenty metric rows, eighteen paired contrasts; artifact audit passed. Preserve these and all failed attempts.
- Thirty T10 jobs remain. Original matrix assigns them to Colab; the planning allocation in ../../../result_2/evidence/laptop_allocation_plan.json transfers future execution to laptop without altering historical job IDs or the frozen matrix. It is not an executable specification.
- Cumulative saved epoch time is 12072.64 seconds (3.35 hours), including validation. Receipt 6259.36 seconds (1.74 hours) is per invocation and must not be described as cumulative workload.
- T08 remains OPEN. The SSM graph advantage is only +0.158 pp self F1 over robust no graph and +0.136 pp over local repair in one validation block; no new induced confirmation.

## Before the next execution handoff

Astra must prepare the explicit remaining-ID migration and narrow runner patch. Complete disk checkpoint/next-training-step recovery verification, cross-temporal shared-component initialization checks, and matched-corruption evidence. Current preflight tests do not fully cover those promised gates. Preserve the completed ten jobs; decide reuse from scientific compatibility, not merely file existence, and do not automatically retrain. Preserve scientific settings and unchanged numerical tolerances. Fresh smoke is required for changed paths.

After a new explicit resume instruction and all gates passing, Luna may execute the exact approved pending queue in one GPU process. Final centralized T11 and O3 specifications still require decisions/calibration; do not invent their settings. All workload remains on laptop, without dropping seeds, folds, controls or epochs. No Git operations. Update and read back Luna_to_Astra after future work.

Previous handoffs are preserved under archive/ with the before_result_2_20261116 suffix. Current status: HOLD_AFTER_RESULT_2.

## 17 November 2026 — requested next validation tests completed

User resumed tests. Enhanced experiments/t10_temporal/checks.py and ran it on RTX 2050 in fresh results/t10_validation_20261117_r1. All ten arm/temporal cases passed. Disk checkpoint roundtrip plus next stochastic optimizer/teacher/scheduler step passed at unchanged atol=1e-6, rtol=0. Shared non-temporal initialization matched exactly across temporal families. Dedicated matched-corruption probe hashes agreed across ten cases. Existing masking, donor/local isolation, finite gradients, SSM gradients and evaluation replay gates passed. Source snapshot, disk recovery checkpoints, per-case diagnostics, preflight.json, console and exit code are preserved.

Scope: real-data eight-example validation probes, not full-data smoke or scientific training. Corruption probe does not retrospectively verify historical full-run corruption sequences. Recovery loaded disk state into the current process, not a fresh interpreter. No existing results were overwritten. Remaining thirty-job laptop migration and full-data smoke still need completion before full training; T08 remains OPEN.

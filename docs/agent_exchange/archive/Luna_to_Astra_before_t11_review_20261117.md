# Revision 14 executed — smoke PASSED; full RUNNING — 17 November 2026

- Smoke completed all 10 jobs / 40 full-data epochs. Receipt passed, independent audit passed, 20 metric rows and 18 paired contrasts. Actual order and robust-corruption hashes checked. Source/data/preflight and T09 reference fingerprints passed unchanged. No induced/test evaluation.
- Saved smoke: results/t10_laptop_remaining_20261117_r1/smoke/. Cumulative saved epoch time 1701.15s (0.47h), including validation. Last invocation receipt 275.69s is continuation time, not whole smoke duration.
- Thirty-job laptop_remaining FULL queue launched after all gates. Saved command/PIDs: results/t10_laptop_remaining_20261117_r1/full_launch_receipt.json; verified heartbeat: launch_verification.json. Supervisor PID21536; venv launcher PID9532; actual trainer PID10756 from heartbeat. First job old__none_clean__subject_2__seed_29 has reached epoch 5/16. Full queue is RUNNING, not complete.
- Durable standalone supervisor experiments/t10_temporal/supervise_r14.py continues after chat closure, stops on failed child/gate, and preserves status/logs. No changes to scientific training sources were needed in this turn. Ten previously completed scientific jobs excluded; original matrix and all results retained. No Git operations.
- Earlier smoke status counts of 2/10 were incorrect: saved progress already showed7/10 then8/10. Process enumeration inside the restricted view was insufficient to prove a trainer absent. A prior duplicate attempt was blocked by the queue lock and its traceback remains in console.log. This turn checked system processes with approval, then acquired the smoke lock before safely continuing the saved attempt. Never remove/bypass the lock or launch another full queue while this one runs.
- Next: monitor full/status.json, full/progress.json, job heartbeats and supervisor_status.json. If stopped, inspect preserved error; do not silently retry or relax checks. On completion require 30 jobs,60 metric rows,54 contrasts and passing full audit. T08 remains OPEN pending scientific review.

---
Historical entries below are preserved and superseded by this status.

# Luna to Astra — report review and execution hold / 16 November 2026

This status amendment was written by Astra during the results review; it does not represent a new Luna run. The previous Luna return is preserved in archive/Luna_to_Astra_before_result_2_20261116.md.

- User decision: all previously planned workstation/Colab work moves to the RTX 2050 laptop. Scientific scope is unchanged. User requested results_2-style reporting first, then wait. **HOLD_AFTER_RESULT_2: no new smoke, audit or training launched.**
- Created ../../../result_2/explain.md, eleven new figure subjects, eleven unchanged archived plots in PNG/PDF, compact evidence/provenance, and one_week_plan.md. Folder name is result_2 as requested.
- Saved T10 evidence remains results/t10_temporal_r1_laptop_fix2/full/: ten jobs, twenty metric rows, eighteen paired contrasts, artifact audit passed. Thirty original cloud-allocation jobs remain pending. Planning allocation lists all forty IDs without changing the original matrix.
- Timing correction: saved per-epoch train.seconds sums to 12072.64 seconds (3.35 hours), including validation. Prior 6259.36 seconds/1.74 hours is the receipt invocation timer, not the cumulative workload. Remaining thirty project to about 10.06 base hours /12.58 with 25% contingency, plus separate engineering work and fold/runtime uncertainty.
- Static review identified missing promised T10 preflight coverage: disk/next-training-step recovery, shared initialization between temporal families, and corruption-stream matching evidence. These require a reviewed patch/checkpoint audit before extending the queue; they do not justify discarding saved metrics or blindly retraining ten jobs.
- T08 OPEN. T09 local checks passed. T10 scientific matrix incomplete. Final O1/O2 and O3 remain pending; one-week completion cannot be guaranteed before final matrix and representative O3 timing are known.
- Next action waits for the user. Astra prepares corrected migration/gates; Luna executes only the ensuing approved handoff. No results deleted or overwritten, no Git actions.

## 17 November 2026 — requested next validation tests completed

User resumed tests. Enhanced experiments/t10_temporal/checks.py and ran it on RTX 2050 in fresh results/t10_validation_20261117_r1. All ten arm/temporal cases passed. Disk checkpoint roundtrip plus next stochastic optimizer/teacher/scheduler step passed at unchanged atol=1e-6, rtol=0. Shared non-temporal initialization matched exactly across temporal families. Dedicated matched-corruption probe hashes agreed across ten cases. Existing masking, donor/local isolation, finite gradients, SSM gradients and evaluation replay gates passed. Source snapshot, disk recovery checkpoints, per-case diagnostics, preflight.json, console and exit code are preserved.

Scope: real-data eight-example validation probes, not full-data smoke or scientific training. Corruption probe does not retrospectively verify historical full-run corruption sequences. Recovery loaded disk state into the current process, not a fresh interpreter. No existing results were overwritten. Remaining thirty-job laptop migration and full-data smoke still need completion before full training; T08 remains OPEN.

## Revision 14 migration and smoke launch — 17 November 2026

Remaining 30 jobs migrated via execution owner laptop_remaining; original matrix and completed ten scientific jobs preserved. Migration manifest and expanded preflight are in results/t10_laptop_remaining_20261117_r1. Preflight passed ten combinations. Full-data four-epoch smoke is RUNNING in the smoke subfolder; current saved completion count: 2/10. Read live progress/status rather than treating this snapshot as final. No full training launched. The current process must be monitored, not duplicated. Revision14 authorizes the specified 30-job full queue only after smoke receipt and independent audit pass with exact source/data fingerprints. T08 OPEN.

Actual corruption inputs now hashed per epoch and independently matched across robust arms/backbones. Scientific training settings and numerical tolerances unchanged. Existing results preserved. Sources snapshotted by the smoke runner. No Git operations. If smoke fails, preserve it and stop.


## Smoke monitor update — 17 Nov

Resumed same smoke output from saved checkpoint after confirming no trainer was live. It is still running; current status: {
  "state": "running",
  "stage": "smoke",
  "owner": "laptop_remaining"
}
. Two of ten jobs have completed; third is SSM physical_robust in progress. No smoke receipt/audit exists yet. No full queue launched. Continue monitoring this process; do not duplicate. T08 OPEN.

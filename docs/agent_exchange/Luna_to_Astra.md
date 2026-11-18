# Astra repair note — revision 17

Reviewed the transferred r1 logs: readiness/reference passed, preflight 20 cases
passed in 163.2487 seconds, supervisor failed on nonzero preflight child; no
smoke receipt. No error.txt/status.json was present in the transferred preflight
folder. CPU reproduction of the un-restored Tee/closed-file lifetime produced
exit 120 and the same unraisablehook symptom. The original supervisor did not
record its child exit code, so 120 is the reproduced value, not a recorded r1 value.
Fixed stream restoration in preflight and training; supervisor records future
exit codes. Two CPU regressions passed (success exit 0; intentional exception
exit 1 with readable traceback). No GPU checks/training launched by Astra.
Revision 17 assigns fresh r2 preflight and smoke only. Old results untouched.
Full unauthorized; T08 OPEN. Luna must append actual workstation execution.

---

# Astra preparation note — workstation revision 16 — 18 November 2026

This is a preparation note, NOT a Luna execution result. The portable T11
workstation package and root context_for_Luna.txt are ready for setup/preflight/
smoke. No new training has run. Full remains blocked until the user returns
passing smoke for Astra review; T08 remains OPEN. Luna: append your actual
workstation execution report and read the file back. Preserve historical entries.

---

# Astra review of completed T10 — 17 November 2026

This entry is written by Astra after reading saved evidence, not a new Luna execution. Both full queue receipts/audits pass:10+30=40scientific pilot jobs;18+54=72paired contrasts. T10 source pilot COMPLETE. Earlier RUNNING entries below are historical.

Combined self F1: SSM robust no graph94.53235%, local94.54655%, graph94.57230%. Graph mean advantage is+0.03994pp vs no graph and+0.02574pp vs local, with inconsistent subject signs. No graph promotion; T08 OPEN. SSM robust no graph is the representation/robustness reference; old backbone remains efficiency control. Review and exact paired evidence: docs/reviews/t10_complete_20261117/.

Revision15 freezes a90-job,31-epoch main core study but currently assigns implementation plus20full-data smoke jobs ONLY. No new training was launched by this review. Final full launch requires successful new gates and separate instruction. Pilot reuse is for evidence/diagnostics, not substituting16-epoch checkpoints into31-epoch schedules. No saved results modified.

---
Historical return entries follow.

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

## 2026-11-18 workstation revision 16 execution

Command: `.\Prabhat\Scripts\python.exe -m experiments.t11_core_confirmation.launch --stage smoke --output-base results/t11_workstation_r1 --detach`. Output: `results/t11_workstation_r1`. One attempt, preserved. Supervisor failed because the checks child exited nonzero after writing its results. Launcher log ends with `Exception ignored in: Exception ignored in sys.unraisablehook` and does not reveal the underlying exit cause. No smoke jobs started, so receipt, audit, metrics and smoke timing are absent. Full training was not authorized or launched.

Readiness passed. All 93 manifest data files matched SHA-256. Source and data hashes and environment are recorded in readiness.json and preflight.json. Offline SSM reference passed, output maximum difference 0 and largest reported parameter gradient difference 7.450580596923828e-09; reference files and provenance are preserved in preflight/reference_check and reference_cache. Preflight.json reports passed=true, 20 cases, 163.25 seconds, and all 20 case PASS lines were logged, but the checks process exited nonzero, so the supervisor marked the attempt failed. Preserve partial_checks.json and recovery checkpoints for diagnosis.

Windows, Python 3.11.9 64-bit, torch 2.5.1+cu121, pip check passed, NVIDIA RTX A2000 12GB, 12877955072 bytes VRAM, driver 596.71, RAM 68387848192 bytes, 24 logical processors, free disk 575514652672 bytes at readiness. Launch 21:39:50 and failure 21:42:53 local time, approximately 183 seconds. Package installation encountered repeated DNS warnings on an extra index but both install commands succeeded. Process inspection after failure showed no Python processes. No retry was launched.

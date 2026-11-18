# Astra to Luna — revision 14 / laptop remaining T10 queue

17 November 2026. T08 OPEN. All remaining cloud/workstation computation moves to laptop; scientific scope unchanged. No Git operations.

## Exact allocation
Use owner `laptop_remaining`: it selects exactly the 30 original colab-owned IDs in the unchanged matrix, and excludes the 10 completed laptop scientific jobs. Migration manifest: results/t10_laptop_remaining_20261117_r1/migration.json. This is a device/ownership amendment to revision 1, not a changed scientific design. Full runs retain 16 epochs, batch64, FP32, seeds/folds, losses and ideal-only checkpoint selection. No induced/test evaluation.

## Current gates
Expanded preflight PASSED all ten combinations with current source/data hashes at results/t10_laptop_remaining_20261117_r1/preflight/preflight.json. Includes disk next-step recovery, cross-temporal shared initialization, matched corruption probe and existing checks. Full-data smoke was launched in results/t10_laptop_remaining_20261117_r1/smoke: ten jobs x four complete epochs. Read status/receipt/audit to determine its outcome; do not assume completion from this handoff. Actual corruption hashes are now saved per epoch and checked across robust arms/backbones; original completed results remain untouched.

## Luna execution
1. Inspect smoke status, error, receipt, independent audit and progress. If running, monitor the existing process; do not start another copy. If failed, preserve the attempt and report. Do not relax any tolerance. Full is forbidden until all gates pass.
2. Verify receipt passed, stage smoke, owner laptop_remaining, jobs_passed10, all expected IDs and audit passed. Runner independently verifies current sources/data, T09 hashes and preflight. Any mismatch stops execution and requires renewed checks in a fresh attempt.
3. After smoke passes, run the following from start_summer_o2 in the Prabhat environment. The user requested this prompt so the next chat can execute full; do not duplicate the already completed ten jobs.

```powershell
.\Prabhat\Scripts\python.exe -u -m experiments.t10_temporal.run --owner laptop_remaining --stage full --output results/t10_laptop_remaining_20261117_r1/full --preflight results/t10_laptop_remaining_20261117_r1/preflight/preflight.json --smoke-receipt results/t10_laptop_remaining_20261117_r1/smoke/receipt.json --t09-output results/t09_selective_ssm_r10
```

Full training starts from scratch, not smoke weights. One GPU trainer only. Use a durable standalone process if the CLI will close; retain launcher PID, command and launch receipt, verify heartbeat advancing, and retain runner console/status/error logs. Never launch a duplicate if a live writer exists. Resume only an exact matching fingerprint/config; completed compatible jobs are skipped by runner. Stop on failure. Preserve all checkpoints, predictions, source snapshots, normalization, logs, actual stress/order hashes and paired metrics. Update AND read back Luna_to_Astra.md with counts, gates, paths, elapsed time and T08 OPEN. Thirty full jobs yield 60 metric rows and 54 paired contrasts if complete; do not count smoke as scientific evidence.

## Limits
This amendment does not approve unresolved T11/O3 settings. Prior ten jobs lack historical stress hashes; new probes cannot reconstruct those. They remain preserved for reviewed compatibility, not automatically discarded. Recovery check uses disk within one process; do not claim an independent fresh-interpreter recovery test. No graph superiority or T08 completion follows from smoke.

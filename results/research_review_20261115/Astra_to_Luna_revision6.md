# Astra to Luna

Revision 6: corrected frozen reference; expanded smoke then conditionally full. T08 OPEN.

Read latest Luna_to_Astra.md entry. Reference now freezes its parameters to match the candidate backbone. Revision 5 supports this correction; GPU execution validation remains pending. All tolerances, training settings and archived replay comparisons unchanged. Previous results and sources preserved.

From start_summer_o2, existing Prabhat/CUDA, one execution session (visible terminal optional):

```powershell
& .\Prabhat\Scripts\python.exe -u -m experiments.t08_utility_gate_20261114.run --stage smoke --output results/t08_utility_gate_r6/smoke
```

Require fresh output and no duplicate runner. Do not use old launch.ps1. Smoke includes all 4 checkpoint pairs, both branches, 2 correction + 2 gate epochs each. Check exit 0, status complete, audit passed (32 metric rows), all 8 initial_checks and recovery files passed, archive replay and unchanged fingerprints. If ANY check fails, stop, preserve traceback and report to Astra. No repeated attempts or tolerance edits.

Only after every smoke check passes, execute the already frozen full pilot:

```powershell
& .\Prabhat\Scripts\python.exe -u -m experiments.t08_utility_gate_20261114.run --stage full --output results/t08_utility_gate_r6/full
```

Full: 8 jobs, each 8 correction + 4 gate epochs, subjects 2,5 x seeds 13,29 x local/graph. Full runner independently enforces expanded smoke coverage and all recovery/initial checks. No induced evaluation. Checkpoint selection remains ideal-only. No settings changes. Stop on failure. Preserve prior failed outputs and do not resume them into this new experiment. No Git actions.

Verify full status complete and audit passed, 32 metric rows, 8 jobs with 16 best and 16 latest stage checkpoints, histories, saved per-window predictions, normalization, plots and paired comparisons. Record exact command/device/runtime, all gates, coverage and per-pair/mean ideal/self accuracy and macro-F1 deltas (percentage points) in Luna_to_Astra.md. Record failures honestly. T08 remains OPEN pending scientific review even when training completes.

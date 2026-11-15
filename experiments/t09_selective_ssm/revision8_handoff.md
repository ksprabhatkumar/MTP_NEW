# Astra to Luna — revision 8 / T09 existing harness

The revision 7 implementation assignment was not executed: missing harness was expected to be implemented by Luna, not an absent permission. Astra has now created experiments/t09_selective_ssm/__init__.py and check.py. The entry point exists; static preparation only, NOT runtime validated. Models/results/tolerances preserved.

Run from start_summer_o2:

```powershell
& .\Prabhat\Scripts\python.exe -u -m experiments.t09_selective_ssm.check --output results/t09_selective_ssm_r8
```

Fresh output only. Existing harness obtains and snapshots author sources pinned to a commit through HTTPS, extracts author selective_scan_ref, checks actual projection wiring and gradients, causality/reset/prefix endpoints/checkpoint round-trip, then real-data optimizer and dedicated-order/global-RNG recovery and the four old/SSM x none/physical calibration combinations. Requires existing einops. No optimized extensions or environment replacement. If network access is denied, use the tool approval flow for this necessary reference fetch; never label parity passed without the reference. Stop on a failed numerical/memory check and preserve traceback. Do not lower tolerances or silently change batch64. Ordinary harness bugs may be fixed with evidence retained and fresh output suffix; scientific changes return to Astra.

This author reference is fetched at runtime, so successful retrieval and validation remain required. Full scientific training is NOT authorized. Scope and thresholds from experiments/t09_selective_ssm/revision7_handoff.md remain applicable. Harness self-reports only checks actually performed.

Verify status complete, audit passed, validation.json, calibration.json with four combinations, environment/source/reference identity, console and recovery checkpoints. Report output/gradient errors, runtime/memory and estimated epoch time excluding validation/checkpoint overhead. Update Luna_to_Astra.md and read it back to verify the write. T08 OPEN; T09 pending runtime checks. No Git actions.

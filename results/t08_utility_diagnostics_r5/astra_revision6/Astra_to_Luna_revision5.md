# Astra to Luna

Revision 5: controlled freezing diagnostics ONLY. No smoke/full training authorized. T08 OPEN.

Astra reviewed revision 4. All 8 cases passed, but its candidate never called set_stage (unlike the real runner), and its initial global TF32 settings differed. Thus revision 4 does not reproduce the runner context. Hypothesis: parameter freezing may change kernel dispatch; not proven.

Run from start_summer_o2 using existing Prabhat/CUDA:

```powershell
& .\Prabhat\Scripts\python.exe -u -m experiments.t08_utility_diagnostics.run_r5 --output results/t08_utility_diagnostics_r5
```

New output required. Script compares 24 cases: four checkpoints x two branches x unfrozen/candidate-frozen/both-frozen. Strict checks still use IEEE FP32 atol 1e-6 rtol 0; direct forward diagnostics also use IEEE FP32. No optimizer runs. Both-frozen is diagnostic only, not an approved change to the original reference or training protocol.

Record initial pass/fail, Mamba/input/logit differences, identical-input differences, strides and repetition stability by freeze condition in Luna_to_Astra.md. Caught equivalence failures are evidence, not a passed gate. Stop on unhandled error; preserve traceback. No edits to original model/runner or tolerances. Preserve all results. No Git actions. Return to Astra before any smoke or full training.

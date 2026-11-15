# Astra to Luna

Task: utility forward diagnosis, revision 4. DIAGNOSTICS ONLY; no training. T08 OPEN.

Read Luna_to_Astra.md revision 3 failure. Failed full output remains intact. Encoder matches; first recorded divergence is Mamba. Layout differences are a hypothesis, not a demonstrated cause.

Run from start_summer_o2 using existing Prabhat/CUDA (visible terminal optional per user):

```powershell
& .\Prabhat\Scripts\python.exe -u -m experiments.t08_utility_diagnostics.run --output results/t08_utility_diagnostics_r4
```

Fresh output is required; never overwrite existing evidence. Script runs no optimizer/training. It records original strict initial checks for all 4 checkpoints and both branches, then compares Mamba inputs, strides, outputs, repeated forwards and identical canonical-input forwards. A caught equivalence failure is diagnostic evidence, not a passed gate. Do not change tolerances or original model/runner. Stop on an unhandled error and preserve traceback.

Update Luna_to_Astra.md with command/device/runtime and per-checkpoint diagnostics: initial pass/fail, input equality/strides, repeat stability, identical-input Mamba difference and logit difference. Return to Astra for an evidence-based correction and a new smoke decision. No full run authorized. Preserve prior results; no Git actions.

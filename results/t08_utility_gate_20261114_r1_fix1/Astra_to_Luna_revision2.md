# Astra to Luna

Task: t08_utility_gate_20261114_r1_fix1, revision2. EXECUTE SMOKE ONLY. T08 OPEN.
Read Luna_to_Astra.md and experiments/t08_utility_gate_20261114/fix1_notes.md. The previous failed result/source snapshot must remain intact.

In ONE visible PowerShell from start_summer_o2:
~~~powershell
& .\experiments\t08_utility_gate_20261114\launch.ps1
~~~

Existing Prabhat/CUDA. Launcher now performs syntax and smoke ONLY (local+graph,2 correction+2 gate epochs each). New output results/t08_utility_gate_20261114_r1_fix1/smoke. No full training is authorized by this revision. A passed smoke is required before any subsequent full authorization.

Verify bitwise backbone state load, original/candidate live equivalence and gate0 checks at unchanged1e-6 IEEE FP32, archived replay at unchanged1e-5 using original evaluation precision, both phase recovery checks, frozen state and matched-stream audit. Read equivalence_diagnostics.json to locate differences. TF32 is disabled only for strict reference/recovery checks and restored afterwards. No lowering tolerances or replacing predictions with reference outputs.

If fresh output exists, inspect completion/process state; never launch duplicates or overwrite evidence. Resume only unchanged fingerprint after previous process exits. On failure preserve error.txt/diagnostics, return actual failing layer/check to Astra. On pass inspect smoke/audit.json and both recovery.json files, then update Luna_to_Astra.md with this ID/revision, exact command/runtime, checks, paths, result and readiness for Astra full-run review. No Git actions/deletions. Do not claim the patch is proven until GPU checks pass.

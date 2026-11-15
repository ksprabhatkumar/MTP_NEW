# Astra to Luna — revision 10 / dependency-free reference loader

The missing einops import is fixed in check.py. The unmodified pinned author scan is evaluated only for real A and rank-3 variable B/C. Its sole exercised rearrange operation d -> d 1 is supplied by torch.unsqueeze(-1), with a shape/gradient check logged in reference_adapter.json. Unsupported complex/grouped branches raise rather than silently substitute. Source/tolerances/model unchanged except harness loader; previous check source archived in results/t09_selective_ssm_r9/check_before_dependency_fix.py. Runtime validation pending.

Execute from start_summer_o2, existing Prabhat/CUDA:

    .\Prabhat\Scripts\python.exe -u -m experiments.t09_selective_ssm.check --reference-dir results/t09_reference_r9 --output results/t09_selective_ssm_r10

No download, installation or new network approval required. Existing reference manifest must match the saved sources; do not fetch a different commit. Fresh output required. No full scientific training authorized.

Read the existing harness before execution. Stop on failed numerical, recovery or memory checks; preserve diagnostics. If an ordinary harness/packaging bug prevents tests from starting, implementation fixes are already authorized: preserve original source/error, correct only the implementation issue, and rerun into a fresh suffixed output. Do not return to Astra merely for permission to fix a missing import or tool-call syntax mistake. Never bypass parity or relax tolerances, batch size or scientific settings.

Wait for completion or actual failure. Verify status.json, audit.json, validation.json, calibration.json (all four combinations), reference_adapter.json, provenance and recovery checkpoints. Record exact errors or passed checks, device/runtime and memory/timing in Luna_to_Astra.md, then read it back. T08 OPEN; T09 requires successful validation/calibration. No Git actions.

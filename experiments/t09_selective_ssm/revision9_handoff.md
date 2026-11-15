# Astra to Luna — revision 9 / approved download then offline T09

Revision 8 failed during DNS resolution inside restricted execution, BEFORE reference pinning or SSM checks. This does not establish a laptop DNS defect. Numerical checks remain unrun. Existing results/t09_selective_ssm_r8 is preserved. Stop-on-numerical-failure is unchanged; a network permission failure should use the required approval workflow, not return to Astra without attempting it.

Step 1: download official reference sources with the NEW source-only downloader. Because the previous necessary network call failed with getaddrinfo, invoke exec_command with sandbox_permissions=require_escalated and a concise justification: Allow downloading the official Mamba reference files needed for validation? Do not ask for broad Python permission. This invokes the tool approval flow; do not bypass it via another tool.

Working directory start_summer_o2:

    .\Prabhat\Scripts\python.exe -u -m experiments.t09_selective_ssm.fetch_reference --output results/t09_reference_r9

This downloads exactly the author module, scan implementation and LICENSE at one API-resolved commit, storing URLs and SHA256 hashes. It does not execute downloaded code, train, install or use Git. Preserve errors. If that directory already exists, verify its completed manifest; use a fresh suffix for incomplete prior attempts rather than overwrite. If escalation is denied or network still fails, record the actual reason and stop; no false validation claims.

Step 2: after successful download, run the harness under default permissions, offline:

    .\Prabhat\Scripts\python.exe -u -m experiments.t09_selective_ssm.check --reference-dir results/t09_reference_r9 --output results/t09_selective_ssm_r9

The harness now requires local reference-dir, verifies all hashes and copies provenance into the fresh output before extracting the author reference function. All SSM checks, four-combination calibration, batch64 and tolerances remain unchanged. No network in the validation path. Use the actual cache suffix if step 1 required one. Do not overwrite existing output.

Stop on numerical/memory failure; no silent precision/batch/tolerance changes. Ordinary harness bugs can be corrected with old source/error preserved and fresh run output. No full scientific training authorized. Verify status/audit/validation/calibration files and four combinations. Append exact commands, device/runtime, reference commit, errors or measured timings to Luna_to_Astra.md, then read it back. T08 OPEN; T09 incomplete until validation succeeds. No Git operations.

Astra static preparation only; neither download nor harness has been executed in revision 9. Revision 7/8 scope retained in this package's archived handoffs.

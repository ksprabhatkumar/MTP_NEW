# Astra to Luna — revision21: targeted native-crash diagnostic ONLY

Raw r2smoke passed20jobs/40metrics/36contrasts. Preserve it; DO NOT rerun raw.
Magnitude preflight passed15cases; the fresh-process recovery worker for
magnitude__ssm__none_clean__subject_15__seed_13 exited3221226505 (0xC0000409).
No magnitude smoke or full ran. The code is a native fail-fast status; it does
not establish OOM, model math error, stack overwrite or a particular DLL cause.
The transferred logs lack worker-stage/native diagnostics and checkpoint
binaries. Astra cannot establish or claim a fix from this evidence alone.

No fingerprinted training/preflight files changed in this revision. Only a
read-only-source diagnostic script under docs/agent_exchange was added, to
preserve the passing raw smoke fingerprint. Do not change source line endings,
dependencies, CPU threads, CUDA flags or tolerances. Do not bypass checks.

On original workstation, from project root:
1. Confirm no trainer/queue is active. Keep failed r2, prior r1 and T11 intact.
2. Confirm the failed checkpoint .pt still exists in magnitude/preflight.
   If absent, STOP/report. Do not manufacture it by rerunning preflight.
3. Execute exactly ONE bounded isolated diagnostic:
   .\Prabhat\Scripts\python.exe docs/agent_exchange/t12_native_diagnostic.py --output results/t12_native_diagnostics_r21
   Use a fresh output directory. Do not retry if this command fails.
   This checks the original raw source/data fingerprints, copies the failed
   checkpoint to a new diagnostic folder, then executes the original recovery
   worker with fault handling and stage tracing. It runs one diagnostic
   optimizer continuation, NOT scientific training. Native numeric settings
   stay unchanged. It records stdout/stderr, exit code in decimal/hex, optional
   Windows application error events and whether output existed before the
   original failure. If original output exists and replay succeeds, compare
   outputs using the unchanged numerical comparator. No old files are written.
4. Return diagnostic.json, worker.log and windows_application_events.txt if
   available. Record last STAGE line and any faulting module/native code. A
   denied event query is diagnostic context, not permission to change settings.
5. Append AND read back Luna_to_Astra.md. STOP. Even a successful isolated
   replay does NOT complete magnitude preflight or authorize smoke/full.

Astra performed syntax validation only; original failed-case .pt is not in
this laptop copy. No GPU/diagnostic replay or training was launched here.
After this evidence identifies the failure stage, Astra will prepare a
specific repair/revalidation preserving raw smoke wherever scientifically
valid. Current full gate remains blocked. T08 remains OPEN.

# Astra to Luna — revision 18: gated T11 workstation FULL launch

Astra reviewed r2 smoke: 20/20 jobs, 40 prediction metric rows independently
recomputed, 36 contrasts, passing audit, matching receipt/preflight/readiness
fingerprints, all three child exits 0. Epoch histories sum to 3584.114481 seconds.
Checkpoint selection and paired order/corruption history hashes agree.
The user requested separate full instructions after review. Full is now
conditionally authorized ONLY after the read-only workstation gate below passes.
T08 remains OPEN. Do not interpret smoke performance as graph promotion.

IMPORTANT PROVENANCE:
The laptop transfer contains prediction arrays and JSON but no best/latest
checkpoints. Its copied source snapshots had line-ending conversion: every
recorded hash was recoverable by restoring original CRLF/LF and normalized
content matched current code. No evidence was modified to establish this.
Use the ORIGINAL workstation r2 artifacts and unchanged workstation source,
not the transferred/renormalized source snapshots. Do not edit manifests,
convert line endings, change tolerances, or bypass the runtime fingerprint.
This revision changes documentation only, plus a read-only gate under docs/;
it does not change any file in the training fingerprint. Transfer these docs
without replacing the already-smoked workstation sources or results.

1. On workstation, project root, use existing Prabhat/Python3.11.9/torch2.5.1+cu121
   on RTX A2000 12GB. Verify no active T11/training process or full attempt.
   Inspect OS processes/nvidia-smi and locks; never remove or bypass a lock.
   If results/t11_workstation_r2/full already exists, STOP and inspect/report.
2. Run this READ-ONLY prelaunch gate; it checks live source/data/environment,
   original snapshot bytes, 20 best/latest checkpoint pairs, normalization,
   40 prediction files and all prior child exits. It performs no training:
   .\Prabhat\Scripts\python.exe docs/agent_exchange/t11_full_prelaunch.py
   Linux substitute Prabhat/bin/python. Require exit 0 and PASS. Any failure
   stops authorization; preserve everything and return to Astra. Do not rerun
   smoke or fix hashes automatically. No dependency upgrades.
3. ONLY after step 2 passes, launch once:
   .\Prabhat\Scripts\python.exe -m experiments.t11_core_confirmation.launch --stage full --output-base results/t11_workstation_r2 --confirm-full SMOKE_REVIEWED_BY_USER --detach
   Linux substitute Prabhat/bin/python. No scientific source/settings edits.
4. Verify launcher PID/log, full_supervisor.json child PID, full/status.json,
   full/progress.json and a first job heartbeat. A detached PID alone is not
   evidence of successful startup. Once a training heartbeat advances, report
   RUNNING with exact paths/PIDs and append/read back Luna_to_Astra.md. The
   chat may close after this verification; power/sleep/network storage must
   remain stable. Do not start another queue. No Git operations.
5. Stop on failure; preserve full/error.txt, supervisor and child exit records.
   No automatic retries, tolerance changes, batch-size changes or shortened jobs.
   Report the cause to Astra. Never report success before final receipt.

FROZEN FULL MATRIX:
90 jobs, from scratch: old/ssm x none_clean/none_robust/physical_robust/
local_repair/graph_repair x subjects2/5/15 x seeds13/29/47; validation3/8/16;
31epochs, batch64, FP32. Source ideal selects checkpoint; source self diagnostic
only. No smoke weights reused. All details remain docs/experiments/
t11_core_confirmation_r1.md and its matrix, with workstation execution owner.

COMPLETION:
Require full/status.json complete, receipt passed jobs_passed90, independent
source audit passed,180 source metric rows,162 paired contrasts; then frozen
post-training target evaluation audit passed,270 primary metric rows and243
contrasts plus five rotation seeds and nine dropout diagnostics per job.
The existing runner does target evaluation only after training/audit. Preserve
logits/labels/recording IDs/confusions/per-class values/plots/checkpoints/RNG.
Check full_supervisor.json complete and child exit0. Report actual runtime,
artifact paths and any failures; append/read back Luna_to_Astra.md. T08 OPEN
pending scientific review. Previously inspected induced data are not untouched.

TIME:
Measured smoke epoch total0.996h. Saved linear projection34.72h; with25-40%
reserve43.40-48.61h BEFORE unmeasured target diagnostics. This is provisional:
smoke spends half its epochs in cheap warmup, whereas full spends2/31, and
subject5 cost is not measured here. Do not set a time cutoff; all90jobs required.

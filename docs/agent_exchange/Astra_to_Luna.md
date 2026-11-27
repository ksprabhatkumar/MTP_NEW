# Astra to Luna — revision 20: native readiness repair, smoke only

Revision19 stopped before preflight/training when the optional implementation
choice of a PowerShell CIM RAM/CPU query exited1. Original stderr was not saved;
no particular WMI/permission root cause is established. The failed r1 remains
unchanged. T12readiness now uses native Win32 GlobalMemoryStatusEx and Python
os.cpu_count (POSIX sysconf on Linux), validates positive RAM/CPU values and
still fails on native detection errors. nvidia-smi remains required; its exit,
stdout/stderr and command errors are now recorded. CUDA, package/data hashes,
reference/recovery checks, locks and numerical tolerances are unchanged.

1. User transfers updated code; no Git actions by Luna. Use existing workstation
   Prabhat environment. Inspect active trainers and locks before launching.
2. Run eight CPU tests; require all passed:
   .\Prabhat\Scripts\python.exe -m unittest experiments.t12_representation.test_readiness experiments.t12_representation.test_contract experiments.t12_representation.test_logging -v
   Also run pip check using that interpreter. No dependency upgrades needed.
3. Reuse the successful saved r1/t11_checkpoint_inventory.json: confirm
   passed=true,checkpoint_pairs90,normalization_files3. Do not overwrite it,
   repeat T11 training or delete any checkpoints. If originals have since been
   changed/moved, stop and report rather than assume the old inventory applies.
4. Use a fresh results/t12_representation_workstation_r2. If already present,
   inspect/report before another launch. Preserve the entire failed r1.
   .\Prabhat\Scripts\python.exe -m experiments.t12_representation.queue --stage smoke --output-root results/t12_representation_workstation_r2 --detach
   Linux: substitute Prabhat/bin/python. Raw then magnitude run sequentially;
   readiness, preflight and smoke are rerun with the new source fingerprint.
5. Verify raw/readiness.json has passed=true plus actual system_resources and
   nvidia-smi output. Check child exit0, fresh preflight and advancing smoke
   heartbeats. Verify best.pt and latest.pt exist and latest epoch advances.
6. Stop on any failure; save traceback and native/command diagnostics. No silent
   retries or bypasses. No edits to scientific settings, data or tolerances.
7. Require BOTH representations to finish preflight20cases,smoke20jobs,
   audit40metric rows/36contrasts,passingreceipt,childexit0. Outer smoke batch
   status must be complete. Report timing,checkpoint activity and paths;
   append AND read back Luna_to_Astra.md. No full training in this revision.

Reference scientific specification remains docs/experiments/t12_representation_r1.txt
and its180new-job matrix. Reuse90T11mag_dot results. Smoke totals40jobs. The
original user workflow requires smoke review before a separate full-launch
prompt. T08 remains OPEN. Astra ran8CPU regressions (including real native
RAM/CPU retrieval on the laptop); no GPU readiness/preflight or training ran.

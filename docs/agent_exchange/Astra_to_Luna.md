# Astra to Luna — revision 17: logging fix; fresh workstation smoke ONLY

Read root context_for_Luna.txt for environment, dataset and artifact requirements.
The revision 16 attempt results/t11_workstation_r1 MUST remain unchanged.
It passed 20 preflight cases but its child exited nonzero; no smoke ran.

Diagnosis: checks.py and run.py left sys.stdout/sys.stderr as Tee objects after
closing console.log. A CPU subprocess reproduced exit 120 and the identical
unraisablehook symptom. The common.console_log context now restores streams
before closing the log, on success and failure. launch.py records child exit
codes in child_exits.jsonl. Scientific settings and tolerances are unchanged.

Current assignment (workstation RTX A2000 12GB):
1. Confirm the user has transferred updated code. No Git operations by Luna.
2. Reuse the validated Python 3.11 / torch 2.5.1+cu121 Prabhat environment;
   verify pip check and inspect active trainers. Do not install upgrades.
3. Run CPU regression check from project root:
   .\Prabhat\Scripts\python.exe -m unittest experiments.t11_core_confirmation.test_logging -v
   Both tests must pass. On Linux use Prabhat/bin/python.
4. Use a NEW attempt directory results/t11_workstation_r2. If it already exists,
   inspect and report rather than overwrite or launch a duplicate.
5. Launch (Windows):
   .\Prabhat\Scripts\python.exe -m experiments.t11_core_confirmation.launch --stage smoke --output-base results/t11_workstation_r2 --detach
   Linux: substitute Prabhat/bin/python. The launcher reruns readiness and all
   preflight cases with new source hashes, then smoke ONLY if the child exits 0.
   Do not reuse/edit r1's passing preflight JSON or suppress any nonzero exit.
6. Verify active process/status after detachment. Monitor to completion or failure.
   Require preflight 20/20 and zero child exit, smoke 20/20, receipt/audit passed,
   40 metric rows and 36 paired contrasts. Return actual timings and child exits.
7. Stop on failure, preserve tracebacks/partial output, no silent retries.
   Append and read back Luna_to_Astra.md with command, folder, gates and status.

STOP AFTER SMOKE. No full training, no confirmation flag, no induced evaluation.
User must return smoke results to Astra before a full-launch prompt. T08 OPEN.
Astra ran two CPU subprocess regression tests successfully; no GPU/training ran.

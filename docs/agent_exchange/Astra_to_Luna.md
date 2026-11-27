# Astra to Luna - revision24: repaired sequential FULL launch

User has authorized180new T12full jobs after both smokes passed. The previous
r22sequence supervisor failed before issuing raw training: validate_gates
rejected its own launcher-created status directory. No training started.
Preserve results/t12_full_workstation_r22 and the original r22supervisor script.

Repair: t12_full_sequential_r24.py separates scientific gates from folder
reservation. Parent creates a new output folder and unique reservation; child
claims it exactly once using exclusive started.json creation. Existing status,
wrong reservation and duplicate claims fail closed. One detached outer process
runs raw then magnitude as direct children, waits for actual exit codes, and
requires all raw completion/audit/checkpoint checks before magnitude. No nested
detached trainer polling. Training source/settings/fingerprints unchanged.
Five CPU regressions pass: owned folder, wrong/duplicate reservation, sequential
success, child failure and audit failure (the latter two block magnitude).

EXECUTE on workstation from project root using existing Prabhat:
1. Confirm no active trainer/supervisor and both original raw/magnitude full
   folders are still absent. Preserve all smoke/T11/failed-launch evidence.
   Transfer the two new docs scripts and this handoff only; no Git operations
   by Luna, no dependency/source/line-ending/tolerance changes.
2. .\Prabhat\Scripts\python.exe docs/agent_exchange/test_t12_full_supervisor.py -v
   Require5tests passed. These mocks run no GPU gates or training.
3. .\Prabhat\Scripts\python.exe docs/agent_exchange/t12_full_sequential_r24.py --output-root results/t12_full_workstation_r24 --check-only
   Require passed; verifies BOTH original smoke receipts/preflights/audits,
   current source/data/environment and frozen180job matrix. No artifact writes.
4. Only after gate passes, launch ONCE:
   .\Prabhat\Scripts\python.exe docs/agent_exchange/t12_full_sequential_r24.py --output-root results/t12_full_workstation_r24 --detach
   Linux substitute Prabhat/bin/python. Fresh supervisor folder required.
   Original training destinations remain:
   raw: results/t12_representation_workstation_r2/raw/full
   magnitude: results/t12_representation_workstation_r3/magnitude/full
   Do not use a single-root old queue command. Do not rerun either smoke or T11.
5. Verify status.json running, direct child PID and first actual training
   heartbeat/checkpoint advancement. latest.pt must update every epoch with
   model,EMA teacher,optimizer,scheduler,RNG,normalization/config/history;
   best.pt must track strict source-ideal validation improvement. Report real
   RUNNING status only after this evidence; chat may then close while the
   durable supervisor continues and workstation stays powered/awake.
6. Complete raw90jobs/31epochs, source180rows/162contrasts, target270rows/
   243contrasts plus diagnostics/checkpoint audits BEFORE magnitude90jobs.
   Require zero child exit and passing receipts/audits for each. Final outer
   status must be complete with both stages, not merely a launched PID.
7. Any failure: stop, preserve evidence, no automatic retry/settings changes.
   Append AND read back Luna_to_Astra.md with command/PIDs, progress/checkpoint
   evidence, output paths and actual status. T08 remains OPEN.

Astra verified the startup regression repair locally with5CPUtests; no full
training launched here. This revision authorizes the existing frozen FULL
queue; no further smoke or extra permission step is required if gates pass.

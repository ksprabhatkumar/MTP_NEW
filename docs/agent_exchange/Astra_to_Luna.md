# Astra to Luna — revision 19: T12 workstation smoke

User priority: complete MTP research training, then matched SOTA comparison,
then college presentation. Paper/venue formatting is deferred. T11 is complete;
DO NOT rerun its90jobs. T08 remains OPEN; no graph superiority claim.

Read docs/experiments/t12_representation_r1.txt and its matrix. New runner exists
in experiments/t12_representation; raw and magnitude missing cells total180jobs,
reusing90mag_dot results. No configuration choices delegated to Luna.

1. User transfers code manually; no Git actions. Reuse workstation Prabhat
   Python3.11/torch2.5.1+cu121/requirements_workstation.txt. Inspect running
   processes and locks. Keep original T11 results and checkpoints intact.
2. From project root, run CPU contracts:
   .\Prabhat\Scripts\python.exe -m unittest experiments.t12_representation.test_contract experiments.t12_representation.test_logging -v
   Require4tests passed. No package upgrade or scientific-source edits.
3. Inventory original T11 checkpoints (read-only except NEW output):
   .\Prabhat\Scripts\python.exe docs/agent_exchange/t11_checkpoint_inventory.py --output results/t12_representation_workstation_r1/t11_checkpoint_inventory.json
   Require90best/latest pairs plus3normalization files. If anything is missing,
   stop/report; do not retrain T11 to manufacture the missing artifact.
4. Run the single durable SMOKE queue:
   .\Prabhat\Scripts\python.exe -m experiments.t12_representation.queue --stage smoke --output-root results/t12_representation_workstation_r1 --detach
   Linux: replace interpreter with Prabhat/bin/python. The queue runs raw then
   magnitude automatically, sequentially. Both use fresh readiness/preflight
   and full-data smoke. No duplicate trainers, no hidden full launch.
5. Check smoke_batch_status.json, raw/smoke_supervisor.json and corresponding
   magnitude files, per-job heartbeats, best.pt/latest.pt update times and
   epochs. Confirm advancing heartbeat/checkpoints before closing the chat.
   The durable process continues on workstation while powered on and awake.
6. Stop on ANY failure; preserve partial outputs and traceback. Never alter
   tolerances, overwrite attempts or reduce data. Return exact failing command.
7. Completion needs each representation: preflight20cases, smoke20jobs,
   audit40rows/36contrasts, passingreceipt and child exit0. Outer status complete.
   Total40smokejobs. Save measured timings, disk/VRAM, checkpoints and source/data
   provenance. Append AND read back Luna_to_Astra.md.

STOP AFTER BOTH SMOKES. Existing user workflow requires returned smoke evidence
before the separate full-launch prompt. No full training is launched by this
revision. For the later reviewed full assignment only, queue interface is:
<Prabhat python> -m experiments.t12_representation.queue --stage full --output-root results/t12_representation_workstation_r1 --confirm-full SMOKE_REVIEWED_BY_USER --detach
It runs180newjobs sequentially and checks both smoke fingerprints first.
Astra has run CPU tests only; GPU preflight/recovery/smoke remain workstation work.

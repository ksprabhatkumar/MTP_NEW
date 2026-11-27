# Astra to Luna - revision22: controlled magnitude-only smoke recovery

Reviewed revision21 diagnostic: original recovery worker loaded the failed-case
checkpoint, completed its optimizer continuation, saved output and exited0.
Original output absent; no parent-reference comparison was possible. Event logs
did not identify the native cause. This does not prove a root-cause repair.
It justifies ONE fresh full magnitude preflight and smoke, with native fault
logging enabled and unchanged checks. No automatic retry after another failure.

Raw r2smoke is valid and must NOT be rerun. No fingerprinted training source,
precision, model, dependency, thread count or tolerance changed. This revision
adds only docs/agent_exchange/t12_resume_magnitude.py; transfer this script and
updated handoffs, preserving original workstation source bytes and results.

1. Use existing Prabhat on workstation. Check no active training or T12queue;
   never bypass locks. Preserve r1/r2,T11 and revision21diagnostic evidence.
2. From project root run ONCE (fresh output path):
   .\Prabhat\Scripts\python.exe docs/agent_exchange/t12_resume_magnitude.py --output-root results/t12_representation_workstation_r3 --detach
   Linux substitute Prabhat/bin/python. Do not use the old two-representation
   queue: it would repeat raw or encounter existing outputs.
3. This wrapper verifies passing raw receipt/audit and exact source/data hashes,
   validates successful revision21replay, then runs original magnitude readiness,
   ALL20preflight cases with original parent-reference equivalence checks,
   and20full-data smoke jobs ONLY if preflight exits0. Fault handling propagates
   into child workers. No settings changes. No reuse of15partial passes as20.
4. Verify recovery_status.json, magnitude/smoke_supervisor.json, advancing
   heartbeats and best.pt/latest.pt updates. The detached wrapper continues
   after chat closure while workstation stays awake. Report first real progress.
5. Any failure: STOP, preserve all logs/checkpoints and native fault trace,
   record child_exit.json. No retry, reinstall, driver change or relaxed gate.
6. Success: magnitude preflight20/20 and smoke20/20,40metrics/36contrasts,
   receipt/auditpassed and childexit0. Wrapper validates matching source/data/
   hardware across preserved raw and new magnitude, then writes smoke_pair.json
   referencing raw r2 and magnitude r3. It does not rewrite old receipts.
7. Append AND read back Luna_to_Astra.md; return smoke_pair.json, preflight,
   receipt,audit,timing and child_exit. STOP for the separate full-launch review.

Full training still blocked until both smoke results are reviewed. Later full
commands must use raw r2 and magnitude r3 bases, not assume both are under r3.
No full launch in revision22. T08 OPEN. Astra checked script syntax and replay
artifacts; no workstation GPU run or verified native root-cause fix claimed.

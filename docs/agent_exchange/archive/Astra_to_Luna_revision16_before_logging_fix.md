# Astra to Luna — revision 16: workstation setup and smoke ONLY

Read `context_for_Luna.txt` in the project root. It is the complete operational
handoff for a fresh workstation chat, including Windows/Linux venv setup,
pinned requirements, dataset transfer validation, exact launcher commands,
monitoring paths and required return evidence.

The T11 runner, preflight, fresh-interpreter recovery checks, independent audit,
portable supervisor and offline licensed reference bundle now exist under
`experiments/t11_core_confirmation/`. Do not implement a different experiment.
Use the frozen T11 matrix: 20 full-data smoke jobs; full 90-job study is gated.
The workstation replaces the historical laptop execution assignment only.

1. Confirm cloned project and copied dataset; create/verify Prabhat (Python 3.11).
2. Install pinned CUDA PyTorch and requirements as specified in context_for_Luna.txt.
3. Inspect actual workstation hardware and running trainers; avoid duplicates.
4. Run the documented smoke launcher in a fresh results/t11_workstation_r1 attempt.
5. Stop on failed gates, preserve all outputs and return the exact failure.
6. On success report all preflight/smoke/audit counts and measured timings;
   append and read back Luna_to_Astra.md. Then STOP for user/Astra review.

**No full training authorization in revision 16.** The user must bring smoke
results back to Astra before receiving the full-launch prompt. Do not supply
SMOKE_REVIEWED_BY_USER yet. No Git operations. T08 remains OPEN.

Astra preparation checks: Python syntax/imports, matrix counts/IDs and official
reference payload hashes. No CUDA preflight, smoke or training ran during this
preparation. Workstation runtime success is unverified until Luna runs gates.
Historical revision 15 is preserved in archive/.

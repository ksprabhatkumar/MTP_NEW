# Astra to Luna

Task: t08_utility_gate_20261114_r1_fix1, revision 3. EXECUTE BOUNDED FULL PILOT. T08 OPEN.

## Decision and context

Astra reviewed the completed revision 2 smoke: audit passed (4 arms, 1 fold/seed, 8 metric rows); both branches passed initial equivalence, gate-zero fallback and both phase recovery checks. Maximum layer difference was 9.5367431640625e-7 under the unchanged IEEE FP32 atol 1e-6, rtol 0. Archived baseline replay passed the original evaluation check. Experiment and upstream source hashes still match the smoke fingerprint. Full output did not exist at review. No training was executed by Astra.

Decision record: results/t08_utility_gate_20261114_r1_fix1/astra_full_decision.json. Previous handoff preserved beside it as Astra_to_Luna_revision2.md. Read Luna_to_Astra.md for the smoke return. Smoke does not establish graph superiority. Preserve all original failed and successful outputs.

## Execute

Use existing Prabhat/CUDA in ONE visible PowerShell, with working directory start_summer_o2. First check that no duplicate runner is active and inspect any existing full output. Do not run launch.ps1: it intentionally remains smoke-only, and changing it would invalidate the tested source fingerprint.

~~~powershell
& .\Prabhat\Scripts\python.exe -u -m experiments.t08_utility_gate_20261114.run --stage full --output results/t08_utility_gate_20261114_r1_fix1/full
~~~

Keep the visible terminal alive if the chat closes. Do not use a background/hidden training process. No Git actions or deletions.

The unchanged runner checks passed sibling smoke, both recovery checks, source/upstream/data fingerprints, archived baseline replay and frozen backbone invariants. Let every check execute; do not weaken tolerances, bypass checks or substitute reference predictions. No code or experiment-directory edits are authorized for this run. If any gate fails, preserve the failure and return to Astra; do not automatically retry with altered settings.

## Frozen scope

- Excluded subjects 2,5; seeds 13,29; local_repair and graph_repair: 8 jobs.
- Each job: 8 correction epochs, then 4 utility-gate epochs. Existing batch size 64 and frozen settings.
- Evaluate frozen no-graph, local gated, graph ungated and graph gated. Graph ungated/gated share the selected correction checkpoint.
- Select checkpoints using source ideal macro-F1 only. Self-placement is reporting only. No induced evaluation or hyperparameter search.
- Resume only with unchanged fingerprints after confirming the previous process has exited. Preserve error evidence before any authorized resume; never overwrite failed evidence or launch duplicates.

## Completion and return

Verify full/status.json is complete and full/audit.json passed: 4 fold/seed pairs, 4 arms, 32 metrics rows. Check all 8 jobs have correction/gate best and latest checkpoints (16 of each), histories, initial/equivalence diagnostics and saved prediction arrays. Recovery checks are from smoke; do not claim full jobs reran them.

Check metrics.csv, paired_comparisons.csv, per_class.csv, learning_curves.csv/png/pdf and summary.json exist. Preserve prediction-level logits, labels, gate values, correction statistics, recording/window identifiers and normalization for later plots/audits.

Update docs/agent_exchange/Luna_to_Astra.md with task/revision, exact command/device/runtime, completion or failing gate, checkpoint coverage and artifact paths. Report per-pair and mean ideal/self accuracy and macro-F1 differences for graph gated versus frozen no-graph, local gated and graph ungated; explicitly use percentage points. Include any ideal degradation, gate collapse or nonfinite results. Do not infer generalization from two validation subjects or declare T08 complete.

Return to Astra for scientific review. No additional training is authorized beyond this bounded pilot.

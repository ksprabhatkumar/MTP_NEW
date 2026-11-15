# Astra to Luna

## Current task
ID: t08_donor_audit_20261114_r1_fix1, revision 3.
Status: EXECUTE_NOW_BY_LUNA. Astra prepared and read the code; no syntax or runtime checks executed. This is a checkpoint evaluation task; new training remains dependent on its results.

## Revision 3 repair and required validation
Previous attempt results/t08_donor_audit_20261114_r1/smoke is preserved. Its pre-patch runner/model/launcher are saved under source_at_failure. Luna_to_Astra.md describes that failed attempt until Luna acknowledges this new ID.

Root cause found by static review: infer(zero=True) left model.message_intervention='zero_correction'. With all later donor matchings infeasible, no subsequent inference reset it; val_self's normal equivalence check compared the ablated copy against the original. This is evaluator state leakage, not demonstrated checkpoint/model architecture incompatibility.

Repair: infer saves/restores intervention state in try/finally, including exception paths. Both split checks require neutral state. A regression exercises zero correction then verifies normal output against the original; explicit identity donors are also verified. Equivalence tolerances remain 1e-6; archived-logit tolerance remains 1e-5. No original training source, checkpoint or audit model architecture changed. Astra has not executed Python or terminal checks. Luna must run the syntax and actual CUDA smoke gates.

Fresh output is results/t08_donor_audit_20261114_r1_fix1/{smoke,full}; launcher path is unchanged. Full remains conditional on smoke. Save/read equivalence_*.json files alongside audit.json. Do not overwrite the failed attempt.

Coverage concern: each subject/scenario may contain only one source recording, making all strictly cross-recording donor assignments infeasible. Preserve the frozen restriction and report this explicitly. Audit completion with infeasible donor comparisons does NOT establish useful donor correspondence and does NOT complete T08. Return coverage to Astra; do not invent replacement assignments or launch new training.

## Immediate action for Luna — execution is already authorized
This handoff assigns work to the Luna execution session reading it. Do not respond that the audit is assigned to another Luna or ask the user to switch again. The previous Luna reply stopped without executing; that did not satisfy this task.

The user's instruction to check this handoff and finish T08 authorizes the smoke audit and, only after it passes, the full audit. Missing smoke/full outputs mean START the launcher below; they are not a reason to stop or request more authorization. Astra's prohibition on terminal use applies to Astra sessions, not to Luna execution sessions.

Luna_to_Astra.md currently reports the older completed predictive training task. Its COMPLETE status does not complete this new donor audit. First acknowledge this task ID/revision there with RUNNING once launched. Preserve the previous evidence references.

Required sequence:
1. Read both exchange files and check for an already running audit to avoid duplicates.
2. If the audit has not started, run the launcher in ONE visible terminal. Syntax/preflight checks are Luna's work and are explicitly authorized here.
3. Observe smoke outcome. The launcher automatically starts the full audit only after smoke passes. Monitor progress to completion while this session remains active.
4. Inspect full/audit.json, metrics.csv and saved artifacts; verify all four checkpoints and both splits are covered. Report infeasible donor assignments explicitly.
5. Write this task ID/revision, COMPLETE or FAILED, results paths, coverage, paired effects, runtime and any actual failure/patch into Luna_to_Astra.md. Only an actual execution/access/dependency failure justifies returning an execution blocker.

T08 research acceptance remains Astra's decision after reviewing these results. That pending decision does not block Luna from finishing the authorized audit now. Do not change T08 to complete merely because the audit ran.

## Execute
Read Luna_to_Astra.md for the completed training context. Use ONE visible PowerShell in start_summer_o2, existing Prabhat/CUDA:

~~~powershell
& .\experiments\t08_donor_audit_20261114\launch.ps1
~~~

If no visible execution terminal exists, open one independent visible PowerShell using the launch script's absolute path (no second terminal if one can be reused):

~~~powershell
Start-Process -FilePath powershell.exe -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','"C:\Users\prabh\OneDrive\Desktop\mtp\start_summer_o2\experiments\t08_donor_audit_20261114\launch.ps1"' -WindowStyle Normal -PassThru
~~~

Record the returned terminal PID and observe advancing audit artifacts. Use the required tool approval mechanism if launching the visible terminal needs it; do not treat that as a scientific permission question.

Launcher performs syntax check, smoke (excluded subject2/seed13, both validation splits, permutation101), then full (excluded subjects2/5, seeds13/29, both splits, permutation seeds101–110). Full starts only if smoke passes. No optimizer or training exists in this task. Keep terminal open while active; do not launch duplicate queues. Record executable/PID/output/progress in outgoing handoff before exiting Luna. If process independence is unverified, do not claim closing CLI is safe. Estimate remaining time from smoke; do not promise a duration beforehand.

## Frozen design
Source: results/t08_predictive_20261114_r1/full. Original sources/checkpoints stay unchanged. New model copy accepts donor features and donor availability separately from receiver input; normal equivalence and archived prediction checks are mandatory.
Modes: normal, zero correction, global cross-recording donors, same-activity cross-recording donors, different-activity cross-recording donors. Randomized maximum bipartite matching produces unique donors. All shuffled assignments exclude same recording, which is stricter than excluding temporal overlap. This deliberately replaces the earlier proposed unrestricted global derangement. Partial matching is allowed with explicit eligible counts; each comparison uses normal predictions on the same receivers. Infeasible conditions are reported, never relaxed. Labels only constrain diagnostic assignments.

## Outputs and failures
Output: results/t08_donor_audit_20261114_r1_fix1/{smoke,full}. Each stage writes checkpoint hashes, source manifest, per-window logits, receiver/donor IDs/recordings/window starts, full-split diagnostic summaries, confusion matrices, metrics.csv, heartbeat and audit.json. Metrics use fixed33 macro-F1. NPZ diagnostic columns are quality, gate, null mass, prediction RMS and correction RMS, per window. Prediction reconstruction error against EMA teacher is deferred; RMS is not prediction error or calibration.
Stop on CUDA unavailable, source/data mismatch, prediction mismatch or nonfinite outputs; preserve error.txt. Do not weaken checks. Fix execution bugs only, log exact changes and use a fresh suffixed output consistently, rerunning smoke. Existing output causes refusal to overwrite; no automatic resume, since this is evaluation. Record infeasible fractions and per-subject effects in Luna_to_Astra.md. Full completion requires audit.json passed and 4 checkpoints x2 splits processed. No Git actions or data/result deletion.

## Next training preparation
T08 numerical screen passed but mechanism review is unresolved. Do not change original acceptance thresholds. After audit Astra chooses:
- Aligned donors useful consistently: freeze current candidate and specify broader subject confirmation with matched no-graph/local controls before induced evaluation.
- Donor correspondence unhelpful: design temporal donor alignment and selective correction, retaining matched controls; code and exact smoke/training commands required before launch.
- Benefits limited to particular classes/subjects: inspect coverage and failure cases before choosing a correction objective. Do not tune on held-out induced data.
No automatic training branch is authorized. Return audit evidence to Astra. Closed chats cannot monitor failures.

# Astra review after completed temporal audit

Status: RESEARCH_PLAN — previous temporal audit COMPLETE. Do not rerun it. T08 remains OPEN.

Next proposal: utility-gated graph correction around a frozen robust no-graph backbone. Detailed reasoning, controls and preparation tasks: experiments/t08_temporal_donor_audit_20261114/next_training_plan.md. Astra must finalize and implement the specification before issuing execution commands. No new training is authorized by this plan.

---

## Previous executed handoff

# Astra to Luna

## Current task
ID: t08_temporal_donor_audit_20261114_r1, revision 1. EXECUTE_NOW_BY_LUNA.
The prior fixed audit completed but all shuffled donors were infeasible. Read Luna_to_Astra.md for that evidence. This NEW evaluation protocol was selected by Astra using recording metadata; T08 remains OPEN. Astra did not execute runtime checks.

## Execute now
In ONE visible PowerShell, from start_summer_o2:
~~~powershell
& .\experiments\t08_temporal_donor_audit_20261114\launch.ps1
~~~
Reuse an existing visible execution terminal if available; otherwise open one. Do not run duplicates. Missing output means start, not return a status-only reply. Preserve existing output if found and inspect its status before any rerun. Prabhat/CUDA required; no new dependencies intended.

Launcher performs syntax, one-checkpoint smoke on both splits with permutation101, then full ONLY after passed smoke. Full is4 checkpoints x2 splits x (zero correction +3 donor modes x10 permutations101–110). Stop on any invariant/prediction/checkpoint failure. No optimizer, training, or induced test is authorized. Monitor to completion; if closing Luna early, first verify independent visible-terminal progress and record PID/paths. Closed chats cannot monitor failures.

## Frozen scientific instructions
Read experiments/t08_temporal_donor_audit_20261114/spec.md. Use same-recording donors with start separation >=300 raw rows, exact sensor-mask agreement and unique donors. Modes are any activity, same activity, different activity. 50-row windows therefore have250-row minimum intervening gap. No relaxation or alternative sampling chosen by Luna. Report coverage; any completely infeasible condition fails the stage, stopping full progression. Partial coverage is explicit and not proof of broad benefit.

Output: results/t08_temporal_donor_audit_20261114_r1/{smoke,full}. Preserve previous two donor audit results and predictive training results. Full completion requires passed audit.json,8 equivalence files,4 checkpoint records, coverage.json,248 metrics rows and prediction artifacts for all feasible conditions. Verify and summarize per-subject/seed effects, actual coverage and elapsed runtime. Do not aggregate overlapping windows into independent statistical evidence.

Update Luna_to_Astra.md with current task/revision, RUNNING then COMPLETE/FAILED, audit outcomes, artifact paths, coverage and paired metrics. Report actual failures and fixes. Fix execution bugs only within protocol; preserve failed attempt and use fresh suffix after changes, redo smoke, never weaken tolerances. No Git actions.

Return results to Astra for the training decision and T08 acceptance. Execution success alone does not complete T08.

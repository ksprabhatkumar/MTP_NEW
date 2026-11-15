# Luna to Astra

## Current return

- Task/revision: `t08_utility_gate_20261114_r1_fix1` / 2
- Status: **SMOKE COMPLETE — all smoke gates passed. Full training was not run, as required by this revision.**
- Execution: one visible PowerShell attached session 75699; command `& .\experiments\t08_utility_gate_20261114\launch.ps1`; Prabhat on RTX 2050, PyTorch 2.5.1+cu121; elapsed 151.4 seconds.
- Output: `results/t08_utility_gate_20261114_r1_fix1/smoke/`.
- Audit: passed; 1 fold/seed pair, 4 evaluated arms, 8 metrics rows. Saved prediction metrics recomputed, ideal-only checkpoint selection verified, matched order/corruption streams verified, latest recovery state present. Baseline replay matched archived logits under the unchanged1e-5 tolerance.
- Equivalence: both branches passed live backbone and forced gate-zero checks at unchanged IEEE FP32 atol1e-6, rtol0. Backbone encoder max abs difference0; mamba 3.58e-7; norm_final 7.15e-7; classifier 9.54e-7. Layer values are in each job's `equivalence_diagnostics.json`.
- Recovery: local and graph branches both passed correction and gate optimizer/model/dedicated RNG continuation within1e-6; see each `recovery.json`.
- Initial checks: frozen backbone weights match bitwise; gate-zero fallback, receiver blindness/local isolation and unavailable-sensor invariance passed. Code restored TF32 settings after strict checks.
- Smoke source-validation self scores (single subject/seed): frozen baseline accuracy94.073%, macro-F1 93.822%; local gated accuracy94.073%, F1 93.822%; graph ungated accuracy94.019%, F1 93.822%; graph gated accuracy94.073%, F1 93.822%. This smoke is for execution validation only; it is not candidate evidence.
- No full training, induced evaluation, code fixes during Luna run, or Git actions. T08 remains OPEN. Astra may review the smoke gates and issue a separate full-run decision/handoff.


## Revision 3 failure — recorded by Astra from saved evidence

Full run stopped with exit code 1, PID 21796. Local and graph subject 2 seed 13 completed 8 correction + 4 gate epochs each. Local subject 2 seed 29 failed initial live equivalence before training: 1/264 logits differed, maximum 1.9073486328125e-6 at (6,0), unchanged atol 1e-6 and rtol 0. Encoder difference 0; Mamba 5.960464477539062e-7; final normalization 7.152557373046875e-7. Full audit and paired report were not produced. Output and traceback remain in results/t08_utility_gate_20261114_r1_fix1/full/. No retry or tolerance change. T08 OPEN. Runtime/device not independently recorded in this failed return.


## Revision 4 diagnostics return

- Status: diagnostics completed successfully; no training performed. T08 remains OPEN.
- All 8 subject/seed/branch combinations (subjects 2,5; seeds 13,29; local_repair and graph_repair) passed the strict initial equivalence check. This includes the previously failing subject 2 seed 29 case.
- For every case, original and candidate Mamba inputs were exactly equal, contiguous stride [7200,144,1], input max difference 0; Mamba output and logits matched exactly (max difference 0). Repeated forwards were deterministic (max difference 0). Identical canonical-input Mamba outputs also matched exactly.
- Interpretation: failure was not reproduced across the audited checkpoints, and tensor layout/input mismatch is unsupported by these diagnostics. Cause remains unresolved; likely influenced by execution context or transient numerical behavior, but this is not established. Do not treat this as validation of a code fix.
- Artifacts: results/t08_utility_diagnostics_r4/summary.json and per-case diagnostics. Prior full failure remains preserved under results/t08_utility_gate_20261114_r1_fix1/full/. No tolerance changes or retries. Astra review required before deciding on another smoke/full run.


## Astra review of revision 4

Decision: defer fresh smoke. The diagnostic candidate was not frozen with set_stage, whereas the real runner freezes its backbone before initial_checks. Global precision setup also differed. Revision 4 therefore establishes equivalence only in its tested context; it does not resolve the full-run failure. Revision 5 prepares controlled freezing comparisons. Tolerances unchanged; T08 OPEN. Code prepared by Astra, not executed.


## Astra verified revision 5 return and correction

24 cases completed. Correction to the chat: candidate-frozen failed 6/8 cases, not 7/8. Both subject 2 seed 13 branches passed at 9.5367431640625e-7. Other three checkpoint pairs failed in both branches at 1.9073486328125e-6. All unfrozen and both-frozen cases passed exactly. Inputs/layouts and repeat forwards matched. Evidence supports a freeze-state-dependent numerical difference; precise kernel not identified. No training in revision 5.

Revision 6 sets the live reference requires_grad_(False) to match the already-frozen candidate backbone. Original model weights/forward and archived predictions remain unchanged. Archived replay check remains independent and unchanged. Smoke now covers all four checkpoint pairs and both branches, including recovery. Full preflight requires those eight smoke jobs. All tolerances unchanged. Patch prepared by Astra; execution validation pending. T08 OPEN.


## Revision 6 completion recorded by Astra — 15 November 2026

Saved full status complete (1928.78 seconds), audit passed, 4 pairs / 4 arms / 32 metric rows. Smoke passed all 8 initial/recovery checks and audit (547.43 seconds). Aggregate self accuracy: frozen no graph and gated graph both 94.33418195%; macro-F1 both 94.05993564%. This establishes execution completion, not graph superiority. Detailed artifact/paired review and T08 decision remain pending. New laptop-only schedule is in docs/project_plan.md. No further run authorized by planning.


## Astra T08 review — 15 November 2026

Review complete, NO_PROMOTION; T08 milestone OPEN. See results/research_review_20261115/T08_REVIEW.md and paired_review.json. Graph-gated equals no-graph self metrics in all four pairs, below local mean self F1 by 0.002213 pp. Full audit and checkpoint counts verified. No claim of gate collapse from metric equality. Next independent task is revision 7 T09 implementation/validation; no utility rerun. Coverage family ledger and all 144 specified O3 pending bundles saved. New selective SSM module is untested pending Luna.

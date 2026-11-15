# Astra review — 14 November 2026

Decision: retain receiver-blind predictive repair as a provisional candidate. The original numerical promotion screen passed; it is not redefined retrospectively. T08 remains open pending assessment of whether the graph uses useful aligned donor information. No induced-placement superiority or 4–5 percentage-point gain has been established.

Evidence: full 16/16 complete; both saved full audits passed. Source-validation self accuracy graph 94.529%, robust no graph 94.334%, local repair 94.287%. Graph self macro-F1 94.278%, versus 94.060% and 94.159%. Graph minus robust ideal macro-F1 +0.111 pp. Four fold/seed runs cover only two actual validation subjects (3 and 8; excluded subjects 2 and 5 are not validation identities).

Paired robustness: graph self accuracy versus robust no graph is 0.000, -0.108, +0.474 and +0.415 pp across the four runs. Hence the average gain +0.195 pp is concentrated in validation subject 8. No inferential significance claim is justified from two subjects or overlapping windows.

Interventions (mean over four runs):
{
  "window_shuffle_accuracy_minus_normal_pp": 0.08075512747182101,
  "window_shuffle_macro_f1_minus_normal_pp": 0.020485678969128784,
  "zero_message_accuracy_minus_normal_pp": -0.618335308056861,
  "zero_message_macro_f1_minus_normal_pp": -0.3713554028647814
}

Window-shuffled donors slightly outperform normal donors on average. Zero correction improves accuracy for excluded-subject5/seed29 by 1.126 pp. This suggests inconsistent correction benefit, but does not establish that donor signals are useless. model.py implements window_shuffle using roll(1,0) within each evaluation batch: neighbouring overlapping windows can retain activity and recording information. Zeroing messages changes the inference distribution; it is not equivalent to a separately trained no-graph baseline. Diagnostics currently saved for first16 windows cannot establish whole-dataset quality/gate calibration.

Next step: checkpoint-only donor correspondence audit, before more training.
1. Freeze existing best checkpoints and evaluation splits. Evaluate ideal and self source validation only. Do not select checkpoints or tune gates on self validation outcomes.
2. Establish deterministic normal predictions identical to the archived outputs within the existing numerical tolerance.
3. Build donor assignments across the complete validation split, not within batches. Keep the receiver local branch unchanged and move donor features, quality and availability together. Preserve sensor identities. Use seeded derangements and exclude same-recording overlapping windows wherever metadata permits; report any infeasible strata instead of silently relaxing constraints.
4. Compare normal, zero correction, global donor shuffle, same-activity/different-recording donor shuffle and different-activity donor shuffle. Activity labels are diagnostic assignment constraints only, never inference inputs or training targets. Use ten prespecified permutation seeds 101–110, shared across checkpoints. Report eligible counts and normal metrics on identical eligible receiver subsets for restricted interventions.
5. Export per-window predictions and donor/receiver metadata, fold/seed accuracy and fixed-33 macro-F1, paired differences, class confusion, and full-validation summaries of quality, gate, null mass, prediction error and correction norms. Report subject effects separately. Repeated permutations measure intervention variability, not independent subjects.
6. Astra reviews whether aligned donors reliably outperform irrelevant donors, whether corrections hurt particular activities, and whether auxiliary prediction explains any benefit. If correspondence helps, freeze the candidate for a separately specified broader subject confirmation protocol. If it does not, investigate temporal alignment and correction selectivity before proposing another architecture; no immediate large training sweep.

Execution status: RESEARCH_PLAN_ONLY. No new executable audit module or command has been prepared or run in this review. Astra must prepare the evaluator and exact Luna smoke/evaluation instructions before handoff execution. No new training or induced evaluation is authorized by this plan. Existing source/results remain intact.

# T08 utility-gate scientific review — 15 November 2026

Decision: REVIEW_COMPLETE_NO_PROMOTION. T08 candidate-selection milestone remains OPEN. Execution is complete: 8 jobs, 16 best and 16 latest stage checkpoints, passed audit with 32 rows. Full runtime 32.15 minutes; expanded smoke 9.12 minutes. No training is rerun by this review.

Mean self-placement accuracy: frozen no graph 94.33418%, graph gated 94.33418%; macro-F1 both 94.05994%. Graph gated minus baseline = 0.00000 percentage points for both metrics in EVERY pair. Graph gated minus local gated mean self F1 = -0.002213 pp (accuracy ties). Ideal graph-gated minus baseline mean accuracy +0.030491 pp and F1 +0.020224 pp. The self-F1 win and no-worse-than-local screen fails; ideal noninferiority alone is insufficient.

Graph ungated self accuracy 94.29377%, macro-F1 94.07668%. Gating recovers +0.040409 pp accuracy but loses -0.016744 pp F1 versus ungated. Thus gating does not establish a beneficial activity decision rule or donor correspondence. Equal aggregate/classification metrics do not prove identical logits or gate collapse. Saved gate/probability arrays can diagnose that without retraining.

Evidence is source validation on subjects 3/8 and seeds 13/29; these are two subjects, not four independent subjects. No induced evaluation was run in this pilot. No 4–5 pp gain, graph-superiority, generalization or Mamba contribution is established. Retain as a negative result. Existing temporal block is gated convolution.

All paired graph-minus-baseline/local/ungated effects are in paired_review.json. Source artifacts: results/t08_utility_gate_r6/full. Prior completed models remain historical evidence; do not combine absolute scores across incompatible recipes as matched comparisons. T09 implementation can proceed independently; T10's nominated-candidate arm still needs a scientific decision. No additional graph search is prescribed.

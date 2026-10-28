# Objective 1 full-run analysis, 17 October 2026

The run finished at 03:30:42 local time. All 118 scheduled jobs (53 epochs each),
118 result JSONs and best checkpoints, 326 prediction files, GPU telemetry and
all six final figures are present. The run took 19,609 seconds (5 h 27 min).
No training job failed. The primary placement comparison has three held-out
subjects (2, 5, 15) and three fixed seeds (13, 29, 47). Other ideal-placement
subjects have seed 13 only for no-graph and residual.

## Main result

| Condition | Ideal macro-F1 | Self macro-F1 | Induced macro-F1 | Matched ideal-to-induced F1 loss |
|---|---:|---:|---:|---:|
| No graph | 93.77% | 77.06% | 16.02% | 80.89 points |
| Physical graph | 94.39% | 69.22% | 12.21% | 85.37 points |
| Global graph | 93.59% | 76.66% | 12.77% | 84.01 points |
| Dynamic-only graph | 93.60% | 76.82% | 12.95% | 83.83 points |
| Residual graph | 93.96% | 72.55% | 12.80% | 84.37 points |
| Physical + sensor dropout | 94.41% | 72.49% | 12.55% | 84.86 points |
| Residual + sensor dropout | 94.38% | 76.07% | 12.78% | 84.67 points |
| Residual + GAE + dropout | 93.95% | 77.23% | 12.80% | 83.98 points |
| Residual, sender-mask ablation | 94.41% | 74.90% | 13.50% | 84.04 points |
| Adapted DynamicWHAR | 93.43% | 77.20% | 12.09% | 84.24 points |

These are means of subject means (three seeds within each of three subjects).
The first three F1 columns use all 33 declared labels; a class absent from a
recording contributes zero. The final column uses only classes shared by that
subject's ideal and induced recordings, so it is a different denominator. The
adapted DynamicWHAR architecture is an external comparator, not a matched graph
ablation. No graph treatment consistently solves recorded displacement.

The paired residual-minus-no-graph induced difference is -3.23 F1 points on
average across nine matched subject/seed pairs: negative in eight, positive in
one. It is -7.61 points for subject 2, -0.94 for subject 5, and -1.12 for
subject 15, using subject means. Ideal F1 differs by only +0.19 points on
average. Residual self-placement F1 is 4.50 points lower. These subjects,
rather than 9 seeds or thousands of overlapping windows, are the independent
test units for generalization. Three subjects are too few for a strong
population-level significance claim.

## Placement severity and class coverage

The linked `analysis_displacement_recordings.csv` evaluates each model on the
same raw recording and supported class set. Averaged over subjects/seeds,
no-graph versus residual macro-F1 is 14.28% versus 12.95% with 4 displaced
sensors; 31.08% versus 25.18% with 5; 10.79% versus 7.65% with 6; and 6.60%
versus 5.54% with 7. Scores are not monotone in displacement count. The
recordings have different durations and activity coverage: subject 5's
5-displaced file has only 14 classes, and subject 15's 7-displaced file has 25
classes, 22 shared with its ideal file. Subject 15's 4-displaced file is near
zero F1 for both models. Audit that file before inferring a dose-response.

The ideal-placement seed-13 LOSO mean over all 17 subjects is 88.30% for
no-graph and 87.77% for residual on fixed 33-class macro-F1. The mean of
supported-class F1 is 97.68% versus 97.17%. Subject 7's ideal recording has
only 238 windows and three classes: both models score 9.09% on 33-class F1
and 100% on supported-class F1. Subject 6 has only 21 ideal classes. This
coverage strongly affects all-subject fixed-denominator averages and
validation-subject selection. The parser audit found 20 missing raw feature
values in all 46 recordings and no wholly unavailable sensor/window.

## What the routing actually did

In residual subject-2 seed-13, the learned mixture logits correspond to
78.1% physical, 11.0% global and 11.0% dynamic weights. After accounting for
branch row sums and final row normalization, the effective mean contributions
on 512 randomly sampled windows are 62.6% physical, 31.7% global and 5.7%
dynamic. The final adjacency's average elementwise standard deviation across
time/windows is about 0.0004; the dynamic branch's is about 0.0078. The
Frobenius distance between mean final adjacency on ideal and induced samples is
about 0.0009. These are diagnostics for one checkpoint, not a causal proof
about all folds. They suggest that the current residual model behaves almost
like a fixed graph and motivate the intervention audit in `NEXT_STEP_PLAN.md`.

## Missing sensors are a distinct condition

When each of nine sensors is artificially hidden in turn at ideal test time,
the primary runs' mean F1 drops from 93.77% to 89.66% for no-graph, 93.96% to
91.38% for residual, and 94.38% to 93.91% for residual trained with sensor
dropout. Dropout training helps this synthetic missing-sensor task but did not
repair the recorded active-sensor placement loss. The source ideal and induced
recordings are not time-aligned paired windows.

## Efficiency and limits

The nine primary residual checkpoints have median batch-one GPU latency about
2.07 ms, versus 2.17 ms for no-graph; both count 3,350,709 parameters because
the current no-graph variant retains unused graph parameters. The external
DynamicWHAR adapter has 335,788 parameters and median latency about 1.63 ms.
These measurements use BF16 autocast on this RTX 2000 Ada and exclude offline
normalization/disk IO. They are workstation observations, not wearable-device
deployment evidence. The O1 temporal block is a gated causal convolution, not
the official selective-scan Mamba implementation.

The previous three-subject-only run used a different training pool and implicit
class set, so its values must not be merged with these tables. The present
induced subjects have now been inspected for architecture planning; future
experiments on them should be labeled exploratory. Use independent placement
recordings for a confirmatory claim.

Source artifacts: `status.json`, `summary.json`, `fold_metrics.csv`,
`analysis_overview.csv`, `analysis_paired_seed_differences.csv`,
`analysis_displacement_recordings.csv`, `analysis_ideal_loso_17_subjects.csv`,
`jobs/`, `folds/`, `data_audit.json`, `figures/` and `gpu_telemetry.csv`.

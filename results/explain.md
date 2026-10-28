# Explanation of result plots

This guide explains the plots already stored in the result folders. It links to
those files and their JSON or CSV sources; it does not copy the images or data.
PNG and PDF files with the same name show the same figure.

## How to read the main comparisons

Scores are macro-F1 unless a plot says otherwise. O1's restored summer run and
the earlier O1/O2 runs used different model pipelines and training protocols.
Read each result set on its own; do not combine their scores into one ranking.
REALDISP ideal, self and induced recordings are separate recordings. Induced
placement measures real recorded changes, while synthetic rotations apply a
mathematical rotation to an ideal recording. Neither is a substitute for the
other.

“Reason” below means the observed pattern and a plausible explanation. A bar or
line graph cannot prove its cause. Claims labeled as possible explanations need
targeted follow-up experiments.

## Restored summer-pipeline Objective 1

Source JSON: `summer_o1_session_20261017_232945/training/jobs/<condition>/<subject>/<seed>/result.json`.
The summary is in `summer_o1_session_20261017_232945/training/summary.json` and
the generated table is in `summer_o1_session_20261017_232945/training/ANALYSIS.md`.
This run has 126 jobs, 14 conditions, three held-out subjects and three seeds.

### Ideal versus induced placement

[Open the O1 placement trade-off plot](summer_o1_session_20261017_232945/training/figures/explain_o1_placement_tradeoff.png)

Each point is a condition. Its horizontal position is ideal-placement F1; its
vertical position is recorded induced-placement F1. The original fixed,
normalized fixed, residual and no-graph points cluster near 92–93% on ideal
recordings but between about 9% and 14% on induced recordings. This large drop
shows that success on correctly placed sensors does not transfer to the
recorded shifts in this protocol. The residual point is slightly above the
fixed graph points, but below no graph. The two rotation-augmentation points
move upward on induced placement and far left on ideal placement: this run's
augmentation reduces the gap while imposing a substantial ideal-performance
cost. It does not show that the residual graph caused the improvement.

Possible explanation: active displaced sensors still provide signals, but the
learned sensor-to-sensor messages may transfer poorly when those signals change.
The no-graph result is consistent with the temporal classifier relying more on
per-sensor information and avoiding harmful messages. This is a hypothesis;
the paired and routing plots below support investigating it, but do not prove
the mechanism. Three previously inspected held-out subjects make the evidence
exploratory.

### Paired induced-placement differences

[Open the O1 paired-difference plot](summer_o1_session_20261017_232945/training/figures/explain_o1_paired_induced.png)

The blue bars show residual minus normalized physical graph; green bars show
residual minus no graph. Each bar averages the three seed-matched differences
for one subject. Positive values favor residual. Residual is about +2.12 points
versus normalized fixed for subject 2, essentially tied (+0.08) for subject 5,
and lower (-0.63) for subject 15. Against no graph, residual is lower for all
three: -2.74, -2.76 and -1.44 points. This explains the aggregate result:
residual's small gain over one fixed-graph control is not a general gain over
the no-graph control. Seeds are repeated runs on the same subjects; they do not
turn three subjects into nine independent users.

### Residual routing diagnostic

[Open the O1 routing diagnostic](summer_o1_session_20261017_232945/training/figures/explain_o1_routing_diagnostic.png)

The left bars show effective contribution after branch scaling and graph
normalization: approximately 57.9% physical, 36.4% global and 5.7%
input-dependent. The right panel reports inference interventions on sampled
induced windows. Replacing the input-dependent affinity with uniform or
shuffled affinity changed no sampled argmax predictions in any of the nine
residual runs; measured adjacency variation was at most about 7.8e-6.

The plausible reason for the weak result is that the trained graph behaves
almost like a fixed physical/global mixture, so it does not meaningfully adapt
its edges to the current sensor input. Low effective mass could contribute.
However, zero changed class predictions does not mean logits were identical;
it means any changes did not cross the classifier's decision boundary for
these samples. Check logit changes, affinity entropy, gradients and branch
normalization before deciding why routing collapsed.

### Older Objective 1 placement comparison

[Open the earlier O1 placement comparison](o1_full_20261016_220301/figures/placement_comparison.png)

This is from the earlier modified-pipeline O1 run, not the restored summer
run. Blue and orange bars show ideal and induced macro-F1, with error bars for
spread across three subjects after averaging seeds within each subject. Its
no-graph induced result is about 16.0%; residual is about 12.8%, despite
similar ideal performance. This is another observation that graph messages
did not help in that earlier implementation. A possible reason is that
messages spread changed or misleading sensor features. Since the architecture
and protocol differ from restored summer O1, this plot is corroborating
historical evidence, not a matched replication.

### Older Objective 1 routing heatmaps

[Open the earlier O1 routing comparison](o1_full_20261016_220301/figures/routing_comparison.png)

These heatmaps show mean receiver-to-sender weights for the first 128 windows
in each scenario for one residual checkpoint. They retain visible body-layout
structure, which explains why the learned graph can look anatomically
plausible. A mean heatmap hides variation across windows: similar ideal and
self heatmaps do not establish that routing adapts to placement, and attractive
structure does not show that a branch improves predictions. The old analysis
found the dynamic branch's effective mass near 5.7%; its intervention checks
are more informative about usefulness than the heatmap alone.

### Older Objective 1 validation convergence

[Open the earlier O1 validation curves](o1_full_20261016_220301/figures/validation_convergence.png)

Each line is source-validation macro-F1 over training epochs for held-out
subject 2, seed 13, one line per condition. Curves reaching high values show
that the models learned the source validation distribution. They do not
predict induced-placement performance: the placement plot shows high ideal
scores alongside very low induced scores. One subject/seed is illustrative,
not a measure of variation across the full experiment.

## Earlier modified-pipeline Objective 2

Source JSON: `o2_full_20261017_075315/jobs/<condition>/<subject>/<seed>/result.json`;
aggregate: `o2_full_20261017_075315/summary.json`. The full earlier session has
128 jobs. The three plots below use the primary held-out subjects 2, 5 and 15,
with three seeds each. This O2 model was not the restored summer pipeline.

### Features crossed with graph choice

[Open the earlier O2 feature and graph comparison](o2_full_20261017_075315/figures/explain_o2_feature_graph.png)

The left panel shows ideal performance; the right shows recorded induced
placement. Green is no graph and orange is residual routing. Raw axes score
about 18.2% induced with no graph. Rotation augmentation raises this to about
83.7%, and the magnitude and magnitude-plus-dot features reach about 87.3%
and 87.9%. The likely reason is that raw axis values change under sensor
rotation, while vector magnitudes and within-sensor dot products are preserved
by a fixed rotation. The result supports orientation handling in this earlier
pipeline; it does not show that graph routing produced the gain.

Across all four feature types, residual is slightly below no graph on induced
F1. For magnitude-plus-dot, the values are 87.53% residual and 87.89% no
graph. That small difference should be read alongside subject spread, not as
proof of a real disadvantage or equivalence. Real induced recordings may also
include translation or other placement effects that rotation invariants do
not remove.

### Earlier O2 induced results by subject

[Open the earlier O2 subject comparison](o2_full_20261017_075315/figures/explain_o2_subjects.png)

The bars separate subjects rather than hiding them in one average. Subjects 2
and 5 score around 94–99% with magnitude-plus-dot features. Subject 15 scores
around 70% for both no graph and residual. Its remaining induced recordings
are mutual5–7; mutual4 was excluded because it contains only zero sensor
values. The variation can reflect different recording conditions, activity
coverage or displacement severity; this plot cannot isolate which. It warns
against presenting the overall mean as uniform performance across users.

### Earlier O2 synthetic-rotation check

[Open the earlier O2 rotation comparison](o2_full_20261017_075315/figures/explain_o2_rotation.png)

The bars compare ideal recordings with the same recordings after independent
synthetic rotations of each IMU. Raw-axis/no-graph F1 falls from about 94.2%
to 4.9%. Magnitudes and magnitude-plus-dot remain at their original scores to
the displayed precision, because norms and within-IMU dot products are
unchanged by rotating the vectors together. Rotation augmentation also stays
near 91%, showing that training on rotated raw axes can teach the model some
rotation tolerance. These results concern fixed synthetic rotations; they do
not establish robustness to arbitrary real shifts, translations or
time-varying mount movement.

### Original earlier O2 placement comparison

[Open the original O2 placement plot](o2_full_20261017_075315/figures/placement_comparison.png)

This groups all eight raw/augmentation/invariant-by-graph conditions and shows
ideal, self and induced scores, with error bars for standard deviation across
the three subjects. The pattern matches the newer focused figure: invariant
features and rotation augmentation retain far more induced performance than
raw axes, while no graph is as good as or slightly better than residual.
Large error bars, especially for invariant features, come mainly from the
lower subject-15 result. These are subject differences, not confidence
intervals over a large sample of independent users.

### Original earlier O2 validation curves

[Open the original O2 validation curves](o2_full_20261017_075315/figures/validation_convergence.png)

These lines show each feature/graph condition's selection score by epoch for
subject 2's fold. They rise and stabilize at high values, showing that the
conditions can fit the source validation data. They cannot explain the
placement gain by themselves; use the induced test comparison and synthetic
rotation plot for that. The plotted fold is one example, and the full primary
result averages multiple subjects and seeds.

## Smoke-test figures and source illustrations

`workstation_smoke_20261016/figures/` contains early smoke-test versions of
the O1 placement, routing and convergence plots. They are for checking that
the pipeline ran and emitted sensible artifacts; they are not full-run
estimates and should not be used as final evidence.

The copied DynamicWHAR source folder also has `overview.png`, `initialFE.png`
and `dynamicII.png`. These are source-paper/repository illustrations, not
plots generated from this project's results. Their explanations belong to
the cited DynamicWHAR work, so they are not interpreted as project outcomes
here.

## Overall interpretation

Across the completed comparisons, orientation-stable inputs give a large
benefit in the earlier O2 pipeline. Graph routing has not yet shown an
additional benefit over no graph, and the restored summer O1 diagnostics
indicate little effective input-dependent routing. Those findings motivate
testing why the dynamic branch is ineffective and repeating O2 on the restored
summer pipeline. They do not establish that all dynamic graphs fail. The three
recorded-displacement subjects are limited and previously inspected; a new
independent placement cohort is needed for a confirmatory population claim.

## Completed summer-based O2 placement plot (21 October 2026)

[Open the matched summer O2 plot](summer_o2_session_20261021_173341/training/figures/placement.png). The bars compare ideal, self and recorded induced placement for the completed summer-backbone O2 study. The large induced improvement from raw axes to magnitude/dot features is consistent with the latter's invariance to a fixed rotation of each IMU. Within both invariant feature families, residual routing improves slightly over the physical graph but trails no graph. The paired subject/seed values, source-only representation choice, rotation checks, routing interventions, latency, and per-recording values are in [the O2 review](summer_o2_session_20261021_173341/training/O2_REVIEW.md). Nearly constant dynamic affinities and zero prediction flips under uniform/shuffled interventions limit any claim that sample-conditioned graph routing caused the feature gain. These placement subjects were previously inspected, so this is exploratory evidence.

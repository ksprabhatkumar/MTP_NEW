# Explanatory plots from completed JSON results

Generated from the completed per-job `result.json` files and run `summary.json`
with `plot_result_explanations.py`. Each chart is saved as PNG and PDF beside
its run's results. Scores are fixed 33-class macro-F1 percentages. Plots use
the three held-out subjects 2, 5 and 15, average three seeds within each
subject, then compare subjects. O2 supplementary jobs are excluded from these
three-subject comparisons.

## Summer-based Objective 1

Source: `summer_o1_session_20261017_232945/training/jobs/*/*/*/result.json`.
This is the restored summer GIN-attention pipeline, 126 completed jobs.

- [Ideal versus induced placement](summer_o1_session_20261017_232945/training/figures/explain_o1_placement_tradeoff.png): shows the clean/shift trade-off for fixed, residual, no-graph and rotation-augmented controls. The induced data are recorded REALDISP displacements, not synthetic rotations.
- [Paired induced differences](summer_o1_session_20261017_232945/training/figures/explain_o1_paired_induced.png): residual minus normalized fixed and residual minus no graph, paired by subject and seed. Positive favors residual.
- [Routing diagnostic](summer_o1_session_20261017_232945/training/figures/explain_o1_routing_diagnostic.png): effective physical/global/input-dependent branch mass from nine residual runs, plus uniform/shuffled dynamic-affinity interventions. Zero changed sampled predictions does not establish useful adaptation.

The residual model averages 11.31% induced macro-F1 versus 10.79% for
normalized fixed and 13.63% for no graph. Its intended superiority is not
verified. The original O1 supervisor's plotting stage failed due a Python
3.13/3.11 NumPy environment mix; these independently generated figures do
not by themselves change that outer status or complete the scientific audit.

## Earlier modified-pipeline Objective 2

Sources: `o2_full_20261017_075315/summary.json` and
`o2_full_20261017_075315/jobs/*/*/*/result.json`. The full earlier run has
128 completed jobs; these charts use its 72 primary jobs. Its architecture
is not the restored summer pipeline, so its numbers must not be joined to
the O1 table as a matched comparison.

- [Features crossed with graph choice](o2_full_20261017_075315/figures/explain_o2_feature_graph.png): ideal and recorded induced scores for raw, rotation-augmented, magnitude, and magnitude-plus-dot features, each with no graph and residual graph.
- [Variation by subject](o2_full_20261017_075315/figures/explain_o2_subjects.png): induced placement for three strong orientation-handling conditions. Subject 15 remains notably harder.
- [Synthetic rotation check](o2_full_20261017_075315/figures/explain_o2_rotation.png): ideal recordings before and after independent fixed IMU rotations. This tests orientation stability, not every real placement change.

Magnitude-plus-dot/no-graph reaches 87.89% induced macro-F1 in the earlier
pipeline, while its residual counterpart reaches 87.53%. This supports a
feature benefit there, not a residual-routing benefit. The same three induced
subjects had been examined previously, so the evidence is exploratory. No
summer-based O2 GPU session has run; the CPU preflight JSON is a numerical
invariance check and is not an accuracy study.

To regenerate from the workspace root in a PowerShell with Anaconda available:

```powershell
$env:PYTHONPATH = ''
& 'C:\ProgramData\anaconda3\python.exe' plot_result_explanations.py
```

The script reads existing JSON and writes only plot files. It does not run
training or alter the result JSON.

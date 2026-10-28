# Objective 2 data gate (2026-10-17)

The audit read all 46 REALDISP source recordings (7,355,749 rows) and checked the 81,666 cached Objective 1 windows against source labels and scenario names. Three recordings have activity labels but all 117 sensor channels are zero at every raw row: `subject6_self.log`, `subject13_self.log`, and `subject15_mutual4.log`. Their 5,398 cached windows are excluded from Objective 2, leaving 76,268. The original Objective 1 results are retained; `corrected_o1_primary.csv` recomputes the relevant comparison without the invalid recording. Corrected induced macro-F1 is 16.74% for no graph and 13.54% for the residual graph.

Minor timestamp resets in ordinary files occur at null-label boundaries. The three all-zero files account for nearly all nonmonotone counts. Subject 7 has only three ideal classes and remains in ideal leave-one-subject-out testing, but is ineligible as a validation subject. Subject 6 ideal has 21 classes and is also ineligible for validation. Validation subjects require at least 30 classes in both ideal and valid self recordings.

The routing intervention on nine Objective 1 residual checkpoints showed that uniformizing dynamic attention changed a mean 0.26% of sampled induced predictions and no ideal predictions. Its effective mass was approximately 5.68%, versus 62.78% physical and 31.55% global. This motivates comparing orientation representations against the no-graph control before changing graph architecture.

See `raw_recordings.csv`, `raw_summary.json`, `invalid_recordings.json`, `routing_intervention.csv`, and `corrected_o1_primary.csv` in this directory. Objective 2 is exploratory because these held-out subjects were already inspected during Objective 1.

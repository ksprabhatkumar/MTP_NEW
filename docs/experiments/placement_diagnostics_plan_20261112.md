# Placement diagnostics and conditional experts: research plan

Date: 12 November 2026. State: RESEARCH_PLAN; not a frozen executable training specification. Model branches mean architecture paths, not Git branches. No new training is launched by this plan.

## Evidence driving the decision

The audited historical raw and mag+dot experiments and the latest repair pilot are summarized in `results/research_review_20261112/RESEARCH_REVIEW.md`. No-graph mag+dot leads induced recognition; gated pairwise has a small self-placement advantage. Raw is not the strongest ideal expert on average. Latest graph repair is tied with robust no graph and below local repair on source validation. Ordinary additional neighbor mixing has not earned promotion.

An oracle choosing between existing raw-no-graph and mag+dot-no-graph predictions can recover only 0.397 pp induced accuracy, 1.383 pp self accuracy and 0.970 pp ideal accuracy beyond mag+dot. These are label-assisted upper bounds for those fixed experts and those historical data, not achieved routing scores and not macro-F1 bounds. They do not bound newly trained/fused experts. A 4-5 pp induced accuracy target therefore requires a materially better expert or representation, not merely a switch between these existing two.

## Direction A: two input experts and deployment-time routing

Retain both input representations from the same raw window X=[A_x,A_y,A_z,W_x,W_y,W_z]: a raw branch and an invariant mag+dot branch. Begin with no-graph experts and a graph-controlled counterpart with matched losses, seeds, splits, capacity and augmentation. Do not hard-code raw=ideal or invariant=induced; that premise is unsupported by the saved means.

Let p_raw(y|X) and p_inv(y|phi(X)) be calibrated probabilities. A soft router produces g(X) in [0,1] and p(y|X)=g*p_inv+(1-g)*p_raw. Gate features can include sensor anomaly scores, disagreement, entropy and stable raw/invariant summaries. Labels, file names, scenario IDs and subject IDs must not enter deployed inference. A scenario classifier is a diagnostic auxiliary task; detecting induced does not prove one expert is better for every window.

Train gate from out-of-fold expert predictions on development subjects/recordings. The expert must not have fitted the samples used to train its gate. Select gate/temperatures only on separate development validation. Freeze expert parameters first; jointly fine-tune only as a separately budgeted later experiment. Compare invariant-only, raw-only, constant probability average, learned router and label-assisted oracle upper bound. A router must beat the stronger single expert and constant ensemble; report ideal degradation and compute cost. Keep graph versus no-graph comparisons inside the same complete routing recipe.

Separate the two possible research claims:
- Ideal-only/domain-generalization: fit experts and anomaly models using development ideal recordings plus declared synthetic perturbations; natural displaced recordings are evaluation data.
- Placement-aware supervised training: use self/induced recordings from development subjects for detector/router training. This is legitimate with disjoint subjects but changes the original ideal-only setting. Report it separately and give the no-graph controls the identical data.

Historical subjects 2,5,15 have already influenced research choices and are exploratory. Before further model selection, audit provenance and designate a locked evaluation protocol; any subjects previously used to train an existing checkpoint are not untouched tests for that checkpoint. Retrain from scratch with the final held-out subjects excluded. Nested subject-wise evaluation with documented development decisions is an option; truly fresh participants/placements are stronger confirmation. Never randomly split overlapping windows from the same recording across train/calibration/test.

## Direction B: detect placement inconsistency, then test graph usefulness

Three outputs must be distinguished:
1. Recording/window alarm: signals are inconsistent with expected placement. Feasible first target using recording scenario metadata as weak labels.
2. Sensor localization: which of nine sensors is inconsistent. Requires actual per-sensor displaced/not-displaced labels, or initially synthetic masks for a limited proxy test.
3. Exact position or displacement vector: where the sensor was moved. Not supported by activity labels and scenario filenames alone; requires verified position/orientation annotations, calibration or new controlled data.

The local manual identifies self placement of three sensors and mutual displacement involving four to seven sensors. Filenames identify scenario and count, not the sensor identities in the inspected metadata. Do not infer that sensor 4 is wrong in mutual4, mark all nine nodes wrong, or label every self-placement sample as erroneous. Search original protocol/annotations before asserting sensor-level ground truth; if unavailable, collect known-sensor perturbations and retain the recording-level task separately.

Use a detector with raw orientation-sensitive descriptors as well as invariant relational descriptors. With a fixed shared rotation R within each IMU/window, ||RA||=||A|| and (RA)^T(RW)=A^TW; temporal dot products are also invariant when the rotation is constant over their compared samples. Thus mag+dot alone cannot detect an otherwise pure constant orientation change. Invariance does not remove translation, time-varying attachment motion, bias or gain changes. Raw orientation evidence has to be conditioned/calibrated for activity so legitimate movement is not called misplacement.

Proposed graph diagnostic (hypothesis, unimplemented): masked cross-sensor prediction. Mask receiver i's current input completely; predict a fixed or stop-gradient target embedding z_i from the other sensors and optional past-only state. Prevent self/skip leakage. Standardize residuals with train-only scale:

e_i(t) = mean_k [(z_i,k(t)-pred_i,k(t))^2 / (sigma_i,k^2+epsilon)].
s_i(t) = calibrated combination of e_i(t), raw orientation inconsistency and cross-sensor coherence residuals.

Avoid collapse by a frozen teacher or a separately constrained target encoder. Condition expected relations on predicted activity or marginalize over activity probabilities, never use ground-truth activity at inference. Compare graph prediction with local past-only prediction, simple per-sensor statistical distance, a no-graph multichannel predictor, and the same detector without graph edges. High residual is evidence of distribution shift, not proof of wrong placement: new activities, unseen users, faults and noise are alternative causes.

Calibrate thresholds on held-out normal development recordings, respecting subject/recording groups. Start with an empirical target normal false-alarm rate of 5%; this is a design target, not a guaranteed rate on unseen users. Freeze temporal smoothing/persistence before test. Report AUROC, AUPRC, TPR at the target FPR, realized FPR, false alarms/hour, detection delay, per-subject and severity-4/5/6/7 results. For sensor localization report precision/recall and top-k only when ground-truth identities are available. Do not treat overlapping windows as independent confidence-interval samples.

If detection works, use q_i=calibrated reliability(s_i) to downweight unreliable senders and trigger expert routing. Preserve the local/invariant fallback and a null-neighbor option. With four to seven disturbed sensors, a clean majority cannot be assumed; test multiple simultaneous faults and the all-low-reliability case. Initially output 'placement inconsistency suspected; check these sensors' with uncertainty, not an exact location claim.

## Dependency-ordered execution plan and budget gates

1. DONE: verify historical predictions, current full pilot and oracle complementarity; save review/plots (this turn).
2. Astra, 0.5-1 working day: freeze evaluation regime, target metric (accuracy versus fixed-33 F1), minimum useful gain, maximum ideal loss and label scope; confirm original sensor annotations. Resolve fresh-test provenance. This is the main prerequisite.
3. Luna/Terra, 0.5-1 day: implement development manifests and simple recording-level anomaly baselines; no new data audit/reprocessing beyond needed placement-label mapping. Smoke preprocessing, threshold calibration, label leakage and saved scores. Astra reviews false alarms and failure examples.
4. Terra, 1-2 days: implement masked relational predictor plus matched local/no-graph controls. Run smoke, then a small predeclared development pilot. Include correct-neighbor versus shuffled/zero-neighbor tests. Do not add genuine Mamba or federation in the same ablation.
5. Astra, 0.5 day: proceed to routing only if diagnostics are informative and development experts show meaningful complementary errors. Otherwise improve the invariant expert or the corruption model first; stop router training if its attainable gain is negligible.
6. Luna/Terra, 1-2 days: fit/calibrate a minimal router on out-of-fold development predictions, compare fixed ensemble and strongest expert, then freeze the complete pipeline. Track raw/invariant x graph/no-graph evidence without selecting using final induced labels.
7. Workstation, measured after smoke: perform the frozen multi-seed subject-disjoint comparison, paired subject-level uncertainty and severity analysis. Wall-clock budget must be based on that machine's measured epoch time and job count, not a guessed 5-7 hour fill. The previous 8-job laptop repair pilot took about 32 minutes; new architecture times are unknown.
8. Astra, 0.5-1 day: write conclusion, plots and limitations; accept negative graph findings. Then revisit genuine selective Mamba and the pending O3 federation work as separately controlled objectives.

These are effort estimates, not a new deadline; the earlier 15-day calendar is stale as of 12 November. The next executable brief must replace these ranges with specific commands and a measured budget. The current requested work is review and planning.

## Required evidence for every later implementation

Source/config/data hashes; source-only normalization; full split/recording identifiers; per-epoch losses, accuracy and fixed-33 macro-F1; best/latest optimizer/EMA/scheduler/RNG checkpoints; scores/probabilities/labels/indices per window; expert probabilities/gate weights/anomaly scores; threshold and calibration manifest; graph diagnostics and interventions; runtime/memory; CSV and PNG/SVG figures; clear status/errors and a final artifact audit. Preserve historical results and use `docs/agent_exchange/protocol.md` for both directions of communication.

## Sources and scope

Dataset scenario semantics: https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset and local `har_data/REALDISP/dataset manual.pdf` (27 March 2014). Sensor displacement and robust fusion precede this project: Banos et al. 2014, https://pubmed.ncbi.nlm.nih.gov/24915181/ . The proposed detector/router is a research hypothesis, not a verified novel method or demonstrated performance gain. The algebraic invariance argument above follows directly from orthogonality of R and the implemented feature definition in o2_features.py.

# Training plan after the active summer-based Objective 1 run

Status: planning only, 18 October 2026. `entire_plan.md` remains the authoritative
completion checklist. This plan covers the immediate P4.6/P4.7 closeout and the
next two training phases, P5 (Objective 2) and P6 (Objective 3). No O2 or O3
training has started.

## Gate 0: close the running O1 session

The active run is `results/summer_o1_session_20261017_232945`. It has a frozen
126-job, 31-epoch schedule and must finish before the GPU is used for another
full study. Check the outer `status.json`, `training/completed.json`, all 126
`result.json` files, `training/summary.json`, `training/ANALYSIS.md`, placement
figure, predictions, per-recording class coverage, routing interventions,
latency, telemetry, source snapshots and logs. Complete P4.6 only after the
artifacts are present; complete P4.7 only after paired subject/seed analysis.
Keep the recorded-displacement result even if residual routing loses. The
three induced subjects were already examined, so this is exploratory evidence.

## Phase 1: P5 / Objective 2, rotation-stable routing

**Question.** Do magnitude and within-IMU dot-product features stabilize the
residual graph's adjacency and improve recorded placement performance beyond
fixed fusion, raw-axis residual routing and rotation augmentation?

**Data and matching.** Use the corrected six-axis REALDISP cache: 50 samples,
stride 25, nine named IMUs. Use the same held-out subjects 2, 5 and 15,
source validation subjects 3, 8 and 16, excluded zero-signal recordings,
seeds 13/29/47, batch 64, recovered summer model, CE main loss, Adam/OneCycle
recipe and the O1 schedule of 31 epochs. Fit all scalar normalization only on
the fold's training subjects. Select the best epoch from source ideal
validation alone. Do not select variants using held-out placement scores.

**Feature arms.** Raw six axes and its raw-vector rotation augmentation are
controls. Magnitude has two scalars per IMU/time step: acceleration and gyro
norms. Magnitude+dot has five: those norms, same-time acceleration dot gyro,
acceleration dot previous acceleration, and gyro dot previous gyro. The
existing `o2_features.py` uses the first vector as the lag-one predecessor at
the first time step; retain and document this convention. Apply independent
fixed SO(3) rotations to each IMU, using the same rotation on its acceleration
and gyro vectors across the window. Transform raw physical vectors before
training-only scalar normalization. Keep nominal sensor identity and the
availability mask. Classification is the main comparison; any GAE follows
only as a separate invariant-target ablation.

**Graph arms.** Cross each invariant feature arm with normalized physical
static graph, residual physical/global/dynamic graph, and no-graph control.
The no-graph arm tests whether a gain is due to the features alone; residual
routing remains the target method. Add raw-axis augmentation with no graph.
Raw physical, residual, no-graph and raw-augmentation physical/residual are
already being trained in the active O1 session. Reuse their scores only after
checking identical folds, preprocessing, model, optimizer, seeds and epoch
budget. If the new feature pipeline changes a raw control, rerun its matched
control rather than combine incompatible results. Keep summer static-sum as a
historical architecture reference, distinct from normalized physical static.

**Provisional full matrix.** Six invariant conditions plus raw-augmentation
no-graph = seven new conditions. Seven x three held-out subjects x three seeds
= 63 sequential GPU jobs, each 31 epochs. This is an estimated 2.5-3.5 hours
from O1 calibration, not a promise; run a source-only speed calibration before
locking the time estimate. Add any raw bridge/reruns to the explicit job count
before launch. Do not overlap this study with active O1 training.

**Preflight and smoke.** First validate per-fold labels and recording membership.
Numerically test feature invariance under independent fixed rotations
(FP32 atol/rtol 1e-5) and adjacency/prediction consistency in evaluation mode
(atol/rtol 1e-4); investigate any failure before training. Confirm no raw-axis
bypass, and record parameter counts for two-, five- and six-feature inputs.
Run one short GPU job per new condition, checking finite loss, nonzero useful
gradients, checkpoint/resume, source-validation learning and graph sensitivity.
Then run all 63 jobs in a single sequential, resumable session with per-job
logs and an isolated result directory.

**Analysis and decision.** Report fixed 33-class and common supported-class
macro-F1, ideal/self/induced scenarios separately, each induced recording,
synthetic rotation separately, subject and seed spread, prediction/adjacency
invariance, effective graph branch mass and intervention sensitivity, batch-one
latency and memory. Compare paired residual-vs-static and residual-vs-no-graph
differences within each representation; compare raw versus invariant within
each graph. Credit dynamic routing only if it adds a meaningful placement gain
beyond fixed fusion, no graph and raw augmentation without an unacceptable
ideal-placement or cost penalty. Report negative or mixed results directly.

## Phase 2: P6 / Objective 3, subject-client federation

**Entry gate.** Freeze the O1/O2 interpretation and O3 model arms first. The
planned three arms isolate the contributions: raw + residual graph, chosen
invariant feature + normalized physical graph, and the same invariant feature
+ residual graph. Choose the invariant feature by the mean best source-ideal
validation macro-F1 across the nine residual runs (subjects 2/5/15 x seeds
13/29/47); break an exact tie in favor of magnitude+dot. This rule reads no
held-out placement metric. If its placement
result is negative, still run the predeclared mechanistic comparison and label
it accordingly. Keep a no-graph control if O2 indicates that it is needed to
explain the representation effect; record that as a matrix expansion.

**Clients and tracks.** For each of the three folds, each source subject is one
client with that subject's ideal training recordings only; the validation and
test subjects are never clients in the unseen-user track. Preserve the exact
window/label/recording exclusions. The server receives model updates and
counts, not raw windows. Unseen-user evaluation tests the held-out subject's
ideal, self and available induced recordings without target adaptation.
Known-client evaluation uses each participating client's disjoint self and
available induced recordings with its frozen local state. These tracks must
not be pooled into one score. Record client class and recording coverage,
including source clients with unavailable self recordings.

**Matched methods.** For each model arm, run (1) centralized training,
(2) local-only training for the known-client reference, (3) plain FedAvg,
and (4) FedAvg with client-local normalization state. Place the same
normalization module after per-IMU scalar projection in the plain and local
normalization models; change which state is shared, not the layer layout.
Share encoder, graph, temporal and classifier weights. For the unseen-user
local-normalization evaluation, freeze a reference normalization state built
from training sources only; never fit target-subject statistics. Describe
this as a FedHealth-2-inspired adaptation, not a reproduction of that method.
Because this normalization layer changes the O2 architecture, first run a
centralized bridge on one source fold/seed and record the clean validation
difference from the frozen O2 model. All four O3 methods then use the same
bridged architecture for their paired comparisons.

**Compute-matched initial schedule.** Use all available source clients each
round, one local pass per client, for 31 rounds. Centralized and local-only
controls get 31 passes over their respective training windows. FedAvg weights
client updates by their number of training windows. Use the same three seeds,
folds, initialization policy, classifier loss and batch size across arms.
Freeze the local optimizer reset/state rule and learning-rate schedule before
full training. Full participation is the initial protocol for comparable
sample exposure; a partial-participation study is a separately named ablation.

Three model arms x four methods x three folds x three seeds = 108 run bundles.
One local-only bundle trains all source-client models for that fold/seed/arm.
This count is provisional until O2 chooses its representation and the source
client manifest is checked. O1's per-job timing suggests roughly 4.6 hours
of bare model work for 108 comparable bundles; client switching and evaluation
could make the actual O3 run materially longer. Treat 5-10 hours as an early
planning range and replace it with a measured estimate after the O3 pilot.
Split execution into two
sequential resumable sessions if calibration puts one session beyond the
desired 5-6-hour window: centralized/local-only/FedAvg first, then matched
client-local-normalization jobs. Never change the model or splits between
those sessions. Do not run O3 concurrently with O1 or O2 on the GPU.

**Preflight, smoke and results.** Validate subject-client membership and label
coverage; test sample-weighted aggregation, local-state isolation, source-only
unseen normalization, checkpoint/resume of server/client/optimizer/RNG state,
and communication-byte accounting. Smoke one fold/seed with two rounds and
limited batches, then a short source-validation pilot and measured speed
calibration. The full run logs each round's participating clients, weights,
source validation, per-client metrics, wall time, GPU memory, upload/download
bytes and checkpoint. Report mean and worst client, unseen versus known client,
recorded placement separately from synthetic rotation, centralized-to-FedAvg
loss and whether the O2/O1 gains survive federation. No privacy or on-watch
training guarantee follows from this simulator.

## Time and sequencing

The active O1 run's locked estimate is 5.36 hours from its full-training start;
actual completion can drift. O2's 63 new jobs are preliminarily 2.5-3.5 GPU
hours at the O1 measured pace. O3's early planning range is 5-10 GPU hours;
its client-round smoke and calibration will replace that estimate. The
sequence is O1 artifact audit/analysis -> O2 smoke/calibration/full/analysis ->
O3 client preflight/smoke/calibration/full/analysis. At each gate, update
`entire_plan.md`, `current_step_update.txt`, the run handoff and result files.

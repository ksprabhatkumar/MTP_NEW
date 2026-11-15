# O1 neighbor-value selection: frozen before launch

State: READY_FOR_IMPLEMENTATION / execution gate handled by start_o1_selection.ps1.
Date: 28 October 2026, Asia/Calcutta. Owner: research chat; execution via one visible terminal.

## T01: inherited and remaining decisions

DONE: three objectives, audited REALDISP, nine IMUs, acc/gyro, T=50/stride25, train-only normalization, excluded zero recordings, macro-F1 over 33 classes, subject separation, physical/no-graph/augmentation controls and three-objective research story. See resources/objectives_methodology.tex and docs/project_status.md. No new data audit.

Freeze final O1 target: mean induced-displacement fixed-33 F1 advantage >=0.5 percentage points over matched no graph, positive mean effect in at least 2/3 test subjects (2/5/15), ideal F1 loss <=0.5 points. Report accuracy, self separately, every seed/subject effect and uncertainty, and whether effect exceeds seed variation. These margins are research decisions made today, not historical resource claims. Three already-inspected subjects imply exploratory evidence; no false significance claim or target-driven checkpoint selection.

Today is SOURCE-ONLY candidate selection, not the final superiority trial. Preserve source-ideal checkpoint selection. For promotion: mean source-self F1 > no graph, no worse than local control, and source-ideal deficit <=0.5 points versus no graph. Also report each fold/seed and mechanism interventions. If criteria fail, select only a clearly marked provisional candidate or keep O1 unresolved; do not claim a win. No further unplanned search today.

## T03: observed mechanism

Full validation-set intervention audit of the 27 October checkpoints is saved at results/o1_t01_t08_20261028/operator_audit.json. Graph normal -> shuffled source-self F1: fold2 90.072 -> 90.085; fold5 95.685 -> 95.759. Zero-message scores 90.178/95.702. Removing correction produces 88.365/96.115. Neighbor content is weakly useful at best; the existing concatenation MLP can obtain benefit from its local input. This is evidence motivating a stricter value path, not proof that all neighbors are misinformation.

## T04/T06: operator contract

Input X [B,50,9,5], mag_dot; availability [B,9]. Compute h [B*T,9,16] with the existing zero-adjacency summer encoder. Preserve the same temporal block and head.

v_j = W_v h_j; m_i = sum_j A_ij v_j; g_i = sigmoid(w_g [h_i,m_i,abs(h_i-m_i)] + b_g); delta_i = g_i W_o SiLU(m_i); h'_i = h_i + delta_i.

A uses existing physical/global/dynamic residual graph with zero diagonal and missing-sender masks. No local input enters the correction VALUE network. Local h controls the gate only. W_o has no bias and starts zero; gate weights start zero and bias -2 (initial gate about 0.119). First update learns W_o; later updates must reach W_v and routing. Empty neighborhoods produce zero correction; reject all-missing windows.

Arms allocate identical modules: none (delta=0); local_control (m_i=v_i with the same gate/output); gated_neighbor (m_i=sum A_ij v_j). Allocated counts alone do not imply equal active capacity: report trained parameters, and use local control for value-path capacity. Do not change historical models or results.

Files: models/summer_gated_neighbor.py; run_o1_selection.py; check_o1_selection.py; analyze_o1_selection.py; start_o1_selection.ps1. Existing helpers imported; old pilot unchanged.

## T07: exact training matrix and gates

- Smoke: 3 arms x excluded subject 2 / validation 3 x seed13 x 4 full epochs. Separate output smoke/.
- Full selection pilot: 3 arms x excluded subjects 2,5 / validations 3,8 x seeds13,29 x 16 full epochs = 12 jobs. No held-out test evaluation. All remaining ideal subjects train each fold, retaining the inherited subject-disjoint protocol.
- Adam lr1e-4, weight_decay5e-4, existing scheduler_for with fixed epoch budget, clip norm1, hidden16, dropout0.5, batch64, BF16 if supported otherwise FP32. Same CPU-staged batches, seed, features, training exposure for all arms.
- Selection: highest source-ideal fixed33 F1, first maximum. Source-self is an explicitly declared candidate diagnostic, not checkpoint selection. No warm starts from historical arms.
- Source/data/schedule fingerprints and exact seed/RNG/optimizer/scheduler resume mandatory. Save original source snapshots and per-fold normalization/indices. Never overwrite historical results.
- Preflight: no-graph equality at initialization, missing-sensor masks, no self edges, zero-message => zero correction even after nonzero output initialization, finite actual-data gradients, two-step message gradient and exact interrupted/uninterrupted checkpoint recovery.
- Smoke must finish all three arms, reload latest/best, recompute metrics from saved logits/indices and produce plots before full launch. Full preflight rechecks source hashes. A failed smoke stops the queue.
- Full command from project root: .\Prabhat\Scripts\python.exe -u run_o1_selection.py --output results/o1_t01_t08_20261028/full --epochs 16 --subjects 2,5 --seeds 13,29 --batch-size 64
- Before direct full/resume command, run check_o1_selection.py --output results/o1_t01_t08_20261028/full. Resume requires original fingerprinted sources/config. Initial automation is start_o1_selection.ps1 in the single visible terminal already opened by start_t01_t08.ps1.

Artifacts: best/latest, per-epoch JSONL/heartbeat, source/data hashes, normalization, fold metadata, logits/labels/predictions/recording/scenario/window starts, confusion/per-class metrics, learning_curves.csv, metrics.csv, paired effects, adjacency/gate samples, zero/shuffle/correction interventions, plots PNG/PDF, timing/memory, software lock and completion audit. Training does not call an LLM. Failure saves traceback/status and stops.

Runtime is calibrated from smoke; 16 epochs are the pilot budget, not a 5-7-hour minimum. Mamba and final displacement confirmation are subsequent workstation experiments.

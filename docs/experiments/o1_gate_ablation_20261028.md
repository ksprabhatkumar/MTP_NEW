# Bounded T08 follow-up: gate initialization

State: READY_FOR_EXECUTION after fresh smoke. User requested continuation on28 October after first pilot completed.

Evidence: first neighbor-only pilot failed source-self promotion (graph92.2214%, no graph92.3278%, local92.5451%). Distant donors reduce graph F1 on average, so neighbors carry some useful information. Gate initialization0.119 produces small correction amplitudes. Hypothesis: accepting more neighbor signal initially improves optimization; this is not a claim that misinformation is solved.

Change ONLY gate bias from -2 to +2 (initial sigmoid0.119 ->0.881). Keep zero output projection, masks, value operator, temporal model, features and optimizer unchanged. The same gate change applies to local_control. None allocates identical modules as before but never uses the gate. Fresh matched controls are trained, not inferred or cherry-picked from old runs.

Exact matrix inherited from o1_neighbor_selection_20261028.md: smoke3 arms x fold2/validation3 x seed13 x4 epochs; then full3 arms x folds2,5/validations3,8 x seeds13,29 x16 epochs, batch64, source-ideal checkpoint selection. Train ideal source subjects only; no held-out test evaluation. Same promotion thresholds and no extra search based on target scores.

New isolated files: models/summer_gated_neighbor_open.py, run_o1_gate_ablation.py, check_o1_gate_ablation.py, start_o1_gate_ablation.ps1. Base implementation and old results remain unchanged. New run directory results/o1_gate_ablation_20261028; source/data/config fingerprints and checkpoints retain exact identity. New config explicitly records initial_gate_bias2.0 and operator version.

One dedicated visible terminal runs smoke then full automatically. Previous session is complete; its owned terminal may be replaced to avoid simultaneous training terminals. Same finite-value, availability, initialization, checkpoint recovery and artifact/plot checks required. Smoke failure stops full launch. Failure never changes hyperparameters.

After full completion: compare paired graph/no-graph/local effects; inspect zero and distant-donor interventions. If still unsuccessful, record failed optimization hypothesis and do not promote. A subsequent change to the training objective or graph structure requires a separate evidence-backed specification; this gate ablation is not a license for an unlimited search.

Full/resume: .\Prabhat\Scripts\python.exe -u run_o1_gate_ablation.py --output results/o1_gate_ablation_20261028/full --epochs 16 --subjects 2,5 --seeds 13,29 --batch-size 64 . Executable launcher start_o1_gate_ablation.ps1 runs smoke and then this command. Original fingerprinted source/config required for resume.

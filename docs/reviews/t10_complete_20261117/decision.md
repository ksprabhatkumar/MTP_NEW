# T10 complete scientific review — 17 November 2026

Decision: **no graph promotion; T08 OPEN**. T10's 40-job source pilot is complete. Both saved queue audits pass (10+30 jobs,18+54 contrasts). This review reads saved results and recomputes aggregates in JavaScript; it does not reload checkpoints or rerun training. No induced evaluation was performed for this review.

## Combined results

Equal weights across the four subject/seed blocks, equivalent here to seed means within subject then equal subject means. These are only TWO validation subjects. F1 is fixed33; subject3 ideal has no support for class31, so absolute ideal/self scores also reflect support differences.

|Temporal|Arm|Ideal F1 %|Self F1 %|Self accuracy %|Mean job minutes|
|---|---|---:|---:|---:|---:|
|old|none_clean|93.877|92.720|92.881|3.20|
|old|none_robust|94.821|94.192|94.454|7.52|
|old|physical_robust|93.330|91.473|92.157|8.10|
|old|local_repair|94.818|94.323|94.388|7.76|
|old|graph_repair|94.863|94.148|94.310|7.92|
|ssm|none_clean|93.572|93.206|93.017|13.15|
|ssm|none_robust|94.603|94.532|94.435|29.03|
|ssm|physical_robust|93.633|92.397|93.006|29.52|
|ssm|local_repair|94.659|94.547|94.447|35.87|
|ssm|graph_repair|94.590|94.572|94.482|34.41|

|Temporal|Excluded / validation|Seed|Graph − robust no graph, self F1 pp|Graph − local, self F1 pp|Ideal graph − no graph pp|
|---|---|---:|---:|---:|---:|
|old|2/3|13|0.1208|-0.0850|0.0717|
|old|2/3|29|-0.4704|-0.3269|0.0200|
|old|5/8|13|0.1400|-0.4618|-0.1363|
|old|5/8|29|0.0341|0.1743|0.2140|
|ssm|2/3|13|0.1583|0.1359|0.0562|
|ssm|2/3|29|-0.3009|0.1511|0.0378|
|ssm|5/8|13|0.1964|-0.0778|-0.0834|
|ssm|5/8|29|0.1060|-0.1063|-0.0627|

SSM graph's mean self-F1 advantage is +0.03994pp over robust no graph and +0.02574pp over local. Its ideal difference versus robust no graph is -0.01300pp. It therefore passes the original aggregate numerical source screen, but that screen only grants CANDIDATE_REQUIRES_ASTRA_REVIEW, not promotion. The full evidence is insufficient for promotion: graph-minus-no-graph is negative after averaging seeds for validation3 (-0.07129pp), and graph-minus-local is negative for validation8 (-0.09206pp). It wins versus no graph in3/4 blocks and versus local in2/4. This is fragile, very small source evidence, not recorded-displacement superiority.

Old-temporal graph loses on average to robust no graph (-0.04387pp) and local (-0.17484pp); it fails the existing aggregate source screen. Do not add a post-hoc significance threshold to change that conclusion.

SSM robust no graph gains +0.34079pp self F1 over old robust no graph, but loses -0.21783pp ideal F1 and approximately-0.01883pp self accuracy. Its self-F1 gains are positive after seed averaging for both validation subjects (+0.56790pp for3,+0.11368pp for8). This supports retaining SSM for final confirmation, not claiming broad superiority. Measured mean job time is29.03 versus7.52 minutes, about3.86x. Keep the old backbone as the efficiency control. SSM local adds just+0.01420pp self F1 versus SSM no graph and changes model capacity; retain it as the graph capacity control.

## Compatibility and reuse

Both runs use identical dataset fingerprints and model/SSM/feature source hashes. Normalized-line source comparison shows training.py only added observational stress hashing; run.py added ownership selection and hash logging. checks.py/audit.py expanded verification. Thus aggregate these as the same scientific16-epoch pilot, disclosing instrumentation changes. Earlier ten jobs lack historical actual-stress hashes; do not claim them retrospectively audited. No blind source-hash whitelist is justified for future changes.

Best epochs range9–15, none selected16. This is not proof of convergence. The final study is planned at31epochs as a separately frozen longer schedule, not a continuation of16-epoch OneCycle. All pilot checkpoints remain useful scientific pilot evidence, timing and diagnostic anchors; they are not valid31-epoch matched results. No redundant pilot rerun is planned.

Summed saved epoch times for40jobs=42355.02s=11.765h including validation. The new30-job receipt reports about8.43h; distinguish receipt duration from epoch totals. Per-job timings are noisy, not controlled throughput benchmarks.

## Next decision

Freeze the next main matrix: both temporal backbones × all five existing spatial/robustness arms × excluded subjects2,5,15 × seeds13,29,47,31epochs (90jobs). This preserves clean and fixed-graph controls rather than quietly dropping them based on pilot results. Validate on3,8,16 respectively, with explicit data eligibility checks. Main anchor: robust no graph. Graph remains a challenger and local repair its matched capacity control. All training uses ideal source subjects; select checkpoints on source ideal F1 only.

No new utility gate, physics augmentation, multi-lag feature or hyperparameter sweep in this round. Physics proposals require unit/physical-assumption validation before becoming an experiment; adding them now would confound the temporal/graph conclusion. The alternative-method research file remains a separate candidate plan.

This is the CORE mag+dot temporal/spatial confirmation round, not all remaining thesis work. Raw/rotation/magnitude representation comparisons, external comparator, GAE/augmentation and any required donor ablations, O3 and thesis delivery remain tracked obligations. Their final additional matrices are not silently replaced by this90-job round.

Recorded induced subjects2/5/15 were inspected historically: report their final scores as retrospective exploratory evidence. They cannot become untouched confirmation. Graph can only earn an evidence-qualified advantage after consistent matched recorded-placement results and mechanism analysis; no numerical gain is promised.

See ../../experiments/t11_core_confirmation_r1.md and ../../agent_exchange/Astra_to_Luna.md for implementation/smoke handoff. No new training launched.

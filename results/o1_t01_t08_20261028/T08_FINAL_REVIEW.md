# T08 final review and handoff

Reviewed 29 October 2026 (Asia/Calcutta). State: REVIEW_COMPLETE_NO_PROMOTION.

## Decision

T08 review is complete. None of the three candidates meets the frozen promotion rule: source-self F1 must exceed its matched no-graph control and be at least as high as its local-capacity control, with source-ideal deficit no greater than0.5 percentage points. Do not relax that rule after seeing these results.

Retain original message_correction as the reference graph candidate for the next research decision because it has the largest observed gain over its own no-graph control. This is NOT approval as a validated graph improvement. Cross-family module allocation changes data-order RNG consumption, so absolute family rankings are not a clean causal comparison.

| Study | Graph self F1 % | Matched no graph % | Matched local control % | Graph minus no graph pp | Graph minus local pp | Promoted |
|---|---:|---:|---:|---:|---:|---|
| neighbor_gate_low | 92.2214 | 92.3278 | 92.5451 | -0.1064 | -0.3236 | No |
| neighbor_gate_open | 92.5689 | 92.3278 | 92.8400 | +0.2411 | -0.2711 | No |
| original_correction | 92.7117 | 92.2249 | 93.0090 | +0.4867 | -0.2973 | No |

## Interpretation and limitations

The original correction reaches92.7117% source-self F1 versus92.2249% no graph, but local control reaches93.0090%. Thus the measured no-graph advantage does not establish a graph-specific advantage beyond additional local processing. Accuracy values are saved separately in T08_FINAL_DECISION.json; F1 and accuracy must not be conflated.

Distant-donor tests show that message content affects predictions, but effects vary across folds/seeds. For the original correction the mean self-placement donor effect is slightly positive, so its source-self gain cannot be confidently attributed to correct neighbor content. This weakens a message-quality claim; it does not prove that every neighbor is harmful.

Only source-validation subjects3 and8 were evaluated, using excluded-subject folds2 and5 and seeds13/29. Neither held-out induced displacement nor the final three-subject protocol was evaluated. Four fold/seed pairs are not four independent subjects. Graph superiority on the final displacement task remains OPEN.

## Verification

All36 full jobs and9 smoke jobs completed. Each full job has16 epochs, best inference checkpoint and latest resumable model/optimizer/scheduler/RNG checkpoint. Saved completion audits verified histories, loaded checkpoints, recomputed prediction metrics and regenerated plots. Frozen training source hashes were checked again at closeout. No historical results were deleted and no Git commands were run.

The last training completed28 October2026 at22:44:19 IST. The 29 October work is inference/review/documentation only. Graph diagnostic plots, paired effects, CSV metrics, logits, window/recording identifiers and donor indices are retained. analysis_source snapshots preserve the postprocessing scripts.

## Research-chat handoff (GPT-6 Astra Medium)

1. Do not rerun these completed pilots or promote a candidate solely to obtain a positive conclusion. The frozen candidate-comparison budget is closed.
2. Next O1 research decision: specify a graph-specific mechanism or training objective that distinguishes neighbor utility from local capacity. Evidence motivates checking relation-conditioned values or source-side sensor-corruption training with identical augmentation exposure for controls; neither approach is yet implemented or authorized as an unspecified full matrix.
3. Keep no-graph and local-capacity controls, source-only checkpoint selection, independent data-order RNG for future cross-family comparisons, and untouched-test discipline. Preserve the final displacement target and uncertainty reporting.
4. T09 genuine-Mamba implementation may proceed independently using docs/experiments/mamba_temporal_spec_20261028.md. Its implementation/training is not completed by T08.

## CLI handoff (GPT-6 Sol Medium)

No new experiment is READY_FOR_EXECUTION from this review alone. Read the next frozen research specification, implement it, run fresh smoke, then start its approved full queue in one independent visible terminal. Save checkpoints/plot data and stop safely on errors. Do not automatically launch a large displacement study with the rejected candidates.

T02 laptop launcher-exit survival was verified; actual desktop-app/CLI shutdown was not exercised during this chat. Workstation environment verification remains required. User performs all Git synchronization manually.

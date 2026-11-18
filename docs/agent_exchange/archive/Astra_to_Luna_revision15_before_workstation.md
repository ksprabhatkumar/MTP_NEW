# Astra to Luna — revision 15: implement T11 core and run smoke only

T10 source pilot COMPLETE:40/40jobs, both audits pass. T08 OPEN / NO_GRAPH_PROMOTION. The SSM graph aggregate source screen passes narrowly but its mean gains (+0.03994pp vs robust no graph,+0.02574pp vs local) reverse across subjects/controls; do not claim graph superiority. See docs/reviews/t10_complete_20261117/decision.md and summary.json.

The next90-job scientific design is frozen in docs/experiments/t11_core_confirmation_r1.md and t11_core_confirmation_r1_matrix.json. Implement it in a NEW experiments/t11_core_confirmation namespace. All needed matrix/optimization/evaluation choices are in the spec. Preserve T10 and all other saved evidence. No Git; no existing results overwritten.

## Current assignment
1. Read spec, matrix and complete review. Confirm no trainer is active using reliable process visibility and locks. Empty restricted process output is not sufficient.
2. Implement the new runner, explicit FP32 evaluator, generalized subject fold loader including excluded15/validation16, preflight, independent artifact audit and durable launcher. Verify existing data exclusions and normalization. Do not silently call the old build_fold which only supports2/5. Keep scientific settings frozen.
3. Implement syntax/config/geometry/gradient/shared-initialization/stream/recovery gates exactly as specified; disk and fresh-interpreter next-step recovery, unchanged tolerances. Verify T09 reference/source provenance. Use unique fresh output attempt paths under results/t11_core_confirmation_r1/. No module is claimed to exist yet: finish implementation BEFORE invoking it.
4. Run20full-data smoke jobs (4epochs): all10combinations at subject2/validation3 and subject15/validation16, seed13. No reduced batches or windows. Stop on any failed check; preserve tracebacks and partial artifacts. Routine implementation fixes use fresh attempts and renewed gates. Do not drop conditions or change tolerance/schedule.
5. Independently audit predictions, checkpoints, metadata, selection, actual corruption/order hashes and expected counts. Save timing projections for90jobs, plots and exact next full-launch command. Update AND read back Luna_to_Astra.md, including sources/data/reference hashes and failed attempts.

**Do not launch the90-job full queue in this revision.** Return smoke status and actual timing for a separate full-launch decision. No induced evaluation during current implementation/smoke; final evaluation follows the frozen spec after training authorization. No re-running the completed40-job pilot. No physics/Gram/new-gate changes in this round. T08 stays OPEN; other representation/external comparator/GAE/augmentation/O3 tasks remain required after explicit scope checks.

Astra performed static/JSON review and arithmetic on saved results only; no new Python tests, checkpoint loading or training in this review. Revision14 launch instructions are superseded: its30-job queue is complete.

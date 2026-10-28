# Documentation cleanup guide

Prepared: 27 October 2026 (Asia/Calcutta). **Nothing was deleted, renamed or overwritten.** Only new documentation and a historical archive were created.

## Working documents

- `project_plan.md`: consolidated roadmap, including graph message-passing proposals and a separate genuine-Mamba experiment before O3.
- `project_status.md`: completed experiments, current results and unresolved issues.
- `project_handoff.md`: concise context for continuing work, paths and operational boundaries.
- `documentation_cleanup.md`: this manual deletion list and dependency exceptions.
- `archive/planning_history_2026-10-27.zip`: exact bytes of all 19 root planning/status files below plus SHA-256 manifest; prevents losing historical details during manual cleanup.

Use lowercase snake_case for working documents. Date immutable archives with YYYY-MM-DD. Do not create duplicate next-step/update documents.

## Manual deletion candidates: root copies only

These 11 files are consolidated and archived. No direct references were found in current executable source/configuration during this review. Older Markdown/TeX documentation may still link to them; after deletion use the new working documents or archive to resolve historical references. This is not authorization to delete other same-named files in subfolders.

| Root filename | Why this standalone copy can be retired |
|---|---|
| `plan.md` | Early task breakdown superseded by the consolidated plan. |
| `steps.md` | Early O1-only scope superseded by the consolidated three-objective plan. |
| `PROJECT_UNDERSTANDING.md` | Historical project assessment superseded by current status. |
| `FURTHER_RESEARCH_PLAN.md` | Research decisions preserved in the archive; next work consolidated in project_plan.md. |
| `NEXT_TRAINING_SESSION.md` | Completed O2 schedule with stale interim statuses; historical details archived. |
| `NEXT_PHASE_PREP_HANDOFF.txt` | Historical O2/O3 preparation handoff consolidated into current status and handoff. |
| `O2_TRAINING_HANDOFF.txt` | Historical O2 handoff; run-local evidence remains preserved. |
| `TRAINING_HANDOFF.txt` | Root historical workstation handoff; do not remove any same-named file inside results. |
| `TRAINING_RUN.md` | Historical workstation instructions preserved in archive; paths are not valid local instructions. |
| `handoff_message.txt` | Overlapping cross-chat history replaced for daily use by project_handoff.md. |
| `update.txt` | Accumulated historical updates replaced for daily use by project_status.md. |

## Retain these legacy filenames

These eight files are archived too, but are excluded from the deletion list because scripts or preserved pointers still use their names. Changing those integrations is outside this cleanup. Their narrative may be stale; use project_status.md for current conclusions.

| Root filename | Dependency / reason to retain |
|---|---|
| `entire_plan.md` | run_summer_o1.py copies this into new runs; other scripts link to it. |
| `NEXT_STEP_PLAN.md` | run_o2.py includes this in source snapshot/hash inputs. |
| `NEXT_TWO_OBJECTIVES_TRAINING_PLAN.md` | run_summer_o2.py copies it into new runs. |
| `O1_ALTERNATIVES_PLAN.md` | run_summer_o1_alternatives.py copies it; the session JSON references it. |
| `TRAINING_PLAN.md` | run_training.py includes it in source snapshot inputs. |
| `current_step_update.txt` | Written by run_long_training.py and run_summer_o1.py; machine-facing status file. |
| `SUMMER_O1_HANDOFF.txt` | Written by run_summer_o1.py; machine-facing handoff file. |
| `SUMMER_PHASE_HANDOFF.txt` | Referenced by results/summer_o1_active_run.json; retain historical pointer target. |

## Everything else stays

Keep `SUMMER_O1_AUDIT.md` as distinct architecture/provenance evidence, `context.text`, `README.md`, and all files under `resources/`. Keep all results and run-local handoffs/plans/source snapshots; every program, model, launcher, test and executable JSON configuration; datasets/raw data; checkpoints; environment and installation files; logs including `results_log.txt` and `pytorch_cuda_install.log`; figures and split manifests. No cleanup of those categories was performed or proposed.

## Verification

All 19 archived payloads were checked byte-for-byte against their originals after writing. ZIP integrity passed. No existing file was modified. This guide lists candidates for the user to remove manually; no deletion command was executed.

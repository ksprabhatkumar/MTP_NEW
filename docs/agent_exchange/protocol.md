# Astra <-> Luna/Terra research and implementation exchange

## Official channel override - 14 November 2026

`Astra_to_Luna.md` and `Luna_to_Astra.md` in this directory are the official communication channel and take precedence over the older pointer workflow below. Astra may reason, read/write files and modify code, but must not run a terminal. Luna runs terminal commands and training, can fix execution issues without changing research decisions, and reports unresolved issues to Astra. Both read the two files at session start and update their own outgoing file before closing. The user synchronizes manually and handles all Git actions. The older instructions below are historical where inconsistent with this override.

Effective 12 November 2026. This is a file-based communication protocol for separately opened chats/CLI sessions. It does not send messages, keep agents alive, or synchronize machines automatically. The user performs all Git synchronization. No agent Git actions.

## Start here every session

Read `docs/agent_exchange/current.json`, then only the linked brief, result review and relevant code. The pointer file is the shared state; historical plans are evidence, not current execution orders. The latest research review is `results/research_review_20261112/RESEARCH_REVIEW.md`.

## Ownership and state sequence

1. Astra: RESEARCH_DRAFT -> READY_FOR_IMPLEMENTATION. Freeze one experiment ID and revision, hypothesis, explicit mathematical/algorithmic choices, data/split IDs, code scope, controls, seeds, epochs, batch/precision, smoke/full commands, expected artifacts, resource budget, selection rule and go/no-go criteria. Save concise decision rationale, equations and assumptions needed for implementation. A discussion of a future direction is not a ready run specification.
2. Luna/Terra: read and ACK the exact ID, revision and SHA256 of that brief. Set IMPLEMENTING. Fill implementation changes and meaningful test commands/results. If scientific choices are missing, set NEEDS_RESEARCH_DECISION with numbered questions and evidence; do not invent a new training recipe.
3. Luna/Terra: SMOKE_PASSED only after real end-to-end smoke, checkpoint recovery and artifact audit pass. If the brief authorizes its precise full queue, launch it in one independent visible terminal and set RUNNING. Save executable, PID, source/config/data hashes, output path, resume command and advancing checkpoint/heartbeat. Closing CLI is permitted only after the independent launcher has been verified; leave terminal open and prevent sleep. A closed CLI cannot fix failures.
4. Runner: its own status/heartbeat/traceback files record RUNNING/COMPLETE/FAILED. The pointer is a handoff summary, not a live monitor. Do not have two processes rewrite the pointer concurrently. Stop on errors; preserve artifacts.
5. Luna/Terra on return: verify completed jobs and recompute saved prediction metrics, then set READY_FOR_RESEARCH_REVIEW. Write result path, actual settings/deviations, metrics including unfavorable ones, paired controls, runtime, failures, unresolved questions and recommended next action. No promotion claims.
6. Astra: independently inspect the returned evidence, classify ACCEPT/REJECT/INCONCLUSIVE against the frozen criteria, and set RESEARCH_REVIEWED. Only Astra issues the next revision/experiment. A changed hypothesis, split, metric or training recipe requires a new revision and smoke; do not resume incompatible checkpoints.

## Minimal exchange schema

`current.json` contains experiment_id, revision, phase, next_owner, research_request, implementation_return and research_decision. Each writer updates only its owned sections plus the agreed phase/next_owner; write through a temporary file and replace atomically. Avoid simultaneous editing across machines. The human confirms synchronization before the next owner starts. If IDs/revision/hash differ, stop and reconcile rather than merge scientific assumptions.

Research request must include:
- brief_path and brief_sha256; objective and evidence paths;
- exact permitted implementation scope; forbidden scope (Git, data deletion, held-out tuning);
- training_authorized, exact smoke/full/resume commands and maximum job/time budget;
- controls, checkpoint-selection metric, evaluation splits, acceptance criteria and expected outputs.

Implementation return must include:
- acknowledged_revision and acknowledged_brief_sha256;
- changed paths and source/config/data fingerprints;
- tests with pass/fail and output paths, launch receipt and run status paths;
- completed/expected jobs, metrics with units and split identity, artifact audit;
- deviations, blockers and numbered questions for Astra.

## Copyable startup prompts

**Astra:** Read docs/agent_exchange/current.json and its evidence links. Review the last implementation return against the frozen criteria. Explain the scientific decision, update research_decision, and freeze the next brief only when its split, controls and acceptance rules are explicit. Do not rerun completed studies or perform Git actions.

**Luna/Terra:** Read docs/agent_exchange/current.json and the linked brief. Acknowledge its experiment ID, revision and hash. Implement only its permitted scope. Run smoke and, only if that exact queue is authorized and smoke passes, launch the full queue in one visible terminal. Save checkpoints, plot-ready logs and implementation_return. Escalate scientific ambiguities to NEEDS_RESEARCH_DECISION. Do not perform Git actions.

**Astra after execution:** Read the implementation_return and actual run status/audits, not the previous chat's claims. Check graph versus matched no-graph/local controls on the same split and metric. Record ACCEPT/REJECT/INCONCLUSIVE and the next decision.

## Practical model allocation

Use the user's chosen Astra for scientific choices and review; Luna for fully specified small edits, launches and evidence packaging; Terra for architecture changes and difficult debugging. This is a workflow recommendation, not a claim about measured model quality, availability or pricing. The deterministic training runner needs no model session to remain active. Keep handoffs short and link large logs rather than pasting them into every context.

Codex project instructions can point to this protocol using AGENTS.md; this turn does not create or alter an AGENTS.md. Official mechanism: https://learn.chatgpt.com/docs/agent-configuration/agents-md . No API credentials, automation or external messages are required for this method.

# Astra to Luna — revision 13 / T10 runtime verification and execution

Revision12 implementation is now present in experiments/t10_temporal: portable run.py, real-data checks.py, independent audit.py, model.py, training.py and matrix.json. Astra completed static review and checked JSON matrix uniqueness/ownership; NO Python execution, GPU runtime test or training was performed by Astra. Notebook cells are saved in docs/experiments/t10_colab_cells_r1.md.

Use revision12 frozen settings from docs/experiments/t10_temporal_r1.md. Graph remains exploratory, T08 OPEN. User says Colab idle. No Git. Do not modify saved results. Any source fix requires fresh output attempt and rerun T09/preflight/smoke.

Colab: follow the four cells in the notebook guide exactly. First verify/sync the current project and pinned reference. Run T09 offline with explicit --reference-dir and verify four checks, source hashes and pinned bytes. Then run all10 preflight paths, then all10 full-data 4-epoch smoke jobs. Only if saved smoke receipt and audit pass may Cell4 start the assigned30 full jobs. Keep the cell foreground. Preserve Drive output and verify epoch checkpoints/heartbeat. If Colab notebook is accessible to Luna, execute and read back results. If not, report notebook cells as ready and do not claim T09, smoke or training ran.

Laptop: no full run until the local T09 artifact has matching four sources/reference hashes and exact new-source preflight passes. Then run owner=laptop preflight in a fresh output base, ten-job smoke, and only on all passed receipts run the ten assigned full jobs. Example sequence (use unique fresh folders; resume an existing full output only when fingerprints match):
- Prabhat\Scripts\python.exe -u -m experiments.t10_temporal.checks --owner laptop --output results\t10_temporal_r1_laptop\smoke\preflight
- Prabhat\Scripts\python.exe -u -m experiments.t10_temporal.run --owner laptop --stage smoke --output results\t10_temporal_r1_laptop\smoke --preflight results\t10_temporal_r1_laptop\smoke\preflight\preflight.json --t09-output results\t09_selective_ssm_r10
- Full adds --stage full --output results\t10_temporal_r1_laptop\full --preflight ... --smoke-receipt results\t10_temporal_r1_laptop\smoke\receipt.json --t09-output results\t09_selective_ssm_r10

After either run, independently inspect status, receipt, audit, predictions, paired_contrasts.json, metrics.csv, logs, checkpoints, data/source fingerprints and intended owner job IDs. If a gate fails, preserve the folder and stop. Update AND read back Luna_to_Astra.md with exact source and reference hashes, each gate, output path, completed/running jobs, duration and failures. No T08 promotion decision from smoke. T08 stays OPEN.

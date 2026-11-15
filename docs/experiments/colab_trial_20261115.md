# Colab trial brief — 15 November 2026

Decision: YES to a bounded readiness/engineering trial; NOT yet ready to authorize T10 scientific training. T08 remains OPEN; utility-gate review NO_PROMOTION. Laptop T09 reference validation/calibration already accepted. User-reported Colab T09 pass is encouraging but actual Colab source/reference artifacts have not been inspected in this chat. Its command lacks --reference-dir, unlike the current local harness: reconcile source version before accepting it as the same experiment. Never infer exact parity from rounded error numbers alone.

## Executable first step

User synchronizes files manually. Do not issue Git commands. In the Colab notebook:

~~~python
from google.colab import drive
drive.mount('/content/drive')
~~~
~~~python
%cd /content/MTP_NEW
!python -u -m experiments.colab_readiness.check --t09 /content/drive/MyDrive/mtp/results/t09_selective_ssm_colab_r1 --output /content/drive/MyDrive/mtp/results/colab_readiness_20261115_r1
~~~

This is an inventory, not a training gate or a success claim. Fresh output required; use a fresh suffix if it exists. Current preflight is statically prepared, not executed here. Read readiness.json, actual T09 JSONs and current harness. Match model/backbone/source/reference hashes; normalize Windows separators for comparison. Missing/mismatched sources require review and possibly rerunning T09 under the current harness and same pinned local reference. Avoid reinstalling torch; use Colab's compatible CUDA environment, record versions. Required Python imports include torch, numpy, scipy, pandas, sklearn and matplotlib plus actual repository imports. No Windows venv/PowerShell or optimized Mamba extension installation.

## Data and persistence

Code root /content/MTP_NEW (no extra start_summer_o2). Raw recordings at har_data/REALDISP; local prepared cache dataset/workstation_o1. Verify Drive cache exists and hashes match local cache; if absent, COPY the completed local cache to /content/drive/MyDrive/mtp/dataset/workstation_o1 and verify before relying on runtime restart. Do not regenerate a cache merely because the runtime changed. Preserve audit and per-recording metadata.

Train from local cache to reduce Drive I/O. Save every epoch's best/latest checkpoints, history, status, config/data/source/reference hashes, normalization and RNG/order state into a fresh Drive run folder. Use generation-numbered durable checkpoint files and retain the preceding valid generation; verify saved files before advancing the resume manifest. Drive FUSE cannot be assumed to have local-disk atomic guarantees. After runtime loss restore verified latest generation, same config/hash/environment; skip audited completed jobs only. Never mix laptop and T4 trajectories silently; cross-device continuation is not bitwise reproducibility. New incompatible config gets a new run ID.

Linux runner must replace Windows msvcrt lock with a Linux-compatible lock, preserve single writer, handle ordinary errors and resume. Forced VM deletion cannot execute a finally handler: only already-persisted checkpoints survive. No anti-idle/limit-bypass scripts. Use notebook execution, not remote shells. Closing chat is distinct from Colab disconnect/lifetime limits.

## Next training rounds and unresolved gate

Keep complete T10 design: old/new temporal x no graph/physical/nominated graph, plus matched local-capacity control where needed. T08 has no promoted candidate. Its graph may enter as an explicitly exploratory reference after a documented decision, not by claiming T08 succeeded. Do not launch an arbitrary reduced matrix or replace scientific full runs with calibration.

Before T10 authorization Astra must freeze nominated graph role, exact paired loss/augmentation recipe, folds/seeds/epochs and all required controls. Source-only checkpoints use ideal macro-F1; no self/induced selection. Preserve train-only normalization, subject separation and independent initialization/order RNG. Source validation and induced confirmation are distinct. A full numerical matrix is intentionally not fabricated while these decisions are unresolved.

Then implement portable runner and full-epoch smoke: all materially different arms, finite forward/backward, current reference parity, paired order, save/load/continuation, actual GPU memory/RAM, Drive checkpoint roundtrip and prediction/plot regeneration. Only passed exact-source smoke permits the preapproved full queue. Required artifacts: manifest, best/latest checkpoints, optimizer and RNG, normalization, histories, per-window labels/logits/IDs, class/recording metrics, paired effects, timings/memory, audit, plots and failure status.

## Runtime and resource assessment

User-reported T4 step extrapolations (not verified here): old none4.04s/physical3.29s; SSM none24.17s/physical18.94s per epoch. Laptop SSM38.29/39.61s implies about1.58x/2.09x speedup in these microbenchmarks. GPU16GB vs4GB increases capacity, not a guaranteed4x speedup. Colab RAM12.7GB is LESS than laptop15.7GB; keep CPU corpus copies bounded.

For scale only: four calibrated arms x four fold/seed pairs x16 epochs would be64*(4.04+3.29+24.17+18.94)=3228s, about54min of extrapolated training steps, excluding graph/local candidate arms, transfer/evaluation/checkpoints/preprocessing. This is NOT the authorized T10 matrix or an end-to-end ETA. Final ETA=sum measured full-epoch job costs + smoke/startup/audit +25% contingency. First session: readiness plus engineering smoke; schedule subsequent resumable blocks around measured job time, without assuming a fixed free quota.

Official Colab FAQ: https://research.google.com/colaboratory/faq.html — free limits and GPU availability vary; notebooks can run at most12 hours depending on usage/availability, not a guaranteed allowance. Local VM files disappear; Drive can be slow or hit I/O quotas. Record assigned GPU each session. Do not promise remaining account GPU hours, which are not provided here.

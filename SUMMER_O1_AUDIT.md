# Summer pipeline versus the previous workstation implementation

Authority confirmed by user: summer colloquium PDF plus
`resources/objectives_methodology.tex`. The PDF's chapters 3-4 and Tables 5.3/5.4
were reread from the existing local text extraction in `tmp/pdfs/colloquium_report.txt`.
`context.text` contains partial historical source, not an untouched repository.

| Component | Summer evidence | Previous workstation O1/O2 |
|---|---|---|
| Spatial encoder | GIN aggregation multiplied by parallel tanh GIN attention | Additive self/message projections + BatchNorm + ReLU; attention branch removed |
| Reconstruction | GIN decoder, joint focal + full clean-input MSE | Decoder disabled for most variants; masked-target MSE only in one O1 ablation (this loss split follows the later methodology, but is not the summer recipe) |
| Temporal/head | Gated causal local source, flattened sensor embeddings, mean pooling, LayerNorm, classifier | Same temporal algebra and overall pooling/head pattern; different capacity |
| Affinity input | O1 methodology specifies encoded per-IMU H | Raw standardized channels with tanh before Q/K |
| Training | PDF Adam, focal + MSE, StepLR; differing tables for capacity/loss weights | AdamW + cross-entropy, fixed learning rate, hidden width 64 |
| Channels | Historical local code had acc/gyro/magnetometer | Six acc/gyro channels, consistent with later controlled O1 methodology |
| Masking | O1 methodology zeros unavailable inputs and outputs and abstains if all missing | Sender mask exists; complete receiver/output/abstention contract not implemented |
| No-graph meaning | Required control in O1 methodology | Removes cross-sensor graph messages, but flattened temporal projections still fuse sensors |

Therefore the earlier suite is a controlled experiment on a modified descendant
of the summer pipeline. It is not a verified change of only static edges to
dynamic edges. Marking the corrected summer-based O1 complete from that run would
be premature. Existing results remain valid observations for their saved code.

## Adjacency comparison

The requested formula is a nonnegative physical/global/dynamic mixture, sender
mask, no graph self-edges, final row normalization and a separate residual self
path. The current mixture resembles that contract but computes affinity before a
learned per-IMU representation, masks dynamic scores before softmax rather than
applying the stated mask after C, and uses a different message operator. The
older full report contains an extra identity-loop formula; the user selected the
methodology file as authority, so that older version must not override it.

## Why no graph may win in the existing runs

Measured: raw no-graph induced F1 exceeds residual; the dynamic branch's effective
mass is about 5.7%; uniformizing it changes very few predictions. Also the
no-graph temporal module already performs learned cross-sensor fusion.
Interpretation to test: a nearly fixed graph can mix displaced raw signals into
incorrect neighbors while the strong temporal fusion path needs no such mixture.
The different branch row sums dilute nominal dynamic gate weights. These are
diagnoses and hypotheses, not evidence that graph-based recognition cannot work.

The intended scientific goal remains a residual dynamic graph that improves
placement robustness. We must restore the baseline and test the specified
mechanism. Adding dynamic edges alone does not mathematically guarantee better
test F1; the methodology itself requires comparison with static, augmentation
and no-graph controls and allows a reproducible negative result.

## Resolved source and decisions (later in this session)

The user provided the local original repository:
`C:\Users\General\Desktop\prabhat\mamba_GT_WHAR\Mamba_WHAR_Phase1`.
Its clean working tree is at `792ef355e8ba94da638c6b14b103aea0dabb0e07`.
An archive and SHA-256 file manifest are in `baselines/summer_original`.
The original PyG environment generated a reference fixture; the restored dense
static GIN matched logits, reconstruction, input gradients and every original
parameter gradient within 2e-5 absolute/relative tolerance. This is actual source
parity, replacing the earlier partial-excerpt uncertainty.

The user selected **cross-entropy main, GAE ablation**. Original main.py supplies
hidden 16, dropout .5, Adam 1e-4, weight decay .0005, batch 64 and OneCycleLR.
The user asked for the worse-accuracy recipe, but the PDF tables do not identify
attributable accuracy. We use the recovered actual source configuration for all
controls; we do not fabricate a ranking or weaken only the static control.

The original training script evaluates test_loader each epoch and selects best
weights using that score. That behavior is not copied: the new study selects on
the distinct source validation subject. Raw six-axis data, corrected sensor
order, invalid-file exclusions and train-only per-sensor normalization are also
explicit corrections. Consequently, source architecture parity does not imply
numerical reproduction of the old reported accuracy.

`models/summer_dynamic.py` preserves the GIN aggregation-times-attention operator,
auxiliary GIN decoder and temporal/head algebra. It changes topology through a
learned per-IMU causal descriptor and the authoritative mixture equation. The
methodology's illustrative alternative message layer is not substituted for GIN;
the graph-only comparison retains summer GIN's self term exactly once. A separately
named residual_balanced ablation normalizes branch scales before mixing.

All 14 real-data smoke conditions passed. Six-epoch source-only pilots reached
best ideal validation F1 91.02% static-sum, 91.12% normalized-static and 91.11%
residual. These are readiness checks, not placement results or a superiority
claim. A simulated interruption test reproduced the uninterrupted final model
exactly, including restored optimizer, scheduler and RNG state.

## Historical uncertainties recorded before source recovery

1. Original source location or user authorization to reconstruct the partial
   source from `context.text` and PDF.
2. PDF Table 5.3 (REALDISP hidden 32, dropout .5, weight decay .001, MSE 2.0)
   versus Table 5.4 (hidden 16, dropout .4, weight decay .0005, MSE .10).
   Existing main.py has another variant (dropout .5, focal gamma .5, OneCycle).
   Do not silently treat these as the same reported experiment.
3. The authoritative methodology explicitly requests classification cross-entropy
   for the principal study with GAE as an ablation. This conflicts with reading
   "change only the graph" as preserving the summer joint focal/MSE loss. The
   user has been asked which should govern the new main comparison. Earlier
   cross-entropy training was consistent with that methodology instruction;
   removing GIN attention and using raw rather than encoded affinity remain
   separate architectural differences.

Next: resolve these through the pending questions, implement a labelled baseline
and an isolated weighted graph change, and use source validation before full training.

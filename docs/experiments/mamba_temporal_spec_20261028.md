# T05: genuine selective-SSM temporal implementation specification

State: READY_FOR_IMPLEMENTATION for T09; not implemented or trained by T05. Date 28 October 2026.

## Purpose and protected interfaces

Replace only the temporal module in the matched O1/O2 model. Current PurePyTorchMambaBlock is gated convolution and must retain that label in historical comparisons. Input/output [B,50,144], formed by flattening 9 sensor embeddings of width16. Keep node order, masking, features, spatial encoder, pooling, final normalization and classifier unchanged. Never migrate historical checkpoints by silently interpreting old weights as SSM weights.

Project decision: one pre-normalized residual block, model width144, expanded width288, state size16, causal depthwise kernel4, delta rank9, output dropout0.5. Save all settings in config. Wrapper is x + dropout(Mamba(LayerNorm(x))). Reset recurrent state for each independent window; never carry state across overlapping windows, recordings, batches or subjects. Existing backbone remains the matched reference, with parameter counts and compute reported rather than claiming identical temporal capacity.

## Selective recurrence contract

Use Mamba-1 as reference, not Mamba-2 or a newly substituted architecture. Project onto signal u and gate z; apply causal depthwise convolution and SiLU to u. Input-dependent projections produce B_t, C_t and positive delta_t. A=-exp(A_log). For channel d and state k, start s_0=0 and compute:

s[t,d,k] = exp(delta[t,d]*A[d,k])*s[t-1,d,k] + delta[t,d]*B[t,k]*u[t,d]

y[t,d] = sum_k C[t,k]*s[t,d,k] + D[d]*u[t,d]

Gate y by SiLU(z), then output-project. This specifies the real-valued selective_scan_ref recurrence, including its input discretization; do not substitute an exact-ZOH B term and claim reference equivalence. Keep recurrence/exponentials and A_log in FP32; cast outputs for surrounding mixed precision. Preserve reference initialization and no-weight-decay treatment where specified for A_log/D; disclose optimizer grouping and match non-SSM controls.

References checked 28 October:
- Gu and Dao, Mamba: https://arxiv.org/abs/2312.00752
- Author module and initialization: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
- Author reference recurrence: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py

## Files and platform decision

Implement models/selective_mamba_temporal.py, with a public [B,T,D] interface and explicit backend selection. Add backbone choice to a NEW experiment runner/config; do not change the frozen laptop pilot's sources. Save exact reference/version/source hashes. Native Windows must not be assumed to support the optimized extension.

First use a transparent PyTorch recurrence for correctness at T=50. On the workstation, inspect OS, CUDA/toolchain and benchmark it. Use the official optimized kernel only when installation and parity succeed in a separate environment; otherwise the reference implementation is a genuine selective SSM but must be reported as an unoptimized backend. No operating-system migration or dependency replacement is implied by this spec. Do not spend the 15-day budget on indefinite kernel installation.

## Required validation before T10

1. Tiny fixed-input FP32 comparison to reference scan: maximum absolute error <=1e-5 and relative tolerance1e-4; compare parameter/input gradients <=1e-4 absolute,1e-3 relative. Record errors, dimensions and version.
2. Temporal causality test on the temporal module: change suffix, verify earlier outputs unchanged in eval mode. This does not assert full-model training-mode causality through BatchNorm.
3. Reset-state determinism; sequence versus stepwise recurrence equality; finite outputs/gradients; nonzero gradients to delta/B/C/A and graph paths after warmup.
4. Checkpoint/resume and real-data forward/backward; smoke all selected graph/backbone combinations. Profile warmup and repeated batch-one latency plus actual training epoch time on named hardware.
5. Freeze batch/precision/epochs after smoke and before comparison. Apply the same optimization exposure to matched arms. Report every change from the old backbone.
6. Use an explicit, separately seeded generator for training-window order. Model construction must not change that order merely by allocating more parameters. Save/restore this generator's state in checkpoints. Keep model initialization and dropout seeds explicit. Today's families have internally matched controls, but their different module allocations consume different CPU RNG draws; do not treat their absolute scores as a perfectly paired cross-family experiment. The workstation factorial must remove that confound and retrain its matched controls.

T10 matrix: old temporal block versus selective SSM, each with no graph, physical graph and T08 candidate; add matched local-value control where needed. T08 supplies the selected model; it does not remove the need for physical/none controls. T09 implementation and T10 training remain workstation tasks.

# Research direction for placement-aware routing and rotation-stable multi-IMU HAR

Research checked: 18 October 2026. No implementation code or training was produced.

**Recommendation:** first establish whether the current dynamic adjacency has a measurable causal effect on messages and logits. If branch dilution is confirmed, normalize each masked graph branch separately before mixing it. Then evaluate the existing magnitude-plus-dot representation in the restored summer pipeline with fixed, residual, and no-graph controls. Treat a learned TRI-HAR-style representation as a second-stage option, not a prerequisite for recovering useful routing.

Evidence labels used throughout:

- **[M] Established mathematical property**, under the assumptions stated.
- **[P] Finding or implementation detail reported by a cited primary source.** It is not an independently reproduced result.
- **[O] Observation supplied in your experimental brief.** Logs and checkpoints were not independently audited.
- **[I] Inference** from the available evidence.
- **[H] Proposed hypothesis or experiment.** All candidate architectures and experiment recommendations below carry this label unless explicitly marked otherwise.

The requested Deep Research connector was not exposed among the available tools and was not found by plugin discovery. This review instead used web-accessible primary papers and official repositories. Only the pasted brief was attached; `colloquium_report.pdf`, `objectives_methodology.tex`, and local experiment artifacts were not available in this workspace. Their contents and implementation details therefore remain unverified. Some publisher pages were inaccessible; unresolved details are identified rather than filled in from secondary summaries.

## 1. The objectives and the present evidence

**O1 — placement-aware residual dynamic inter-sensor routing.** Determine whether physical, globally learned, and input-dependent interactions reduce degradation under recorded placement changes relative to normalized physical routing, augmentation-matched physical routing, no graph, and a documented DynamicWHAR adaptation. Preserve acceptable ideal-placement performance and measure inference cost. The research question concerns useful adaptation, not simply a learned adjacency.

**O2 — rotation-stable representations for dynamic routing.** Determine whether representations stable to independent IMU mounting rotations stabilize affinities without discarding excessive activity information, improve recorded-placement performance, and enable a dynamic graph to add value beyond invariant features with fixed fusion or no graph.

Retain the summer encoder, its GIN aggregation multiplied by a tanh GIN attention branch, its self contribution, pooling, classifier, and recovered temporal block for controlled comparisons. **[O] The temporal block is gated convolution/projection code, not a verified official selective-scan Mamba implementation.** Describe it that way in the paper. Keep graph reconstruction outside the primary cross-entropy experiment.

| Supplied result | What it supports | What it does not establish |
|---|---|---|
| Residual induced F1 11.31%, normalized physical 10.79% | [O] A small aggregate difference of about 0.52 percentage points | A consistent subject-level advantage or a practically useful adaptation mechanism |
| No graph induced F1 13.63%; residual is worse for each subject mean | [O] The current graph does not outperform this control | That every possible graph architecture must fail |
| Dynamic effective mass about 5.7%; negligible adjacency variation | [I] Input-conditioned routing appears weak in the measured runs | Whether the cause is normalization, optimization, a measurement bug, or task redundancy |
| Uniform/shuffled dynamic affinity changes no sampled predictions | [O] No observed argmax sensitivity in that sample | Zero change in logits, probabilities, loss, or margins |
| Fixed+rotation 60.67% versus residual+rotation 66.55% induced | [O] A 5.88-point augmented-setting difference worth investigating | A dynamic-routing mechanism; both runs also lose substantial ideal F1 |
| Earlier magnitude+dot/no-graph 87.89%, residual 87.53% induced | [O] Strong representation evidence in the earlier pipeline | A graph benefit, or a matched comparison with summer O1 |

**[I] Confidence assessment:** high confidence that the supplied results do not currently support a dynamic-graph superiority claim; moderate confidence that the measured dynamic path is functionally weak; low confidence in any particular causal explanation. The large augmentation benefit makes orientation sensitivity a leading explanation for part of the failure, but recorded displacement is not a pure rotation experiment.

## 2. The most relevant literature

### Verified TRI-HAR identification

**[P] Seungyeol Baek, Yoonbyung Chai, Yonghyeon Lee, Sungjoon Choi, and Sungho Suh (2026), “Rotation-Invariant Multi-IMU Activity Recognition under Independent Per-Location Orientation Shifts.”** The authors expand TRI-HAR as **Truly Rotation-Invariant HAR**. The arXiv record identifies ISWC 2026 and links the proceedings DOI; the official repository confirms acceptance. Sources: [arXiv:2608.15621](https://arxiv.org/abs/2608.15621), [DOI 10.1145/3830727.3834824](https://doi.org/10.1145/3830727.3834824), [official code](https://github.com/mbaek01/TRI-HAR). I found no competing relevant paper under TRI-HAR, Tri-HAR, or TriHAR.

### Evidence table

Each row separates what a source actually evaluates from what would need testing in your setup. No headline scores are compared across protocols.

| Paper and official implementation | Inputs, graph, and evaluation | Transfer assessment |
|---|---|---|
| **Baek et al., TRI-HAR, ISWC 2026**, identified above; [paper](https://arxiv.org/html/2608.15621v1), [code](https://github.com/mbaek01/TRI-HAR) | [P] Six-axis streams; PAMAP2/DSADS/Opportunity/REALDISP use 3/5/7/9 locations. Per-location equivariant encoding precedes invariant fusion. Dynamic neighbors connect time steps within an IMU, not different IMUs. Reports fixed independent-rotation stability and a REALDISP mutual-displacement fold. Other datasets use LOSO; REALDISP uses four ideal/self groups plus a mutual fold, with 2.4-second windows. | [I] Highly relevant representation source, but its REALDISP protocol is not your ideal-only, held-out-user protocol. Subject exclusion for the mutual fold needs auditing. Its reported comparison does not establish superiority over your handcrafted invariants. Host latency on 3/5 IMUs is not a nine-IMU RTX 2000 Ada measurement. |
| **Shenghuan Miao, Ling Chen, Rong Hu, Yingsong Luo (2022), “Towards a Dynamic Inter-Sensor Correlations Learning Framework for Multi-Sensor-Based Wearable Human Activity Recognition,” IMWUT 6(3), Article 130.** [DOI](https://doi.org/10.1145/3550331), [official repository](https://github.com/wdkhuans/DynamicWHAR) | [P] The repository implements input-dependent pair interactions from sensor descriptors. Its REALDISP configuration has nine nodes, nine channels per node, 40/100-sample windows and ten users; the loader holds out a user. Other configurations include five-node Opportunity, seven-node RealWorld and ten accelerometers for Skoda. A placement-specific evaluation was not verified from the accessible primary material. | [H] Adapt to nine six-axis nodes and 50 samples using your split and training budget. This is a documented adaptation, not reproduction of the published score. The original environment specifies old Python/PyTorch versions; port its operations rather than downgrading your CUDA environment. |
| **Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu (2019), “Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition,” CVPR, 12026–12035.** [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf), [DOI](https://doi.org/10.1109/CVPR.2019.01230), [official code](https://github.com/lshiwjx/2s-AGCN) | [P] Anatomical, global, and sample-dependent graph components; joint/bone streams on NTU-RGBD and Kinetics skeletons. NTU includes cross-subject and cross-view evaluation. Skeleton coordinates are not wearable six-axis signals. It does not establish independent mounting-rotation or recorded IMU-placement robustness. | [I] Strong architectural precedent for the three branches, not evidence that your convex normalized variant will work. Bone coordinates/directions are unavailable here. Transfer the graph principle only, with identical temporal processing and features across controls. |
| **Haoyu Xie, Haoxuan Li, Chunyuan Zheng, Haonan Yuan, Guorui Liao, Jun Liao, Li Liu (2025), “Decomposing and Fusing Intra- and Inter-Sensor Spatio-Temporal Signal for Multi-Sensor Wearable Human Activity Recognition,” AAAI 39(13), 14441–14449.** [DOI/paper](https://ojs.aaai.org/index.php/AAAI/article/view/33582), [full text](https://arxiv.org/html/2501.10917v1), [official code](https://github.com/Anakin2555/DecomposeWHAR) | [P] Input-dependent cross-sensor attention after temporal processing. Opportunity: five nine-axis IMUs; REALDISP: nine nine-axis IMUs and ten retained users; Skoda: ten accelerometers. Reports LOSO on Opportunity/REALDISP and within-user holdout on Skoda. No isolated mounting-rotation guarantee or matching induced-placement study is established. | [I] Supports descriptor-based directed interactions as a design option. Its magnetometers, temporal module and preprocessing prevent direct score transfer. The official repository warns that settings differ from DynamicWHAR and some baseline scores are quoted. Full-model replacement would confound your O1 test. |
| **Aras Yurtman and Billur Barshan (2017), “Activity Recognition Invariant to Sensor Orientation with Wearable Motion Sensors,” Sensors 17(8), 1838.** [DOI](https://doi.org/10.3390/s17081838), [primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5579846/), [author-hosted PDF](https://www.ee.bilkent.edu.tr/~billur/publ_list/sn17_2.pdf) | [P] Geometric and SVD-based orientation-removing transformations, evaluated across five datasets and four classifiers. Sensor configurations vary, including inertial/magnetic data. Both person-dependent and person-independent validation are discussed in this research line; exact dataset-wise splits should be extracted before reproducing it. No learned inter-sensor graph. Official code not verified. | [I] Relevant precedent for simple invariant preprocessing. It does not prove that invariance preserves all useful information. Geometric features transfer readily; SVD canonicalization needs degeneracy/sign handling. Do not imply that six-axis results reproduce magnetic-reference methods. |
| **Congyue Deng, Or Litany, Yueqi Duan, Adrien Poulenard, Andrea Tagliasacchi, Leonidas J. Guibas (2021), “Vector Neurons: A General Framework for SO(3)-Equivariant Networks,” ICCV, 12200–12209.** [Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_Vector_Neurons_A_General_Framework_for_SO3-Equivariant_Networks_ICCV_2021_paper.html), [DOI](https://doi.org/10.1109/ICCV48922.2021.01198), [official code](https://github.com/FlyingGiraffe/vnn) | [P] Equivariant vector-valued neural operations demonstrated on point-cloud tasks, including dynamic-graph backbones. Object rotation evaluation, not wearable placement or held-out-user HAR. | [I] A useful mathematical/implementation foundation for a small per-IMU learned representation. Adapting it to ordered time signals is a new experiment; point-cloud performance is not evidence of HAR superiority. Vector activations and neighborhood search add cost. |

**DynamicWHAR implementation detail that matters:** [P] the official model uses sigmoid-dependent edge modulation and adds the modulated edge feature to an unmodulated edge feature. Its aggregation uses both endpoint incidence matrices. It is therefore not equivalent to replacing your adjacency with a row-softmax Q/K matrix. Preserve these details in the external comparator and document any deviation. [Official model source](https://github.com/wdkhuans/DynamicWHAR/blob/main/modules.py).

**Protocol detail:** [P] the official training script evaluates its test loader during training; that fact alone does not prove test-based checkpoint selection. Your adaptation must use separate source validation and keep target scores out of training decisions. The original REALDISP loader contains 117 stored channels, while model configuration uses nine channels per node; inspect the channel-selection path instead of assuming every stored channel reaches the model. [Training configuration](https://github.com/wdkhuans/DynamicWHAR/blob/main/main.py), [loader](https://github.com/wdkhuans/DynamicWHAR/blob/main/data.py).

### Approaches screened but not prioritized

- **Hypergraphs:** Nafees Ahmad and Ho-fung Leung, “HyperHAR: Inter-sensing Device Bilateral Correlations and Hyper-correlations Learning Approach for Wearable Sensing Device Based Human Activity Recognition,” IMWUT 8(1), Article 1 (2024), [DOI 10.1145/3643511](https://doi.org/10.1145/3643511). The publisher text was inaccessible and official code was not verified. Exact modalities, splits and sample-conditioned hyperedge construction remain unresolved. **[I] Do not promote this to a preferred candidate from the title alone.** First demonstrate a useful pairwise interaction; higher-order relations do not directly diagnose your inactive routing.
- **Quality/transferability attention:** [P] Ling Chen et al., *SALIENCE: An Unsupervised User Adaptation Model for Multiple Wearable Sensors Based Human Activity Recognition*, aligns local/global source and target-user distributions and weights discriminative, transferable sensors. [Primary preprint](https://arxiv.org/abs/2108.10213), [official code](https://github.com/wdkhuans/SALIENCE). **[I] This motivates sensor-specific weighting, but its target adaptation setting is not your source-only protocol.** Do not silently use unlabeled test-user data.
- **Canonicalization:** [P] Yurtman–Barshan provide a primary precedent; [Gil-Martín et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346883/) evaluate gravity/forward-reference preprocessing with subject-wise validation. **[I] A one-second window, vigorous motion, uncertain forward direction and gravity-only yaw ambiguity make canonicalization a less straightforward first intervention than within-IMU invariants.** It is a meaningful later comparator if gravity-relative direction appears essential.
- **Large graph transformers, multiple graph scales and hypergraphs:** [H] defer. Nine nodes make dense pairwise routing inexpensive. Greater expressiveness does not resolve a disconnected, diluted or unnecessary dynamic path. Try one directed descriptor graph before extra heads or scales.

## 3. Separate the nuisance transformations

| Phenomenon | What changes | What the proposed methods handle |
|---|---|---|
| Fixed axis/mounting rotation | Coordinate representation of an otherwise equivalent local motion | Within-IMU invariants handle independent fixed rotations; augmentation only approximates coverage |
| Translation along a segment | Lever arm, tangential/centripetal acceleration, tissue coupling | No rotation-invariance guarantee; selective fusion may help empirically |
| Different anatomical location | The sensor observes different segment motion | Requires identity/location information or suitable training variation; current proposals do not guarantee recovery |
| Sensor identity/permutation error | Input channels no longer correspond to the assumed body locations | Rotation invariance does not fix it; physical/global graphs and ordered fusion can fail |
| Dropout | An observation is absent | Availability masking plus matched dropout training; separate from reliability |
| Loose/time-varying mounting | Coordinate changes and additional sensor motion relative to the body | Pointwise invariants remove only the coordinate component; temporal invariants generally break, and actual slip adds physical motion |
| New users | Anatomy, execution style, speed and other distributions | Requires held-out-user evidence; no mathematical guarantee from rotation invariance |

**[M] Translation is not rotation.** Under a rigid-segment approximation, changing position by a fixed segment vector Δr changes acceleration by terms α × Δr + ω × (ω × Δr). Angular velocity is shared by points on an ideal rigid segment, after coordinate alignment. Real tissues and loose mounts violate that approximation. **[H] A useful later diagnostic is acceleration-versus-gyroscope ablation by recording, rather than labeling all induced displacement as orientation shift.**

For independent fixed rotations R_i ∈ SO(3), applied consistently to acceleration and gyro at sensor i:

**[M]** ‖R_i a_t‖ = ‖a_t‖ and (R_i a_t)ᵀ(R_i g_t) = a_tᵀg_t.

**[M]** Temporal dots remain invariant when the rotation is constant over the compared times:

(R_i a_t)ᵀ(R_i a_(t−ℓ)) = a_tᵀa_(t−ℓ).

With time-varying rotations, the middle factor becomes R_i(t)ᵀR_i(t−ℓ), generally not identity. Norms and contemporaneous aᵀg remain invariant to a purely mathematical pointwise common rotation of the two streams. That does not make them invariant to the extra physical motion caused by strap slip.

**[M] Cross-sensor raw-vector dots generally fail:** (R_i a_i)ᵀ(R_j a_j) = a_iᵀR_iᵀR_j a_j. Cancellation requires the same rotation or other alignment assumptions. Conversely, dot products between already invariant scalar descriptors from different sensors are invariant to independent rotations.

**Information loss:** magnitudes remove direction and trajectory geometry. Temporal Gram features recover some within-IMU angular relationships but not absolute mounting orientation or cross-IMU relative frames. For ordinary vector streams, a complete Gram matrix determines geometry only up to an orthogonal transform; proper-rotation invariants such as scalar triple products can retain additional handedness information. Your sparse set of five features is not a complete invariant description. Acceleration is a polar vector and angular velocity an axial vector under reflections; restrict mounting tests and claims to proper SO(3) rotations rather than casually extending them to reflections.

**[M] Normalization order matters.** Build scalar invariants from physical vectors before scalar standardization. An anisotropic xyz standardizer generally does not commute with rotation. For learned equivariant inputs, use one rotation-compatible scalar scale per vector modality and avoid a fixed nonzero xyz mean subtraction unless equivariance is explicitly preserved. Test the complete raw-input-to-logit pipeline, not just rotations applied after preprocessing.

## 4. Diagnose the graph before changing it

### First experiment: an end-to-end routing intervention audit

**[H] Use existing checkpoints, evaluation mode and a fixed source-validation sample.** Include multiple recordings/classes and identical windows under independent rotations. Target recordings may be analyzed later as descriptive evidence, but must not decide the repair.

Record per-example, per-receiver values before averaging: descriptors, Q/K, affinity logits, C, masked branch row sums, final A, weighted messages, self-path outputs and classifier logits. Keep full precision for this audit. Repeat identical inputs first to establish a numerical/stochastic baseline.

Intervene at two distinct locations:

1. Replace C before mixture normalization with legal uniform affinities, a source-training mean affinity, and affinities from different examples. Preserve receiver/sender identity and compatible availability masks.
2. Replace the final A actually consumed by message passing with deliberately different legal graphs. Hold input features and all other activations fixed where possible.

Measure maximum/RMS adjacency change, message change, centered-logit change, probability KL, loss, class margins, F1/accuracy and prediction flips. Centering logits removes irrelevant common offsets. Repeat cross-example swaps; a batch-order shuffle that also shuffles the corresponding inputs is not a destructive intervention. A same-class swap is a stricter mechanism probe than a cross-class swap and can be reported separately.

Interpretation:

- C changes but A barely changes → branch scaling/masking/normalization is implicated.
- A changes but messages barely change → aggregation axis, identical values, caching or weighting is implicated.
- Messages change but logits barely change → downstream redundancy or cancellation is implicated.
- Logits change but argmax does not → the old diagnostic was too coarse.
- All stages change but performance does not improve → a functioning dynamic path may still be unnecessary.

### Falsifiable diagnostic tree

All causes below are **[H] unconfirmed**. A successful numerical repair is not evidence of better placement robustness.

| Hypothesis | Small test and measurements | Supporting/rejecting evidence | Minimal intervention and matched control |
|---|---|---|---|
| Branch scale mismatch | Log each branch's masked off-diagonal row sum and mixture coefficient | Effective mass diverges from nominal weights and tracks degree/scale; similar row sums weaken this explanation | Normalize branches separately; compare old/new normalization at identical weights, then matched retraining |
| Dynamic dilution after masking | Compare C before/after diagonal and sender removal; inspect C_ii | Large self probability removes dynamic mass; small diagonal mass rejects this specific cause | Mask logits before dynamic softmax; compare with post-softmax masking and renormalization |
| Temperature too high/low | Affinity-logit spread, entropy normalized by log(valid sender count), top-two gap, saturation | Almost uniform or near one-hot rows plus poor gradients support it; neither alone proves failure | Small source-validation temperature study; identical schedule/capacity and a fixed/uniform control |
| Query/key collapse | Across-example variance, singular values, cosine similarities and Q/K norms | Distinct inputs yield identical Q/K or row-constant score shifts | Change descriptor pooling or normalization only; give fixed/no-graph models the same descriptor capacity |
| Weak or broken gradients | Gradient norms, update/weight ratios, optimizer membership; finite-difference check | Detached tensors, absent parameters, or near-zero updates identify a concrete defect | Repair wiring or change affinity weight decay/LR; optimizer-matched static/no-graph controls |
| Mixed-precision effects | Identical forward/backward in FP32 versus current precision | FP32 restores differences/gradients beyond baseline error | Keep affinity path FP32; same precision change in relevant controls |
| Self path dominates | Compare self/neighbor activation norms and local logit sensitivity to their scales | Neighbor interventions have little influence; norms alone are insufficient | Small nonzero neighbor gain or reduced self gain; apply the same gain treatment to fixed routing |
| Temporal/classifier bypass | Disable neighbor messages while preserving self and downstream paths | Negligible loss change despite substantial graph changes supports redundancy | One-path-at-a-time intervention; retain exact temporal block in graph/no-graph comparisons |
| Ideal-only CE gives no incentive to adapt | Compare gradients and intervention sensitivity on clean versus rotated/corrupted source validation | Adaptation helps only under training-time variation; constant routing may be optimal on clean data | Matched augmentation across no/fixed/dynamic graphs; do not use target-placement validation |
| Mask/dimension/diagnostic error | Tiny hand-checkable graphs, one active sender, all absent, batch size one; unique sensor markers | Wrong aggregation axis, leaked absent values, NaNs, or a batch-averaged “constant” graph identify defects | Correct code first; rerun baselines affected by the defect |
| Oversmoothing | Measure node feature similarity before/after each spatial layer and margin changes | Distinct nodes become indistinguishable and reduced propagation helps | Shallower/gain-controlled propagation; same depth/gain for fixed and dynamic controls |
| Corruption spreads | Corrupt one source-validation sensor; track outgoing influence and other nodes' errors | Bad sender contaminates otherwise useful nodes | Reliability gating or sparse routing; compare identical gating/sparsity on fixed and no-graph fusion |
| Shuffled intervention ineffective | Verify donor differs, same sensor indices retained, A actually changes | Identical donors/graphs or permutation equivariance explain zero effect | Independent cross-example donor assignment and mean-graph replacement; repeat donors |

**[M] Effective graph mass is computable exactly.** Let b_ri be the valid off-diagonal row sum of branch r at receiver i, with coefficient π_r. Away from the epsilon-clamped case:

η_ri = π_r b_ri / Σ_s π_s b_si.

Thus γ = 0.3 does not imply 30% dynamic mass. If a physical row has two unit edges, a dense global row has eight unit edges, and C has a masked row sum below one, equal nominal weights produce unequal effective contributions. This is an illustrative algebraic example, not a claim about your implementation.

**[M] Adjacency mass is not message or decision contribution.** Feature magnitudes, nonlinear gates and cancellation also matter. Moreover, your parallel tanh GIN branch can modulate effective messages even with a constant A. Log adjacency and actual gated message contributions separately. Your observations concern weak input dependence in the routing matrix, not proof that the entire encoder is input-independent.

Do not add entropy regularization merely to make heatmaps vary. **[H] Any regularizer must improve source-validation performance or a prespecified robustness objective beyond the same model without it.** Constant matrices satisfy rotation-consistency losses perfectly, so consistency alone cannot establish useful routing.

## 5. Ranked O1 candidates

Shared conventions: X has shape B×9×50×6; encoded values H have shape B×9×T′×d; a temporal descriptor z has shape B×9×d_z. Use a single graph per window initially, A of shape B×9×9, broadcast over T′. This is sample-adaptive routing, not necessarily a graph changing at every sample inside a window. Keep the existing temporal descriptor unless the audit implicates it.

### O1-A — separately normalized physical/global/dynamic branches: first choice

**Mechanism:** prevent branch magnitude and node degree from silently deciding the mixture. This is a controlled repair of your current method, inspired by the adaptive-graph principle of 2s-AGCN, not a claim of reproducing that architecture.

Let M_ij = m_j·1(i≠j). Define RowNorm to return a zero row if no valid neighbor exists:

P = RowNorm(M ⊙ A_phys)

G = RowNorm(M ⊙ softplus(U))

Q = zW_Q, K = zW_K

C_ij = masked_softmax_j(Q_iK_jᵀ/(τ√d_a)), excluding unavailable senders and self before softmax.

A = π_p P + π_g G + π_d C, with π = softmax(θ).

For a row where P has no valid anatomical neighbor, set that branch to zero and renormalize the mixture weights over nonempty branches; log the resulting row-specific weights. If no neighbors exist at all, return zero adjacency and use the existing self path. Do not invent a uniform anatomical edge as a silent fallback.

**Shapes and cost:** U is 9×9; Q/K are B×9×d_a. Relative to your existing residual method, this can add zero learned parameters. Absolute routing parameters include approximately 2d_zd_a projection weights, 81 global entries and three gate logits, plus existing descriptors. Dense affinity work is O(B·9²·d_a); matrix storage is O(B·9²).

**Summer changes:** adjacency construction only. Keep GIN/tanh aggregation, temporal block and classifier identical. Start with a small nonzero dynamic initialization so it receives gradients; do not force a permanent dynamic share to manufacture a result.

**Why it might help:** [H] directly addresses measured low dynamic mass if dilution is the cause. **Cannot solve:** rotation-sensitive descriptors, translation, wrong sensor identity, or a task that does not need interaction.

**Falsifier:** after repair, a trained global-only mixture or source-mean graph matches its validation performance and interventions still show negligible useful sample dependence. Compare dynamic-only and physical+global controls. Runtime change should be small relative to the current residual model, but must be measured.

### O1-B — reliability-weighted directed routing: conditional second choice

**Mechanism:** distinguish an available sensor from a useful sender. Only pursue this if one-sensor corruption analysis shows harmful information propagation.

Let q_j = sigmoid(w_qᵀz_j+b_q), shape B×9×1. Construct:

A_ij = RowNorm_j(M_ij q_j A_base,ij).

Also pass a weighted neighbor-support quantity s_i = Σ_j A_base,ij m_j q_j into a bounded neighbor gain g_i, so that:

neighbor input_i = g_i Σ_j A_ij H_j.

Without this second mechanism, equal low q values cancel during row normalization and cannot reduce total reliance on neighbors. Keep the existing self contribution separate. A simple fixed function of s_i should precede a learned gate to minimize new confounds.

**Shapes/cost:** same A and H; a linear q head adds d_z+1 parameters. Extra compute O(B·9·d_z + B·81); no special dependency. A small MLP costs more and requires a capacity control.

**Training:** use CE with matched source-only dropout/noise corruption. Synthetic corruption labels may supervise a corruption detector, but that head must not be called a calibrated placement-reliability estimator. Rotation alone should not be labeled “bad quality” once features are invariant.

**Controls:** static sensor weights; fixed graph with the same q/gain; no-graph fusion with the same q/gain and parameter budget; dynamic graph without q. These distinguish quality selection from dynamic edge selection.

**Cannot solve:** unseen corruption without detectable signal, confident but misleading sensor features, or incorrect anatomical identity. **Falsifier:** gate scores fail held-out source-corruption calibration, or q-weighted no-graph fusion explains the entire benefit. That result can still support quality-aware fusion, but not dynamic-graph superiority.

### O1-C — sparse directed dynamic neighbors: optional third choice

**Mechanism:** remove weak senders only if dense propagation is empirically harmful. Apply top-k to valid dynamic scores, renormalize, then mix with P/G. Begin with one prespecified small k and a dense control rather than a large search.

**Shapes/cost:** B×9×9 adjacency; no additional parameters. With nine nodes, top-k selection may increase latency and dense physical/global branches keep the total graph dense. Do not claim speedup unless propagation actually uses sparse operations and measurements support it.

**Controls:** learned top-k versus fixed degree-matched neighbors, and dense versus sparse at equal neighbor-message magnitude. A secondary two-descriptor-scale version can compare whole-window and half-window descriptors, but must have equal-width single-scale controls.

**Cannot solve:** collapsed scores, unstable neighbor ties or wrong representations. **Falsifier:** edge choices change but validation performance and destructive-intervention results do not improve. Defer if O1-A already shows that dynamic routing is unnecessary.

## 6. Ranked O2 candidates

### O2-A — within-IMU scalar invariants, with a restrained temporal Gram extension

**First run the features already implemented:** magnitudes and the five-channel magnitude-plus-dot representation. Their historical benefit makes them the lowest-risk matched experiment, not a guaranteed result in the restored pipeline.

Input X is B×9×50×6. Output F is B×9×50×p, with p=2 or p=5:

f_it = [‖a_it‖, ‖g_it‖, a_itᵀg_it, a_itᵀa_i,t−1, g_itᵀg_i,t−1].

Standardize each scalar feature with source-training statistics and map p channels to the summer encoder's existing width. Specify first-sample lag handling; use a validity mask or deterministic zero padding, never a previous sample from another recording.

**[H] Only if five channels lose useful discrimination**, add a short set of temporal lags L. An example is:

f_it = [‖a_it‖, ‖g_it‖, a_itᵀg_it, {a_itᵀa_i,t−ℓ, g_itᵀg_i,t−ℓ, a_itᵀg_i,t−ℓ, g_itᵀa_i,t−ℓ}_ℓ∈L].

Then p=3+4|L|; L={1,2,4} gives 15 channels. These lag choices are a pilot proposal, not a literature-derived optimum. Retain magnitudes if cosine-normalized dots are tried; otherwise intensity information disappears. Fit modality scales on training data only to control unequal units.

**Summer changes:** feature construction and input projection only. Use invariants for both routing descriptors and message values if the full classifier is claimed invariant. Invariant affinity with raw-axis values does not ensure invariant predictions.

**Parameters/memory:** feature construction adds no learned parameters. A linear p→d projection adds pd+d parameters; relative to six-axis input, the change is (p−6)d. Feature memory scales with p; a 15-channel stream uses 2.5 times the raw six-channel feature storage before projection. Do not materialize the full 100×100 per-IMU Gram matrix unless needed.

**Controls:** p=2 versus p=5; no/fixed/residual graph; a width-matched raw projection; identical scalar encoder and temporal block. Compare any extra-lag representation under no graph as well as the graph candidate.

**Falsifiers:** invariance error exceeds floating-point tolerance, clean class separation deteriorates unacceptably, or dynamic routing adds nothing over fixed/no-graph fusion. This candidate addresses fixed mounting rotations, not physical translation or identity errors. It is supported by the mathematical argument, your earlier observations and geometric-invariant precedents, not by a claim that Yurtman–Barshan used these exact five channels.

### O2-B — a learned per-IMU equivariant-to-invariant encoder

**Mechanism:** learn within-IMU directional relationships before removing orientation, potentially retaining information omitted by sparse hand-designed dots. TRI-HAR and Vector Neurons are the relevant sources; superiority over O2-A is an open hypothesis.

Reshape X to B×9×50×2×3. A shared equivariant encoder E processes each IMU separately, yielding vector features V of shape B×9×50×c×3 in a simple VN variant. Form invariant scalars from learned vector norms or V_tF_tᵀ with a co-rotating learned frame F, then project to B×9×50×d. Feed these scalar sequences into the same summer graph/temporal pipeline.

**[M]** If both V and F transform with the same orthogonal representation, their product V Fᵀ is invariant. This proof does not cover arbitrary normalization, scalar activations applied independently to xyz, or raw-axis bypasses.

**Implementation scope:** this time-resolved adapter is a proposed adaptation, not the unmodified TRI-HAR classifier. It should preserve sequence order for the summer temporal block. A smaller VN temporal encoder is also a new method and must not be labeled a TRI-HAR reproduction.

**Cost:** exact parameter count depends on c, frame width and depth and must be measured. A bias-free vector-channel linear map c_in→c_out uses c_in c_out shared weights; vector activations hold three coordinates. A dense per-IMU temporal neighbor search scales as O(B·9·50²·c), much more than a nine-node graph. Begin with small width; do not assume the published configuration fits the desired session length.

**Controls:** learned invariant encoder with no graph and fixed graph; matched scalar temporal encoder with similar parameters; O2-A; original TRI-HAR as a separate external baseline if affordable. Changing the local encoder requires these controls even when the downstream summer block stays fixed.

**Cannot solve:** translation, arbitrary location permutation or unmodeled time-varying mounting. **Falsifier:** no improvement over O2-A at acceptable clean performance/cost, or gains reproduced by the scalar capacity control. Dependency compatibility with Windows/Python 3.11 remains to be checked in a separate environment.

### What TRI-HAR can supply

| Requested role | Assessment |
|---|---|
| Representation module | [H] Yes: test its location encoder as an alternative to handcrafted invariants |
| Temporal module | [H] Its within-location processing is relevant, but substituting it changes another component and needs matched temporal controls |
| Inter-sensor graph | [I] Not a drop-in O1 solution; retain a separate sensor graph after invariant features |
| Consistency objective | [H] Do not attribute a new graph-consistency loss to TRI-HAR; structural invariance is the relevant transferable principle |
| External baseline | [H] Yes, using your six-axis inputs and subject/placement split, explicitly labeled as an adapted evaluation |

The [official repository](https://github.com/mbaek01/TRI-HAR) provides location-wise and joint variants, fusion options and rotation-test utilities. Pin a commit and audit preprocessing before transferring them. The paper's reported synthetic invariance should not substitute for testing physical rotations before your own normalization.

**Optional consistency experiment:** for a non-exact representation, compare CE alone against CE plus λ_f‖z(X)−z(RX)‖² + λ_A‖A(X)−A(RX)‖², using independently rotated paired views with the same dropout mask. Give graph-free controls the same feature consistency and two-view training exposure. For exact invariants, these losses should already be near zero and generally add no useful supervision. Constant graphs are a trivial solution; require task benefit and destructive-intervention evidence.

## 7. Preferred integrated design and simpler fallback

**[H] Preferred first integrated candidate: O2-A with five invariant channels + O1-A with separately normalized branches.** This changes the representation and fixes a diagnosed normalization problem without simultaneously introducing a new temporal model, reconstruction loss, hypergraph or large attention stack.

Data flow:

1. Raw nine-IMU six-axis window: B×9×50×6.
2. Within-IMU invariants, then training-only scalar normalization: B×9×50×5.
3. Shared input projection to the existing summer feature width.
4. Existing per-IMU temporal descriptor: B×9×d_z.
5. Masked Q/K affinity and normalized physical/global/dynamic mixture: B×9×9.
6. Existing summer weighted GIN and tanh-attention branch, retaining its self path.
7. Existing gated convolution/projection temporal block, pooling, normalization and classifier: B×33 logits.

Symbolically, H = EncoderInput(F(X)); A = Router(Descriptor(F(X)), m); logits = Summer(A,H). Preserve the actual encoder order and dimensions during implementation rather than imposing a guessed graph/temporal order from this notation.

**[M] Conditional invariance proof:** if F(RX)=F(X), availability/identity are unchanged, and all later inference operations are deterministic functions of F, then A(RX)=A(X) and logits(RX)=logits(X). This ensures orientation stability, not discrimination or usefulness of the graph.

**[H] Routing mechanism claim:** across different activities/windows, A may vary; across rotated versions of the same window, it should not. Both properties must be tested separately. Increased between-window variance is useful only if it improves a prespecified outcome compared with static alternatives.

**Simpler fallback:** the same five-channel representation with normalized fixed anatomical routing. Keep the no-graph model as a mandatory comparator. If no graph is consistently better, report that finding; an O2 representation contribution can stand without claiming O1 success. This preserves a clear graph-based investigation without forcing a positive graph conclusion.

## 8. Staged experiment plan

### Stage A — numerical and data checks, before training

- Verify the corrected 76,268-window manifest, the three excluded zero recordings, and subject 15 mutual5–7 with mutual4 excluded. These are supplied constraints, not newly audited facts.
- Split by subject and recording before windowing; never let overlapping windows cross split boundaries. Confirm 50 samples, stride 25 and six channels per IMU.
- Confirm source-training-only normalization and separate source-validation subjects. Check fixed 33-class labels and the class support per recording.
- Run the routing audit, including actual logits/messages and the parallel tanh gate.
- Run independent rotations at physical-input level, whole-pipeline invariance tests, mask edge cases, FP32/AMP comparison and batch-one inference.
- Check augmentation applies a single proper rotation per sensor/window consistently to acceleration and gyro. Check augmentation occurs in physical coordinates, not anisotropically standardized coordinates. Log angle sampling and rotation determinants.

**[I] The large summer clean-performance loss under rotation augmentation deserves this audit.** It could reflect a real information/optimization trade-off, overly aggressive augmentation, or a preprocessing defect. The existing results do not distinguish these explanations.

### Stage B — short pilots selected only on source validation

**[H] Begin with one prespecified source-validation split and seed 13, using a short fixed pilot budget such as 5–8 epochs.** This duration is a screening choice, not sufficient evidence of converged superiority. Maintain identical optimizer steps and augmentation exposure across paired runs.

Pilot five-channel invariants with no graph, normalized fixed graph and existing residual graph. Add repaired residual normalization only if Stage A supports it. Compare clean validation and a preregistered independent-rotation stress set of the same validation subjects; keep other corruption tests separate. Confirm a promising result on a second source split or seed before full training. If the clean source-validation checkpoint rule must remain identical to historical O1, retain it and use synthetic tests only as reported diagnostics; any new selection rule requires matched reruns.

Do not use the induced-placement results to choose k, τ, graph gain, feature lags, early stopping or reliability supervision.

### Stage C — minimal matched matrix

Core matrix: retain the original residual definition to anchor historical comparisons.

| Representation/training | No graph | Normalized physical | Existing residual |
|---|---:|---:|---:|
| Raw axes, no rotation augmentation | Required | Required | Required |
| Raw axes + independent rotation augmentation | Required | Required | Required |
| Magnitudes, no rotation augmentation | Required | Required | Required |
| Magnitudes + dots, no rotation augmentation | Required | Required | Required |

This is 12 conditions × 3 subjects × 3 seeds = **108 jobs** if all were new. It is a comparison matrix, not a recommendation to rerun valid completed cells. If the six raw/augmented cells already match exactly, the six invariant cells require **54 new jobs**. Verify that a raw+augmentation/no-graph cell actually exists; it was not included in the supplied O1 excerpt.

Run the five-channel row first (27 full jobs) only after pilots pass; then magnitudes (27) to quantify what the dot features add. Use 31 epochs, seeds 13/29/47, subjects 2/5/15 and the established source-validation checkpoint policy. Retain every per-seed and per-recording output.

Necessary additions for a strong dynamic-routing claim, on the source-selected representation:

| Addition | Purpose | Cost if new |
|---|---|---:|
| Repaired residual O1-A | Test normalization repair against existing residual | 9 jobs |
| Dynamic-only | Determine whether the anatomical/global mixture suppresses useful dynamic routing | 9 jobs |
| Physical+global, no dynamic branch | Separate global learning from sample adaptation | 9 jobs |
| Documented DynamicWHAR adaptation | External comparator required for O1 | 9 jobs |

These are 36 additional jobs if all required and none reusable. Do not expand every additional condition across every representation. Source-mean/uniform/cross-example interventions require inference, not retraining. However, inference replacement alone is insufficient: the trained physical+global control tests whether the model can learn equally well without sample adaptation.

**Capacity controls:** if an architectural change adds a descriptor, gate, width or temporal block, add a corresponding fixed/no-graph model that actually uses the extra capacity. Unused dummy parameters are not a useful capacity control. Optimizer or schedule changes must be mirrored. Keep reconstruction weight zero in the main matrix.

**Deferred:** TRI-HAR encoder/full baseline, quality gating, sparse graphs, multiple scales and reconstruction. Promote only the candidate that survives its mechanism test. Original unnormalized fixed-graph runs remain historical context unless an implementation repair invalidates them.

### Stage D — recorded placement and mechanism analysis

After freezing choices, evaluate ideal, self and induced placement separately. Report:

- Fixed 33-class macro-F1 and accuracy, plus supported-class macro-F1 with the class set and zero-division convention stated. Define supported classes from ground truth, not a model's predictions.
- Per-subject, per-recording and per-seed results. Preserve your primary average: seeds within subject, then equal-weight subjects. State how recordings are combined within a subject; concatenated-window F1 and average recording F1 answer different questions.
- Paired differences relative to fixed/no-graph/global-only controls, and ideal-to-placement degradation. Always show absolute clean and shifted performance: a small degradation can be caused by poor clean performance.
- Synthetic independent fixed rotations separately from recorded displacement. Add shared rotations as a sanity test, and time-varying coordinate rotations as a separate stress test without claiming physical loose-mount simulation.
- Graph variance across distinct windows; invariance error on rotated copies; valid-neighbor-normalized entropy; branch masses; gradients/update ratios; self and neighbor message magnitudes.
- Performance and centered-logit effects of interventions, alongside heatmaps.
- Batch-one latency, parameter count, peak training/inference memory and preprocessing time.

**Uncertainty:** seeds are repeated training realizations, not new people. Overlapping windows are not independent observations. Show the three paired subject means and their range; seed dispersion characterizes optimization variability. Subject-cluster bootstrap intervals may be presented as exploratory but are unstable with three subjects. Recording/block resampling describes within-subject uncertainty, not population generalization. Do not report narrow window-bootstrap intervals as evidence about unseen users.

All three induced subjects have already been inspected. Subsequent decisions are therefore development-informed, even if training excludes their data. Freeze the new plan now and label this evaluation exploratory. A confirmatory generalization claim needs new untouched users/recorded placements or a genuinely independent dataset with a documented compatible protocol. More ideal-only held-out users can strengthen clean user-generalization evidence but cannot manufacture new induced-placement subjects.

## 9. What evidence permits each claim?

| Claim | Required evidence |
|---|---|
| “The features are rotation-stable” | Algebraic argument plus raw-input-to-feature/logit tests over independent proper rotations, including normalization and lag boundaries; separate fixed from time-varying rotations |
| “The model is more robust to recorded placement” | Paired recorded self/induced results at an explicitly acceptable clean-performance cost, with per-subject/recording support and no claim that all changes were rotations |
| “The dynamic graph contributes beyond fixed fusion” | Better matched training results than fixed/no-graph/global-only models, plus selective damage from destroying sample dependence; graph variation alone is insufficient |
| “The result generalizes to new users” | Subject-disjoint preprocessing/training/selection and evidence on genuinely held-out users; fresh confirmation needed after repeated inspection of the three induced subjects |

## 10. Compute plan for the 16 GB RTX 2000 Ada

**[H] Calibrate instead of promising runtimes.** Reuse complete-job times from the 126 existing jobs if available. Otherwise measure warm-up, data loading, several steady-state epochs, validation and final evaluation separately for no/fixed/residual models.

For condition c and fold s, estimate:

t_job(c,s) = t_setup(c,s) + 31·t_epoch(c,s) + n_validation·t_validation(c,s) + t_final_eval(c,s).

Sum estimates across jobs; include variability across folds and checkpoint overhead. A 5–6-hour session fits only the number of jobs supported by measured t_job, with explicit reserve for saving and interruptions. Do not confuse windows/second during training with whole-job throughput.

| Work item | Resource expectation; all timing uncalibrated |
|---|---|
| Checkpoint routing/invariance audit | No training; bounded evaluation batches and modest CPU/GPU use |
| Two/five-channel invariant features | No learned feature parameters; usually small arithmetic overhead, but preprocessing/cache timing must be measured |
| Separate graph normalization | Same parameter count and O(81) graph storage per example; likely minor overhead |
| Quality head/top-k | Small arithmetic additions; top-k is not automatically faster for nine nodes |
| Learned VN/TRI-style representation | Highest implementation and activation-memory risk; profile separately at small batch size |

**[M] Storage illustration:** one FP32 B×9×9 adjacency with B=256 occupies 82,944 bytes, about 0.079 MiB. Several such matrices remain small. One B×9×50×64 scalar activation occupies about 28.1 MiB; its three-coordinate vector counterpart about 84.4 MiB, before gradients, optimizer state, temporary tensors or other layers. This explains why the representation/temporal encoder is more likely than nine-node adjacency to determine memory use.

Preserve the established batch size when feasible. If memory forces gradient accumulation, note that it does not reproduce the batch statistics of a larger BatchNorm batch. Use matched batch treatment for affected controls. Keep affinity FP32 if the audit shows AMP sensitivity; otherwise benchmark both before changing precision policy.

For batch-one latency, warm up the GPU, synchronize timing, report median and tail latency over repeated windows, and separate device inference from end-to-end loading/preprocessing/transfers. Measure CPU only if it is a deployment target. At REALDISP's 50 Hz, stride 25 corresponds to a 0.5-second update interval, but the acceptable application latency still needs specifying. Dataset sampling details: [official REALDISP record](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset).

Suggested session order: audit/calibration → source pilots → only the first validated matched full-training block → remaining seeds/folds and analysis. Save model, optimizer, scheduler, RNG and preprocessing state so sessions can resume without changing the experiment. Windows/Python 3.11 compatibility of external repositories is not yet validated; isolate their dependencies and avoid replacing your functioning environment.

## 11. Pilot go/no-go criteria and expected outcomes

Agree before full runs on δ_clean (maximum acceptable ideal-F1 loss), δ_robust (minimum practically useful robustness improvement) and latency/memory ceilings. These are decision thresholds, not predicted accuracy targets. Their values require your application priorities.

**Go to matched full training only if:**

1. Data/mask/rotation checks pass and the actual consumed adjacency is verified.
2. Affinity parameters receive measurable updates; any residual constancy has an understood explanation.
3. The representation passes whole-pipeline invariance checks at a precision-appropriate tolerance.
4. A candidate has acceptable source clean performance and a repeatable prespecified validation advantage or clear mechanism evidence worth testing at convergence.
5. Runtime and memory have been measured and fit a resumable schedule.

**No-go for a claimed dynamic-routing improvement if:** only heatmaps improve; variation rises without task benefit; mean/global routing performs equally; gains come entirely from quality weighting or extra capacity; or clean loss exceeds the agreed tolerance. If short pilots are inconclusive because all models are still learning, extend a small matched pilot rather than treating early ranking as definitive.

**Expected qualitative outcomes [I/H]:** invariant features are more likely to fix the dominant orientation sensitivity than graph complexity alone. Separate normalization may reveal a previously suppressed dynamic branch, but it may also confirm that static/global fusion is sufficient. Useful dynamic graphs might yield small, activity-specific benefits rather than universal gains. A learned invariant encoder could recover some lost temporal geometry, at higher cost, or fail to beat five simple features. No numerical accuracy improvements are promised.

## 12. Remaining questions requiring your input or project artifacts

1. Where are the authoritative report, methodology file, current implementation and per-job logs/checkpoints? These are needed to verify the exact tensor paths and determine which completed cells are reusable.
2. Which source-validation subjects are used for each held-out subject, and what is the exact checkpoint/tie-breaking rule?
3. What ideal-performance loss and batch-one deployment latency are acceptable for the application?
4. Does the metadata distinguish orientation change from translation/location change within each induced recording, and are any new placement subjects or recordings available for untouched confirmation?
5. What are the measured epoch/job times and batch sizes from the completed summer runs? These determine the number of jobs per 5–6-hour session.

**Direct recommendation:** run the frozen-checkpoint routing intervention audit first, tracing C → masked branch masses → actual A → gated messages → centered logits. If it confirms dilution, first change branch normalization and pre-softmax masking, keeping the summer network fixed. Then test the existing five-channel invariants under no/fixed/residual routing. Commit to another long training block only after correctness, usable gradients, whole-pipeline invariance, source-validation performance and measured runtime support it. A positive O1 claim additionally requires trained static/global controls and evidence that destroying sample dependence removes a reproducible benefit.

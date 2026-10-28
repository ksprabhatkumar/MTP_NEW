# Alternative approaches for O1 and O2

Prepared: 21 October 2026.

This document consolidates the alternatives discussed in this conversation and the earlier research-direction report. It is not an exhaustive survey of every HAR model. Each entry follows fields 0–7: objective, paper, formula, architecture, rationale, results, links, and incorporation instructions.

**Evidence conventions:** “Paper equation” means verified in the primary text, sometimes with clearer notation. “Implementation equivalent” means derived from official source code. “Proposed adaptation” means our suggested experiment, not a published formula or result. Missing information is marked explicitly. Accuracy and macro-F1 are different metrics; scores across different splits, modalities and label sets are not directly comparable.

## Shared model and evaluation contract

The incorporation instructions below are proposals based on your supplied model description, not an audit of the current implementation.

- Input: batch × 9 IMUs × 50 samples × 6 channels; channels are acceleration xyz and gyroscope xyz. Stride: 25.
- Preserve the summer GIN aggregation multiplied by its parallel tanh GIN attention branch, separate self contribution, temporal block, pooling and classifier unless an entry explicitly proposes a replacement.
- The recovered temporal block uses gated convolution/projection operations. It is **not a verified official selective-scan Mamba implementation**.
- Main training loss: cross-entropy. Keep graph reconstruction as a separate ablation.
- Train on source-subject ideal recordings; fit preprocessing on training data only; use separate source validation for all tuning and checkpoint selection.
- Keep subjects 2, 5 and 15 and seeds 13, 29 and 47 paired across comparisons. Subject 15 mutual4 remains excluded. Preserve the documented recording exclusions.
- Match training budget, augmentation and downstream capacity. Report accuracy and fixed 33-class macro-F1 for ideal, self and induced placement, plus subject/recording results, latency, parameters and memory.
- Synthetic rotations test orientation stability. Recorded displacement can also change position and physical motion. Rotation invariance alone does not solve translation, wrong anatomical location, sensor permutation, missing sensors or loose mounting.

Notation: i is the receiving sensor, j the sending sensor; h_i is a learned sensor descriptor; H stacks descriptors; m_j is sender availability; omega denotes angular velocity; epsilon is a numerical safeguard. Adjacency orientation must be checked against the existing message-passing convention.

# Objective 1 — Placement-aware residual dynamic inter-sensor routing

Determine whether physical, globally learned and input-dependent interactions improve recorded-placement robustness beyond normalized physical routing, augmentation-matched physical routing, no graph and a documented DynamicWHAR adaptation, with acceptable ideal performance and inference cost.

## O1-A. Physical + global + sample-dependent graph: 2s-AGCN

### 0. Objective addressed

O1: architectural precedent for the three graph branches. Wearable placement robustness is a proposed transfer, not an established result of this paper.

### 1. Paper and year

Lei Shi, Yifan Zhang, Jian Cheng and Hanqing Lu. **Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition. CVPR, 2019.**

### 2. Mathematical formula

Paper Eqs. (3)–(5):

$$
f_{out}=\sum_{k=1}^{K_v}W_k f_{in}(A_k+B_k+C_k)
$$

$$
C_{k,ij}=\frac{\exp(\theta_k(v_i)^\top\phi_k(v_j))}
{\sum_{r=1}^{N}\exp(\theta_k(v_i)^\top\phi_k(v_r))}.
$$

A_k is the anatomical adjacency, B_k a globally learned matrix, and C_k an input-dependent affinity. The paper's B_k is unconstrained; its additive rule is not your nonnegative convex mixture.

### 3. Original architecture

Joint/bone streams → adaptive graph and temporal convolution blocks → global pooling → stream classifiers → score fusion.

### 4. Why it could address the problem

Learned connections can supplement anatomy and vary by input. It does not itself prevent a weak dynamic branch, nor establish robustness to independent IMU rotations.

### 5. Datasets and reported accuracy

| Dataset/protocol | Accuracy |
|---|---:|
| NTU-RGBD, cross-subject | 88.5% |
| NTU-RGBD, cross-view | 95.1% |
| Kinetics-Skeleton, top-1 | 36.1% |
| Kinetics-Skeleton, top-5 | 58.7% |

Source: paper Tables 3–4. These are skeleton results, not REALDISP results.

### 6. Paper and code links

- [Paper PDF](https://arxiv.org/pdf/1805.07694)
- [Official code](https://github.com/lshiwjx/2s-AGCN)

### 7. Incorporation into our model — proposed adaptation

1. Transfer only the graph-construction principle; retain nine IMU nodes and the summer encoder. Do not invent a bone stream from six-axis IMU measurements.
2. Use existing temporal descriptors H of shape batch × 9 × d to produce C of shape batch × 9 × 9. Keep Q and K separate so directed interactions are possible.
3. Evaluate a clearly labeled normalized adaptation using O1-F below. An exact additive, potentially signed B implementation would be a separate ablation, since it changes the graph semantics.
4. Feed the resulting adjacency into the existing weighted aggregation and parallel attention paths; preserve the separate self path.
5. Compare physical-only, physical+global, dynamic-only, full residual and no-graph controls. Replace C at evaluation with its source-training mean and with a cross-window shuffled C; measure logit and F1 changes.

## O1-B. Cross-sensor descriptor attention: DecomposeWHAR

### 0. Objective addressed

O1: input-dependent directed interaction between temporal sensor descriptors.

### 1. Paper and year

Haoyu Xie et al. **Decomposing and Fusing Intra- and Inter-Sensor Spatio-Temporal Signal for Multi-Sensor Wearable Human Activity Recognition. AAAI, 2025.**

### 2. Mathematical formula

Paper Eqs. (8)–(9):

$$
A_{ij}=\frac{\exp(Q(X_i)^\top K(X_j))}{\sum_{r=1}^{N}\exp(Q(X_i)^\top K(X_r))},
\qquad O_i=W\left(\sum_j A_{ij}V(X_j)\right).
$$

The text adds O to the original features through a residual connection. The printed affinity does not include a square-root dimension scaling factor.

### 3. Original architecture

Modality embedding → local temporal extraction → channel/variable fusion → Mamba temporal aggregation → cross-sensor attention → residual fusion → classifier.

### 4. Why it could address the problem

Temporally informed descriptors may produce more useful interactions than noisy instantaneous inputs. It does not guarantee nonconstant attention or placement robustness.

### 5. Datasets and reported accuracy

Table 1, 800 ms window setting:

| Dataset | Accuracy |
|---|---:|
| Opportunity | 78.28 ± 0.68% |
| REALDISP | 96.64 ± 0.75% |
| Skoda | 97.61 ± 0.52% |

Opportunity/REALDISP use LOSO; Skoda uses a within-user holdout. Nine-axis inputs in the first two differ from our six-axis setting.

### 6. Paper and code links

- [Primary full text](https://arxiv.org/html/2501.10917v1)
- [AAAI publication](https://ojs.aaai.org/index.php/AAAI/article/view/33582)
- [Official code](https://github.com/Anakin2555/DecomposeWHAR)

### 7. Incorporation into our model — proposed adaptation

1. First borrow only the descriptor-to-affinity operation. Compute H from the existing temporal descriptor extractor and use its attention matrix as the dynamic branch.
2. Apply availability and self-edge masks before softmax. Combine branches using O1-F; do not replace the temporal block in the same experiment.
3. Keep attention-weight generation separate from a full attention-message replacement. Replacing GIN messages with W(sum A V) is a second experiment requiring a matched GIN control.
4. If implementing the complete published network, run it as an external six-axis comparator with the same source split and label set. Document changed modalities, windows and training budget.
5. Compare dynamic attention against a learned static matrix using the same descriptors and classifier. Improvement over physical routing alone does not prove sample dependence is useful.

## O1-C. Pairwise interaction modulation: DynamicWHAR

### 0. Objective addressed

O1: the specifically requested external dynamic inter-sensor comparator.

### 1. Paper and year

Shenghuan Miao, Ling Chen, Rong Hu and Yingsong Luo. **Towards a Dynamic Inter-Sensor Correlations Learning Framework for Multi-Sensor-Based Wearable Human Activity Recognition. IMWUT, 2022.**

### 2. Mathematical formula

**Official-implementation equivalent; printed paper equations were not accessible:**

$$
e_{ij}=\phi([h_i\Vert h_j]),\qquad r_{ij}=\sigma(w^\top e_{ij}+b),
\qquad \widetilde e_{ij}=(1+r_{ij})e_{ij}.
$$

With E stacking edge features and receiver/sender incidence matrices R_rec and R_send, aggregation is:

$$
H_{agg}=\frac{1}{N}(R_{rec}^{\top}+R_{send}^{\top})\widetilde E.
$$

The sigmoid modulation retains an unmodulated edge contribution. It is not a row-softmax adjacency.

### 3. Original architecture

Per-sensor CNN → pairwise edge features → sigmoid edge modulation → endpoint aggregation → concatenate self/interaction features → FC classifier.

### 4. Why it could address the problem

It explicitly constructs sample-dependent pair interactions and provides a distinct comparator to Q/K routing. Its modulation alone cannot eliminate an edge because 1+r lies between 1 and 2.

### 5. Datasets and reported accuracy

Official configurations include Opportunity, RealWorld, REALDISP and Skoda. The following numbers are reported for DynamicWHAR in the later DecomposeWHAR comparison table; they were **not independently checked against the original DynamicWHAR table**:

| Dataset | Accuracy |
|---|---:|
| Opportunity | 74.27 ± 0.46% |
| REALDISP | 92.58 ± 0.50% |
| Skoda | 93.80 ± 0.54% |
| RealWorld | Not verified |

### 6. Paper and code links

- [Original paper DOI](https://doi.org/10.1145/3550331)
- [Official code](https://github.com/wdkhuans/DynamicWHAR)
- [Verified model implementation](https://github.com/wdkhuans/DynamicWHAR/blob/main/modules.py)
- [Source of quoted comparison scores](https://arxiv.org/html/2501.10917v1)

### 7. Incorporation into our model — proposed adaptation

1. Implement the external comparator using nine nodes, six channels per node, 50 samples and 33 classes. Port operations to the current environment rather than downgrading CUDA/PyTorch to the old repository environment.
2. Preserve pair-feature construction, sigmoid modulation and both-endpoint aggregation in the comparator. Calling a Q/K replacement “DynamicWHAR” would be inaccurate.
3. Use the project data loader and source validation rules. Do not select epochs using the held-out user.
4. For a modular summer-model experiment, replace only the inter-sensor interaction block and project its output to the existing feature width. This is a DynamicWHAR-inspired hybrid, separate from the comparator.
5. Match additional projection capacity in the control. Disable or shuffle the learned modulation to test whether it contributes beyond edge features alone.

## O1-D. Higher-order sensor relationships: HyperHAR

### 0. Objective addressed

O1: candidate group-level interactions beyond pairwise edges; currently an incompletely verified candidate.

### 1. Paper and year

Nafees Ahmad and Ho-fung Leung. **HyperHAR: Inter-sensing Device Bilateral Correlations and Hyper-correlations Learning Approach for Wearable Sensing Device Based Human Activity Recognition. IMWUT, 2024.**

### 2. Mathematical formula

**Exact paper formula not verified.** Publisher full text was inaccessible. A generic hypergraph equation would not establish what HyperHAR actually implements, so none is attributed to it here.

### 3. Architecture

Conceptual overview only: per-device features → bilateral and higher-order inter-device correlations → activity classifier. Precise layers and hyperedge construction remain unverified.

### 4. Why it could address the problem

Our hypothesis is that some activities depend on coordinated groups of sensors rather than independent pairs. This does not directly diagnose inactive routing; increased expressiveness could remain unused.

### 5. Datasets and reported accuracy

Original full-model dataset/result table not verified. HyperHAR-Lite is a separate variant; its scores must not be substituted for full HyperHAR.

### 6. Paper and code links

- [Paper DOI](https://doi.org/10.1145/3643511)
- [Publisher PDF](https://dl.acm.org/doi/pdf/10.1145/3643511)
- [Author publication page](https://nafees-060.github.io/Nafees/)
- **Official code:** not verified online; this does not establish that no code exists.

### 7. Incorporation into our model — proposed direction, not a reproduction recipe

1. Obtain the original method section before claiming a HyperHAR implementation. Resolve whether hyperedges are fixed, globally learned or input-dependent.
2. Keep the existing nine per-IMU descriptors as nodes. A verified multi-order module could replace only the interaction block, with an output projection back to the summer feature width.
3. Preserve anatomical identities and availability masking in all group operations.
4. Require pairwise-only, higher-order-only and combined controls at matched capacity. Defer implementation until a useful pairwise interaction has been demonstrated; this entry is not ready for faithful reproduction from current evidence.

## O1-E. Transferability-guided sensor weighting: SALIENCE

### 0. Objective addressed

O1-adjacent: adaptive sensor fusion. It is not an inter-sensor graph and the full method requires target-user adaptation.

### 1. Paper and year

Ling Chen et al. **SALIENCE: An Unsupervised User Adaptation Model for Multiple Wearable Sensors Based Human Activity Recognition. 2021 preprint.**

### 2. Mathematical formula

Paper Eqs. (5)–(6):

$$
v=\sum_{i=1}^{n}\alpha_i x_i,\qquad
\boldsymbol\alpha=\operatorname{softmax}\left(\frac{Q(\widetilde Y)K(X)^\top}{\sqrt h}\right).
$$

X contains sensor features; flattened local domain-discriminator outputs form Y-tilde; h is the embedding width.

### 3. Original architecture

Per-sensor CNNs → alignment-guided attention → weighted fusion → two BiLSTMs → FC/softmax; local/global domain discriminators train alongside.

### 4. Why it could address the problem

It weights sensors using discrimination and domain alignment. This motivates selective fusion, but domain transferability is not the same as physical placement reliability.

### 5. Datasets and reported accuracy

| Dataset | Accuracy | Macro-F1 |
|---|---:|---:|
| PAMAP2 | 89.4% | 82.8% |
| Opportunity | 74.4% | 74.1% |

Table 2 uses user adaptation: half the new user's data is unlabeled adaptation data, half is testing. Opportunity uses five high-level activities.

### 6. Paper and code links

- [Paper PDF](https://arxiv.org/pdf/2108.10213)
- [Official code](https://github.com/wdkhuans/SALIENCE)

### 7. Incorporation into our model — proposed adaptation

1. Keep full SALIENCE outside the source-only primary comparison. If evaluated, declare a separate target-adaptation protocol and give all adaptation baselines equivalent access.
2. For source-only experiments, borrow only the idea of a sensor gate q_j=sigmoid(g(h_j)). Train it from source classification and controlled source corruptions. This is our quality-gating hypothesis, not SALIENCE's published alignment mechanism.
3. Apply q_j to sender messages; use a separate receiver-level message-strength gate when every neighbor is unreliable. Row normalization alone can cancel a common reduction in all sender weights.
4. Compare availability-only, learned static weights, input-dependent gates and gate+graph. Check whether gains remain when sample-dependent graph routing is removed.

## O1-F. Separately normalized branches, reliability gates and optional top-k

### 0. Objective addressed

O1: minimal repairs for suspected branch dilution, unreliable senders and overly diffuse affinities.

### 1. Paper and year

**Project proposals, not additional published approaches.** Architectural inspiration: 2s-AGCN (2019), DecomposeWHAR (2025), and SALIENCE. Their papers do not report this combined design.

### 2. Mathematical formula — proposed adaptation

Let M_ij=m_j 1[i != j]. For a nonnegative branch Z define:

$$
\mathcal N_M(Z)_{ij}=\frac{M_{ij}Z_{ij}}{\sum_r M_{ir}Z_{ir}}
$$

when the denominator is positive; otherwise the row is zero. Define:

$$
P=\mathcal N_M(A_{phys}),\quad G=\mathcal N_M(\operatorname{softplus}(B)),
\quad C_{ij}=\operatorname{maskedSoftmax}_j\left(\frac{(\hat h_iW_Q)^\top(\hat h_jW_K)}{\tau\sqrt{d_a}}\right).
$$

For nonempty branch rows, mix using:

$$
(\pi_p,\pi_g,\pi_c)=\operatorname{softmax}(u),\qquad
A=\pi_pP+\pi_gG+\pi_cC.
$$

Renormalize mixture weights over nonempty branches per receiver. When no sender exists, use zero neighbor contribution and the existing self path. Do not run softmax on an all-masked row.

Optional quality extension:

$$
q_j=\sigma(g(h_j)),\qquad A^q=\mathcal N_M(A\operatorname{diag}(q)),
\qquad \mu_i=\rho_i\sum_j A^q_{ij}\psi(h_j),\quad \rho_i\in[0,1].
$$

Optional top-k: retain the k highest eligible affinity logits per row before softmax. It changes sparsity, not the requirement to demonstrate useful input dependence.

### 3. Proposed architecture

Sensor descriptors → separately normalized physical/global/dynamic branches → mixture → optional quality/sparsity control → summer graph encoder → temporal block → classifier.

### 4. Why it could address the problem

Separate normalization prevents raw branch row sums from silently deciding branch contribution. Quality gating could limit corrupted messages. These remain hypotheses; the supplied 5.7% dynamic mass does not identify the cause by itself.

### 5. Datasets and accuracy

No published results for this exact proposal. Your current residual model reports 11.31% induced macro-F1; that is neither accuracy nor a result for this repair.

### 6. Paper and code links

Use O1-A/B/E links for inspiration. No verified public code implements this exact project proposal.

### 7. Incorporation into our model

1. Audit current masking, branch row sums, gradients, message norms and centered logits before modifying the graph.
2. Change only branch normalization first; preserve optimizer, initialization policy and all downstream blocks.
3. Test optional reliability and top-k separately after normalization. Do not introduce all three together.
4. Measure effective branch contribution after every mask/gate, not merely learned mixture coefficients.
5. Require a reproducible advantage over physical+global and no-graph controls, and loss of that advantage when sample dependence is destroyed. Increased adjacency variation alone is insufficient.

# Objective 2 — Rotation-stable representations for useful dynamic routing

Determine whether per-IMU representations stable under independent mounting rotations preserve activity information, stabilize affinities, improve recorded-placement robustness and enable a dynamic graph to help beyond invariant features with fixed fusion or no graph.

## O2-A. Learned per-location invariants: TRI-HAR

### 0. Objective addressed

O2 directly; O1 only through a proposed downstream graph over invariant sensor features.

### 1. Paper and year

Seungyeol Baek et al. **Rotation-Invariant Multi-IMU Activity Recognition under Independent Per-Location Orientation Shifts. ISWC, 2026.** TRI-HAR means Truly Rotation-Invariant HAR.

### 2. Mathematical formula

Paper Eqs. (5)–(7):

$$
\bar V=\frac1T\sum_t V_t,\qquad F_t=\operatorname{VN\!-!MLP}([V_t,\bar V]),\qquad z_t=V_tF_t^\top.
$$

$$
(V_tD(R))(F_tD(R))^\top=V_tF_t^\top,\qquad
h_l=\operatorname{Pool}(\{z_t^{(l)}\}_{t=1}^T).
$$

D(R) is an orthogonal lifted rotation action. Features and learned frame co-rotate, so their inner products are invariant.

### 3. Original architecture

Per-IMU vectors → shared FER-VN-DGCNN → invariant projection → temporal pooling → fixed-order location concatenation → MLP.

### 4. Why it could address the problem

Invariant projection precedes cross-location fusion, allowing independent IMU rotations. The internal dynamic graph connects time steps within each IMU, not different sensors.

### 5. Datasets and reported metric

Table 2 reports **macro-F1, not accuracy**; original/fixed-rotation scores are identical:

| Dataset | Macro-F1 |
|---|---:|
| PAMAP2 | 83.64 ± 5.56% |
| DSADS | 89.07 ± 4.49% |
| Opportunity | 38.06 ± 3.77% |
| REALDISP | 91.94 ± 4.99% |

REALDISP mutual-displacement fold: 85.29% macro-F1. Its five-fold protocol is not our ideal-only held-out-user experiment. Accuracy is not provided in these tables.

### 6. Paper and code links

- [Full paper and equations](https://arxiv.org/html/2608.15621v1)
- [Proceedings DOI](https://doi.org/10.1145/3830727.3834824)
- [Official code](https://github.com/mbaek01/TRI-HAR)

### 7. Incorporation into our model — proposed adaptation

1. Reshape each window to batch × 9 × 50 × 2 × 3. Apply the same encoder separately to each location; the two vector streams are acceleration and gyro.
2. For minimal integration, retain invariant time-step features z before pooling. Project their scalar channels to the summer input width and feed them to both graph descriptors and message features.
3. Preserve the 50-step sequence and existing temporal block. Using only pooled h and repeating it over time would discard temporal information; do not do that.
4. Compare this sequence-preserving hybrid with the complete pooled TRI-HAR external baseline. The hybrid changes the original architecture and must be named accordingly.
5. Apply independent fixed rotations jointly to acceleration and gyro at each IMU. Verify raw-input-to-feature, adjacency and logit invariance, including normalization. Use scalar modality scales rather than anisotropic xyz standardization when asserting exact equivariance.
6. Run invariant features with no graph, fixed graph and residual graph. Measure compute on nine IMUs and the actual GPU; published host latency does not predict our runtime.

## O2-B. Gravity/forward-frame canonicalization

### 0. Objective addressed

O2: align input axes to an estimated consistent reference frame before graph construction.

### 1. Paper and year

Manuel Gil-Martín et al. **Reducing the Impact of Sensor Orientation Variability in Human Activity Recognition Using a Consistent Reference System. Sensors, 2023.**

### 2. Mathematical formula

Paper Eq. (1), compact equivalent notation:

$$
v_{new}=Tv_{orig},\qquad T_{pq}=u_{p,new}^{\top}u_{q,orig},\quad p,q\in\{x,y,z\}.
$$

The new axes use forward direction, gravity and their cross product. The method estimates gravity and forward direction using five-second sliding means.

### 3. Original architecture

Acceleration → reference-frame estimation/transformation → spectral processing → CNN classification, with a movement/posture distinction in the enhanced variant.

### 4. Why it could address the problem

A consistent frame can reduce axis mismatch while retaining direction. Weak motion or unreliable gravity makes frame estimation ambiguous.

### 5. Datasets and accuracy

Table 3, rotated data with activity-type-specific processing:

| Dataset/location | Accuracy |
|---|---:|
| WISDM_lab | 91.36 ± 0.24% |
| WISDM_wild | 81.54 ± 0.20% |
| MotionSense | 92.09 ± 0.31% |
| USC-HAD | 59.35 ± 0.58% |
| PAMAP2 chest / wrist / ankle | 68.12 / 74.71 / 67.93% |
| HARTH back / thigh | 87.52 / 87.56% |

Subject-wise evaluation. The abstract's WISDM number differs from Table 3; this document uses the explicitly labeled table setting.

### 6. Paper and code links

- [Full text, Eq. (1) and Table 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346883/)
- [DOI](https://doi.org/10.3390/s23135845)
- **Official code:** not verified online.

### 7. Incorporation into our model — proposed adaptation

1. Estimate the frame separately for each IMU, before scalar/channel normalization. Apply the same proper rotation to acceleration and gyro; extending this accelerometer preprocessing to gyro is our adaptation.
2. The original five-second context exceeds our one-second window. Either use recording-local causal history with an explicit warm-up policy, or declare a separately validated short-context adaptation. Do not silently increase context for only this method.
3. Define a fallback for near-zero forward motion. Any posture/movement switch must be predicted from input, never supplied from true test labels.
4. Keep the summer network and graph controls matched. Log frame discontinuities and performance on stationary activities as well as moving activities.

## O2-C. Within-IMU magnitude and temporal Gram features

### 0. Objective addressed

O2: inexpensive exact invariants for fixed independent mounting rotations; the first recommended representation baseline.

### 1. Paper and year

**Existing project approach, not a uniquely identified paper implementation.** Yurtman–Barshan (2017), below, is related geometric-invariance literature; the exact five-feature combination here comes from your experiment brief.

### 2. Mathematical formula — project definition

Magnitude-only:

$$
z_{i,t}=[\|a_{i,t}\|_2,\|\omega_{i,t}\|_2].
$$

Magnitude plus dot products:

$$
z_{i,t}=[\|a_{i,t}\|_2,\|\omega_{i,t}\|_2,
a_{i,t}^{\top}\omega_{i,t},
a_{i,t}^{\top}a_{i,t-1},
\omega_{i,t}^{\top}\omega_{i,t-1}].
$$

For one fixed proper rotation R_i per IMU:

$$
(R_i u)^\top(R_i v)=u^\top v,\qquad \|R_i u\|_2=\|u\|_2.
$$

Optional richer temporal Gram features use G_i(p,q)=v_{i,p}^T v_{i,q}. Cross-sensor raw-vector dots generally fail independent-rotation invariance because R_i^T R_j need not be identity.

### 3. Proposed architecture

Per-IMU raw vectors → invariant scalar features → scalar standardization → summer encoder and optional routing → temporal block → classifier.

### 4. Why it could address the problem

It removes axis orientation from graph inputs and messages. Magnitudes discard direction; temporal dots retain some within-sensor geometry. Temporal invariance assumes a fixed rotation across compared samples, not arbitrary time-varying mounting.

### 5. Dataset and existing results

User-supplied REALDISP results from the earlier modified spatial pipeline:

| Representation / routing | Ideal macro-F1 | Induced macro-F1 |
|---|---:|---:|
| Magnitudes / no graph | 93.78% | 87.27% |
| Magnitudes / residual | 94.07% | 86.87% |
| Magnitude+dot / no graph | 94.49% | 87.89% |
| Magnitude+dot / residual | 94.51% | 87.53% |

These are not published accuracy values and are not matched to the restored summer O1 runs. They support representation benefit, not graph superiority.

### 6. Paper and code links

- [Related geometric-invariance paper](https://www.ee.bilkent.edu.tr/~billur/publ_list/sn17_2.pdf)
- **Exact project implementation:** no public repository verified; existing local features are described in your brief.

### 7. Incorporation into our model

1. Compute features from physical vectors before normalization. Fit scalar feature statistics on source training subjects only.
2. Produce batch × 9 × 50 × 2 or × 5. Handle the first temporal dot with a fixed documented convention, such as repeating the first sample as its predecessor; never borrow samples across recording boundaries.
3. Change the input projection width only; retain the summer spatial and temporal processing. Use invariant features for both affinity and message pathways if claiming end-to-end invariance.
4. Cross both representations with no graph, normalized physical and residual routing. Include raw axes and matched rotation-augmentation controls.
5. Only add more lags after the five-feature baseline. Match additional projection capacity so a gain is not attributed solely to extra parameters.

## O2-D. Geometric and SVD orientation-invariant preprocessing

### 0. Objective addressed

O2: nonlearned orientation-removing preprocessing; a further alternative from the earlier review.

### 1. Paper and year

Aras Yurtman and Billur Barshan. **Activity Recognition Invariant to Sensor Orientation with Wearable Motion Sensors. Sensors, 2017.**

### 2. Mathematical formula

Verified SVD branch, paper Eqs. (8), (11)–(12):

$$
V=U\Sigma W^\top,\qquad T_{SVD}(V)=\Sigma W^\top=U^\top V.
$$

V stores three-axis samples as columns. The paper also proposes a heuristic geometric transformation; its full nine-component formula is not reproduced here. Our five-feature definition in O2-C is not that exact transformation.

### 3. Original architecture

Windowing → heuristic or SVD transform → statistical features → normalization/PCA → BDM, k-NN, SVM or ANN classifier.

### 4. Why it could address the problem

Principal axes co-rotate with the data, removing a fixed coordinate orientation up to sign/degeneracy ambiguities. Those ambiguities require explicit treatment.

### 5. Datasets and accuracy

The paper evaluates five datasets labeled A–E: an eight-person/19-activity multi-IMU dataset, four-person/five-activity accelerometers, 30-person/six-activity smartphone data, 14-person/12-activity inertial data and 15-person/seven-activity chest acceleration. Figure 7 reports classifier-specific P-fold and leave-one-person-out accuracies. Exact chart values were not reliably extracted; no single headline accuracy is substituted.

### 6. Paper and code links

- [Author-hosted full paper](https://www.ee.bilkent.edu.tr/~billur/publ_list/sn17_2.pdf)
- [DOI](https://doi.org/10.3390/s17081838)
- **Official code:** not verified online.

### 7. Incorporation into our model — proposed adaptation

1. Build a 3 × 100 matrix per IMU/window by concatenating its 50 acceleration and 50 gyro vectors after rotation-compatible modality scaling fitted on training data.
2. Estimate one common SVD frame per IMU; transform both modalities with that frame and restore the six-channel time sequence.
3. Implement deterministic sign handling and test nearly repeated singular values, stationary windows and adjacent-window frame flips. Plain library SVD is not automatically a stable canonicalization.
4. Retain the summer model and compare with magnitude+dot preprocessing under identical graph choices. Do not import whole-dataset normalization from an older protocol.

## O2-E. Small learned equivariant encoder: Vector Neurons

### 0. Objective addressed

O2: mathematical building blocks for a smaller learned alternative to the full TRI-HAR encoder.

### 1. Paper and year

Congyue Deng et al. **Vector Neurons: A General Framework for SO(3)-Equivariant Networks. ICCV, 2021.**

### 2. Mathematical formula

The paper's vector-linear operation and equivariance relation, using V with vector rows:

$$
f_{lin}(V)=WV,\qquad f_{lin}(VR)=WVR=f_{lin}(V)R.
$$

An invariant projection uses a co-rotating learned frame T:

$$
(VR)(TR)^\top=VT^\top.
$$

Scalar coordinate-wise ReLU does not preserve this property; vector-compatible nonlinearities and normalization are required.

### 3. Original architecture

Point cloud → VN-PointNet or VN-DGCNN → invariant features → classification; related heads address segmentation/reconstruction.

### 4. Why it could address the problem

It supplies operations that preserve directional structure before invariant projection. Adapting them to ordered IMU sequences is a new experiment, not a published wearable result.

### 5. Dataset and reported accuracy

ModelNet40, VN-DGCNN: 89.5% for z-axis-rotation training/arbitrary-rotation testing; 90.2% for arbitrary-rotation training/testing (Table 1). ShapeNet is also used for segmentation/reconstruction; those metrics are not HAR accuracy.

### 6. Paper and code links

- [Paper PDF](https://arxiv.org/pdf/2104.12229)
- [Official code](https://github.com/FlyingGiraffe/vnn)

### 7. Incorporation into our model — proposed adaptation

1. Use a small shared per-IMU vector encoder, preserving acceleration/gyro as 3D vectors. Avoid mixing locations before invariant projection.
2. Preserve temporal indexing with shared temporal vector operations, or explicitly document time-neighbor construction. Treating a sequence as an unordered point cloud can discard activity order.
3. Project each time step to invariant scalar channels, then feed the unchanged summer pipeline. Compare against a scalar encoder with matched width/parameter budget.
4. Verify every preprocessing, activation, pooling and normalization operation under random independent rotations. Start with a small encoder and measure memory before scaling.

## O2-F. Rotation augmentation and rotation-consistency training

### 0. Objective addressed

O2, with optional O1 affinity consistency; these are training alternatives, not exact structural invariance.

### 1. Paper and year

Independent-location rotation augmentation is explicitly evaluated in TRI-HAR (2026). The following combined classification/affinity consistency loss is a **project proposal**, not a TRI-HAR paper equation.

### 2. Mathematical formula — proposed adaptation

For each window, sample R_i independently from SO(3), fixed across time, and rotate acceleration and gyro together:

$$
a'_{i,t}=R_i a_{i,t},\qquad \omega'_{i,t}=R_i\omega_{i,t}.
$$

An optional objective is:

$$
\mathcal L=\tfrac12[CE(y,p(X))+CE(y,p(RX))]
+\lambda_p JS(p(X),p(RX))
+\lambda_A\frac{\|A(X)-A(RX)\|_F^2}{N(N-1)}.
$$

Here JS is Jensen–Shannon divergence; set lambda_A=0 in no-graph controls. Start with classification plus augmentation alone.

### 3. Proposed architecture

Original/independently rotated source windows → shared summer model → classification losses → optional prediction/affinity consistency.

### 4. Why it could address the problem

It discourages orientation-sensitive predictions and affinities. Consistency can also be satisfied by a constant graph, so it cannot establish useful routing or guarantee invariance to every rotation.

### 5. Datasets and existing results

Your summer REALDISP results: fixed+rotation 60.67% induced macro-F1; residual+rotation 66.55%, with ideal macro-F1 64.86% and 68.56%. These are supplied experimental observations, not accuracy or results for the proposed consistency loss. No result exists here for that additional loss.

### 6. Paper and code links

- [TRI-HAR augmentation protocol](https://arxiv.org/html/2608.15621v1)
- [Official TRI-HAR code](https://github.com/mbaek01/TRI-HAR)
- **Proposed consistency objective:** no public project implementation verified.

### 7. Incorporation into our model

1. Generate rotations on physical vectors before normalization; use valid proper rotations, not independent arbitrary 3 × 3 matrices.
2. Keep augmentation probability/distribution matched across physical, residual and no-graph runs.
3. Add prediction consistency first; test affinity consistency separately. Select all weights from source validation.
4. For already exact invariant features, first check whether the consistency loss is numerically redundant.
5. Evaluate clean-performance trade-offs and use shuffled/constant-graph interventions. A low consistency loss plus a constant adjacency is not an O1 success.

# Recommended integration order

1. **Diagnose O1 first:** trace dynamic affinity → masked mixture → actual messages → logits. Confirm whether branch dilution, numerical issues or bypassing actually occur.
2. **Minimal O1 change:** O1-F branch normalization only, if the audit supports it. Compare against physical+global and no graph.
3. **First O2 experiment:** run O2-C magnitude and magnitude+dot features in the restored summer pipeline, crossed with no/fixed/residual graph.
4. **External comparator:** adapt DynamicWHAR faithfully to the same six-axis input, split and training budget.
5. **Learned O2 option:** add TRI-HAR-derived sequence-preserving invariant features only if simple invariants leave a meaningful gap. Treat the pooled original as a separate baseline.
6. **Later experiments:** descriptor attention, reliability gates, canonicalization, small VN encoders and consistency losses, one change at a time. Defer HyperHAR until its exact method is verified.

Preferred proposed combined flow:

**Nine six-axis IMUs → within-IMU invariant sequence → temporal descriptors → separately normalized physical/global/dynamic graph → summer gated GIN interaction → existing temporal block → pooling/classifier.**

A positive representation result requires preserved activity information and verified rotation stability. A positive graph result additionally requires gains over invariant fixed/no-graph controls and evidence that removing sample dependence removes a reproducible benefit. Previously inspected placement subjects cannot serve as completely untouched confirmation data.

## Verification gaps retained deliberately

- HyperHAR: exact equations, layer sequence, full-model result table and official code remain unverified.
- DynamicWHAR: formulas are implementation-derived; listed scores are from a later comparison, not independently verified original tables.
- Yurtman–Barshan: exact classifier-specific chart values and the complete heuristic feature formula were not reliably extracted.
- Gil-Martín and Yurtman–Barshan: official public code was not verified.
- Proposed hybrids/repairs have no published accuracy. No improvement target is invented.


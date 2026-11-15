# O1 proposal: reliability-aware graph repair

Date: 29 October 2026. State: RESEARCH_PROPOSAL, NOT READY_FOR_EXECUTION. No implementation/training launched. This is a proposed synthesis, not an established named method or a verified novelty claim.

## Target and evidence

Interpret the requested4-5 percent advantage as4-5 percentage points of induced-displacement fixed33 macro-F1; report accuracy separately. The historical matched no-graph induced F1 is87.12%, implying91.12-92.12% against that historical reference. The actual claim must use a freshly matched, equally trained baseline: if augmentation improves no graph, the target rises with it. No current evidence supports a promise of this gain.

T08 found original correction source-self92.7117%, no graph92.2249%, local control93.0090%. Gate changes increased neighbor sensitivity but did not beat local capacity. Thus the next experiment should teach cross-sensor repair explicitly and distinguish it from extra per-sensor processing. The current no-graph temporal projections already mix sensors; graph repair is a structural learning bias, not access to information unavailable to no graph.

## Design hypothesis

During ideal-only training, a graph has little incentive to learn when a present sensor is unreliable. Train on clean/corrupted pairs of the SAME source window, so there is a clean target for repair. Use receiver-specific transformations because a wrist and ankle need not have similar features. Allow the graph to refuse every sender, rather than forcing a row-normalized average of bad messages.

These are hypotheses motivated by our experiments. Useful corruption sensitivity is not proof of real-displacement generalization.

## Data and corruption contract

- Retain existing audited data, splits, exclusion list, mag_dot features, train-only normalization and strong no-graph architecture. Do not repeat the data audit or modify target recordings.
- Make corruption copies from TRAINING windows only. Perturb a subset of sensors while retaining at least one trustworthy sender. Keep a substantial clean fraction. Source validation may include a separately seeded, frozen synthetic stress set; never train on it.
- Distinguish missing sensors (availability=false) from present-but-displaced sensors (availability=true). Never supply the synthetic corruption mask to the deployed classifier/router as an oracle.
- Rotation alone is inadequate: mag_dot is designed to cancel fixed mounting rotation. Initial corruption design must address positional effects and attachment instability; generic noise/dropout is a separate robustness control, not a substitute for real displacement.
- Candidate physical transform for a fixed shift r on the SAME approximately rigid body segment: a_shift = a + alpha cross r + omega cross (omega cross r), then apply a mounting rotation consistently to a_shift and omega. This is a mechanics-derived approximation, not a reproduction of soft-tissue/slippage effects. Check existing unit/sample-rate metadata; require SI units, consistent coordinates, filtered angular acceleration and justified r bounds before implementing. Do not move sensors between unrelated body segments using this formula. No displacement ranges are frozen in this proposal.
- Model attachment disturbance separately using bounded, temporally smooth perturbations. Calibrate plausible ranges from documented acquisition assumptions and source information, not held-out induced scores. Synthetic corruptions may fail to represent real misplacement; evaluate those domains separately.

## Spatial operator

Input after invariant features: [B,T=50,N=9,F=5]. Keep hidden width16 initially and the current temporal backbone for the first attribution experiment. Add short per-sensor temporal descriptors for reliability and edge selection; do not change to Mamba simultaneously.

Let h_i(t) be a local embedding, e_ij a learned embedding of the nominal receiver/sender sensor identities, q_j in(0,1) a predicted sender reliability, and g_i in(0,1) a predicted need for correction. Reliability can use a temporal summary rather than a single sample. It is learned from available sensor data at inference.

Use a receiver-conditioned VALUE that cannot create a pure local correction when sender information is zero:

v_ij = phi(h_i, h_j, e_ij) - phi(h_i, 0, e_ij).

Use one shared small MLP with sensor-pair embeddings, not81 independent networks. v_ij is signed; it may reinforce or oppose feature components. The subtraction guarantees v_ij=0 for a zero sender input. This prevents the unconditional local-value shortcut present in the old concatenation correction, while permitting different transformations for different sensor relationships.

For valid j != i, define l_ij = s(h_i,h_j,e_ij) + log(q_j+epsilon) + log(pi_ij+epsilon), where pi is the physical/global prior and s is a learned compatibility score. Append a learned null-neighbor logit l_i0 and value v_i0=0. Normalize logits over valid senders AND the null neighbor:

p_ij = softmax_j(l_ij, l_i0); r_i = sum_(j!=i) p_ij v_ij;
h'_i = h_i + g_i W_o r_i.

No self edges; missing senders have probability zero. A present receiver with no valid neighbors receives zero correction. All sensors missing still requires explicit abstention/error handling. Retain a separate unchanged local path. Start W_o at zero; do not also initialize g near zero without a measured gradient/warmup check.

Why the null matters: if every q_j is small, renormalizing only across senders can still force a total neighbor weight of one. A null competitor allows total real-neighbor mass below one. Do not renormalize real-neighbor probabilities again after removing the null entry.

## Repair targets and training loss

An EMA teacher sees the CLEAN source view in eval mode; the student sees the corrupted view. Teacher weights are a moving average of student weights, with explicit consistent treatment of normalization buffers. No teacher gradients and no teacher requirement at inference. Do not train or update the teacher on validation/test windows.

Use a clean supervised warmup to prevent a collapsed teacher. Then optimize:

L = CE(y,p_clean) + CE(y,p_corrupt)
    + lambda * KL(stopgrad(p_teacher_clean) || p_corrupt)
    + mu * L_masked_latent_repair + nu * L_quality.

Latent repair compares repaired representations at affected sensors with stopped-gradient clean teacher targets, using normalized latent cosine error rather than demanding exact raw-axis reconstruction from invariant features. Sensor-specific motion is not perfectly recoverable from other sensors; keep this auxiliary loss modest and ablate it. It must not dominate activity discrimination.

Quality supervision is a synthetic TRAINING proxy. Prefer a bounded target derived from clean-versus-corrupted teacher local-feature discrepancy, calibrated on training data, rather than declaring every rotation unreliable. Do not claim it measures true real-world sensor reliability. Log calibration and uncertainty under source stress, and test generalization to recorded displacement.

Loss weights, teacher decay/warmup, perturbation schedule and selection rule must be frozen in the subsequent executable specification. They are not disguised as already tuned constants here.

## Fair controls and staged test

Stage1 keeps the current temporal backbone. Minimum controls:

1. No graph under the existing recipe, for continuity.
2. Strong no graph with the SAME corruptions, teacher/consistency objectives, auxiliary targets and training exposure.
3. Local-capacity control with the same repair modules but receiver-only inputs. Edge scores/reliability must not secretly read other sensors in this control; preserve the pre-existing downstream mixing in all arms.
4. Proposed graph repair with the matched augmented recipe.

Compare graph primarily with2 and3. A gain over1 alone could be augmentation/distillation rather than graph benefit. Report parameters, active modules, wall time and teacher cost. Use independent data-order/corruption RNG streams shared by matched runs and save their states for exact resume.

Start with a real-data smoke covering full loss, masks, gradients, null behavior, teacher updates, checkpoint resume and plot reconstruction. Then a small source-only paired pilot; calibrate runtime on the actual GPU. Do not infer cost from the old simple graph pilot. The paired views/teacher/pairwise messages add material training cost.

If promising, ablate relation conditioning, null/reliability, latent repair and teacher separately; then test genuine Mamba as a separate factorial change. Preserve no graph and local controls there as well. Freeze the final model before recorded-displacement evaluation. The existing inspected test subjects remain exploratory evidence; no independence claim is recovered by relabeling their splits.

## Required evidence and failure conditions

- Classification and clean-to-displaced degradation, per-subject/recording/class accuracy and fixed33 F1, paired seeds and uncertainty.
- Teacher/repair/quality loss components and gradient norms; sender quality, null mass, correction/local norm ratio, edge type and temporal variation.
- Test value zeroing, distant donors, static/uniform routing and oracle-versus-predicted quality only as explicitly labeled diagnostics. Oracle quality is not an eligible deployable result.
- Save logits/indices, corruption seeds/parameters, masked sensors, teacher/student states, optimizer/RNG, metadata and plot CSVs. Preserve old results.
- Reject a graph-specific claim if only augmentation produces the gain, if local control matches it, if the model ignores neighbors, or if benefits vanish on real displacement. A4-5pp target is ambitious; no guaranteed range or fabricated expected score is assigned.

## Prior art and novelty boundary

- GraphMAE motivates masked FEATURE recovery with a cosine-style objective; it does not establish a wearable misplacement gain: https://arxiv.org/abs/2205.10803
- Mean Teacher motivates weight-averaged teachers and consistency targets; the proposed clean/corrupted fully supervised adaptation is our design: https://arxiv.org/abs/1703.01780
- The REALDISP displacement study distinguishes real sensor-position changes from ideal placement and examines robust fusion: https://pmc.ncbi.nlm.nih.gov/articles/PMC4118358/

This proposal combines established ideas for our specific graph-repair question. Its novelty and4-5pp advantage are unverified. Next deliverable is an executable specification after source-side corruption/unit/architecture choices are resolved; this document does not authorize a blind full training queue.

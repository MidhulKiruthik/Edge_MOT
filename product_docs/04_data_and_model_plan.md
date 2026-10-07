# RACE-MOT Data, Labels, and Model Plan

**Status:** Draft v0.2 — dataset access, terms, and exact splits must be confirmed before training.  
**Purpose:** Prevent leakage and ensure the model is trained on the same event the product policy is meant to manage.

## 1. Data roles

| Data | Role | Use boundary |
|---|---|---|
| MOT17 training sequences | Main development benchmark | Group sequences by source video; keep detector variants of the same source in one fold. Never randomly split frames or tracklets. |
| MOT20 training sequences | Frozen crowd/generalization evaluation | Do not tune risk thresholds or feature selection on MOT20 if claiming transfer from MOT17. |
| Optional authorized local clip | Product workflow demonstration / domain-shift observation | Separate from benchmark scores; consent/permissions, annotation protocol, retention, and sharing terms required first. Not a dataset contribution by itself. |

Record the exact download URL/version, license/terms, annotation format, detector source, sequence list, checksums, and transformations. Do not package restricted benchmark videos or annotations in the product release.

## 2. Label definition and counterfactual generation

Use the mathematically specified avoidable identity-failure target in the main research report. Before code, freeze and document: anchor matching rule and IoU threshold; visibility/occlusion handling; future horizon $K$; persistence length $M$; detector/tracker versions; tracker initialization; and the two branch actions.

For each eligible state, generate paired offline rollouts from the same track state:

- Skip detector on the next frame, then follow the agreed future rollout protocol.
- Run detector on the next frame, with all other settings matched.

Use annotations only to create offline training/evaluation labels. Runtime features must be available by the current frame. Publish pseudocode and manually inspect edge cases before training. This is an offline counterfactual simulation; do not describe it as causal proof about every physical deployment.

## 3. Feature protocol

### Track-history baseline

Normalized box state/velocity by elapsed time, Kalman residual/covariance, detector confidence history, missed-frame count, time since last detector call in both seconds and source frames, consecutive detector skips, track age, association margin, and track-local overlap/crowding. Keep detector skips distinct from source frames lost before inference. Preserve timestamps; report risk calibration and policy metrics by detector-gap bin, with bins frozen before held-out evaluation.

### Scene-discovery safety guard

The per-track predictor cannot assign risk to a person who has not yet produced a track. Test a deterministic, low-resolution scene activity guard that measures frame difference outside padded active-track regions and may upgrade a scheduled `SKIP` to full-frame `DETECT`. Share the same thumbnail/frame-difference calculation with the optional scene-context ablation instead of computing it twice. The guard may not downgrade a planned detection. Tune thresholds on policy-validation sequences only. This is product robustness, not a novelty claim. [ALBIREO](https://arxiv.org/abs/2609.29648) already includes an empty-scene screen and rescue path, so compare the guard against both risk-only and ALBIREO-like scheduling; test incremental new-track discovery while other tracks are active. Report discovery delay/recall, false-trigger rate, full-pipeline energy, and latency. Remove the guard if it adds no reliable value or its cost erases savings.

For offline evaluation, define a **new-entry event** as a ground-truth person identity becoming visible (including re-entry after a predeclared absence interval) while another predicted track is active and no active predicted track matches that identity under the frozen evaluator's IoU rule. Exclude sequence initialization, where full detection is mandatory. Define **track-acquisition delay** as the number of source frames from event onset to the first output track matched to that identity; report undiscovered/censored events separately rather than assigning an arbitrary delay. Define **guard precision** as the fraction of guard overrides on frames containing a new-entry event; report false overrides per 1,000 input frames as well. Freeze matching, re-entry interval, and event rules before held-out evaluation. These annotations are for offline scoring only.

### Scene-context and fusion ablations

Use only current/prior-frame values available online: thumbnail luminance/contrast, Laplacian-variance sharpness/blur proxy, frame-difference activity, and active-track count. First select/confirm the primary TCN against GRU and temporal MLP using track history only. Then keep that TCN fixed and compare three variants with identical labels, folds, calibration protocol, and policy: (1) track-history-only; (2) simple concatenation of track history and context; and (3) a small shared context encoder whose frame embedding is gated into each per-track representation. Compute the shared context once per frame and reuse it across tracks. The gated variant is optional and should be attempted only after the basic pipeline works. Do not include optical flow in the core. These features are proxies; measure parameter count, extraction/fusion time, memory, energy, calibration, and policy value on the Jetson. Keep the more complex variant only if held-out benefit exceeds its cost. Do not impose an unsupported fixed parameter or sub-millisecond target.

### Training boundaries

- Fit scaling/normalization only on the training fold.
- Use training-fold weighting/mining only; calibrate on natural-prevalence calibration sequences.
- Select thresholds on policy-validation sequences distinct from final evaluation.
- Use grouped/cross-fitted sequence splits if the small number of MOT17 source sequences prevents a stable fixed three-way split.
- Keep all detector versions of one source sequence within the same fold.

## 4. Model comparison

One tiny causal TCN implements the primary novelty claim; compare compact GRU and temporal MLP on identical labels/folds. Compare against static logistic regression/MLP and confidence-only and Kalman-uncertainty-only rules. Compare policy results with fixed skipping, an ALBIREO-like per-object uncertainty scheduler, EMO-like and RT-MOT-like strategies, and HSFSO or a clearly documented approximation. SDOF-Tracker is related prior art for skip plus optical-flow propagation; do not add flow to the proposed policy and attribute its effect to risk prediction. Evaluate scene-discovery guard separately from model architecture.

The three input/fusion variants are paired ablations, not independent novelty claims. Keep the temporal model family, labels, sequence folds, calibrator, and policy choices stable while testing them. A 0.70 score is not guaranteed to correspond to a 70% event rate: evaluate calibration on held-out sequences at natural prevalence and report reliability plots, Brier score, ECE, and sequence-aware confidence intervals.

## 5. Stress conditions

Use clean MOT17/MOT20 as the primary conditions. Secondary, predeclared stress tests may apply brightness/contrast reduction, motion blur, compression, and timestamp-preserving source-frame subsampling. Keep source-frame loss distinct from the policy's detector skip. For each actual detector-gap bin, report HOTA/IDF1, ID switches, fragmentation, risk recall/calibration, latency, and energy. Apply training augmentations only within training sequences; freeze test corruption levels before held-out evaluation. Report each condition separately. Augmenting MOT17/20 does not create a new dataset.

## 6. Data quality and integrity checks

- Verify annotations, frame indices, timestamps, and identity continuity on sampled sequences.
- Compare label-generation output to hand-audited paired rollouts for easy, crowded, occluded, and camera-boundary cases.
- Check branch label prevalence and per-sequence prevalence before choosing class weights.
- Detect duplicate frames and related sequence leakage.
- Save deterministic configuration, random seeds, checksums, and software version for every dataset/model artifact.
- Keep test labels inaccessible to model/policy selection code until frozen evaluation.

## 7. Data-use and reporting rules

Cite benchmark papers and official dataset pages. Respect terms for storage, derived annotations, and distribution. Local footage needs permission and an explicit retention plan. Never publish recognizable raw footage without a valid authorization basis. If local labels are too small or unreliable, describe the clip only as a limited demo and do not use it to support a generalization claim.

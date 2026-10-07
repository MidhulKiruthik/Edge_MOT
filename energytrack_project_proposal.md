# Research Project Proposal: Risk-Calibrated Edge MOT

> **Historical proposal:** Retained for audit and idea history. Where this file
> conflicts with the binary-action RACE-MOT scope, current device evidence, or
> gate status, `deep_research_mot_edge_merged.md`, `todo.md`, and the decision
> log control.

**Working title:** **RACE-MOT: Risk-Calibrated Adaptive Compute and Explanations for Multi-Object Tracking on Edge Devices**  
**Status:** Proposal draft for supervisor review; no implementation claims or results are made.  
**Prepared:** 5 October 2026

> **Academic integrity note:** This is a planning document. The student must verify every source, write the final report in their own words, record code/data provenance, and never report unmeasured results as facts. Similarity percentage is not a substitute for proper attribution.

## 1. Strict faculty assessment of the current idea

The starting idea—lightweight CNN detection plus MOT on an edge device—is useful, feasible, and too broad to earn a high novelty score by itself. “YOLO + ByteTrack/OC-SORT on Jetson,” INT8 conversion, and FPS comparison are established engineering exercises. A student could build a working demo and still have a weak research contribution.

The merged note already points toward adaptive detection, but it does not sufficiently distinguish that direction from prior work. EMO specifically studies context-aware detector skipping for edge MOT. RT-MOT studies confidence-aware scheduling of detection and association under real-time constraints. Edge MOT pruning has also been studied. Consequently, “we skip detections when tracks are confident,” “we add an adaptive controller,” “we measure energy,” and “we use a smaller CNN” are not safe standalone novelty claims. The proposal must compare directly with these ideas and state exactly what is different. [EMO project and paper](https://github.com/git-disl/EMO), [RT-MOT paper](https://arxiv.org/abs/2210.11946), [edge MOT pruning paper](https://arxiv.org/abs/2410.08769)

### Provisional marks against the supplied rubric

These are faculty-style estimates based only on the current idea document, not final grades.

| Evaluation component | Weight | Current estimate | Strict assessment |
|---|---:|---:|---|
| Problem selection and research gap | 10 | 6 | Important problem, but scope is broad and the gap is phrased too generally. Existing adaptive and edge MOT work needs stronger treatment. |
| Novelty and innovation | 20 | 7 | The baseline combination is not novel. A sharply defined risk predictor, policy, and evaluated explanation layer could improve this, subject to literature verification. |
| Literature survey | 10 | 6 | Broad coverage, but the document contains claims/leads requiring verification and needs a direct related-work comparison with EMO and RT-MOT. |
| Methodology and AI model development | 20 | 11 | Feasible pipeline, but the learned contribution, training/calibration protocol, and data split are not fixed. |
| Experimental results and comparison | 20 | 12 | Good candidate metrics, but too many datasets, devices, stressors, and policies for one undergraduate project. Fair baselines are not yet specified. |
| Publication/patent potential | 10 | 4 | A systems evaluation may make a strong course project; publication needs a clear differentiated method and compelling repeatable results. Patent potential is presently low. |
| Report, demo, presentation | 10 | 8 | The deployment demo is tangible and explainable, provided the project remains scoped and reproducible. |
| **Total** | **100** | **54** | **Promising direction, not yet a high-scoring research proposal.** |

The main weaknesses are not solved by adding a fashionable CNN block or a saliency heatmap. The proposal needs (1) one deployment context, (2) one falsifiable technical contribution, (3) a direct comparison to the closest prior work, and (4) an evaluation capable of disproving the contribution.

## 2. Refined research direction

### Proposed problem

On a fixed-camera edge stream, can a small learned **track-failure risk estimator** decide when the system should spend extra computation—by running the detector now, using a higher input resolution, or invoking a lightweight appearance cue—while keeping identity quality and tail latency within stated limits?

The key distinction to investigate is **risk prediction for near-future identity failure and calibrated, auditable compute escalation**, rather than periodic or confidence-only frame skipping. This is a **novelty hypothesis**. It becomes a defensible contribution only if a systematic literature review shows that the closest approaches do not already do substantially the same thing, and experiments show benefit over them.

### Recommended application boundary

Use **fixed-camera pedestrian flow monitoring** (for example, a campus corridor or entrance) as the target scenario. The output is anonymous short-term trajectories and counts, not facial identity. Fixed-camera footage keeps the camera-motion problem manageable and makes a small, ethically collected external test set realistic. Public MOT sequences remain the primary benchmark; local footage is a clearly separated domain-shift demonstration, not the only evidence.

If suitable device access is limited, target one device first (Jetson Orin Nano *or* Raspberry Pi with an available accelerator). A second device is an extension, not a requirement. Do not promise 15 FPS, p95 latency, memory, or energy targets until a baseline is measured on the actual hardware.

## 3. Novel contribution statement (proposal version)

> We propose and evaluate a calibrated, track-level risk estimator that predicts imminent localization or identity-association failure from inexpensive temporal and visual cues. Its output selects among a small set of detector/tracker compute actions. For each escalation, the system records a human-readable evidence trace (for example, rising motion uncertainty, falling detector confidence, ambiguous association margin, or likely overlap) and measures whether that evidence predicts actual tracking errors. We compare the policy against always-on inference, fixed-rate skipping, a confidence-only policy, and the closest published adaptive edge MOT methods under a common accuracy, latency, and energy protocol.

This statement does **not** claim a new detector architecture or a first-ever adaptive MOT system. If the risk estimator cannot outperform simple threshold rules or the closest prior work, narrow the claimed contribution to a careful negative result and reusable measurement/evaluation protocol.

### Contributions to aim for

1. A small risk estimator that predicts a defined near-future track failure event, trained using only training sequences and calibrated on a disjoint validation split.
2. A compute policy with a bounded action set, explicit safety fallback, and no use of future frames at inference.
3. An explanation record that identifies the evidence behind each escalation and is evaluated for event detection and calibration, rather than judged only by attractive heatmaps.
4. A controlled edge study reporting tracking quality, tail latency, energy, thermals, and failure regimes against relevant baselines.
5. Optionally, a small consented and de-identified local evaluation set with documented scene conditions and annotation protocol. It is supplementary unless enough varied footage and reliable labels can be collected.

## 4. Research questions and hypotheses

**RQ1.** Does a learned track-failure risk score predict identity breaks better than track confidence, age, or Kalman uncertainty alone?

**RQ2.** Does risk-triggered compute escalation reduce energy per input frame compared with always-on inference while preserving HOTA and IDF1 within pre-registered tolerances?

**RQ3.** Are the predictor and policy useful under crowding, brief occlusion, motion blur, and detector misses, or do they fail in identifiable conditions?

**RQ4.** Do the explanation records faithfully identify the cues that caused escalation and correlate with subsequent ground-truth failures?

**H1 (predictive value).** A calibrated risk model will improve precision-recall for near-future tracking failures over the strongest single-cue threshold baseline on held-out sequences.

**H2 (system value).** At a matched tracking-quality target, the adaptive policy will reduce measured joules per processed input frame relative to always-on detector execution. The target reduction is to be set after a pilot; it is not a promised result.

**H3 (limitations).** The benefit will shrink or reverse in dense scenes and long occlusions, where the system must escalate frequently or an appearance-free tracker cannot recover identity.

## 5. Methodology

### 5.1 Baseline pipeline

```text
Input frame
  -> lightweight CNN detector (batch 1)
  -> ByteTrack association baseline
  -> track state + inexpensive risk features
  -> optional compute escalation on the next eligible frame
  -> tracks, counts, latency/energy log, explanation record
```

Use one frozen, documented small detector as the main baseline (e.g. YOLO nano variant supported by the chosen runtime) and ByteTrack. Add OC-SORT as a second tracker only if time permits. Detector weights, code commit, thresholds, runtime, precision, input size, and power mode must be frozen before final comparisons. The project is not proposing a new CNN backbone. A MobileNetV3/depthwise CNN comparison belongs in an ablation only if it is deployable on the chosen target and does not displace the research contribution.

### 5.2 Risk event and model

Define the prediction target before training. A practical initial target is: **for each active track, will its identity assignment become incorrect, fragmented, or unmatched within the next K frames?** Derive labels from ground-truth identities on training/validation sequences using an explicitly written matching rule. Do not use benchmark test annotations to choose thresholds, train the risk model, or tune policy parameters.

Candidate inputs available online:

- Kalman covariance and normalized prediction residual history;
- detector confidence trend and box scale;
- age, missed-frame count, and time since last high-confidence observation;
- best-versus-second-best association cost margin;
- local box overlap/crowding and an occlusion proxy;
- optional low-cost image crop features from a tiny depthwise CNN.

Start with logistic regression or a small MLP as a transparent baseline. Then test a tiny MobileNetV3-small/depthwise CNN crop encoder only if it adds measurable value. Calibrate scores using a validation-only method (e.g. temperature scaling or isotonic regression), and report Brier score, expected calibration error with binning details, and precision-recall for the failure event. A learned model that is not better calibrated or useful than simple thresholds should not be retained merely to make the project sound more novel.

### 5.3 Policy and compute actions

Keep action choices limited and measurable:

- **A0 — normal:** continue lightweight tracking and use the planned detector cadence;
- **A1 — detect now:** run the base detector on the current frame;
- **A2 — refine:** use a higher resolution or crop-based detector pass around uncertain tracks;
- **A3 — appearance check:** optionally compute a tiny embedding only for a track with high overlap/association ambiguity.

Compare the learned policy with: detector every frame; fixed skipping at several intervals; confidence-only threshold; uncertainty-only threshold; and the closest reproducible published adaptive approach (including EMO/RT-MOT where code/data/runtime permit a fair comparison). Include a cooldown or compute budget to prevent repeated expensive actions. If risk is high and the budget is exhausted, use the safest available detector action and log that the budget was exceeded; do not silently drop the track.

### 5.4 Explainability design

Do **not** describe Grad-CAM or SHAP alone as the XAI contribution. Provide a per-track, per-action evidence record such as:

```text
Track 12: refine on frame 184
Reason evidence: association margin low; predicted box covariance high; overlap proxy high
Risk estimate: 0.81 (calibration bin observed frequency: ...)
Action: crop refinement; measured added latency/energy: ...
Outcome check: matched / identity break within K frames
```

The system must preserve the actual feature values and policy threshold that caused the action. Evaluate explanation utility through: (i) risk calibration, (ii) precision/recall of explanations for subsequent failures, (iii) feature ablation or counterfactual checks showing whether removing a stated cue changes the action, and (iv) a small structured review by the project team or supervisor for clarity. Treat saliency visualizations as optional qualitative figures only.

## 6. Dataset plan and leakage controls

### Main benchmark

- **MOT17**: primary public benchmark for pedestrian tracking and standard comparability.
- **MOT20**: one crowded-scene stress test, if compute and annotation-compatible data handling fit the schedule.
- **Local fixed-camera clips**: optional external-domain prototype evaluation, collected with institutional rules/consent, without face recognition, and with a documented retention/de-identification plan.

Do not use MOT17 and MOT20 interchangeably as train and test without reporting sequence-level splits and domain overlap considerations. Split by full sequence, never by randomly sampling neighboring frames. Keep test sequences untouched until model/policy choices are frozen. If a local dataset cannot be ethically collected and released, omit it rather than making a dataset novelty claim.

### Optional dataset contribution

A small supplementary **EdgeMOT-Risk** annotation set could label frame intervals for detector miss, overlap/occlusion, motion burst, identity switch, and recovery. This is only worth doing if the event definitions are reproducible, two annotators can check a subset, annotation agreement is reported, and redistribution is permitted. Otherwise derive risk labels from existing identities and contribute the risk-label generation protocol instead of claiming a new dataset.

## 7. Experimental protocol

### Baselines and ablations

| ID | Method | Purpose |
|---|---|---|
| B0 | Detector every frame + ByteTrack | Quality/energy reference |
| B1 | Fixed interval detector skipping + tracker | Simple compute-saving reference |
| B2 | Confidence threshold scheduling | Tests whether the learned model is needed |
| B3 | Kalman uncertainty threshold scheduling | Tests value beyond motion uncertainty |
| B4 | Closest published adaptive method reproducible under same conditions | Direct related-work comparison |
| P1 | Proposed risk estimator + action policy, without crop CNN | Main interpretable method |
| P2 | P1 + optional crop CNN cue | Tests visual cue benefit and extra cost |
| P3 | P1 with one feature group removed at a time | Risk model ablation |

Use identical detector weights, video decoding, tracker parameters, frame order, and device settings for comparable variants. Separate algorithmic measurements from deployment measurements; report both offline benchmark replay and live-camera timing if a camera is available. Run enough repeated trials to quantify runtime variability and use sequence-level paired comparisons for tracking scores. Do not claim statistical significance from frames treated as independent samples.

### Metrics

**Tracking:** HOTA plus DetA/AssA, IDF1, MOTA, ID switches, fragmentation, and recall of tracks through occlusion. HOTA is appropriate because it balances detection and association components; MOTA alone can obscure association quality. [HOTA paper](https://doi.org/10.1007/s11263-020-01375-2)

**System:** end-to-end p50/p95 latency, sustained input/output FPS, dropped frames, detector invocation rate, peak RAM, model size, average and idle power, dynamic power when measurable, joules per input frame, temperature, and thermal throttling. State exactly whether decoding, preprocessing, display, and logging are included. Avoid promising that parameter count or TOPS predicts actual energy.

**Risk/XAI:** failure-event precision/recall and PR-AUC, calibration curve, Brier score, expected calibration error, escalation rate, budget-exceeded rate, explanation evidence coverage, and counterfactual consistency.

### Decision criterion

Set tolerances after a small pilot and before the final test run. A reasonable *candidate* criterion is non-inferiority within a small predeclared HOTA/IDF1 margin while reducing joules per input frame; the exact margins must be justified by repeatability and application needs. Report the full Pareto curve instead of cherry-picking one threshold. The system fails its main claim if it saves energy only by materially damaging identity continuity or missing deadlines.

## 8. Literature positioning and gap test

| Prior work / family | Existing contribution | What this project must add to be distinct |
|---|---|---|
| EMO (Ganesh et al., 2023) | Periodic and context-aware detector skipping, including similarity-based cues, plus edge tracking measurements | Predict a defined near-future track failure, calibrate risk, explain action evidence, and compare directly; verify the full method before asserting the boundary. |
| RT-MOT | Confidence-aware scheduling of detection/association with real-time constraints | Focus on track-level error-event risk and energy-aware action selection; distinguish scheduling assumptions and metrics explicitly. |
| Edge MOT channel pruning (2024 preprint) | Compression/pruning for edge MOT | This project studies conditional runtime allocation and calibrated explanations, not another pruning method. |
| ByteTrack / OC-SORT | Strong lightweight association baselines | Use them as baselines, not claimed contributions. |
| HOTA | Balanced tracking metric with detection/association submetrics | Report the component metrics and identity errors, not HOTA alone. |

Before proposal submission, update this table with verified publication venue/year, method details, hardware, dataset, metrics, and limitation from the original paper. Search IEEE Xplore, ACM DL, CVF, arXiv, and Google Scholar using combinations of: `edge multi-object tracking adaptive detection`, `track failure prediction`, `confidence-aware MOT scheduling`, `energy-aware tracking`, `MOT uncertainty calibration`, and `explainable tracking failure prediction`. Record the search date and inclusion criteria. Do not write “first,” “unexplored,” or “no prior work” unless the review supports it.

## 9. Feasibility, risks, and scope cuts

| Risk | Consequence | Scope response |
|---|---|---|
| Close prior work already uses the same risk-triggered idea | Novelty claim collapses | Reframe toward calibrated failure labels/explanation evaluation or choose a different gap before implementation. |
| Risk labels are noisy or event rare | Predictor learns dataset artifacts | Define events precisely, report prevalence and sequence split, compare to trivial baselines, and keep a non-learned fallback. |
| Edge power readings are unstable | Energy conclusion is weak | Fix power mode, warm up, repeat long runs, record ambient/temperature, use external meter or vendor telemetry with stated limits. |
| Re-ID/appearance branch grows scope | Training/deployment work dominates | Make appearance escalation optional; core project works with detector-now/refine actions only. |
| Local video permission or privacy issue | Dataset cannot be used/released | Use public datasets only; do not capture identifiable footage without approval. |
| Too many datasets/devices/stress tests | No controlled study completes | One primary dataset, one stress set, one target device; extensions only after core table is complete. |
| New CNN architecture distracts from research question | More code, little novelty | Do not invent backbone blocks unless profiling shows the bottleneck and an ablation proves a benefit. |

**Explicitly out of scope for the first implementation:** custom YOLO backbone, transformer tracker, multi-camera Re-ID, large-scale new dataset, patent filing, cloud inference, and claims of general privacy preservation. On-device processing reduces transmission; it does not by itself anonymize the video.

## 10. Work plan and deliverables

| Phase | Work | Deliverable / gate |
|---|---|---|
| 1. Proposal week | Verify closest prior work; lock scenario/device/dataset; obtain supervisor feedback | One-page novelty statement and approved scope. Stop or reformulate if gap is not distinct. |
| 2. Baseline | Reproduce detector + ByteTrack and benchmark protocol | Frozen baseline results and configs. |
| 3. Risk labels/model | Define event labels, sequence splits, simple baselines, train/calibrate risk model | Held-out predictive evaluation; keep model only if it beats thresholds. |
| 4. Policy/XAI | Implement limited action policy and evidence logs | Working system with action budget and trace records. |
| 5. Evaluation | Run ablations, edge measurements, stress set, uncertainty analysis | Full results tables, Pareto plot, failure examples. |
| 6. Finalization | Documentation, reproducibility bundle, demo, report/presentation | Source code, configs, run instructions, model/data attribution, limitations, demo. |

## 11. Expected outcome and publication potential

The minimum credible outcome is a working, reproducible edge MOT prototype with a defensible experiment and a clear account of when adaptive computation helps or fails. A publication is possible only if the related-work distinction is real and the evaluation produces a sufficiently strong, repeatable finding. A patent is not a sensible current objective: the idea is not yet a specific patentable invention, and novelty/ownership require separate review.

## 12. Faculty verdict

**Recommendation: approve conditionally as a proposal direction, not as a novelty claim yet.** The original idea likely earns around **54/100** at idea stage under the supplied rubric. The refined version can target a stronger score because it defines a measurable learned component, meaningful XAI, baselines, and failure conditions. It still faces close prior art, and there is no guarantee the contribution is novel or that the method will improve results.

Before implementation, the student should be able to answer in one sentence: **What measurable capability does the proposed risk-calibrated policy provide that EMO, RT-MOT, and a confidence/uncertainty threshold baseline do not?** If that answer is not supported by papers and an executable evaluation plan, do not start building the model yet.

## References to verify and cite in the final proposal

1. Ganesh et al., *Fast and Resource-Efficient Object Tracking on Edge Devices: A Measurement Study*, [arXiv:2309.02666](https://arxiv.org/abs/2309.02666); implementation and described context-aware skipping methods in the [EMO repository](https://github.com/git-disl/EMO).
2. *RT-MOT: Confidence-Aware Real-Time Scheduling Framework for Multi-Object Tracking Tasks*, [arXiv:2210.11946](https://arxiv.org/abs/2210.11946).
3. *Efficient Multi-Object Tracking on Edge Devices via Reconstruction-Based Channel Pruning*, [arXiv:2410.08769](https://arxiv.org/abs/2410.08769). Verify publication status; do not call it peer reviewed without checking.
4. Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-object Tracking*, [IJCV/DOI](https://doi.org/10.1007/s11263-020-01375-2).
5. Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, [ECCV 2022 paper](https://arxiv.org/abs/2110.06864).
6. Cao et al., *Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*, [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html).
7. [MOTChallenge datasets and evaluation](https://motchallenge.net/). Follow the selected dataset's current terms of use.

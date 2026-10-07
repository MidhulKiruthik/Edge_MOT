# RACE-MOT: Risk-Calibrated Adaptive Compute for Edge Multi-Object Tracking
**Document type:** Living research proposal and evidence review for a final-year undergraduate project  
**Last reviewed:** 8 October 2026
**Status:** Product-first research proposal; bounded G1 local feasibility exists, while acceptable detector recall, data/checkpoint terms, grouped roles, G4, and the baseline remain open. Phone work is temporarily deferred under D-33.
**Core scope:** Online, single-camera pedestrian MOT using one detector, ByteTrack, one physical edge device, MOT17, and MOT20.

> **Novelty claim:** RACE-MOT's methodological contribution is a calibrated, per-track estimate of avoidable future identity failure under a specified detector-skip action, used to schedule the next detector call and evaluated against tracking quality and measured edge-system cost. Its defining combination is the failure target, action-conditioned labels, probability calibration, per-track decision evidence, and detector scheduling in a single-camera MOT pipeline. This is the project's claimed novelty; the proposal does not claim that its individual building blocks (TCN/GRU, calibration, skipping, or ByteTrack) are new.

> **Evidence discipline:** This report separates established evidence, partial support, engineering hypotheses, and unverified leads. Published desktop/server results are not edge-device measurements. Verify changing hardware specifications and all lead citations against primary papers or manufacturer documentation before submission.

## Document status and maintenance

This file is the working source of truth for the proposal. Its purpose is to preserve the research evidence, record decisions, and make future novelty suggestions reviewable without silently expanding the core project.

| Decision item | Current position |
|---|---|
| Research question | Can future identity failure be predicted early enough to selectively spend compute while preserving MOT quality? |
| Claimed contribution | One primary method: predict calibrated per-track, skip-action-conditioned avoidable identity failure and use that risk to schedule the next detector call. Make detector-gap history explicit. Evaluate on a physical Orin Nano with complete-pipeline energy and latency accounting. |
| Supporting deliverables | Paired-rollout label-generation code, auditable decision records, local Orin Nano prototype, and measured energy/latency/thermal report. These validate and operationalize the primary novelty claim; they are not additional independent method claims. |
| Novelty position | **Claimed as a focused methodological contribution.** ALBIREO, HSFSO, RT-MOT, EMO, Split and Connect, GLoMOT, and LUKF-Track are explicitly distinguished below by target, timing, input, action, and evaluation. Reproduction and a systematic search strengthen the defense and publication case; they do not change the project into a generic implementation. |
| Accepted core | One detector, ByteTrack, one edge device, MOT17 grouped sequence evaluation, MOT20 stress/generalization evaluation, one temporal risk-policy contribution, and binary DETECT/SKIP actions with a scene-discovery guard that can upgrade SKIP to DETECT. |
| Product deployment decision | The target is the Jetson Orin Nano Engineering Reference Developer Kit Super with the D-23 software inventory. Local MOT17 is the active input; the eventual phone demo remains in the MVP but further phone work is deferred under D-33. |
| Detector implementation candidate | YOLOX-Tiny, person class, TensorRT FP32 on the current verified board path. FP16 is blocked on TensorRT 10.16.2. Bounded OpenCV parity exists; acceptable recall and checkpoint terms remain open. |
| Deferred ideas | Re-ID, patch/ROI inference, learned motion replacement, pseudo-depth association, thermal feedback control, multi-camera tracking, continual learning, and new dataset creation. Reconsider only through the idea register and scope review. |
| Proposal gate | Use the focused comparison below to defend the novelty claim. Inspect source/code and complete the targeted search before final proposal submission and before making any “first” or exhaustive-priority claim. |

### How to maintain this report

For every new idea, add a dated entry to [Appendix A](#appendix-a-future-novelty-idea-register) before changing the core proposal. Record its source, closest prior art, evidence status, scope and evaluation impact, and decision. Promote an idea into the core only after it has a defined research question, a defensible distinction, and a feasible comparison plan. Update the relevant proposal section and [Appendix B](#appendix-b-revision-history) at the same time. Keep rejected and deferred ideas in the register so they are not repeatedly rediscovered or accidentally described as accepted contributions.

### Contents

- [Executive conclusion](#executive-conclusion)
- [1. Application context and market signals](#1-application-context-and-market-signals)
- [2. MOT systems and edge state of practice](#2-mot-systems-and-edge-state-of-practice)
- [3. Research gaps](#3-research-gaps)
- [4. Literature review and claim audit](#4-literature-review-and-claim-audit)
- [5. Focused prior-art comparison](#5-focused-prior-art-comparison)
- [6. Novelty contribution and supporting opportunities](#6-novelty-contribution-and-supporting-opportunities)
- [7. Project directions and scope](#7-project-directions-and-scope)
- [8. Proposed project: RACE-MOT](#8-proposed-project-race-mot)
- [9. Dataset and split protocol](#9-dataset-and-split-protocol)
- [10. Edge platform and measurement](#10-edge-platform-and-measurement)
- [11. Experimental protocol](#11-experimental-protocol)
- [12. Risks and scope controls](#12-risks-and-scope-controls)
- [13. Publication framing](#13-publication-framing)
- [14. Novelty verification gate](#14-novelty-verification-gate)
- [15. Final recommendation](#15-final-recommendation)
- [Appendix A. Future novelty idea register](#appendix-a-future-novelty-idea-register)
- [Appendix B. Revision history](#appendix-b-revision-history)

---

## Executive conclusion

Edge AI is moving video analytics toward local inference because cloud-only pipelines introduce network delay, bandwidth cost, privacy exposure, and availability risk. Lightweight detectors and trackers are already strong: YOLO-family nano models with ByteTrack or OC-SORT are practical baselines, while BoT-SORT, StrongSORT, Deep OC-SORT, and occlusion-aware methods improve identity continuity at additional compute cost.

The product problem is how to preserve identity continuity under strict **latency, memory, energy, thermal, and model-size budgets**. RACE-MOT's novelty claim is the specific use of a calibrated per-track probability of avoidable identity failure under an explicit upcoming detector-skip action to schedule the next detector invocation. This distinguishes it from context/workload scheduling (EMO/RT-MOT), motion-assisted skipping (SDOF-Tracker), aggregate skip-ratio accuracy prediction (HSFSO), per-object state-uncertainty triggering (ALBIREO), post-hoc tracklet repair (Split and Connect), and sparse-frame association/motion filtering (GLoMOT/LUKF-Track). The scene-discovery guard protects entrants before a track exists; it is a product safety feature, not a second novelty claim. Matched comparisons establish the size and practical value of the contribution.

### Recommended project

**RACE-MOT: Risk-Calibrated Adaptive Compute for Edge MOT**

**Core research question:** *Can future identity failure be predicted early enough to selectively spend compute while preserving MOT quality?*

**Novel contribution:** a **temporal, per-track predictor of avoidable identity failure conditioned on the proposed detector-skip action, calibrated as a probability and used to schedule detector invocation under full-pipeline resource constraints**. The method is evaluated against ALBIREO-like per-object uncertainty scheduling, HSFSO, and simple policies. Actual detector-gap features make the failure estimate operationally meaningful across variable gaps; gap-aware tracking alone is not claimed as new. The scene-discovery guard is a separate product-safety branch for entrants that have no active track yet; neither the guard nor generic scene cues are additional novelty claims.

Use one detector and ByteTrack on the available Jetson Orin Nano. The live product demonstration uses one stationary mobile phone as an H.264/RTSP source on a private local network; use MOT17/MOT20 recorded sequences for repeatable benchmark comparisons. YOLOX-Tiny is the provisional person-detector candidate, with TensorRT FP16 deployment if the installed JetPack/runtime supports the export. Inventory the exact Orin Nano RAM/SKU, carrier, cooling, and software before choosing the inference environment. A lightweight temporal model consumes per-track history, including actual elapsed seconds/source frames and consecutive detector skips. A controlled variant also receives low-cost scene-quality/activity descriptors. After processing frame $t$, the calibrated predictor plans one of two actions for the next arriving frame; when that frame arrives, a scene-discovery guard may upgrade SKIP to DETECT using low-resolution activity outside active-track regions:

- invoke the detector; or
- skip detector inference and propagate existing tracks, subject to a hard maximum skip interval.

The novelty claim is **not** adaptive skipping alone, generic scene features, a new YOLO block, or XAI decoration. It is calibrated per-track failure prediction for next-frame scheduling, with an explicitly paired skip/detect label, compared directly with an ALBIREO-like object-wise uncertainty scheduler, HSFSO, EMO/RT-MOT-like policies, and simple rules. The scene-discovery guard is product robustness; compare it against risk-only operation and an ALBIREO-like policy with its empty-scene screening/rescue behavior, and include its full cost. Scene-context fusion is a separate optional ablation. Evidence logs and physical edge measurements support the product and reproducibility; they are not automatically a new XAI or optimization method. If comparisons show no meaningful advantage, report that result and narrow the performance claim while preserving the precise method contribution.

Suggested initial constraints:

- establish device-specific baseline latency, RAM, and power before fixing performance targets;
- set a predeclared HOTA/IDF1 non-inferiority margin based on baseline repeatability and application needs;
- treat energy reduction as an experimental outcome, not a promised 30% gain;
- report p50/p95 latency, sustained FPS, dropped frames, average/idle power, joules/input-frame, temperature, throttling, HOTA, IDF1, MOTA, ID switches, and track fragmentation.

### Strict faculty assessment of the current proposal (idea-stage)

Assessment dated **6 October 2026** against the supplied rubric. These marks judge the current proposal's readiness, not a completed implementation. The experimental-results score reflects the quality of the planned comparison only; no experimental results, edge-device measurements, or demo have been produced yet.

| Component | Weight | Estimate | Strict assessment |
|---|---:|---:|---|
| Problem selection and research gap | 10 | 6 | The product problem is clear, but ALBIREO directly schedules detector calls from per-object state uncertainty, while GLoMOT and LUKF-Track address low-frame-rate motion. The remaining distinction needs direct matched comparison. |
| Novelty and innovation | 20 | 8 | The contribution is now narrowly defined and differentiated by its target, action conditioning, calibration, and scheduling use. Closest comparators are explicit. No trained predictor, matched evaluation, or measured frontier exists yet, so the claim has not earned experimental validation. |
| Literature survey | 10 | 8 | The review now includes ALBIREO, GLoMOT, LUKF-Track, and prior adaptive MOT/video systems with primary links. Direct implementation/code comparison and a systematic search are still outstanding. |
| Methodology and AI model development | 20 | 15 | Failure labels, temporal-model comparisons, calibration, the binary policy, sequence-level splits, and full-pipeline cost accounting are specified. Feasibility still depends on the available device, small number of independent MOT17 sequences, counterfactual-rollout validity, and measured cost of scene features. |
| Experimental results and comparative analysis | 20 | 10 | The planned baselines, ablations, stress tests, metrics, and uncertainty reporting are substantially clearer. No results exist yet, and the closest baselines have not been reproduced under a common edge-device protocol. |
| Publication/patent potential | 10 | 4 | A paper is possible if a substantive distinction survives review and the method improves the measured frontier. Patent potential remains low and unsupported. |
| Report, demo, presentation | 10 | 8 | The proposal is now structured and traceable, with scope and evidence limits stated. Source code, a working edge demo, and presentation materials are not yet available. |
| **Total** | **100** | **59** | **The proposal now has a clear, defensible novelty claim and a realistic product path. It remains idea-stage: the predictor, matched baselines, edge measurements, and experimental gains are still to be produced.** |

The next score increase must come from completing the closest-prior-art comparison and producing reproducible edge measurements, not from adding more model components. The novelty claim is defined; implementation and measurements must establish its practical value and performance limits.

---

## 1. Application context and market signals

| Sector or trend | Edge MOT relevance | Engineering implication |
|---|---|---|
| Smart surveillance | Counting, dwell time, anomaly alerts, perimeter monitoring, and privacy-preserving analytics need persistent identities | Measure identity switches and track fragmentation, not only object detection AP |
| Autonomous vehicles and ADAS | Pedestrians, vehicles, cyclists, and obstacles must be tracked with predictable latency | Report p50 and p95 latency and dropped frames; average FPS alone is inadequate |
| Robotics and warehouses | AGVs, pick-and-place systems, and mobile robots require local perception when connectivity is unreliable | Low power and thermal stability matter over long runs, not just short demos |
| Retail and public-space analytics | Queue length, occupancy, customer flow, and heatmaps require trajectories | Privacy, on-device processing, and stable identity assignment are commercially relevant |
| UAV and drone analytics | Aerial surveillance, search and rescue, and inspection contain small objects and camera motion | VisDrone-style small-object stress tests are valuable, but aerial MOT is more difficult than ordinary pedestrian MOT |
| Sports analytics | Player tracking and formation analysis require long identity continuity under similar appearance and occlusion | Appearance features help, but their edge cost must be measured |
| Edge-cloud split architectures | Local filtering and urgent decisions can happen on-device while selected metadata or clips are uploaded | Define exactly what leaves the device before claiming privacy preservation |
| Heterogeneous accelerators | CPU, GPU, NPU, DSP, and memory bandwidth affect real speed more than parameter count alone | Compare TensorRT, ONNX Runtime, TFLite, or Hailo runtime only when the same model and input conditions are used |
| Model compression | INT8, pruning, distillation, lower resolution, and smaller backbones reduce cost | Compression can damage small-object recall and Re-ID identity features; evaluate MOT metrics after compression |
| Thermal and energy constraints | A device that is fast for five minutes but throttles later is not a reliable edge product | Log temperature and power over a sustained run and report behavior before and after thermal stabilization |

### Hardware and market signals

The supplied studies identify the following industry signals:

- Jetson Orin Nano-class devices provide a realistic low-power GPU target, with vendor claims around tens of INT8 TOPS depending on configuration and power mode.
- Raspberry Pi 5 plus an AI accelerator such as Hailo-8L provides a useful NPU-oriented comparison, but the exact TOPS, power, and supported operators must be cited from the current product documentation.
- Qualcomm and MediaTek continue to integrate NPUs into mobile and embedded SoCs.
- TensorRT, ONNX Runtime, OpenVINO, NCNN, TFLite, and vendor NPU toolchains make deployment more accessible than it was a few years ago.

Public market reports forecast rapid edge-AI growth, but estimates differ because "edge AI," video analytics, and hardware are defined differently. Use market figures only for motivation. The defensible academic motivation is the convergence of hardware roadmaps, real-time detector research, privacy needs, and operational latency/energy constraints.

Useful industry sources:

- [NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
- [AWS IoT Greengrass](https://aws.amazon.com/greengrass/)
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)
- [ETSI Multi-access Edge Computing](https://www.etsi.org/technologies/multi-access-edge-computing)

---

## 2. MOT systems and edge state of practice

### 2.1 Tracking-by-detection

The dominant practical pipeline is:

```text
Camera frame -> detector -> detections -> association -> persistent tracks
```

The detector generally consumes most of the compute, often estimated in practice at roughly 85-95% of pipeline cost, although the exact fraction depends on input resolution, Re-ID usage, runtime, and rendering. This creates a natural opportunity: reduce detector invocations while protecting identity continuity.

### 2.2 Lightweight detector families

| Family | Main idea | Edge relevance | Important limitation |
|---|---|---|---|
| MobileNetV2/V3 with SSD or YOLO head | Depthwise separable convolutions and hardware-aware design | Mature mobile deployment and small models | Small objects and crowds can be weak; parameter count does not predict device latency by itself |
| EfficientNet/EfficientDet | Compound scaling and efficient feature extraction | Strong accuracy-efficiency baseline and TFLite ecosystem | EfficientNet is mainly a backbone/classifier family; detector and export cost still matter |
| NanoDet-Plus, PicoDet, PP-YOLOE-S | Purpose-built small one-stage detectors | Appropriate for ARM/NPU experiments | Accuracy and speed vary by implementation; some references are repositories rather than peer-reviewed papers |
| YOLOX and YOLOv7 | Anchor-free or re-parameterized one-stage detection | Strong community support and real-time baselines | NMS and post-processing can dominate small-device latency |
| YOLOv8/YOLO11 | Modern small YOLO models with broad export support | Easy baseline for TensorRT/ONNX deployment | Version and license details must be frozen; official technical releases are not necessarily peer-reviewed papers |
| YOLOv10 | NMS-free dual assignment and efficiency-driven design | Potentially reduces post-processing and supports efficient scaling | Desktop GPU results do not guarantee NPU/ARM gains; tracking still needs association |
| RT-DETR | Efficient hybrid encoder and query selection | NMS-free real-time DETR alternative | Transformer memory and kernels may be less suitable for low-end ARM devices |
| Lite DETR | Interleaved multi-scale encoder and efficient attention | Reduces detection-head compute | Attention and memory behavior still need validation on the target board |
| YOLO-NAS | NAS-optimized detector family | Good accuracy-efficiency candidate | Larger variants may be excessive for a 5-15 W target |
| EfficientDet-D0 | Efficient feature pyramid and compound scaling | Mature TFLite comparison point | Small-object and runtime performance depend strongly on implementation |

The first supplied study lists indicative figures such as YOLOv8n around 3.2M parameters, YOLOv10n around 2.3M, NanoDet-Plus around 1.2M, and RT-DETR-R18 around 20M. Treat these as configuration-specific orientation values, not universal facts. Edge FPS must be measured on the actual board, runtime, input size, precision, and power mode.

### 2.3 Tracker families

| Tracker | Association signal | Strength | Edge cost/risk |
|---|---|---|---|
| SORT | Kalman motion plus IoU/Hungarian assignment | Extremely fast and easy to reproduce | Identity switches under occlusion, camera motion, and similar-object crossings |
| DeepSORT | SORT plus CNN appearance embedding | Better recovery after short occlusion | Additional CNN inference and memory; older appearance model |
| ByteTrack | Two-stage association using high- and low-confidence detections | Strong accuracy with simple association and low tracker cost | Depends on detector recall and thresholds; low-confidence boxes can create false tracks |
| OC-SORT | Observation-centric Kalman correction and virtual trajectory | Strong under nonlinear motion and prolonged occlusion with cheap association | No appearance cue for similar objects crossing in dense scenes |
| BoT-SORT | Motion, camera-motion compensation, improved Kalman state, Re-ID | Strong MOTChallenge identity results | Re-ID and camera compensation increase latency, memory, and energy |
| StrongSORT | Stronger appearance, AFLink, and GSI | Strong identity metrics | Post-processing and appearance modules are unsuitable for strict online edge budgets unless simplified |
| Deep OC-SORT | Adaptive appearance integration over OC-SORT | Better identity preservation while retaining motion baseline | More compute than pure OC-SORT; edge measurements are limited |
| FairMOT | Joint detection and Re-ID in one network | Shared representation and established baseline | High-resolution shared features and older backbone assumptions |
| CenterTrack | Joint center detection and motion from previous frame | Avoids separate association stage | Sensitive to large motion, occlusion, and backbone/input cost |
| SparseTrack and pseudo-depth methods | Depth-like cues or depth cascade matching | Helps resolve 2D overlap ambiguity | Depth estimation adds compute and has limited edge validation |
| SMILEtrack and occlusion-aware methods | Siamese similarity or plug-in occlusion modules | Better crowded-scene identity continuity | Extra appearance/similarity inference can violate an edge budget |
| Joint/query methods such as MOTR/MOTRv2 | Transformer track queries and temporal propagation | Elegant end-to-end temporal modeling | Training and inference memory are generally too high for a first undergraduate edge project |

**Established baseline:** small detector + ByteTrack and small detector + OC-SORT. A project that only combines YOLO with DeepSORT, ByteTrack, or OC-SORT is an implementation exercise, not a novelty claim.

---

## 3. Research gaps

| Selected gap | Prior-art boundary | What remains to establish |
|---|---|---|
| Adaptive scheduling and low-frame-rate MOT are established. RACE-MOT's contribution is a **calibrated per-track probability of avoidable identity failure, conditioned on a specified detector-skip action and actual elapsed time, used to schedule the next detector call**. | EMO and RT-MOT cover adaptive/context/confidence scheduling; HSFSO predicts tracking accuracy under skip ratios; ALBIREO schedules detector calls from per-object Kalman uncertainty; GLoMOT and LUKF-Track address low-frame-rate or nonlinear-motion MOT; DeepScale and Adaptive-EVOD cover resolution/frame-rate/resource adaptation. | Measure whether this formulation preserves HOTA/IDF1 while improving the energy/deadline frontier against matched ALBIREO-like uncertainty scheduling, HSFSO, and simple baselines. Scene features, Mamba/SSM, pseudo-depth, ROI crops, and thermal feedback are outside the novelty claim. |

This is one research gap. The motion-gap and occlusion papers make broad claims about nonlinear motion or pseudo-depth untenable. The broader device, thermal, small-object, compression, and domain-adaptation topics remain evaluation context or future work, not additional claimed gaps.

### Already solved versus underexplored

| Topic | Status | Correct project framing |
|---|---|---|
| YOLO + ByteTrack/OC-SORT on edge | Established baseline / engineering | Use as infrastructure and comparison, never as the novelty claim |
| Periodic/context skipping and confidence/workload scheduling | Established prior art (EMO, RT-MOT) | Required baselines; generic adaptive scheduling is not new |
| Per-object state/uncertainty-based detector skipping on edge devices | Established adjacent prior art (ALBIREO, accepted SEC 2026) | Required matched uncertainty-scheduler comparison; identity metrics/calibration remain to be measured under this project protocol |
| Low-frame-rate MOT, nonlinear motion handling, and pseudo-depth cues | Established prior art (GLoMOT, AAAI 2026; LUKF-Track, Entropy 2026) | Use as strong related-work boundaries and frame-gap/occlusion stress motivation; do not claim SSM or pseudo-depth novelty |
| GRU/TCN/Transformer for motion, association, or ID-switch localization | Established prior art | Temporal architecture alone is not new; the target and resource policy need differentiation |
| Tracklet switch prediction/repair | Existing work (e.g. Split and Connect) | Major threat to the proposed novelty; compare task timing, output, and use in compute allocation directly |
| Quantization and saliency visualizations | Generic techniques | Optional evaluation tools, not the main contribution |
| Scene/context-conditioned adaptive processing | Established direction; a specific risk-conditioned feature test remains to be evaluated | Test track-only versus track-plus-scene inputs as an ablation; feature addition alone is not novelty |

Do not claim that no paper exists unless the final systematic search supports that wording. Document databases, terms, dates, and exclusions, and phrase incomplete coverage cautiously.

---

## 4. Literature review and claim audit

The following list consolidates the named primary works from both studies. Peer-reviewed venues are identified where reliable official links are available. arXiv, repositories, technical releases, ResearchGate records, and unspecified "2025/2026 series" references are retained as leads but must not be presented as peer-reviewed evidence without verification.

### 4.1 Detection and efficient backbones

| Paper | Year/venue | Method and data | Key result or value | Edge/MOT limitation | Official link |
|---|---|---|---|---|---|
| Searching for MobileNetV3 | 2019, ICCV | Hardware-aware NAS and NetAdapt; ImageNet, COCO, Cityscapes | MobileNetV3-Large reports improved ImageNet accuracy with lower latency than V2; detection and segmentation adaptations | Backbone paper, not full MOT; latency depends on detector head and kernels | [ICCV paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html), [DOI](https://doi.org/10.1109/ICCV.2019.00140) |
| EfficientNet | 2019, ICML | Compound scaling of width, depth, and resolution; ImageNet and transfer tasks | Strong accuracy-efficiency scaling | Scaling does not guarantee lowest latency or memory on ARM; not MOT | [PMLR](https://proceedings.mlr.press/v97/tan19a.html) |
| YOLOX | 2021/2022, preprint | Anchor-free YOLO, decoupled head, SimOTA; COCO | Strong one-stage detector family with tiny/nano variants | NMS and post-processing remain; speed is GPU-centric | [arXiv](https://arxiv.org/abs/2107.08430) |
| YOLOv7 | 2022, preprint | E-ELAN, trainable bag-of-freebies, re-parameterization; COCO | Reports 56.8 AP at 30+ FPS on V100 for its real-time range | Headline variants are not all edge-sized; no identity or energy metrics | [arXiv](https://arxiv.org/abs/2207.02696) |
| YOLOv6 | 2022, preprint | Hardware-friendly backbone/head and self-distillation; COCO | Targets industrial real-time deployment | TensorRT/GPU-specific speed and limited edge MOT evaluation | [arXiv](https://arxiv.org/abs/2209.02976) |
| Lite DETR | 2023, CVPR | Interleaved multi-scale encoder and key-aware deformable attention; COCO | Reports 60% detection-head GFLOP reduction while retaining approximately 99% performance | GPU efficiency does not ensure ARM/NPU efficiency; not a tracker | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Lite_DETR_An_Interleaved_Multi-Scale_Encoder_for_Efficient_DETR_CVPR_2023_paper.html), [DOI](https://doi.org/10.1109/CVPR52729.2023.00435) |
| RT-DETR, "DETRs Beat YOLOs on Real-time Object Detection" | 2024, CVPR | Efficient hybrid encoder, uncertainty-minimal query selection; COCO | R50/R101 report 53.1/54.3 AP and 108/74 FPS on T4 | T4 results are not edge results; transformer memory and tracking integration add risk | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html), [DOI](https://doi.org/10.1109/CVPR52733.2024.00066) |
| YOLOv8 | 2023, technical release | Anchor-free split head and modern training; COCO/user datasets | Broad ONNX/TensorRT support and useful nano baseline | Not a conventional peer-reviewed paper; version and licensing must be frozen | [Ultralytics repository](https://github.com/ultralytics/ultralytics) |
| YOLOv10 | 2024, NeurIPS version/preprint | Consistent dual assignment, NMS-free inference, efficiency design; COCO | YOLOv10-S is reported faster and smaller than comparable RT-DETR; YOLOv10-B reports lower latency than YOLOv9-C at similar performance | Desktop GPU comparisons; edge kernels and MOT compatibility need direct tests | [arXiv](https://arxiv.org/abs/2405.14458) |
| RTMDet | 2022, technical/preprint | CSPNeXt and real-time detector scaling; COCO | Strong accuracy-speed trade-offs | Detector-only benchmark; no complete energy-aware MOT evaluation | [arXiv](https://arxiv.org/abs/2212.07784) |
| NanoDet-Plus | 2021, repository/preprint lead | Lightweight anchor-free detector, Ghost-PAN, DSLA, GFL; COCO | Designed for mobile CPU and low memory | Peer-review status and headline FPS must be checked; small/dense-object accuracy can be weak | [Repository](https://github.com/RangiLyu/nanodet) |
| EfficientDet-D0 | 2019, detector family | Efficient feature pyramid and compound scaling | Useful TFLite comparison point | Runtime and small-object behavior are implementation-dependent | [TensorFlow model family](https://github.com/google/automl/tree/master/efficientdet) |
| Benchmarking object detection models on edge devices | 2024, preprint lead | YOLO/EfficientDet/SSD comparisons on Jetson/Raspberry Pi; COCO/custom | Useful motivation for edge detector comparison | Detection-only; must verify official metadata before citation | [arXiv lead](https://arxiv.org/abs/2410.04173) |
| EUAVDet | 2024, preprint lead | Lightweight aerial detector; VisDrone | Reported 20+ FPS on Jetson Nano in the supplied study | Detection-only, not MOT; paper metadata should be verified before thesis use | **Verify official source before citing** |

### 4.2 Tracking and association

| Paper | Year/venue | Method and data | Key result | Edge/MOT limitation | Official link |
|---|---|---|---|---|---|
| DeepSORT | 2017, IEEE ICIP | Kalman filter plus CNN appearance descriptor; MOT16 | Established appearance-aware baseline | Older descriptor and additional inference cost | [DOI](https://doi.org/10.1109/ICIP.2017.8296962) |
| Tracktor++ | 2019, ICCV | Detector box regression as tracker, motion model, Re-ID; MOT17 | Shows detector quality can dominate tracking performance | Repeated detector/regression cost and older design | [ICCV paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Bergmann_Tracking_Without_Bells_and_Whistles_Tracktor_++_ICCV_2019_paper.html), [DOI](https://doi.org/10.1109/ICCV.2019.00141) |
| CenterTrack | 2020, ECCV | Center detection plus previous-frame centers and motion; MOT17, KITTI, nuScenes | Joint detection and tracking without separate association | Sensitive to large motion/occlusion; backbone cost remains | [ECCV paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3255_ECCV_2020_paper.php), [DOI](https://doi.org/10.1007/978-3-030-58548-8_26) |
| FairMOT | 2021, IJCV | Joint detection and identity embedding; MOT15/16/17/20, KITTI | Strong detection/Re-ID balance | Shared high-resolution features cost memory; older backbone | [arXiv](https://arxiv.org/abs/2004.01888), [DOI](https://doi.org/10.1007/s11263-021-01408-4) |
| ByteTrack | 2022, ECCV | Two-stage association uses high- and low-score detections; MOT17/20, KITTI, DanceTrack, BDD100K | Widely reported 80.3 MOTA and 77.3 IDF1 on MOT17 | No appearance model; detector recall/thresholds dominate; no energy objective | [ECCV paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php), [DOI](https://doi.org/10.1007/978-3-031-20047-2_1) |
| MOTR | 2022, ECCV | Transformer track queries and temporal propagation; DanceTrack, MOT17/20 | End-to-end query-based tracking | Transformer memory and training complexity are high for edge | [ECCV paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1739_ECCV_2022_paper.php) |
| MOTRv2 | 2022, NeurIPS lead | Bootstrapped detector plus MOTR query tracking; MOT17/20, DanceTrack | Strong DETR-based tracking | Compute-heavy and operationally complex | [arXiv](https://arxiv.org/abs/2204.00776) |
| BoT-SORT | 2022, preprint/benchmark system | Motion, camera-motion compensation, Re-ID, improved Kalman state; MOT17/20 | Reports 80.5 MOTA, 80.2 IDF1, 65.0 HOTA on MOT17 | Re-ID and CMC add compute and memory; not a low-power deployment study | [arXiv](https://arxiv.org/abs/2206.14651) |
| OC-SORT | 2023, CVPR | Observation-centric re-update and virtual trajectory; MOT17/20, KITTI, DanceTrack | Official project reports 700+ FPS for tracker-only CPU association and strong benchmark results | Tracker FPS excludes detector; no appearance cue for similar-object crossings | [CVPR paper](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html), [DOI](https://doi.org/10.1109/CVPR52729.2023.00571) |
| StrongSORT | 2022/2023, IEEE TMM lead | Stronger Re-ID, camera compensation, AFLink and GSI; MOT17/20 | Strong identity metrics on server benchmarks | AFLink/GSI may be offline or costly; unsuitable for strict online budgets without simplification | [arXiv](https://arxiv.org/abs/2202.13514), [DOI](https://doi.org/10.1109/TMM.2022.3217698) |
| Deep OC-SORT | 2023, preprint/benchmark | Adaptive Re-ID over OC-SORT; MOT17/20, DanceTrack | Reports 64.9 HOTA on MOT17, 63.9 on MOT20, 61.3 on DanceTrack in the supplied source | Appearance improves identity but reduces the minimal-compute advantage | [arXiv](https://arxiv.org/abs/2302.11813) |
| SparseTrack | 2023/2025 lead | Pseudo-depth and depth-cascade matching; MOT17/20 | Competitive HOTA using overlap/depth cues in the supplied study | Pseudo-depth overhead and publication metadata require verification | [DOI lead](https://doi.org/10.1109/TCSVT.2024.3524670) |
| SMILEtrack | 2024 lead | Siamese similarity and patch self-attention; MOT17/20 | Supplied study reports 65.2 HOTA on MOT17 | Similarity network overhead; verify venue/results before final citation | [arXiv lead](https://arxiv.org/abs/2401.07722) |
| OA-SORT | 2026 lead | Occlusion-aware modules described as OAM/OAO/BAM; MOT17/20 | Supplied study reports gains over existing trackers | 2026 venue and results must be checked against official proceedings; not yet edge validated |
| OCCTrack/OcclusionTrack | 2025 lead | Confidence Kalman filter, depth-cascade matching, CMC; MOT20/DanceTrack | Supplied study describes dense-scene robustness | Multi-component pipeline and unverified metadata; edge cost unclear |
| PD-SORT | 2025 lead | Pseudo-depth states and DVIoU; MOT17/20 | Supplied study reports improved overlap disambiguation | Depth overhead and source metadata require verification |

### 4.3 Adaptive inference, edge systems, and temporal identity prediction

| Work | Year/status | Method and data | Finding | Limitation or use |
|---|---|---|---|---|
| Fast and Resource-Efficient Object Tracking on Edge Devices: A Measurement Study (EMO) | 2023, arXiv | Periodic and context-aware detector skipping, including similarity-based cues; MOTChallenge | Studies reduced detection compute while retaining comparable tracking quality | Direct prior art for generic detector skipping; compare methods and assumptions rather than claiming adaptive skipping is new. [Paper](https://arxiv.org/abs/2309.02666), [code](https://github.com/git-disl/EMO) |
| RT-MOT: Confidence-Aware Real-Time Scheduling Framework for MOT Tasks | 2022, arXiv | Confidence-aware detection/association scheduling under real-time constraints | Establishes a scheduler and confidence-driven trade-off between execution and tracking quality | Direct prior art for confidence-aware scheduling; distinguish prediction target, policy actions, objective, and device-energy protocol. [Paper](https://arxiv.org/abs/2210.11946) |
| Hierarchical Surrogate-Based Frame Skip Optimization for MOT (HSFSO) | 2026, *Information Sciences*, article 123610 | MLP surrogate predicts tracking accuracy; a global skip-ratio stage is refined by frame-level optical-flow selection; experiments use ByteTrack and YOLOv8-small on MOT17/20 | Closest comparator for learned tracking-quality prediction and adaptive frame skipping | Direct comparator. RACE-MOT's target is per-track avoidable identity failure under a particular next-frame skip action, with probability calibration and action scheduling; compare on common data and edge constraints. [Journal paper](https://doi.org/10.1016/j.ins.2026.123610), [article page](https://www.sciencedirect.com/science/article/abs/pii/S0020025526005414) |
| SDOF-Tracker: Fast and Accurate Multiple Human Tracking by Skipped-Detection and Optical-Flow | 2022, IEICE Transactions on Information and Systems | Interpolates detector results with optical flow and terminates tracks using interest-point distribution; evaluated on MOT17/20 | Direct prior art combining periodic skipped detection with motion propagation on the same benchmark families | Optical-flow propagation is not a novelty claim here. It could be a future comparison/alternative skip-propagation mechanism, but adding it now would change the tracker and confound the risk-policy evaluation. [Journal paper and abstract](https://www.jstage.jst.go.jp/article/transinf/E105.D/11/E105.D_2022EDP7022/_article), [DOI](https://doi.org/10.1587/transinf.2022EDP7022) |
| Continuous, Real-Time Object Detection on Mobile Devices without Offloading (AdaVP) | 2020, IEEE ICDCS | Mobile Parallel Detection and Tracking pipeline runs detector and tracker concurrently; adapts DNN model settings to video change rate; implemented on Jetson TX2 | Prior art for parallel mobile detection/tracking and content-adaptive detector configuration | The proposed core remains sequential detector-invocation control with one detector configuration. Do not claim parallel detection/tracking or dynamic model configuration as new; defer those because they alter scheduling, action space, and latency/energy attribution. [Paper PDF](https://sites.ucmerced.edu/files/wdu/files/icdcs_2020_adavp.pdf), [DOI](https://doi.org/10.1109/ICDCS47774.2020.00085) |
| DeepScale: Online Frame Size Adaptation for Multi-object Tracking on Smart Cameras and Edge Servers | 2022, IEEE/ACM IoTDI | Selects input frame size based on visual-content complexity; evaluates MOT trade-offs and camera/edge-server partitioning, including a Jetson TX2 testbed | Direct prior art for variable-resolution computation in MOT | A low-resolution/full-resolution action is not a new contribution by itself. Keep resolution fixed in the core study; cite DeepScale if a later multi-resolution policy is proposed. [Conference paper](https://conferences.computer.org/cpsiot/pdfs/IoTDI2022-2rrtCzgVqtMewzqOmTALX0/964100a067/964100a067.pdf), [DOI](https://doi.org/10.1109/IoTDI54339.2022.00010) |
| Edge-Assisted Video Object Detection With Adaptive Resource Allocation and Model Selection (Adaptive-EVOD) | 2026, IEEE Transactions on Mobile Computing, early access | Adapts detection resolution, frame rate, model placement/selection, and GPU resources in an edge-assisted video-detection system | Recent adjacent prior art for multi-mode resource-aware video inference | It is not the same single-device MOT task, but it rules out broad novelty claims around jointly adapting resolution, frame rate, and compute resources. Inspect full paper and match scope before any extension. [IEEE DOI](https://doi.org/10.1109/TMC.2026.3719713) |
| ALBIREO: Adaptive, Energy-Efficient Inference Framework for Video Object Detection on the Edge | Accepted at ACM/IEEE SEC 2026; arXiv preprint submitted 29 Aug 2026 | Triggers detector calls from per-object 10-D Kalman uncertainty; includes rescue and empty-scene screening; reports edge energy and AP@50 on BDD100K MOT validation | Very close adaptive detector-invocation prior art, including object-state-conditioned scheduling and Jetson measurements | Its abstract reports detection AP rather than HOTA/IDF1 and does not establish calibrated identity-failure prediction. Implement an Albireo-like matched uncertainty policy with the project detector/ByteTrack and compare full-pipeline energy and identity metrics. Do not claim per-object adaptive skipping is new. [Preprint](https://arxiv.org/abs/2609.29648), [code](https://github.com/amirtaherin/albireo) |
| GLoMOT: Efficient Online GNN-based Low-Frame-Rate Multi-Object Tracker | AAAI 2026 | Online GNN association with a dynamic node buffer, adaptive context-aware feature weighting, and a pseudo-depth feature; evaluates MOT17, DanceTrack, SportsMOT, and VisDrone under frame gaps | Direct prior art for low-frame-rate MOT, motion/appearance ambiguity, temporal memory, and pseudo-depth cues | GLoMOT takes low-rate input and is a tracker, not the proposed detector-invocation policy. Cite as a method boundary and frame-gap/occlusion reference; do not claim nonlinear LFR MOT or pseudo-depth as new. A full GNN tracker comparison is outside the fixed-ByteTrack core. [AAAI paper](https://doi.org/10.1609/aaai.v40i6.42500), [official PDF](https://ojs.aaai.org/index.php/AAAI/article/download/42500/46461) |
| LUKF-Track: A Multi-Object Tracking Method with an Unscented Kalman Filter on a Lie Group Manifold | *Entropy* 2026, 28(1), 103; published 15 Jan 2026 | Appearance-independent motion variant uses a Lie-group UKF; reports MOT17, MOT20, and DanceTrack results | Direct prior art for nonlinear motion filtering and motion-only MOT under occlusion | The supplied note misidentifies the venue as Sensors. LUKF-Track is a different tracker, not an adaptive detector scheduler; cite it as a strong motion-model boundary and keep a tracker swap outside this study. [Journal article](https://doi.org/10.3390/e28010103), [full text](https://www.mdpi.com/1099-4300/28/1/103) |
| Confidence-Guided Frame Skipping to Enhance Object Tracking Speed (Lee) | 2024, Sensors | Uses confidence in lightweight single-object tracking to invoke a more robust tracker when needed; evaluated for SOT, including VOT2018 | Related precedent for confidence-triggered selective computation | It is not a multi-object tracking or detector-scheduling method, so it is not a direct MOT baseline. It does rule out broad claims that confidence-gated compute is new; cite it as adjacent SOT work and compare only if a faithful, meaningful adaptation can be defined. [Journal paper](https://www.mdpi.com/1424-8220/24/24/8120), [DOI](https://doi.org/10.3390/s24248120) |
| Split and Connect: A Universal Tracklet Booster for MOT | 2022, IEEE Transactions on Multimedia (preprint 2021) | Temporal dilated convolutions predict potential ID-switch positions; a splitter and attention-based connector repair tracklets; evaluated on MOT17/20 | Directly studies temporal prediction of ID-switch locations and tracklet-level identity repair | Key prior-art boundary: distinguish online skip-action-conditioned failure prediction and detector invocation from predicted switch localization and tracklet repair. Cite the journal publication year/venue accurately. [IEEE journal record](https://ieeexplore.ieee.org/abstract/document/9672670), [DOI](https://doi.org/10.1109/TMM.2022.3140919), [preprint](https://arxiv.org/abs/2105.02426) |
| GAKP: GRU Association and Kalman Prediction for MOT | 2020, arXiv preprint | GRU learns association costs from motion and appearance features | Shows GRU use in MOT association predates this project | A GRU is not a novelty claim. [Paper](https://arxiv.org/abs/2012.14314) |
| ETTrack: Enhanced Temporal Motion Predictor for MOT | 2024, Applied Intelligence | TCN plus temporal Transformer predict object motion from historical trajectories | Shows TCN and temporal sequence modeling are established for MOT | The proposal must predict calibrated identity-failure risk for compute allocation, not merely future box position. [DOI](https://doi.org/10.1007/s10489-024-05866-4) |
| QPilot: Reinforcement Learning-Based Adaptive Scheduling for Object-Sparse MOT | 2026, PerCom WIP lead | RL model switching using device capacity and scene complexity | Recent adaptive MOT scheduling | Obtain proceedings/full paper and inspect methods/results before final novelty claims. [Lab announcement](https://www.jn.sfc.keio.ac.jp/%E3%80%90%E6%8E%A1%E6%8A%9E%E3%83%BB%E7%99%BA%E8%A1%A8%E3%80%91%E4%BF%AE%E5%A3%AB%E8%AA%B2%E7%A8%8B1%E5%B9%B4-%E7%BE%85%E5%90%9B%E3%81%8Cieee-percom-2026-wip-session%E3%81%A7%E7%99%BA%E8%A1%A8/) |
| Reconstruction-based channel pruning for Edge MOT | 2024 lead | Structured pruning preserving tracking accuracy; MOT17/custom | Supplied study reports up to 70% model-size reduction | Single ByteTrack-style setting and no reliable energy accounting identified; verify primary paper |
| Prune-Quantize-Distill ordered pipeline | 2026 lead | Pruning, QAT, and knowledge distillation; COCO | Promising size-latency frontier | Detection-focused and unverified metadata; use as motivation for a controlled ablation |
| QuantEdge hybrid quantization | 2025 IEEE Access lead | Dynamic precision adaptation for hardware | Relevant to heterogeneous hardware | Detection-focused, not full MOT; verify DOI and hardware details |
| Knowledge distillation plus QAT for edge anomaly detection | 2024 lead | MVTec/custom | Shows compression workflow on Jetson-class hardware | Single-image anomaly detection, not temporal identity tracking |
| A deep learning driven framework with ObjTrackNet for real-time object and human detection in smart video surveillance | 2026, *Discover Computing* | Reports difficult cases involving occlusion, low illumination/contrast, blur, and low-rate sampling; discusses variable frame rate, illumination-aware enhancement, and motion compensation as future directions | Useful motivation for stress conditions, not evidence that scene features improve risk prediction | The paper does not evaluate a scene-conditioned skip controller or establish novelty for using brightness/blur/motion features. Treat its recommendations as hypotheses and test them with a controlled ablation. [Springer paper](https://link.springer.com/article/10.1007/s10791-026-10143-8) |
| Lightweight and real-time object detection model on edge devices with model quantization | 2021, *Journal of Physics: Conference Series* 1748, 032055 | Uses MobileNetV2/SSD-Lite and TensorFlow Lite float32-to-int8 quantization for object detection; reports detector-oriented COCO/edge results | Confirms quantization as established edge-detector optimization | It is not an MOT study and does not establish identity preservation. Cite only as detector-compression background; keep quantization as a late ablation, not novelty. [DOI](https://doi.org/10.1088/1742-6596/1748/3/032055) |
| Structured pruning and quantization for traffic detection | 2026 lead | PTQ/QAT on YOLO; UA-DETRAC | Useful traffic deployment baseline | ResearchGate-only source in supplied study; not suitable as sole scholarly evidence |
| E4 energy-efficient early exit | 2025, AAAI lead | Attention-based cascade and DVFS co-optimization; video classification | Supplied study reports 2.8x speedup and 26% energy savings | Classification, not MOT; supports the adaptive-compute motivation but not the MOT novelty claim |
| Dynamic reconfiguration survey | 2025 lead | Early exits, routing, selective activation | Provides adaptive edge-inference taxonomy | Survey/position paper, not MOT implementation |

**Important citation rule:** the lead rows above should be retained in a project bibliography only after the team obtains the official PDF, DOI, venue, dataset protocol, and exact result table. They are useful research leads, but they are not equivalent evidence to the established papers with official links.

### 4.4 Audit of the additional problem claims

The following points came from an informal online scan. They are hypotheses to verify, not established facts simply because they sound technically plausible.

| Claim from the scan | Evidence-based verdict | Project response |
|---|---|---|
| Severe occlusion causes identity errors; lightweight embedding pruning makes Re-ID fail | **Partly supported, causal wording overstated.** Crowded/occluded scenes challenge association and identity continuity; MOT20 was created to test extremely crowded scenes. But the scan gives no evidence that edge systems generally prune Re-ID layers or that pruning is the cause of the ID switches. ByteTrack has no Re-ID embedding branch to prune. [MOT20 benchmark](https://arxiv.org/abs/2003.09003), [MOTChallenge benchmark analysis](https://doi.org/10.1007/s11263-020-01393-0) | Keep ByteTrack fixed. Report IDF1, ID switches, fragmentation, and risk-model recall by predeclared crowd/visibility strata where annotations support it; use MOT20 as the crowd stress set. Do not add Re-ID or claim to solve re-identification in this project. There is no universal requirement that every edge tracker exceed 30 FPS: set the deadline from the actual input cadence and application. |
| Lower/variable frame rates create harder displacements, blur, and Kalman errors | **Mechanism is plausible; the scan conflates separate causes.** Longer gaps make a constant-velocity model extrapolate farther and can increase association error; exposure-time motion blur is a separate image-formation effect and is not caused automatically by dropping frames. SORT uses a linear motion model, and later work discusses its limits; SDOF-Tracker specifically studies skipped detection and motion propagation. [SORT](https://arxiv.org/abs/1602.00763), [OC-SORT](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html), [SDOF-Tracker](https://doi.org/10.1587/transinf.2022EDP7022) | Separate the effects in a secondary stress test: controlled source-frame subsampling with real timestamps, and independently controlled motion blur. Add elapsed-time/frame-gap and time-normalized motion cues to risk-model inputs if available. Keep input-frame loss distinct from the policy's detector-skip action. Report the stress results separately from the full-rate primary evaluation. |
| Edge MOT lacks context-aware dynamic sampling and should change frame rate/precision with object density | **The absence claim is false.** EMO, RT-MOT, SDOF-Tracker, AdaVP, and especially HSFSO (2026) already cover adaptive skipping, workload/signal-aware scheduling, optical-flow-assisted frame selection, or detector adaptation. HSFSO uses an accuracy surrogate and adaptive skip ratio with ByteTrack on MOT17/20. ObjTrackNet reports failure conditions and suggests future extensions, but does not test a scene-conditioned compute policy. [HSFSO](https://doi.org/10.1016/j.ins.2026.123610), [EMO](https://arxiv.org/abs/2309.02666), [RT-MOT](https://arxiv.org/abs/2210.11946), [ObjTrackNet](https://link.springer.com/article/10.1007/s10791-026-10143-8) | Treat density as a predeclared evaluation stratum. Test whether low-cost scene context improves calibrated per-track skip-failure prediction over track history alone; feature addition itself is not novelty. Keep the controller's two actions (run/skip detector); do not add dynamic resolution or precision switching. Quantization remains a fixed-precision late ablation, not a second controller. |
| Jetson Orin Nano thermal throttling under sustained MOT load is a deployment bottleneck | **Throttling is a real device behavior; the workload-specific claim is unmeasured.** NVIDIA documents fan management and clock throttling when temperature crosses trip points. The quoted 7–15 W range is not universal across versions or software: original Orin Nano 8 GB reference modes include 7/15 W, while JetPack 6.2 adds 25 W and uncapped MAXN SUPER; Orin Nano Super also has 102 GB/s DRAM bandwidth. [NVIDIA thermal/power guide](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html), [NVIDIA JetPack 6.2 modes/specs](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) | Preserve the sustained-run thermal constraint already in the proposal. Record exact module/devkit, JetPack version, power mode, fan/enclosure, ambient temperature, temperature, clocks, and throttling indicators over long runs. Do not claim fanless rugged deployment; the thermal design is an explicit experimental condition. |
| Hundreds of tracks and RTSP decoding saturate the Orin Nano's 102 GB/s memory bandwidth before the GPU | **Plausible bottleneck, but unsupported as a general conclusion.** 102 GB/s is a peak specification for the newer Orin Nano Super configuration; the original 8 GB module is listed at 68 GB/s. Peak bandwidth does not establish achieved bandwidth or prove tracking-state storage is the limiting stage. Decode, copies, resolution, model traffic, and stream count must be profiled. [NVIDIA specifications](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/) | Keep the one-stream scope and report end-to-end latency/RAM. If profiling identifies memory traffic as a bottleneck, report it as an observed result using device counters; do not add a multi-stream claim or optimize for hundreds of tracks without a separate workload. |
| Distributed cross-camera edge mesh exchanging embeddings/coordinates over MQTT or 5G is a novel future direction | **The architecture is established; the transport choice is not the research contribution.** WatchDog studied real-time tracking across geo-distributed edge nodes, and EASE-MCVT describes sending edge-extracted locations/appearance features for cross-camera association. [WatchDog](https://arxiv.org/abs/2002.04597), [EASE-MCVT preprint](https://arxiv.org/abs/2511.13904) | Defer. It requires multiple cameras/nodes, cross-camera identity labels, synchronization, network-loss/latency experiments, and privacy/security analysis. It conflicts with the current single-camera study; MQTT or 5G alone would be implementation choices, not novelty. |
| Hybrid CNN-attention models are a new way to capture global context with edge latency | **The direction is established, and the claim that ViTs are categorically too heavy is too broad.** MobileViT was explicitly designed as a lightweight CNN/transformer hybrid for mobile vision. [MobileViT, ICLR 2022](https://openreview.net/forum?id=vh-0sUt8HlG) | Keep the detector frozen and compare no new backbone. A hybrid detector would change both tracking quality and detector cost, obscuring the central scheduling experiment. Consider it only as a later detector-architecture study with a matched compute budget. |
| On-device unsupervised continual learning is an unaddressed solution to environmental domain shift | **Domain shift is real, but the gap is misstated.** Test-time adaptation for MOT was already studied by DARTH at ICCV 2023; continual updates add pseudo-label drift and forgetting risks. [DARTH](https://openaccess.thecvf.com/content/ICCV2023/html/Segu_DARTH_Holistic_Test-time_Adaptation_for_Multiple_Object_Tracking_ICCV_2023_paper.html), [Re-ID domain-shift survey](https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Nguyen_Tackling_Domain_Shifts_in_Person_Re-Identification_A_Survey_and_Analysis_CVPRW_2024_paper.pdf) | Defer online learning. Test the frozen model on held-out MOT20 and predeclared blur/compression/frame-cadence shifts; report transfer/calibration degradation. An adaptation method needs its own labeled protocol, drift safeguards, and compute/energy budget, so it is a separate research question. |

**Practical conclusion:** the internet scan surfaces useful stress conditions, but it does not justify adding Re-ID, multi-camera networking, a new backbone, or online learning. The in-scope additions are occlusion/density-stratified reporting, a controlled frame-gap/blur/illumination stress test, an ablation for low-cost scene cues, real sustained thermal telemetry, and a direct HSFSO comparison. The primary method remains one calibrated per-track failure-risk policy with a binary detector-invocation action.

### 4.5 Audit of the four additional claims supplied on 6 October 2026

| Suggested claim | Evidence check | Design decision |
|---|---|---|
| Skip-induced nonlinear motion merits a tiny Delta-t-conditioned Mamba/SSM compensator | **The low-frame-rate difficulty is supported; the proposed novelty and numeric budget are not.** GLoMOT (AAAI 2026) directly studies online LFR MOT, large temporal gaps, motion ambiguity, and pseudo-depth. LUKF-Track (Entropy 2026, not Sensors) already proposes appearance-independent nonlinear motion filtering with a Lie-group UKF. Neither establishes that all Kalman predictions fail catastrophically at 5–10 FPS or that a sub-15K-parameter Mamba module will run within an assumed budget on this Orin Nano. [GLoMOT](https://doi.org/10.1609/aaai.v40i6.42500), [LUKF-Track](https://doi.org/10.3390/e28010103) | Keep the existing ByteTrack motion model fixed. Make actual elapsed time since the last detector update, current skip count, and timestamp-normalized displacement explicit inputs. Add a preregistered source-frame gap stress test with separate HOTA/IDF1, IDSW, fragmentation, risk-recall, calibration, latency, and energy results per gap. Defer a learned motion compensator; it changes the tracker and would confound the scheduling study. |
| Spatially sparse RoI/patch detection is a novel third action because 85% of pixels are background and patch inference saves 70% energy | **The numerical premise is unsupported, and the action changes the product design.** Pixel occupancy does not imply the same fraction of CNN FLOPs is wasted; the detector can use scene context, and crops can miss entrants or objects outside currently risky tracks. Adaptive-EVOD already covers adaptive detection location/resolution/rate/resource allocation in edge-assisted video detection. No matched evidence supports the claimed 70% energy reduction for 160x160 crops on this model/device. [Adaptive-EVOD](https://doi.org/10.1109/TMC.2026.3719713) | Defer patch inference. Keep DETECT/SKIP as the two policy actions, one detector and one input size. Treat spatial routing as a future project requiring crop-coordinate validation, new labels, global discovery protection, matched baselines, and full-pipeline measurements. |
| Thermal-aware feedback should change risk threshold and maximum skip count online | **Thermal/clock management is real; the quoted temperatures and 30–50% latency drift are workload/device-specific and unsubstantiated here.** NVIDIA documents thermal sensing, clock/power management, hardware throttling, and tegrastats telemetry; it does not make the attachment's fixed threshold/drift claim universal. [NVIDIA Jetson power/thermal guide](https://docs.nvidia.com/jetson/archives/r36.3/DeveloperGuide/SD/PlatformPowerAndPerformance.html), [tegrastats guide](https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html) | Keep sustained thermal/clock/latency monitoring and deadline-miss analysis in the product evaluation. Do not make temperature alter risk thresholds or skip limits in the core: those choices are frozen from validation and an online controller would add a second policy requiring stability and identity-quality evaluation. Any safety stop uses device limits and is documented separately from adaptive scheduling. |
| Zero-FLOP perspective pseudo-depth is new and can resolve occlusions in ByteTrack | **Pseudo-depth for occlusion association is already explicit in GLoMOT; SparseTrack is also prior art.** GLoMOT fuses normalized object size, foot-point position, and occlusion cues. The supplied single formula is not GLoMOT's method and is not camera-calibrated; its direction/scaling need validation across camera views, truncation, and posture. Arithmetic cost is not zero and the quoted 0.05 ms is unmeasured. [GLoMOT](https://doi.org/10.1609/aaai.v40i6.42500), [SparseTrack](https://doi.org/10.1109/TCSVT.2024.3524670) | Do not add a pseudo-depth association matrix to the current ByteTrack core or call it novel. Retain overlap/crowding and occlusion-stratified evaluation. A geometry cue may enter a future matched association ablation only after its ordering convention and camera assumptions are specified. |

This review changes the detector scheduling design while keeping the tracker and action space focused. The predictor now receives actual detector-gap/timestamp history; a separate low-cost scene-discovery guard may upgrade SKIP to full-frame DETECT for untracked activity or abrupt scene change. ALBIREO already includes an empty-scene screen and rescue path, so the guard's only plausible incremental value is activity outside existing tracks while other tracks are active; evaluate it against a matched ALBIREO-like baseline with its screening behavior, not only against the risk-only policy. GLoMOT and LUKF-Track are close motion/occlusion prior-art boundaries. The supplied components do not create four additional novelty claims. The guard is an engineering safety mechanism whose added cost and benefit must be measured.

---

## 5. Focused prior-art comparison

| Work | Signal/model and purpose | Overlap with this proposal | Required distinction |
|---|---|---|---|
| EMO (Ganesh et al., 2023) | Window-based and similarity-based optimizations to improve on-device MOT throughput | Selective processing and edge MOT | RACE-MOT predicts an action-conditioned future identity-failure event per track; compare the resulting quality/energy frontier with EMO-style policies under matched conditions. [Paper](https://arxiv.org/abs/2309.02666) |
| RT-MOT (Kang et al., RTSS 2022) | Estimates per-task object confidence and tracking-accuracy variation under workload pairs, then schedules tasks under real-time constraints | Prediction-informed workload selection and real-time constraints | RACE-MOT predicts calibrated K-frame identity-failure probability for a detector-skip action; compare with an RT-MOT-like policy and report deadline misses and energy. [Paper](https://arxiv.org/abs/2210.11946) |
| SDOF-Tracker (Nishimura et al., IEICE 2022) | Skips detector calls, propagates results with optical flow, and terminates unreliable tracks; MOT17/20 | Selective detection and skipped-frame tracking on the target benchmark family | Strong direct comparator for skipped detection plus motion propagation. The core study should compare against its released method if reproducible with the fixed detector/tracker protocol; otherwise document the adaptation and limitation. The learned risk policy must show value beyond changing propagation mechanics. [Paper](https://www.jstage.jst.go.jp/article/transinf/E105.D/11/E105.D_2022EDP7022/_article) |
| AdaVP (Liu et al., IEEE ICDCS 2020) | Parallel detector/tracker pipeline and runtime detector model-setting adaptation on a mobile device | Adaptive on-device compute and detection/tracking scheduling | Important systems prior art, but its pipeline and objective differ. Do not claim parallel processing or model adaptation as new; include as related work and defer implementation to preserve comparable sequential end-to-end energy and latency measurements. [Paper](https://doi.org/10.1109/ICDCS47774.2020.00085) |
| DeepScale (Nalaie et al., IEEE/ACM IoTDI 2022) | Adapts frame resolution to visual content for MOT and evaluates camera/edge-server partitioning | Directly overlaps with the proposed medium-risk low-resolution detector mode | Variable-resolution MOT is prior art. Keep resolution fixed in this study; any later resolution action needs a separate comparison against DeepScale and matched quality/energy measurement. [Paper](https://doi.org/10.1109/IoTDI54339.2022.00010) |
| Adaptive-EVOD (Chi et al., IEEE TMC 2026) | Edge-assisted video detection adapts resolution, frame rate, model placement/selection, and GPU resources | Adjacent evidence against broad “risk-aware adaptive compute” novelty | It is not the same single-device MOT setting, but the multi-mode scheduler idea is already being explored. A future extension would need a narrowly specified MOT distinction and direct full-text comparison. [Paper](https://doi.org/10.1109/TMC.2026.3719713) |
| Confidence-Guided Frame Skipping (Lee, Sensors 2024) | Confidence of a lightweight tracker triggers a robust tracker; single-object tracking | Confidence-conditioned computation | Adjacent SOT precedent, not a direct MOT baseline. Its existence means confidence-triggered compute cannot be claimed broadly as new; compare an MOT-specific confidence-only policy and explain the task distinction. [Paper](https://doi.org/10.3390/s24248120) |
| Split and Connect tracklet booster (Wang et al., IEEE TMM 2022; preprint 2021) | Temporal model predicts potential ID-switch positions for tracklet splitting and repair | Deep temporal prediction around identity-switch events | It predicts switch locations for tracklet repair; RACE-MOT predicts a future failure probability before the next detector action and uses it for compute scheduling. Both are compared so the distinction stays precise. [Journal paper](https://doi.org/10.1109/TMM.2022.3140919) |
| GRU association / temporal motion predictors (including GAKP and ETTrack) | GRU/TCN/Transformer models learn association or future motion | Supports use of compact temporal DL; demonstrates temporal models in MOT are established | The proposed GRU/TCN/MLP predicts calibrated failure risk for compute allocation; architecture choice alone is not novel. |
| QPilot (PerCom 2026 WIP lead) | RL model switching for object-sparse MOT using compute capacity/scene complexity | Recent adaptive MOT scheduling | Obtain and review the paper/proceedings and implementation. Compare/position relative to model-switching policies before claiming novelty. |
| HSFSO (Han et al., *Information Sciences*, 2026) | MLP surrogate predicts tracking accuracy; hierarchical method selects a global skip ratio and refines frame-level selection with optical flow; evaluated with YOLOv8-small + ByteTrack on MOT17/20 | Learned tracking-quality prediction and adaptive skipping are close prior art | RACE-MOT predicts track-level avoidable identity failure under a particular skip action and calibrates that probability for a next-frame decision. Compare prediction target, temporal unit, calibration, policy timing, and whole-pipeline edge cost. [Paper](https://doi.org/10.1016/j.ins.2026.123610) |
| ALBIREO (accepted ACM/IEEE SEC 2026; 2026 preprint) | Schedules edge detector calls from per-object Kalman state uncertainty, with empty-scene screening and a rescue path | Closest known edge detector-invocation scheduler at the per-object level | Mandatory matched baseline: adapt its uncertainty trigger/rescue idea to the fixed detector and ByteTrack pipeline, label it as an approximation if it differs from the paper, and compare tracking metrics plus full-pipeline cost. Its reported detection AP/energy is not directly comparable to HOTA/IDF1 without a matched MOT implementation. [Preprint](https://arxiv.org/abs/2609.29648), [code](https://github.com/amirtaherin/albireo) |
| GLoMOT (AAAI 2026) and LUKF-Track (*Entropy*, 2026) | GLoMOT performs online graph association at low frame rates; LUKF-Track applies a Lie-group unscented Kalman filter for motion estimation | Low-frame-rate association and nonlinear motion handling | They improve tracking given sparse detections; RACE-MOT decides whether to request the next detection from predicted identity-failure risk. Keep ByteTrack fixed and compare gap-stratified behavior. LUKF-Track is published in *Entropy*, not *Sensors*. [GLoMOT](https://doi.org/10.1609/aaai.v40i6.42500), [LUKF-Track](https://doi.org/10.3390/e28010103) |

### 5.1 Novelty verification matrix

This matrix states the proposed contribution and the distinction supported by the cited methods. “Not reported as the method's objective” means the paper's stated task is different; it does not imply that an unreported experiment was performed.

| Method | Prediction target and timing | Main inputs | Decision unit and action | Identity-failure target / calibration | Detector scheduling | Physical edge evaluation |
|---|---|---|---|---|---|---|
| **RACE-MOT (proposed)** | After frame *t*, estimate probability of an avoidable identity failure over horizon *K* under a defined skip action | Per-track box kinematics, Kalman residual/covariance, detector-confidence and miss history, association margin, track age, actual detector gap; optional low-cost scene cues | Per-track risk, conservatively aggregated to a frame-level DETECT/SKIP decision for *t+1* | Explicit paired skip/detect rollout label; probability calibration by Brier/ECE and reliability plots is planned | Yes, central objective | Planned on Jetson Orin Nano; complete-pipeline energy is not yet measured |
| ALBIREO (SEC 2026) | Per-object 10D Kalman-state uncertainty before detector invocation | Detector boxes and per-object temporal/Kalman state; lightweight empty-scene visual screen | Object-wise uncertainty informs detector invoke/skip; includes empty-scene screening and rescue behavior | Focuses on box/state prediction and detection quality; does not define calibrated future ID-switch probability | Yes | Paper reports edge experiments on Jetson AGX Orin and AGX Thor; its AP/energy results are not directly comparable to MOT HOTA/IDF1 |
| HSFSO (*Information Sciences*, 2026) | MLP estimates tracking accuracy for adaptive skip-ratio control; optical flow refines frame choice | Image/detection features for global ratio estimation and optical-flow motion cues for local selection | Global skip-ratio selection plus frame-level selection | Predicts tracking accuracy, not the proposed per-track avoidable-failure event; probability calibration is not its stated objective | Yes | Article evaluates MOT17/MOT20 with YOLOv8-small + ByteTrack; use matched device testing for energy comparison |
| EMO (2023) | Window/similarity cues and resource-aware optimizations; no calibrated future ID-failure probability stated | Recent frames/features, visual similarity, and tracker-window state used by its optimization variants | Frame/window heuristics trigger selective processing | No action-conditioned per-track failure probability stated | Yes, through heuristic optimizations | Edge measurement study; compare the policy under matched setup |
| RT-MOT (RTSS 2022) | Predicts object confidence/accuracy variation under workload-pair choices for the next frame | Per-task confidence history and candidate detector/association workload pairs, with timing constraints | Task/workload scheduling under timing guarantees | Confidence and tracking-accuracy prediction; not calibrated per-track skip-caused failure | Schedules workload; not the same binary skip policy | Real-time scheduling evaluation; do not infer Orin energy measurements unless directly reported |
| Split and Connect (TMM 2022) | Temporal network locates potential ID-switch positions in tracklets | Historical tracklet trajectory/box sequence and associated appearance features | Offline tracklet split/connect repair | Identity-switch localization, not a prospective calibrated probability for compute allocation | No | Not an edge detector scheduler |
| GLoMOT (AAAI 2026) | Online graph-based association over low-frame-rate gaps | Track/detection graph nodes, temporal buffer, motion/appearance context, and pseudo-depth cues | Track/detection association | Does not state detector-skip-conditioned identity-failure prediction or calibration | No upstream detector scheduler | Tracking evaluation; no physical edge energy claim used here |
| LUKF-Track (*Entropy*, 2026) | Manifold motion-state filtering | Detection geometry and Lie-group motion state | State propagation and association support | Does not state identity-failure prediction or probability calibration | No upstream detector scheduler | Tracking evaluation; no physical edge energy claim used here |

**Claimed novelty statement:** RACE-MOT combines (1) a precisely defined, skip-action-conditioned avoidable identity-failure target, (2) a lightweight temporal per-track predictor whose output is calibrated as a probability, and (3) a next-frame detector policy evaluated on a physical edge device with whole-pipeline energy, latency, tracking-quality, and deadline metrics. The comparison shows that the closest cited methods address complementary parts of this design—uncertainty-triggered detector calls (ALBIREO), aggregate tracking-quality/skip-ratio prediction (HSFSO), confidence/workload scheduling (RT-MOT), heuristic adaptive processing (EMO), post-hoc switch repair (Split and Connect), or sparse-frame association/motion filtering (GLoMOT/LUKF-Track). RACE-MOT's novelty claim is this specific integrated formulation and evaluation, not a claim that each component or the general idea of adaptive compute is new.

**Research gap statement:** Existing work provides adaptive detector scheduling, uncertainty-triggered inference, confidence/workload scheduling, aggregate tracking-quality surrogates, and post-hoc tracklet repair. RACE-MOT addresses the distinct problem of estimating the calibrated probability that a specific track will suffer an avoidable identity failure under a particular upcoming detector-skip action, then using that risk to schedule the next detector call. The product evaluates whether this formulation preserves HOTA/IDF1 while reducing whole-pipeline energy within latency/deadline limits. Scene context and the discovery guard support product robustness; they are not separate novelty claims.

---

## 6. Novelty contribution and supporting opportunities

**Primary novelty claim:** calibrated prediction of per-track, avoidable identity failure under a specified detector-skip action, used to schedule the next detector call. The table below distinguishes this method contribution from supporting product features and optional studies. Adaptive detector skipping, context-aware skipping (EMO), confidence-aware scheduling (RT-MOT), motion-assisted skipping (SDOF-Tracker), and learned skip-ratio/frame selection (HSFSO) are prior art; the novelty lies in the specific failure target, action conditioning, calibration, scheduling use, and edge MOT evaluation.

| Contribution / idea | Feasibility | Research value | Novelty status and required evidence |
|---|---:|---:|---|
| Standardized edge MOT energy/latency/thermal evaluation | High | High | A useful systems contribution if the protocol is reproducible and includes identity metrics. Do not claim “first” without a systematic review. |
| Calibrated per-track failure-risk prediction with compute escalation | Medium | High | **Primary novelty contribution.** Define paired skip/detect rollout labels for avoidable identity failure, train and calibrate per-track risk, and use it to schedule the next detector call. Compare directly with ALBIREO-like uncertainty scheduling, HSFSO, and simple policies to quantify practical value. |
| Scene-discovery guard for newly entering objects and abrupt scene change | High | Possible product value; not a novelty claim | **Test as a measured product safety mechanism.** ALBIREO already uses per-object state, rescue, and an empty-scene screen. Test whether a shared low-resolution frame-difference/activity signal outside padded active-track regions adds value specifically when tracks are already active, against both risk-only and ALBIREO-like scheduling. Measure new-track discovery delay/recall, false-trigger rate, compute overhead, and energy. Freeze thresholds on validation data. Remove it if it adds no reliable benefit or its cost erases savings. |
| Scene-context ablation for temporal failure risk | High | Useful if the cues add measurable value; not independently novel | Compare the same temporal model with (a) track-history features and (b) track history plus current-scene descriptors. Candidate image descriptors: thumbnail luminance/contrast, Laplacian-variance sharpness/blur proxy, frame-difference motion, and active-track count. Keep track-local overlap/crowding in both variants so the ablation isolates added frame context. Count extraction latency/energy; these are proxies, not semantic truth. ObjTrackNet motivates the conditions but does not validate this controller. Compare against EMO/HSFSO and report calibration/frontier changes. [ObjTrackNet](https://link.springer.com/article/10.1007/s10791-026-10143-8) |
| Three-mode controller: skip / reduced-resolution detection / full-resolution detection | Medium-low | Low as a stand-alone novelty claim; substantial extra evaluation burden | **Defer.** Resolution adaptation for MOT is prior art in DeepScale, and Adaptive-EVOD explores adaptive resolution/frame rate and resource allocation for edge video detection. It expands the action-conditioned labels, policy tuning, and energy/quality comparisons. [DeepScale](https://doi.org/10.1109/IoTDI54339.2022.00010), [Adaptive-EVOD](https://doi.org/10.1109/TMC.2026.3719713) |
| Hardware-state-aware compute controller | Medium-low | Potential product value, but not a second novelty claim | **Keep as a conditional product extension.** First measure sustained thermal/clock/queue behavior. Add closed-loop control only if repeated runs show that throttling or deadline misses materially harm the product, and then evaluate it as a separate ablation against a frozen risk policy. Do not change calibrated risk thresholds online or promise fixed temperature/latency improvements without device-specific evidence. |
| Risk-gated optical-flow propagation | Medium | Low as a stand-alone novelty claim | **Defer.** SDOF-Tracker already combines skipped detection and optical-flow interpolation. It would change track propagation and confound attribution to the risk policy. |
| Auditable decision evidence | High | Supporting value | Log model inputs, calibrated risk, threshold, action, and outcome for audit/debugging. This is good practice, not a separate XAI novelty claim. |
| Triggered lightweight appearance feature | Medium | Medium | Optional extension; compare its accuracy and energy cost against motion-only baselines. Not novel by itself. |
| Quantization impact on identity continuity | Medium | Medium | Useful controlled ablation; INT8 conversion alone is not novelty. The cited MobileNetV2/SSD-Lite quantization paper is detector-only and cannot support an MOT identity claim. [Wang, 2021](https://doi.org/10.1088/1742-6596/1748/3/032055) |
| Small edge-condition MOT stress set | Low-medium | Potentially useful validation resource; high data/annotation burden | **Defer.** Brightness/contrast, blur, compression, and frame-drop augmentation of MOT17/20 is not a new dataset. A captured set needs identity-consistent annotations, camera/condition metadata, permissions, quality checks, release terms, and enough varied sequences to support claims. Keep the current benchmark protocol; consider a small supplemental set only if a documented coverage gap remains. |

Avoid presenting YOLO on Jetson, YOLO plus DeepSORT/ByteTrack, generic scene features, variable resolution, optical flow, hardware telemetry, basic INT8 conversion, pedestrian counting, or a detector-only FPS table as novelty.

---

## 7. Project directions and scope

Scoring uses a 1-5 scale and is provisional. It assumes a 6-9 month undergraduate project; the novelty score reflects the specificity of the claim, while experiments establish measured value.

| Rank | Direction | Novelty | Feasibility | Dataset availability | Compute requirement | Evaluation potential | Placement value | Recommendation |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **Calibrated track-failure risk + binary compute policy, with optional scene-context ablation** | 4 (focused contribution; evidence pending) | 3.5 | 4 | 3.5 | 5 | 4.5 | Recommended core: action-conditioned failure target, calibrated per-track risk, detector scheduling, and measured edge objective; audit logging supports verification, and scene context is an ablation |
| 2 | **Reproducible single-device edge MOT energy/thermal benchmark** | 2.5 | 4.5 | 5 | 4 | 4.5 | 4.5 | Safest fallback contribution; publish a rigorous protocol and honest Pareto results |
| 3 | **Triggered tiny Re-ID for occlusion recovery** | 3 (conditional) | 3 | 4 | 3 | 4 | 4 | Optional extension with data and training risk; appearance cues are not new |
| 4 | **NMS-free YOLOv10 edge MOT** | 2 | 4 | 5 | 4 | 3.5 | 4.5 | Good engineering comparison; weak novelty unless device measurements establish a new practical result |
| 5 | **UAV edge MOT on VisDrone** | 3 (conditional) | 3 | 4 | 3 | 4 | 4.5 | Distinct application but domain-specific and compute-heavy; not the recommended core |

### Recommended baseline stack

- One frozen lightweight detector supported by the physically available device; do not compare multiple detector families in the core experiment.
- ByteTrack as the primary tracker; use the same version and parameters for every policy.
- MOT17 grouped sequence folds for training/model selection; MOT20 frozen as stress/generalization evaluation.
- GRU, causal TCN, and temporal MLP are the learned predictor comparison, not separate detector/tracker systems.
- For the same temporal architecture, compare track-history-only inputs against track history plus low-cost scene-quality/activity cues; include feature extraction overhead in the policy cost.
- Core metrics: HOTA, IDF1, MOTA, ID switches, fragmentation, new-track discovery delay/recall, guard false-trigger rate, full-pipeline joules/input-frame, p50/p95 latency, deadline miss rate, RAM, and temperature. Report identity/calibration/policy behavior by actual detector-gap bin.

### Recommended project combination

Combine a focused version of the measurement study with risk prediction:

> **Measure an every-frame edge MOT baseline, then test whether calibrated prediction of near-future track failure can select compute actions more efficiently than periodic skipping, confidence-only rules, per-object Kalman-uncertainty scheduling (including an ALBIREO-like baseline), HSFSO, and other closest reproducible prior methods. Separately test whether scene context improves prediction and whether a scene-discovery guard improves new-track discovery enough to justify its cost.**

The benchmark establishes a trustworthy baseline. The predictor is a contribution only if it adds measurable value. If it does not, report the negative result and retain the reproducible measurement study.

## 8. Proposed project: RACE-MOT

### Title

**RACE-MOT: Risk-Calibrated Adaptive Compute for Multi-Object Tracking on Edge Devices**

### Research objective

For fixed-camera pedestrian flow monitoring, determine whether a lightweight temporal model can predict an avoidable future identity failure early enough to decide whether to invoke a single detector on the next incoming frame. Test, as a controlled input ablation, whether low-cost scene-quality/activity context improves prediction beyond per-track history alone. Evaluate whether the policy reduces complete-pipeline energy while meeting MOT-quality and real-time constraints. Record auditable decision evidence. Use one detector, one primary tracker, one target device, and two datasets.

### Formal problem

Given a video stream $V = \{I_t\}_{t=1}^{T}$ and an edge device with power, memory, and latency budgets, estimate from information available at time $t$ whether skipping detector inference on the next frame would cause an avoidable identity failure within the next $K$ frames. Choose one of two actions: invoke the single detector now, or skip it and propagate active tracks. Enforce a maximum consecutive-skip count. Minimize measured energy subject to predeclared HOTA, IDF1, latency, deadline-miss, and memory constraints. No future frame or test annotation may be used by the live policy.

Use a constrained objective rather than selecting arbitrary weights for a combined score:

$$
\min_{\pi}\quad E_{\mathrm{total/frame}}(\pi)
$$

subject to:

$$
\begin{aligned}
\mathrm{HOTA}(\pi) &\ge \mathrm{HOTA}_{\mathrm{base}}-\epsilon_H,\\
\mathrm{IDF1}(\pi) &\ge \mathrm{IDF1}_{\mathrm{base}}-\epsilon_I,\\
L_{p95}(\pi) &\le D,\\
\mathrm{DMR}(\pi) &\le \rho,\\
\mathrm{RAM}_{peak}(\pi) &\le B,\\
T_{peak}(\pi) &\le T_{budget}.
\end{aligned}
$$

Here $\pi$ is the policy, $E_{\mathrm{total/frame}}$ includes detector, temporal predictor, tracker, preprocessing, decoding, and other required pipeline work divided by **all input frames**, including skipped-detection frames. The reference is the same detector/tracker running every frame on the same device. $\epsilon_H$ and $\epsilon_I$ are pre-registered non-inferiority margins; $D$ is the input-stream frame deadline, $\mathrm{DMR}$ is deadline-miss rate, $\rho$ is its allowed bound, $B$ is the measured/declared memory budget, and $T_{budget}$ is a sustained operating-temperature limit justified by the device specification and application. Choose constraints from pilot repeatability and application requirements, freeze them before final test, and report if no policy satisfies them. This formulation minimizes energy while preserving HOTA/IDF1 and service limits; the accuracy–energy Pareto frontier remains the comparison tool.

### Proposed pipeline

```text
Frame_t -> online track state/history + low-cost scene-context descriptors
                                      -> GRU / TCN / temporal-MLP risk model
                                                |
                                      frame-level risk aggregation
                                                |
                      risk >= threshold OR skip limit reached?
                               yes -> detector -> ByteTrack
                               no  -> skip detector -> propagate tracks
                                                |
       tracks + auditable decision evidence + full pipeline timing/energy log
```

### Risk target and predictor

Define failure before training. At frame $t$, match ground-truth identity $g$ to an active predicted track using a fixed one-to-one IoU assignment with threshold $\alpha$. Retain only anchors with a valid match at $t$, and let $p_g(t)$ be the anchored tracker ID. For each anchor, fork the same tracker state into two training-only counterfactual rollouts: $a=\text{skip}$ omits the detector at frame $t+1$; $a=\text{detect}$ runs it at frame $t+1$. Both branches then use detector-every-frame updates for the next $K-1$ frames. At each future frame $u$, let $z_g^a(u)$ be the ground-truth identity assigned by the fixed one-to-one IoU matcher to anchored predicted ID $p_g(t)$; set $z_g^a(u)=\bot$ if that predicted ID is absent or unmatched. Evaluate failures only on frames where $g$ is annotated visible. Define branch failure:

$$
F_{g,t}^{a,(K,M)}=\mathbb{1}\left[\begin{array}{l}
\exists u\in\{t+1,\ldots,t+K\}: g\text{ is visible and }z_g^a(u)\notin\{g,\bot\},\\
\text{or }\exists u:\forall v\in\{u,\ldots,u+M-1\},\ g\text{ is visible at }v\text{ and }z_g^a(v)\ne g
\end{array}\right],\qquad a\in\{\text{skip},\text{detect}\}.
$$

The actionable label is an **avoidable failure due to skipping**:

$$
Y_{g,t}^{(K,M)}=F_{g,t}^{\text{skip},(K,M)}\left(1-F_{g,t}^{\text{detect},(K,M)}\right).
$$

This distinguishes failures a detector call could plausibly avert from failures that occur in both branches. Fix $K$, $M$, $\alpha$, matching, visibility/occlusion rules, and branch rollout procedure before training; publish label-generation pseudocode. Train the predictor to estimate $q_{g,t}=P(Y_{g,t}^{(K,M)}=1\mid x_{g,t-L+1:t})$, where the input history contains only information available online through time $t$. Counterfactual futures and ground truth are used only to create offline training labels, never as policy inputs. Include diverse prior skip counts in training rollouts so the predictor sees states encountered by the adaptive policy. Avoid neighboring-frame leakage through sequence-level splits.

Use a temporal deep-learning model to implement the primary novelty contribution. The working implementation name is T-RiskNet; the claim is defined by the prediction target and scheduling policy, not the model name. Here, causal means that runtime features use only current and past observations; it does not mean the offline paired rollout establishes a causal effect in every deployment. Use a small causal 1D TCN as the primary model and compare it with a compact GRU and temporal MLP using identical labels and sequence folds. The base input is normalized track history: box center/scale and velocity per elapsed time; Kalman residual/covariance; confidence history; missed-frame count; actual elapsed seconds and source-frame count since the last detector update; consecutive detector skips; track age; association-cost margin; and track-local overlap/crowding. Preserve source timestamps and report calibration and policy quality in predeclared detector-gap bins. This is gap-aware risk prediction, not a replacement motion model.

For the secondary fusion ablation, first select/confirm the primary TCN against GRU and temporal MLP using track-history-only input. Then keep that TCN fixed and compare three matched variants: track-history-only; the same TCN with low-cost scene descriptors concatenated; and the same TCN with a small shared scene-context encoder and gated per-track fusion. The context branch may use thumbnail luminance/contrast, Laplacian-variance sharpness/blur proxy, mean absolute frame difference, and active-track count. Compute the thumbnail/frame-difference once and share it with the optional model context and scene-discovery guard; do not duplicate this work. Treat these measurements as proxies, not semantic explanations. Do not run optical flow in the core. Measure feature extraction, encoder, fusion, predictor latency, memory, and energy on the target board; retain the more complex fusion only if held-out calibration and policy results justify its overhead. There is no assumed sub-millisecond latency or fixed parameter-count guarantee.

In the gated variant, the asymmetry is computational: a small MLP encodes the frame-level descriptor vector once as $c_t$, while the TCN produces a per-track state $h_{i,t}$. A candidate gated residual is $g_{i,t}=\sigma(W_g[h_{i,t}\Vert c_t]+b_g)$ and $\tilde h_{i,t}=h_{i,t}+g_{i,t}\odot W_c c_t$, followed by the risk head. This is a concrete ablation specification, not a novel operator claim. The shared encoder cost is amortized across active tracks, while per-track gate/head cost still grows with track count; measure both.

Compare learned models with logistic regression/static MLP, confidence-only, uncertainty-only, and the closest reproducible learned scheduling baseline. Use training-only class weighting or negative sampling for rare labels, then calibrate on natural-prevalence calibration sequences. Fit the predictor only on training sequences; fit calibration maps on separate calibration sequences; select policy thresholds on separate policy-validation sequences. Use nested/grouped cross-fitting when the small number of MOT17 source sequences prevents a stable three-way split. Report PR-AUC, operating-point precision/recall, Brier score, ECE with stated binning, reliability plots, parameter count, predictor/fusion latency, memory, and energy. Calibration is an empirical property to test, not a guarantee: a predicted risk near 0.70 should be assessed against observed failure frequency on appropriately sized held-out groups with confidence intervals. The learned method must beat simple risk baselines and the closest reproducible learned scheduling baseline on held-out sequences to justify its cost.

### Exact frame-level compute policy

Run detection on the first frame to initialize tracks. Thereafter, for each active track $i$, infer avoidable-failure score $q_{i,t}$. Aggregate conservatively as $Q_t=\max_i q_{i,t}$; do not assume track failures are independent. Fit a second calibration map to this aggregate on calibration sequences with frame-level labels $Y_t=\max_i Y_{i,t}$, producing calibrated frame risk $\widetilde Q_t$. After processing frame $t$, plan DETECT for the next frame if $\widetilde Q_t\ge\tau$, no tracks are active, or the consecutive detector-skip count has reached fixed maximum $S_{max}$; otherwise plan SKIP. When the next frame arrives, a separate low-resolution scene-discovery guard evaluates frame difference/activity outside padded active-track regions and can upgrade a planned SKIP to full-frame DETECT. It may never downgrade a planned DETECT. Freeze guard thresholds on policy-validation sequences. Select $\tau$ and $S_{max}$ to minimize **measured complete-pipeline energy** subject to validation HOTA, IDF1, latency, deadline-miss, and sustained-temperature constraints, including the guard's cost. Report guard false-trigger rate and new-track discovery delay/recall. Keep the maximum-skip bound even when the guard is inactive. Log the trigger for every detector call; the guard protects new-object discovery, which per-track failure scores cannot model while an object has no track.

### Auditable decision evidence (initial scope)

Action records must distinguish the risk scheduler's **planned** action from the **executed** action after a possible scene-guard override. Include actual elapsed/source-frame detector gap, consecutive skip count, guard trigger/activity score, and per-stage timings. [ALBIREO](https://arxiv.org/abs/2609.29648) already includes an empty-scene screen and rescue behavior; compare the guard's incremental value during scenes with active tracks against that matched baseline as well as the risk-only policy.

For every frame-level action, log planned and executed action, maximum-risk temporary track ID, risk score and calibration-validity status, frame-level risk, threshold, explicit trigger/reason code, actual elapsed/source-frame detector gap, consecutive-skip count, scene-guard activity/override, model/configuration hashes, and component timings. Log only the feature values needed for audit/debugging and privacy policy. Call this **auditable decision evidence**, not a novel XAI method. A logged feature is not proof that it caused the action. Use controlled feature ablations or matched-state counterfactual checks to test whether removing a cue changes risk/action and whether the change helps the measured accuracy–energy frontier; these checks do not prove causal effects. Evaluate calibration on natural-prevalence held-out sequences with Brier/ECE, reliability plots, and sequence-aware confidence intervals. A predicted 0.70 is not guaranteed to mean a 70% event rate; test that relationship empirically on a sufficiently large, relevant held-out group. A post-hoc heatmap is outside the core contribution.

### Assessment of the additional novelty proposals

A newly supplied three-part proposal describes a label formulation, a fused temporal predictor, and a physical edge framework as three claimable novelties. This report adopts one integrated primary method claim: calibrated, action-conditioned per-track identity-failure prediction for detector scheduling. Model fusion is a secondary ablation; physical implementation and audit trail provide product and evaluation evidence. The terminology below keeps the claim specific and testable.

| Suggested addition | Decision | Reason and treatment in this study |
|---|---|---|
| Counterfactual skip-conditioned avoidable-failure target and paired-rollout generator | **Retain as the foundation of the primary novelty claim** | The label asks whether a specified next-frame skip branch fails while the matched detect branch does not, under a frozen offline rollout protocol. The generator is a reproducibility artifact, not a new dataset or causal proof about all real deployments. Compare the target, decision unit, horizon, policy use, and calibration directly with HSFSO and temporal ID-switch prediction in Split and Connect. [HSFSO](https://doi.org/10.1016/j.ins.2026.123610), [Split and Connect](https://doi.org/10.1109/TMM.2022.3140919) |
| Asymmetric kinematic/context fusion for the temporal predictor | **Test only as a secondary architecture ablation; not an independent novelty claim** | Compare track-history-only TCN, the same model with simple context concatenation, and a shared low-cost scene-context encoder whose embedding is gated into each track representation. Compute the shared context once per frame and reuse it for active tracks. Keep labels, folds, predictor family, and policy fixed; measure parameters, p50/p95 model time, memory, energy, calibration, and policy value on Orin Nano. TCNs, gating, scene cues, and context-aware scheduling are established ingredients; novelty requires evidence that this particular design adds held-out value beyond its cost. No fixed sub-millisecond or parameter budget is assumed. [EMO](https://arxiv.org/abs/2309.02666), [HSFSO](https://doi.org/10.1016/j.ins.2026.123610), [Split and Connect](https://doi.org/10.1109/TMM.2022.3140919) |
| Physical edge deployment, decision trace, and energy/thermal report | **Adopt as product deliverable and reproducible systems evidence, not a standalone method novelty** | Run on the available Orin Nano and private-network phone stream; log action, reason code, maximum-risk temporary track, calibrated risk/status, threshold, skip count, model/config hashes, and measured runtime values. A logged feature is evidence available to the policy, not proof that it caused the decision; use matched feature ablations for that question. A local-only design does not by itself prove zero network leakage. Measure energy at a declared Jetson power-input boundary with an external meter as the primary measure when available; use onboard telemetry as diagnostic/cross-check only, report sustained temperature/throttling, and state that camera/router power is excluded. Thermal feedback control remains out of scope. |
| Re-identification compute/battery argument | **Reject from the current rationale** | The core uses ByteTrack and contains no Re-ID branch. Do not claim that this project avoids the cost of a heavy Re-ID CNN or that Re-ID causes a specific memory/battery penalty. Occlusion can be evaluated as a condition, but it does not imply a Re-ID contribution. |
| Name the temporal module T-RiskNet; use normalized velocity, residual, confidence, age, and history features | **Adopt with qualification** | Use T-RiskNet as a working implementation name. The model is the temporal risk predictor already in scope; the name, TCN/GRU architecture, and feature list are not independent novelty claims. |
| Condition failure-risk prediction on actual detector gap | **Adopt as a required feature/schema improvement, not an independent novelty claim** | Store elapsed wall-clock time, source-frame gap, and consecutive detector skips separately. This makes scheduled detector omission distinct from dropped/late input frames and permits calibration/policy reporting by gap. GLoMOT and LUKF-Track establish low-frame-rate motion as prior art; keep ByteTrack's motion filter fixed for this study. [GLoMOT](https://doi.org/10.1609/aaai.v40i6.42500), [LUKF-Track](https://doi.org/10.3390/e28010103) |
| Low-cost scene-discovery guard for untracked activity | **Test as a product safety mechanism; not a novelty claim** | A current-frame downsampled difference/activity test outside padded active-track boxes may upgrade SKIP to full-frame DETECT. Its possible incremental value is during active scenes with untracked motion; [ALBIREO](https://arxiv.org/abs/2609.29648) already includes an empty-scene screen and rescue path. Compare against both risk-only and an ALBIREO-like policy, and measure discovery delay/recall, false triggers, latency, and energy. Remove the guard if it adds no reliable benefit or its cost erases savings. This is a design inference, not an established result. |
| Add scene-quality/activity cues such as luminance, contrast, blur proxy, frame difference, and density | **Adopt as a paired ablation, not as an independent novelty claim** | Compare identical temporal models with track-only versus track-plus-scene inputs; measure feature cost and calibration/frontier changes. ObjTrackNet motivates these stress conditions and suggests future extensions, but does not test this risk policy. Context-aware skipping is already prior art in EMO/HSFSO. [ObjTrackNet](https://link.springer.com/article/10.1007/s10791-026-10143-8) |
| Expand the action set to skip / low-resolution detection / full-resolution detection | **Defer** | DeepScale directly studies adaptive frame sizes for MOT; Adaptive-EVOD adapts resolution/frame rate/resources for edge video detection. This would change action-conditioned labels, controller selection, baselines, and energy attribution. Preserve the binary action space until the core risk study is complete. [DeepScale](https://doi.org/10.1109/IoTDI54339.2022.00010), [Adaptive-EVOD](https://doi.org/10.1109/TMC.2026.3719713) |
| Use frame-difference or optical-flow motion to propagate tracks during skips | **Frame difference may be an input proxy; optical-flow propagation is deferred** | A scalar frame-difference statistic can be included in the scene-context ablation with full cost accounting. Optical-flow detector-skip propagation is already studied by SDOF-Tracker and would alter the tracker behavior. [SDOF-Tracker](https://doi.org/10.1587/transinf.2022EDP7022) |
| Joint identity-quality, energy, delay, memory, and thermal formulation | **Adopt** | Keep the constrained energy-minimization objective; add a sustained temperature bound and measure throttling. Compare the feasible accuracy–energy Pareto frontier rather than optimizing FPS alone. |
| Synthetic edge-artifact degradation mining | **Adopt as augmentation/robustness ablation** | Train with controlled blur, compression, and input-frame-drop corruption on training sequences, and test predeclared corruption levels separately on held-out sequences. This can improve robustness evidence but is not a novelty claim by itself. Keep ingress frame loss distinct from the controller's detector skip action. |
| Sequence-aware hard-negative mining for a conditional Re-ID branch | **Defer** | There is no Re-ID branch in the core project. For rare failure labels, use class weighting or training-fold negative sampling and calibrate on natural-prevalence sequences. |
| Conditional cross-attention Re-ID during crossovers | **Defer as future work** | Adds a second learned module, appearance training, extra latency/energy, and another failure mode. It would create a second contribution and obscure whether the risk policy itself helps. |
| Low-resolution detector plus spatial high-resolution crop router | **Defer as future work** | Adds a second detector path and changes the action space from detector invocation to spatial routing. It is a plausible follow-on after the two-action risk policy has been validated and measured. |
| FP32/FP16/INT8 identity-aware quantization | **Adopt as final ablation if supported by the device** | Keep one detector architecture and tracker; compare supported precisions for detector and temporal predictor, recalibrate each precision on calibration sequences, and report detection AP/recall, HOTA, IDF1, ID switches, fragmentation, ECE, complete energy, and latency. Quantization is not the main novelty. The cited MobileNetV2/SSD-Lite paper is detector-only, so it does not support identity-quality claims. [Wang et al., 2021](https://doi.org/10.1088/1742-6596/1748/3/032055) |
| Let temperature/GPU utilization change the controller mode | **Defer** | Retain device telemetry and thermal limits in evaluation. A hardware-state controller adds a second policy signal and needs its own baselines and stability analysis; adaptive resource/model scheduling is established in related edge-video systems. [AdaVP](https://doi.org/10.1109/ICDCS47774.2020.00085), [Adaptive-EVOD](https://doi.org/10.1109/TMC.2026.3719713) |
| Create a 5–10 minute edge-camera stress set | **Defer; do not claim a dataset contribution yet** | Synthetic illumination/contrast/blur/compression perturbations can test robustness but do not create a dataset. Captured footage would require identity-consistent annotations, permissions, condition/device metadata, quality control, and enough sequences for a useful held-out evaluation. |
| Feature/action logging and ECE/counterfactual checks | **Adopt; do not market as a new XAI method** | Strengthens auditability and validates risk reliability; report confidence intervals and natural-prevalence calibration results. |
| Optical-flow propagation during detector skips (SDOF-Tracker) | **Add as explicit prior art; defer as a core implementation** | SDOF-Tracker already combines skipped human detection with optical-flow interpolation and evaluates on MOT17/MOT20. It is a relevant comparator, not a novelty contribution. First reproduce the agreed ByteTrack baseline and risk-policy comparisons; if time permits, evaluate a faithful SDOF-style propagation variant under matched detector, input, and hardware conditions. Do not silently combine its propagation with the proposed policy and attribute all gains to risk prediction. [Nishimura et al., 2022](https://doi.org/10.1587/transinf.2022EDP7022) |
| Parallel detector/tracker execution or content-adaptive detector configuration (AdaVP) | **Cite as prior art; defer** | AdaVP (Liu et al., IEEE ICDCS 2020) already presents an on-device parallel detection/tracking pipeline and adapts detector settings to video change. Its system is not identical to this MOT study, but it invalidates broad claims to these system ideas. Parallel execution changes the scheduler and power overlap; adding it now would make the central policy comparison harder to interpret. [Liu et al., 2020](https://doi.org/10.1109/ICDCS47774.2020.00085) |
| Confidence-gated escalation from a light tracker (Lee, Sensors 2024) | **Cite as adjacent work; keep confidence-only MOT baseline** | Lee studies single-object tracking, with confidence deciding when a robust tracker is invoked. It is not a direct multi-object detector-invocation baseline, but it is relevant evidence that confidence-gated compute is established in tracking. Keep the proposed baseline MOT-specific and avoid claiming the general idea is new. [Lee, 2024](https://doi.org/10.3390/s24248120) |
| Dynamic spatial routing / low-resolution plus targeted high-resolution crops | **Defer as future work** | Plausible conditional-compute direction, but it changes the action space, detector configuration, crop logic, and measurement protocol. AdaVP supports adaptive detector settings as prior art, not this exact spatial-routing design. It would require a separate hypothesis and controlled ablation; it is not needed to test the current temporal-risk claim. |
| A new edge MOT dataset or ROI-filtered dataset | **Do not claim from preprocessing alone** | Cropping, resizing, synthetic degradation, or filtering MOT17/20 does not create an original dataset. A dataset contribution would require permitted collection, documented capture conditions, identity-consistent annotations, quality control, release/usage terms, and genuinely useful edge measurements (e.g. synchronized power/latency/thermal traces). Keep local data and ROI filtering out of the current core scope. |
| Split-and-connect tracklet repair | **Use as a novelty boundary, not an added module** | The IEEE TMM paper already predicts potential switch locations with temporal convolutions and repairs tracklets on MOT17/20. Its post-tracking repair objective differs from the proposed online skip-action-conditioned detector policy, but makes an unqualified “temporal ID-switch prediction is new” claim indefensible. [Wang et al., 2022](https://doi.org/10.1109/TMM.2022.3140919) |

### Baselines that must be compared

| Baseline | Why it is required |
|---|---|
| Detector every frame + ByteTrack | Quality and resource reference |
| Fixed detector intervals | Tests simple periodic skipping |
| Confidence-only threshold | Tests if a learned temporal risk model is needed |
| Kalman-uncertainty-only threshold | Tests value beyond motion uncertainty |
| Static logistic regression / MLP risk model | Separates temporal modeling value from learned but non-temporal risk scoring |
| EMO-like policy | Direct comparison with the closest reproducible context/similarity skipping strategy |
| RT-MOT-like confidence/workload strategy | Direct comparison with confidence-aware scheduling; match available actions and clearly document approximation |
| HSFSO or faithful reproduction | Required closest learned adaptive-skipping comparator: scene/sequence-level tracking-accuracy surrogate plus adaptive skip ratio and optical-flow frame selection. Match data, detector/tracker where feasible, and energy boundary; document any reproduction differences. |
| ALBIREO-like per-object uncertainty scheduler | Required direct detector-invocation comparator. Adapt its per-object Kalman-uncertainty trigger (and rescue/empty-scene logic where faithfully reproducible) to the fixed detector+ByteTrack path; name deviations as an approximation and compare HOTA/IDF1, ID switches, energy, latency, and deadlines. |
| Risk-only versus risk + scene-discovery guard | Product safety ablation. Also compare against the ALBIREO-like baseline with its empty-scene/rescue behavior where reproduced. Measure new-track discovery delay/recall and false detector-trigger rate, along with whole-pipeline energy and standard MOT metrics; the guard can only upgrade SKIP to DETECT. |
| SDOF-Tracker or a faithful skip-plus-optical-flow comparator | Direct prior art on MOT17/20 that combines skipped detections and optical-flow propagation. Attempt a matched-device comparison if its code can be adapted without changing the main ByteTrack experiments; otherwise document the reproduction barrier and compare its published protocol/results cautiously. |
| GRU, causal TCN, temporal MLP | Main temporal predictor comparison, with identical labels/features/splits |
| Proposed calibrated temporal-risk policy | Candidate method; must improve the measured Pareto frontier at comparable quality/deadline constraints |

### Measurable constraints

| Metric | Target or protocol | Reason |
|---|---|---|
| Tracking quality | Report HOTA, DetA/AssA, IDF1, MOTA, ID switches, fragmentation; define non-inferiority margin before final test | Identity quality matters more than detector AP alone |
| New-object discovery | Report new-track discovery delay/recall and scene-guard false-trigger rate for risk-only vs guard-enabled policy | Tests whether the product guard compensates for the per-track predictor's inability to score not-yet-tracked objects |
| Energy | Joules per input frame and average/idle power; energy reduction is an outcome, not a promised target | Enables a fair resource comparison |
| Runtime | End-to-end p50/p95 latency, sustained FPS, dropped frames | Controls tail behavior rather than average only |
| Deadline miss rate | $\mathrm{DMR}=\frac{1}{T}\sum_{t=1}^{T}\mathbb{1}[L_t>D]$, with $D$ set from the actual input frame period/application requirement | Distinguishes real-time service from offline throughput |
| Memory/thermal | Peak RAM; sustained peak temperature and throttling; enforce device-justified $T_{budget}$ | Makes deployment conditions reproducible and enforces the thermal constraint |
| Batch size | 1, fixed stream | Prevents offline batching from hiding deployment cost |
| Repetition | Repeated runtime trials; sequence-level paired comparisons and confidence intervals where appropriate | Avoid treating correlated frames as independent samples |

**Core scope:** one detector, one primary tracker (ByteTrack), one physically available edge device, MOT17 as the main dataset, and MOT20 as held-out crowd/domain stress data. One temporal risk-prediction novelty. Quantization, second devices, local data, Re-ID, and detector architecture changes are outside the core contribution.

### Expected contributions

1. A temporal GRU/TCN/small-MLP comparison for predicting a mathematically defined future identity-failure event, with actual detector-gap features and a paired test of whether low-cost scene context helps beyond track history.
2. A calibrated, thresholded detector-invocation policy evaluated against ALBIREO-like uncertainty scheduling, HSFSO, EMO/RT-MOT-like baselines, and simple rules, if novelty survives the prior-art gate.
3. A one-device accuracy–energy Pareto evaluation that includes the predictor/controller and reports deadline behavior.
4. Auditable decision traces, reproducible configs, and an honest analysis of regimes where prediction does not help.

---

## 9. Dataset and split protocol

| Dataset | Content and use | Suitability and caution |
|---|---|---|
| [MOT17](https://motchallenge.net/data/MOT17/) | Urban pedestrian sequences with identities | Primary benchmark; use official protocol and distinguish public detections from detector-generated results |
| [MOT20](https://motchallenge.net/data/MOT20/) | Very crowded pedestrian scenes | Held-out stress/generalization set; do not use it to tune the risk threshold if reporting transfer from MOT17 |

### Split and leakage protocol

- Use **sequence-level** splits throughout. Never randomly split frames or tracklets from the same sequence across train/validation/test.
- MOT17 has repeated versions of the same source videos with different detector outputs. Group every detector version of a source sequence into the same fold to prevent scene leakage.
- Use outer grouped sequence folds within annotated MOT17 training data for test estimates. Within each outer fold, keep predictor fitting, probability calibration, and policy-threshold selection on distinct sequences using nested/grouped cross-fitting. The limited number of independent sequences means uncertainty intervals may be wide; report that limitation plainly.
- Freeze model, calibration method, $K$, $M$, $\tau$, and $S_{max}$ before evaluating MOT20. Treat MOT20 as an out-of-domain crowd stress test, not another tuning set.
- Respect dataset terms and cite the official challenge. Report which detection source was used and whether the proposed detector generated detections itself.

### Controlled edge-artifact degradation

As training augmentation and robustness ablations—not separate novelty claims—apply controlled low illumination/contrast, motion blur, video compression, and timestamp-preserving input-frame loss to training sequences. Generate paired skip/detect counterfactual labels under the same corruption so the risk target reflects degraded observations. Keep synthetic ingress-frame loss distinct from the learned policy's detector skipping. Choose corruption ranges from a small pilot or measured deployment conditions, freeze them before held-out evaluation, and report clean MOT17/MOT20 separately from corrupted stress results. Compare risk-model training with and without augmentation; evaluate corrupted held-out sequences at fixed levels without tuning on them. This does not create a new dataset; ObjTrackNet's discussion motivates checking these conditions but does not establish a benefit from this augmentation or policy. [ObjTrackNet](https://link.springer.com/article/10.1007/s10791-026-10143-8)

---

## 10. Edge platform and measurement

At least one **physical edge device is mandatory**; laptop/server-only or simulated timings do not support an edge claim. Select the board actually available before fixing latency or energy targets. The list below contains options, not a requirement to buy or test every device.

| Device | Role | Measurement notes |
|---|---|---|
| Jetson Orin Nano (exact RAM/SKU to inventory) | Sole target device; already available | Record module/carrier identity, JetPack, power mode, clocks, CUDA/TensorRT, cooling, temperature, and throttling. Do not infer performance from TOPS. |
| External meter at Jetson power input; onboard telemetry as diagnostic | Energy measurement for the same physical device | Declare the measured boundary and sample rate; report meter energy as the primary result when available and keep rail telemetry separate as a cross-check. |

Minimum hardware report: exact device/module RAM and carrier board, OS/runtime/driver version, power mode, clocks, input size, precision, batch size, warm-up frames, ambient temperature, fan/enclosure condition, sustained-run duration, rendering/decoding inclusion, and whether power is board-only or whole-system. NVIDIA currently lists multiple Orin Nano RAM/power configurations and multiple JetPack releases; do not assume the installed board is the 8 GB or Super configuration. Record and preserve the installed image for the initial feasibility check. If a reflash is needed, use an official release that supports Orin Nano and verify model/runtime compatibility first. [NVIDIA Jetson Orin Nano module specifications](https://developer.nvidia.com/embedded/jetson-modules), [JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads), [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive), [NVIDIA power modes](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)

Do not compare Jetson, Raspberry Pi, and desktop FPS without these conditions. Do not infer joules from FLOPs.

### Full-pipeline energy accounting

Measure energy from source-frame arrival through emitted tracks for the complete Jetson pipeline, including decode, preprocessing, detector, scene-activity/discovery guard, scene feature extraction/context encoder/fusion, risk predictor, calibration/policy, tracker, and required dashboard/logging. Use an external meter at the Jetson power input as the primary whole-device measure when available; state exactly which board, carrier, fan, and peripherals lie inside that boundary. Use onboard power telemetry as a diagnostic or cross-check and report its rail/sampling limits separately rather than combining readings into a single purportedly exact value. Divide integrated energy by **all input frames**, including skipped detector frames. The phone, Wi-Fi access point, and browser are outside the Jetson energy boundary unless separately metered. Include warm-up, meter/telemetry sampling, idle baseline, ambient/cooling conditions, run duration, repeated runs, temperature, clocks, and throttling. Predictor and scene-guard costs are part of the method/product cost; thermal measurements are evidence about sustained operation, not thermal optimization.

---

## 11. Experimental protocol

Run the experiments in this order. Freeze detector, tracker, preprocessing, stream rate, device settings, and measurement boundary before comparison.

1. **Baseline:** run the detector on every frame with ByteTrack and record tracking quality, new-track discovery latency, complete-pipeline energy, latency, deadline misses, RAM, and temperature.
2. **Simple policies:** test fixed skipping, detector-confidence thresholding, and Kalman-uncertainty thresholding. Tune thresholds on validation sequences only.
3. **Prior-work policies:** implement an ALBIREO-like per-object uncertainty scheduler, then reproduce or implement HSFSO, EMO-like, and RT-MOT-like policies under the same detector/tracker and measurement setup. If exact reproduction is infeasible, document deviations and call the result an approximation; do not claim a win over a paper based on mismatched published tables.
4. **Temporal DL predictor:** compare GRU, causal TCN, and small temporal MLP on identical labels and grouped sequence folds using track-history-only input. Then keep the primary TCN fixed for context concatenation and, if the base pipeline is feasible, shared scene-context gated fusion. Compare with logistic regression/static MLP and single-cue rules; include all context/fusion costs in end-to-end measurement.
5. **Calibrated policy:** fit probability calibration on calibration sequences; select risk threshold $\tau$ and skip limit $S_{max}$ on separate policy-validation sequences under the constrained objective; freeze all choices before outer held-out evaluation.
6. **Ablations:** compare risk-only scheduling with risk plus scene-discovery guard; report new-track discovery delay/recall, false triggers, and energy. Also test track versus scene-context feature groups, history length, prediction horizon, calibration, aggregation, and predictor/controller overhead. Do not add low-resolution patch actions or hardware-state-driven threshold changes to this core experiment.
7. **Stress test:** evaluate the frozen method on MOT20 and predeclared crowd/visibility strata; add controlled source-frame subsampling with timestamps preserved and report every metric/calibration result by actual gap. Independently test motion blur and compression. Do not conflate image blur, lower source-frame cadence, input-frame loss, and the controller's detector-skip action. Do not tune on MOT20 or corrupted held-out data. Compare predictor training with and without training-only degradation augmentation; report these secondary conditions separately from the full-rate primary result.
8. **Quantization last:** if full precision is stable and supported by the device, compare FP32/FP16/INT8 for the same detector and temporal predictor. Recalibrate each precision on calibration sequences and report detector AP/recall, HOTA, IDF1, ID switches, fragmentation, risk ECE, full-pipeline energy, latency, and thermal effects. If only a subset of precisions is supported, report the subset and do not simulate hardware performance.
9. **Statistics and release:** use at least three independent training seeds for the compact temporal model and at least five warm-up-separated full-pipeline runtime runs per policy/dataset/device. Report paired 95% confidence intervals using sequence-aware resampling or grouped held-out folds and run-level variation. MOT17 has few independent source sequences, so state that limitation; do not treat frames as independent samples. Release configs, label-generation code, model checksums, measurement scripts, and traces where dataset licenses permit.

### Minimum ablation matrix

| Variant | Predictor/policy | Purpose |
|---|---|---|
| A | Detector every frame | Quality and resource reference |
| B | Fixed detector intervals | Periodic skipping baseline |
| C | Confidence-only / uncertainty-only | Simple rule baselines |
| D | EMO-like / RT-MOT-like | Established adaptive-prior comparisons |
| D2 | HSFSO or faithful reimplementation | Closest learned adaptive-skipping paper; required comparator |
| D3 | ALBIREO-like per-object uncertainty scheduler | Required object-wise detector-invocation comparator; state adaptation limits |
| E | Logistic regression / static MLP risk | Tests learned but non-temporal scoring |
| F1 | GRU / causal TCN / temporal MLP using track history only | Temporal-model comparison; choose the primary TCN from this comparison |
| F2 | Primary TCN with low-cost scene descriptors concatenated to track history | Tests whether basic scene cues add value after their extraction cost |
| F3 | Primary TCN with a shared scene-context encoder and gated per-track fusion | Optional secondary architecture ablation; test added value and cost against F1/F2; not a separate novelty claim |
| F4 | Calibrated risk scheduler plus scene-discovery guard | Product safety ablation; measure new-track discovery and false-trigger/energy trade-off |
| G | Best full-precision policy, then supported FP32/FP16/INT8 variants | Late-stage identity-aware quantization sensitivity |

### Metrics definitions to keep consistent

- **HOTA:** balance of detection and association quality; use as a primary identity-aware metric.
- **IDF1:** identity precision/recall balance.
- **MOTA:** combines false positives, false negatives, and ID switches but can hide identity behavior.
- **ID switches and fragmentation:** report explicitly, especially under occlusion.
- **New-track discovery:** define entry events on annotated sequences as a visible identity with no matched active track while other tracks are active; exclude initialization. Report source frames to first correctly matched output track, undiscovered/censored events, guard precision, and false guard overrides per 1,000 input frames. Freeze the visibility/re-entry and IoU matching rules before held-out evaluation; use ground truth only for offline scoring.
- **Complete pipeline joules/input-frame:** measured detector + scene-discovery guard + risk predictor + controller + tracker + preprocessing/decoding/output energy divided by all input frames, including frames where detection is skipped. Predictor and guard are part of the product and must not be excluded.
- **Deadline miss rate (DMR):** fraction of input frames whose end-to-end output latency exceeds deadline $D$, where $D$ is set from the configured stream frame period/application requirement. State buffering and timestamp semantics.
- **Latency:** end-to-end p50/p95, sustained throughput, and dropped-frame count. Report model-only component times separately as diagnostics.
- **RAM and temperature:** peak process/system memory as precisely defined, board temperature, and any throttling over sustained runs.
- **Calibration:** PR-AUC, precision/recall, Brier score, and calibration error for the future-failure target. Report failure prevalence and calibration method.
- **Uncertainty:** repeated hardware runs and training seeds; paired comparisons and confidence intervals at sequence/run level. Frames are correlated and are not independent statistical samples.

---

## 12. Risks and scope controls

- Do not train a detector from scratch. Use public weights or fine-tune a small model.
- Core scope is one detector, one primary tracker (ByteTrack), one physical edge device, two datasets (MOT17 main, MOT20 stress/generalization), and one novelty: a temporal failure-risk predictor used for detector invocation.
- The detector-gap features and scene-discovery guard are product/evaluation design choices, not additional novelty claims; retain the guard only if measured benefit justifies its cost.
- Keep Mamba/SSM motion replacement, ROI/patch actions, pseudo-depth association, resolution switching, Re-ID, custom detector architecture, multi-camera tracking, local dataset creation, and cross-device claims outside the core study.
- Do not let video decoding, preprocessing, predictor, tracker, output, or logging costs disappear from total energy. Component timings are supplementary.
- Do not promise an FPS or energy saving before measuring the selected device.
- Do not use detector AP as a substitute for identity quality, and do not treat decision logging as a novel XAI contribution.
- If the adaptive controller loses HOTA/IDF1 or misses deadlines in crowds, report the regime and failure cause rather than tuning the test set.
- If no adaptive policy satisfies the quality/deadline constraints, report that no feasible energy-saving policy was found; this is a valid result.
- Counterfactual training rollouts may not match the learned policy's deployment states. Vary previous skip histories on training sequences, then measure this distribution shift on untouched sequences; do not hide it through frame-level random splitting.

## 13. Publication framing

The paper, if the results warrant one, should be organized around the **accuracy–energy Pareto frontier**, not a headline FPS claim. Plot joules/input-frame against HOTA and in a separate panel against IDF1; mark the feasible region under predeclared latency, deadline-miss, and quality constraints, with confidence intervals. Include the every-frame, fixed-skip, confidence, uncertainty, ALBIREO-like, HSFSO, EMO-like, RT-MOT-like, temporal-risk-only, and temporal-risk-plus-scene-guard policies. Report predictor and guard overhead, new-track discovery behavior, and whether the proposed policy beats a baseline at matched quality/deadline constraints. A working demo is a course deliverable; journal/conference publication is aspirational and depends on verified novelty and robust results.

## 14. Novelty verification gate

### Preliminary search already performed

On **5 October 2026**, an initial web literature search used queries covering `identity-failure prediction`, `ID-switch prediction`, `adaptive MOT scheduling`, `temporal track-quality prediction`, `selective detector invocation`, and `energy-aware MOT`. This was a preliminary search, **not a systematic review**. It surfaced directly relevant prior work: EMO on edge MOT optimization/skipping; RT-MOT on confidence-aware workload scheduling; Split and Connect on temporal prediction of potential ID-switch positions for tracklet splitting/repair; GRU-based association; TCN/Transformer motion prediction; and a 2026 PerCom WIP lead on RL-based adaptive model scheduling. [EMO](https://arxiv.org/abs/2309.02666), [RT-MOT](https://arxiv.org/abs/2210.11946), [Split and Connect](https://arxiv.org/abs/2105.02426), [GAKP](https://arxiv.org/abs/2012.14314), [ETTrack](https://doi.org/10.1007/s10489-024-05866-4), [QPilot announcement](https://www.jn.sfc.keio.ac.jp/%E3%80%90%E6%8E%A1%E6%8A%9E%E3%83%BB%E7%99%BA%E8%A1%A8%E3%80%91%E4%BF%AE%E5%A3%AB%E8%AA%B2%E7%A8%8B1%E5%B9%B4-%E7%BE%85%E5%90%9B%E3%81%8Cieee-percom-2026-wip-session%E3%81%A7%E7%99%BA%E8%A1%A8/)

A follow-up source check on **6 October 2026** verified three additional relevant works: SDOF-Tracker directly combines skipped detection and optical-flow interpolation on MOT17/MOT20; AdaVP (IEEE ICDCS 2020) uses parallel detection/tracking and runtime detector-setting adaptation on a Jetson TX2; and Lee (Sensors 2024) studies confidence-guided fallback for **single-object** tracking. Add the search variants `skipped detection optical flow MOT`, `parallel detection tracking mobile AdaVP`, and `confidence-guided frame skipping tracking` to the systematic review log. [SDOF-Tracker](https://doi.org/10.1587/transinf.2022EDP7022), [AdaVP](https://doi.org/10.1109/ICDCS47774.2020.00085), [Lee](https://doi.org/10.3390/s24248120)

This scan sharpens the novelty boundary. Split and Connect predicts potential switch positions to repair tracklets; RACE-MOT predicts future, skip-action-conditioned identity-failure probability before the next detector decision and uses it to schedule compute. SDOF-Tracker establishes skipped detection with optical-flow propagation as MOT prior art; AdaVP establishes parallel detection/tracking and content-adaptive detector settings as mobile-system ideas. Lee is adjacent SOT prior art and is not described as an MOT result.

**Critical update from the 6 October 2026 search:** ALBIREO adds a close object-wise detector-scheduling comparator using Kalman uncertainty, while HSFSO learns a tracking-accuracy surrogate for skip-ratio selection. GLoMOT and LUKF-Track address low-frame-rate motion; Mamba and pseudo-depth remain outside this project's novelty claim. RACE-MOT's claimed contribution is action-conditioned, calibrated per-track identity-failure prediction used for next-frame detector scheduling. Compare it against ALBIREO-like uncertainty scheduling and HSFSO under matched MOT metrics and device conditions. Detector-gap inputs support the prediction target; the scene-discovery override supports product robustness. [ALBIREO](https://arxiv.org/abs/2609.29648), [HSFSO](https://doi.org/10.1016/j.ins.2026.123610), [GLoMOT](https://doi.org/10.1609/aaai.v40i6.42500), [LUKF-Track](https://doi.org/10.3390/e28010103)

A targeted source check on **6 October 2026** reviewed the newly supplied ObjTrackNet and quantization papers and their suggested extensions. ObjTrackNet's failure analysis and future-work discussion motivate scene-condition stress tests, but it does not demonstrate a scene-conditioned risk scheduler. DeepScale already studies adaptive frame-size selection for MOT; Adaptive-EVOD is adjacent edge-video detection prior art for joint resolution/frame-rate/resource adaptation. The cited MobileNetV2/SSD-Lite quantization paper is detector-only and supports quantization as an established deployment ablation, not an identity-tracking contribution. These checks support a scene-feature ablation and reject a broader three-mode controller as the present novelty. [ObjTrackNet](https://link.springer.com/article/10.1007/s10791-026-10143-8), [DeepScale](https://doi.org/10.1109/IoTDI54339.2022.00010), [Adaptive-EVOD](https://doi.org/10.1109/TMC.2026.3719713), [MobileNetV2/SSD-Lite quantization](https://doi.org/10.1088/1742-6596/1748/3/032055)

### Targeted search to strengthen comparison and publication positioning

Continue the targeted review in IEEE Xplore, ACM Digital Library, CVF Open Access, arXiv, and available citation indexes. Use the phrases above with synonyms and backward/forward citation chasing from ALBIREO, GLoMOT, LUKF-Track, EMO, RT-MOT, HSFSO, SDOF-Tracker, AdaVP, Split and Connect, GAKP, ETTrack, DARTH, and DeepScale. Add query variants for `per-object uncertainty detector scheduling`, `scene-conditioned MOT scheduling`, `new-object discovery detector invocation`, `adaptive frame size/resolution MOT`, `variable frame rate object tracking`, `image-quality-conditioned detector scheduling`, and `thermal/resource-aware video inference`. Inspect ALBIREO's paper/code, HSFSO's full text/supplement/code, GLoMOT/LUKF-Track, DeepScale, and Adaptive-EVOD to strengthen final positioning and define fair comparisons. This review improves the claim's defense and publication case; it does not block product development. The ObjTrackNet future-work discussion is motivation, not a substitute for prior-art search or empirical evidence. Record search dates, exact query strings, inclusion/exclusion criteria, and a related-work extraction table covering prediction target, horizon, inputs, online/offline setting, action selected, datasets, hardware, energy/latency metrics, and released code.

Include work that predicts tracking quality, association errors, ID switches, track fragmentation, or detector need and uses that prediction to alter inference/scheduling. Exclude unrelated railway track quality and most single-object tracking papers from the core MOT comparison, but record them if they affect terminology. Review the full QPilot PerCom 2026 WIP paper/proceedings before submission.

**Evaluation gate:** the novelty claim is defined. Direct comparisons with ALBIREO-like uncertainty scheduling, HSFSO, and temporal ID-switch prediction quantify its practical distinction and value. Test whether calibrated avoidable-failure risk improves the held-out quality/energy/deadline frontier. Separately test whether detector-gap features and the scene-discovery override improve product behavior after their complete cost is counted. Avoid exhaustive “first” or “no prior work” statements.

---

## 15. Final recommendation

The project's novelty claim and core research direction are:

> **RACE-MOT predicts a calibrated, per-track probability of avoidable identity failure under a defined next-frame detector-skip action, then schedules detector compute from that risk while measuring the tracking-quality, energy, and deadline trade-off on a Jetson Orin Nano.**

Use one small CNN detector, ByteTrack, one edge device, MOT17, and MOT20. Establish an every-frame baseline, then compare ALBIREO-like per-object uncertainty scheduling, HSFSO, fixed skipping, confidence-only, uncertainty-only, EMO/RT-MOT-like policies, and the proposed temporal-risk policy. Keep DETECT/SKIP as the only inference actions, but add a cheap scene-discovery guard that can upgrade a scheduled SKIP to DETECT when current-frame motion outside tracked regions or a scene discontinuity is detected. Treat this guard as product robustness, not novelty; measure its entry-detection benefit, false-trigger rate, and full-pipeline cost. Make detector-gap history explicit in the predictor and report results by gap. Keep Mamba/SSM motion replacement, patch detection, pseudo-depth, and thermal feedback outside the core until feasibility and matched evidence justify them. Quantization comes last as an ablation.

**Faculty verdict:** the current idea-stage estimate is **59/100**; it measures proposal readiness, not completed project performance. RACE-MOT claims a focused method contribution: calibrated per-track prediction of avoidable identity failure under a specified detector-skip action, used for next-frame detector scheduling and evaluated under full-pipeline edge constraints. ALBIREO and HSFSO are the closest scheduling/skip-control comparisons; GLoMOT/LUKF-Track define low-frame-rate motion boundaries, while Split and Connect covers post-hoc switch localization and repair. The scene-discovery guard addresses entrants before a per-track risk exists, but its benefit over ALBIREO's empty-scene/rescue path must be measured. Keep the guard only if it improves new-track discovery without erasing energy savings. Do not expand into Mamba, patch inference, pseudo-depth, thermal feedback, Re-ID, multi-camera networking, or continual learning for component count. Report measured results and limits honestly.

---

## Appendix A. Future novelty idea register

This register is the intake point for ideas added after this version. An entry records a proposal; it does not establish novelty. Cite primary sources and include enough detail to reproduce the review. Keep ideas in **proposed** status until the literature and feasibility checks are complete.

| ID | Date / source | Idea | Evidence and closest prior art | Scope / evaluation impact | Decision and destination |
|---|---|---|---|---|---|
| RACE-001 | Earlier proposal discussion; consolidated 6 Oct 2026 | Temporal failure-risk model, conditional appearance/Re-ID, spatial routing, synthetic edge degradations, quantization, and decision explanations | Detailed disposition is recorded in §8, “Assessment of the additional novelty proposals”; relevant prior art is reviewed in §§4–5. | Temporal risk model remains the conditional core hypothesis; degradations and quantization are secondary evaluations; Re-ID and spatial routing are deferred; decision logs are supporting audit evidence. | Partly adopt, partly defer; see §8 and the core-scope decision above. |
| RACE-002 | User-supplied online-scan claims; reviewed 6 Oct 2026 | Occlusion/Re-ID, frame-gap and blur effects, context-aware scheduling, thermal throttling, memory bandwidth, cross-camera edge mesh, hybrid CNN-attention, and continual learning | Evidence audit and source links are in §4.4. The scan contains supported mechanisms, overstated causal claims, and established prior art. | Keep density/occlusion reporting, separate frame-gap and blur stress tests, and sustained thermal telemetry where feasible. Keep multi-camera, backbone changes, continual learning, Re-ID, and multi-stream bottleneck claims outside the core. | Audited and scoped; do not treat the raw scan as validated novelty. See §4.4 and §§9–12. |
| RACE-003 | User-pasted paper review; checked 6 Oct 2026 | Scene-quality/activity cues, three compute modes, optical-flow propagation, quantization, hardware-state adaptation, and a small edge stress set | ObjTrackNet motivates conditions but does not test the proposed policy; DeepScale studies adaptive MOT frame size; Adaptive-EVOD adapts edge-video resolution/frame rate/resources; SDOF-Tracker covers optical-flow propagation; the 2021 MobileNetV2/SSD-Lite paper covers detector quantization. Sources and dispositions are in §§4–6 and §8. | Adopt a paired track-only vs track-plus-scene feature ablation and low-light/contrast robustness stress test. Keep binary skip/detect actions. Defer reduced-resolution mode, policy-driven thermal/utilization actions, optical-flow propagation, and new capture set; retain INT8 as a late ablation. | Partly adopt, defer the rest. Scene features and existing compression/control techniques are not independent novelty claims. |
| RACE-004 | User-provided three-part novelty outline; reviewed 6 Oct 2026 | Skip-conditioned avoidable-failure labels and paired rollouts; asymmetric kinematic/context fusion; auditable Orin deployment with energy/thermal measurement | The core target is compared against HSFSO's learned skip-ratio accuracy surrogate and Split and Connect's temporal ID-switch localization/tracklet repair. TCN/gating/context cues are established building blocks; deployment/auditing are product/system evidence. The proposed performance guarantees and Re-ID cost rationale are unsupported for this scope. | Keep paired skip/detect avoidable-failure prediction as the one primary candidate hypothesis. Test shared scene-context encoding and gated per-track fusion only as an optional ablation. Make the Orin prototype and properly bounded energy/thermal report product evidence; do not claim novel XAI, zero leakage, thermal optimization, or Re-ID savings. See §§6, 8, 10–11. | Partly adopt and substantially qualify; no novelty score increase before evidence. |
| RACE-005 | User-pasted four-gap proposal; reviewed 6 Oct 2026 | Gap-conditioned motion/SSM, ROI patch detection, thermal feedback control, and pseudo-depth association | ALBIREO is close prior art for per-object uncertainty scheduling; GLoMOT and LUKF-Track cover low-frame-rate/nonlinear motion; GLoMOT and SparseTrack cover pseudo-depth; Adaptive-EVOD overlaps resource/location/rate adaptation. The supplied numeric claims (<15K parameters, 70% energy saving, 70–80°C thresholds, 30–50% latency drift, <0.05 ms) are unverified. | Update design with explicit detector-gap/timestamp features and a low-cost scene-discovery guard that can upgrade SKIP to full-frame DETECT; add ALBIREO-like baseline and gap-stratified evaluation. Defer learned motion replacement, ROI/patch action, pseudo-depth, and closed-loop thermal control. Monitor thermal behavior; only trial a governor if sustained device tests show a real product need. | Partly adopt as measured product robustness; reject unsupported novelty/numeric claims; defer other modules. See §§4.5, 5, 8, 11. |
| RACE-___ | YYYY-MM-DD / source or citation | **New idea:** state the precise method or application change. | **Evidence:** closest papers, what they do, and the distinction still to verify. | **Impact:** needed data, model, baselines, metrics, device, time, and any scope trade-off. | **Status:** proposed / adopt / test / defer / reject. Record the sections changed. |

### Review checklist for each new entry

1. State the problem and intended contribution in one or two testable sentences.
2. Search the exact task and its synonyms; inspect primary papers, code, and benchmarks for the closest work.
3. Label each supporting claim as established, partly supported, unverified, or contradicted. Do not convert a gap in the search into a “first” claim.
4. Name the direct baselines, data split, metrics, hardware needs, and expected scope cost.
5. Decide **adopt**, **test as an ablation**, **defer**, or **reject**; then update the proposal and revision history.

---

## Appendix B. Revision history

Rows are chronological. Earlier entries preserve the proposal's prior wording and scores; the latest entry supersedes those novelty-position statements.

| Date | Revision | Evidence / scope effect |
|---|---|---|
| 6 Oct 2026 | Standardized the merged report as a maintained proposal; added document status, navigation, explicit novelty status, future-idea intake, and revision history. Retained the literature tables, claim audits, methodology, dataset protocol, hardware measurements, and experiment plan. | Records HSFSO as the closest known adaptive-skipping comparator; novelty remains unverified pending full-text/code review and systematic search. |
| 6 Oct 2026 | Incorporated the audit of the supplied internet-scan claims. | Kept supported stress conditions and thermal measurement; qualified unsupported causal/bandwidth claims; deferred cross-camera, new backbone, Re-ID, and continual-learning extensions. |
| 6 Oct 2026 | Added temporal risk prediction, counterfactual failure labels, calibrated policy, complete-pipeline energy accounting, and statistical evaluation requirements. | These remain a proposed method and protocol, not completed experimental results or validated novelty. |
| 6 Oct 2026 | Reviewed the newly supplied ObjTrackNet and edge-quantization papers; added scene-context inputs as a paired ablation and updated the prior-art/novelty analysis. | Added DeepScale and Adaptive-EVOD as resolution/resource prior art; kept the controller binary and deferred a new dataset, optical-flow propagation, and hardware-state policy. Cited the quantization paper only as detector-compression background. |
| 6 Oct 2026 | Re-scored the current idea-stage proposal against the course rubric. | Updated estimate to 56/100; kept novelty low because it remains unverified and gave no experimental-results credit beyond the planned protocol. |
| 6 Oct 2026 | Updated product planning after the student confirmed access to a Jetson Orin Nano and a mobile-phone camera feed. | Fixed the single-device target and private-network RTSP demo input; retained MOT17/MOT20 replay for repeatable evaluation. Added YOLOX-Tiny/TensorRT FP16 as an unverified implementation candidate; exact board/software/checkpoint remain subject to the feasibility and licensing check. |
| 6 Oct 2026 | Audited the supplied three-part novelty proposal and updated the model, measurement, and product claim boundaries. | Kept paired skip/detect avoidable-failure prediction as the single unverified methodological hypothesis; added shared scene-context gated fusion only as an optional ablation; classified Orin deployment/logging as product evidence; removed unsupported timing/calibration/privacy/Re-ID claims and clarified the Jetson energy boundary. |
| 6 Oct 2026 | Reviewed the supplied SSM, ROI, thermal-governor, and pseudo-depth proposal against ALBIREO, GLoMOT, LUKF-Track, Adaptive-EVOD, and SparseTrack. | Added actual detector-gap/timestamp features, a measured scene-discovery override, ALBIREO-like scheduling baseline, and gap-stratified evaluation. Kept binary DETECT/SKIP and ByteTrack; deferred SSM replacement, patch actions, pseudo-depth, and thermal feedback pending matched evidence. Revised idea-stage score from 56/100 to 55/100 as the prior-art risk became clearer. |
| 6 Oct 2026 | Added the supplied comparative matrix and adopted a precise positive novelty claim across the report, faculty pitch, decision log, model plan, and SDD. | Defined the claimed contribution as calibrated, per-track, skip-action-conditioned prediction of avoidable identity failure for next-frame detector scheduling; distinguished ALBIREO, HSFSO, EMO, RT-MOT, Split and Connect, GLoMOT, and LUKF-Track. Updated idea-stage estimate to 59/100; experimental gains and physical energy readings remain future evidence. |

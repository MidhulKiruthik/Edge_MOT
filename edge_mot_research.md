# Multi-Object Tracking Using Lightweight CNNs on Edge Devices
## Deep research study for a final-year undergraduate project

> **Historical research input:** Recommendations and numeric targets here are
> not current RACE-MOT requirements. Use `deep_research_mot_edge_merged.md`,
> `todo.md`, and the decision log for the accepted scope and evidence status.

**Scope.** Literature and industry position as of 5 October 2026. The study focuses on online, camera-based multi-object tracking (MOT) where a detector runs on an edge device and an association module maintains identities. Results quoted below are the authors' benchmark results, usually on a GPU; they are not claims about Raspberry Pi performance unless explicitly stated.

## Executive conclusion

The market direction is clear: video analytics is moving from centralized cloud inference toward local inference for latency, privacy, bandwidth, resilience, and operating cost. The academic frontier is also clear: strong tracking accuracy is available from combinations such as YOLO plus ByteTrack, OC-SORT, or BoT-SORT, but the literature still under-reports the **accuracy-latency-energy-memory trade-off on the actual edge target**. A defensible undergraduate contribution is therefore not "use YOLO and DeepSORT". It is an experimentally reproducible **resource-aware adaptive tracker** that changes detector frequency or model size according to scene difficulty and reports HOTA/IDF1, end-to-end latency, peak RAM, model size, power, and energy per frame.

### Recommended project

**Energy-Aware Adaptive Lightweight MOT for Edge Cameras:** deploy a small detector (YOLOv8n/YOLOv10n or MobileNet-SSD baseline) with ByteTrack or OC-SORT on Jetson Orin Nano and Raspberry Pi 5. Use a cheap scene-complexity controller based on detection count, confidence, camera motion, and track uncertainty to choose among detector resolutions or detector skip intervals. Enforce a target such as >=15 end-to-end FPS, <=150 ms p95 latency, <=1.5 GB RAM, and <=10% HOTA loss relative to every-frame inference on a chosen dataset. The contribution is a Pareto evaluation and controller, not a claim that the base detector or tracker is new.

---

## 1. Industry and market trend, 2024-2026

| Trend | Why it matters for MOT | Evidence and engineering implication |
|---|---|---|
| Edge AI is becoming the default location for time-sensitive perception | Sending every frame to a cloud service adds network delay, bandwidth cost, privacy exposure, and failure modes | NVIDIA's Jetson Orin platform, Qualcomm's edge AI portfolio, and AWS IoT Greengrass all position local inference for cameras, robotics, retail, manufacturing, and transport. A project should measure local end-to-end performance, not only model FLOPs. |
| Real-time video analytics is expanding beyond security | Retail occupancy, queueing, traffic, workplace safety, sports, logistics, and industrial inspection need persistent identities, not isolated detections | The commercial value is in counts, dwell time, trajectories, and alerts. Identity switches and missed tracks can be more damaging than a small AP difference. |
| Autonomous systems and robotics need low and predictable latency | Robots and vehicles cannot wait for a bursty remote service; worst-case latency matters for control loops | Report p50 and p95 latency, dropped frames, and thermal throttling. Average FPS alone is insufficient. |
| Privacy and data sovereignty favor on-device processing | Many deployments should avoid uploading identifiable video | Use public datasets for training, but include an on-device privacy and retention discussion. Do not claim that inference is private if raw frames still leave the device. |
| Heterogeneous accelerators are common | CPU, GPU, NPU, DSP, and memory bandwidth may dominate actual speed | Benchmark the same exported model in PyTorch, ONNX Runtime, TensorRT or TFLite where supported. Record input resolution, batch size, precision, and warm-up policy. |
| Energy and thermal budgets are first-class constraints | A camera that is fast for five minutes and then throttles is not an edge solution | Measure wall power with a USB power meter or INA219 and report joules per processed frame plus temperature over time. |
| Model compression is practical but task-dependent | INT8 calibration, pruning, distillation, and smaller backbones reduce cost, but can hurt small-object detection and identity continuity | Evaluate compression on both detector metrics and MOT metrics. Quantization-aware training is an opportunity only if deployment tooling supports it. |

**Important market caveat.** Public market reports often forecast large edge-AI growth but use different definitions of edge AI and video analytics. They are useful for motivation, not for a scientific result. The strongest evidence for this project is the convergence of deployed hardware roadmaps, real-time detector papers, and the operational requirements above.

Useful industry references: [NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), [AWS IoT Greengrass](https://aws.amazon.com/greengrass/), [Qualcomm AI Hub](https://aihub.qualcomm.com/), and [ETSI MEC overview](https://www.etsi.org/technologies/multi-access-edge-computing).

---

## 2. State of the art

### 2.1 Detector families

| Family | Main idea | Edge relevance | Caution |
|---|---|---|---|
| MobileNetV2/V3 + SSD/YOLO head | Depthwise separable convolutions and hardware-aware design | Small model, mature mobile deployment | Small objects and crowded scenes degrade quickly; CPU latency depends on kernels, not only parameters |
| EfficientNet | Compound scaling of depth, width, and resolution | Strong accuracy-efficiency baseline | It is primarily an image classifier; a detector head and export path are still required |
| NanoDet / PicoDet / PP-YOLOE-S | Purpose-built small one-stage detectors | Good starting point for ARM/NPU experiments | Reported speed is hardware and implementation specific; some projects are mainly preprints or repositories |
| YOLOv7 / YOLOX / YOLOv8 | One-stage dense detection with mature training and export ecosystem | Excellent baseline family and community support | NMS, post-processing, and Python overhead can dominate on small devices |
| YOLOv10 | NMS-free end-to-end design and efficiency-accuracy scaling | Reduces post-processing overhead in principle | Published comparisons are mostly desktop GPU; verify edge export and tracking compatibility |
| RT-DETR / Lite DETR | Real-time DETR designs with efficient multi-scale encoding | Removes NMS and offers speed tuning | Transformer memory and kernels can be less friendly to low-end ARM devices |

### 2.2 Tracking families

| Tracker | Association signal | Strength | Edge cost/risk |
|---|---|---|---|
| SORT | Kalman motion plus IoU | Very fast and simple | ID switches under occlusion and camera motion |
| DeepSORT | SORT plus appearance embedding | Better re-identification after occlusion | Extra CNN inference per detection and increased memory |
| ByteTrack | Associates high- and low-confidence detections in two stages | Strong accuracy with simple association; a highly practical baseline | Still depends on detector recall and can propagate false low-confidence detections |
| OC-SORT | Observation-centric correction of Kalman state | Strong under nonlinear motion and occlusion; association is cheap | Appearance-free identity recovery remains weak when similar objects cross |
| BoT-SORT | Motion, appearance, camera-motion compensation, improved Kalman state | Strong MOTChallenge performance | Re-ID backbone and global motion compensation increase latency |
| StrongSORT / Deep OC-SORT | Stronger or adaptive appearance cues over motion trackers | Better identity preservation | Not the first choice for a tight Raspberry Pi budget |
| Joint/query trackers such as MOTR | Detector and track queries are trained together | Elegant end-to-end temporal modeling | Training and inference complexity are usually inappropriate for a first edge project |

**Established baseline.** A small detector plus ByteTrack or OC-SORT is already a strong, publishable-quality baseline. A paper that only replaces ByteTrack with DeepSORT, or only swaps YOLOv5n for YOLOv8n, is not novel by itself.

---

## 3. Specific research gaps

1. **End-to-end accounting gap.** Many papers report detector FPS or tracker FPS separately. Real deployments need capture, resize, inference, decoding, NMS or end-to-end post-processing, association, rendering, and I/O in one pipeline.
2. **Energy gap.** HOTA/IDF1 and FPS are common; joules per frame, average power, temperature, and throttling are uncommon, especially on Raspberry Pi-class devices.
3. **Tail-latency gap.** Mean FPS hides periodic detector stalls, garbage collection, memory pressure, and thermal throttling. p95/p99 latency and dropped-frame rate should be primary metrics.
4. **Adaptive-compute gap.** Detector skipping, variable input resolution, and dynamic model selection are individually known ideas, but a simple controller jointly constrained by identity continuity, energy, and latency is underexplored in undergraduate-accessible MOT experiments.
5. **Occlusion versus budget gap.** Lightweight appearance embeddings may help after occlusion but cost energy. The Pareto boundary between an appearance-free tracker and a tiny Re-ID branch is not consistently measured on edge hardware.
6. **Crowded and small-object gap.** Edge compression often removes the features needed for small or overlapping objects. COCO AP does not fully predict MOT identity quality in MOT20, DanceTrack, or a local crowded-camera dataset.
7. **Cross-device reproducibility gap.** Reported FPS is not comparable without device, runtime, power mode, resolution, precision, batch size, warm-up, and thermal conditions.
8. **Robustness gap.** Lighting, rain, camera shake, compression, and frame drops can cause detector confidence collapse and track fragmentation. A controlled corruption protocol is feasible and valuable.
9. **Data/domain gap.** Most benchmarks are pedestrian-focused and curated. Low-cost projects often report only MOT17; deployment claims should include a second domain such as KITTI, UA-DETRAC, VisDrone, or a small self-recorded sequence.
10. **Accuracy-energy objective gap.** Most methods optimize a single benchmark score. A practical edge study should report a Pareto frontier and a constrained objective, for example maximize HOTA subject to p95 latency and power limits.

---

## 4. Literature review: 23 relevant papers

**Reading rule.** "Peer-reviewed" is stated when a conference or journal venue is known. arXiv-only items are labelled preprint. A reported FPS is never treated as portable across hardware. Dataset names and headline results are condensed from the linked paper or official project page; limitations are an engineering reading of the reported setup.

| # | Paper, year, venue and link | Method and datasets | Key reported result | Limitation relevant to edge MOT |
|---:|---|---|---|---|
| 1 | [MobileNetV3, 2019, ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html), DOI [10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140) | Hardware-aware NAS and NetAdapt; ImageNet, COCO, Cityscapes | MobileNetV3-Large reports 15% lower latency than V2 at higher ImageNet accuracy; detection and segmentation adaptations | Classifier/backbone paper, not a full MOT system; latency depends on mobile kernels and detector head |
| 2 | [EfficientNet, 2019, ICML](https://proceedings.mlr.press/v97/tan19a.html) | Compound scaling of width, depth, resolution; ImageNet and transfer tasks | Strong accuracy-efficiency scaling with fewer parameters than prior models | Scaling rule does not guarantee minimum latency or memory on ARM; detector/tracker cost is outside the paper |
| 3 | [YOLOX, 2021/2022, arXiv preprint](https://arxiv.org/abs/2107.08430) | Anchor-free YOLO, decoupled head, SimOTA; COCO | Strong one-stage detector family, including nano/tiny variants | NMS and post-processing remain; paper speed is GPU-centric and not MOT end-to-end |
| 4 | [YOLOv7, 2022, arXiv/preprint](https://arxiv.org/abs/2207.02696) | Trainable bag-of-freebies, E-ELAN, re-parameterization; MS COCO | Reports 56.8 AP at 30+ FPS on V100 across its real-time range | Large headline models are not edge models; COCO results do not measure identities or energy |
| 5 | [ByteTrack, 2022, ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php), DOI [10.1007/978-3-031-20047-2_1](https://doi.org/10.1007/978-3-031-20047-2_1) | Two-stage association uses almost every detection, including low-score boxes; MOT17, MOT20, KITTI, DanceTrack, BDD100K | 80.3 MOTA and 77.3 IDF1 on MOT17 are widely reported for the official system | Accuracy depends strongly on detector recall and threshold; low-score boxes can add false tracks; original system is not energy-aware |
| 6 | [MOTR, 2022, ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1739_ECCV_2022_paper.php) | Transformer track queries and temporal propagation; DanceTrack, MOT17, MOT20 | End-to-end query-based tracking with competitive benchmark performance | Transformer memory, training cost, and sequence handling are difficult for low-cost edge devices |
| 7 | [YOLOv6, 2022, arXiv](https://arxiv.org/abs/2209.02976) | Hardware-friendly backbone/head and self-distillation; COCO | Several small models target real-time industrial deployment | Official speed depends on TensorRT/GPU; limited evidence for Raspberry Pi-class deployment and MOT identity metrics |
| 8 | [OC-SORT, 2023, CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html), DOI [10.1109/CVPR52729.2023.00571](https://doi.org/10.1109/CVPR52729.2023.00571) | Observation-centric re-update and virtual trajectory during occlusion; MOT17, MOT20, KITTI, head tracking, DanceTrack | Official page reports 700+ FPS for the tracker on one CPU with strong benchmark scores | Tracker-only FPS excludes detector; no appearance branch means similar-object crossings remain hard |
| 9 | [BoT-SORT, 2022, arXiv / benchmark system](https://arxiv.org/abs/2206.14651) | Motion, camera-motion compensation, Re-ID appearance, improved Kalman state; MOT17/MOT20 | Reports 80.5 MOTA, 80.2 IDF1, 65.0 HOTA on MOT17 | Re-ID and camera compensation add compute and memory; benchmark setup is not a low-power deployment |
| 10 | [StrongSORT, 2022, IEEE TMM](https://arxiv.org/abs/2202.13514), DOI [10.1109/TMM.2022.3217698](https://doi.org/10.1109/TMM.2022.3217698) | Stronger detector, camera compensation, appearance extraction, AFLink and GSI; MOT17/MOT20 | Improves identity metrics over DeepSORT-family baselines | Multiple add-on modules make it a poor first baseline for edge power budgets; ablations do not equal a device energy study |
| 11 | [Deep OC-SORT, 2023, ICCV workshop/preprint](https://arxiv.org/abs/2302.11813) | OC-SORT plus adaptive Re-ID integration; MOT17, MOT20, DanceTrack | Reports 64.9 HOTA on MOT17, 63.9 on MOT20, and 61.3 on DanceTrack | Appearance improves robustness but undermines the minimal-compute advantage; reported results are not edge measurements |
| 12 | [MOTRv2, 2022, NeurIPS](https://arxiv.org/abs/2204.00776) | Bootstrapped detector plus MOTR query tracking; MOT17, MOT20, DanceTrack | Stronger DETR-based tracking through detector pretraining/bootstrapping | Joint transformer is compute-heavy and operationally complex for undergraduate edge deployment |
| 13 | [FairMOT, 2021, IJCV](https://arxiv.org/abs/2004.01888), DOI [10.1007/s11263-021-01408-4](https://doi.org/10.1007/s11263-021-01408-4) | Anchor-free detection and identity embedding sharing a backbone; MOT17, MOT20, KITTI | Strong balance of detection and Re-ID with a single network | Shared high-resolution features still cost substantial memory; older baselines may be less competitive than ByteTrack |
| 14 | [CenterTrack, 2020, ECCV](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3255_ECCV_2020_paper.php), DOI [10.1007/978-3-030-58548-8_26](https://doi.org/10.1007/978-3-030-58548-8_26) | Center detection plus previous-frame centers and motion; MOT17, KITTI, nuScenes | Joint detection and tracking without a separate association stage | Requires temporal input and can fail under large motion/occlusion; backbone and input resolution dominate edge cost |
| 15 | [Tracktor++, 2019, ICCV](https://openaccess.thecvf.com/content_ICCV_2019/html/Bergmann_Tracking_Without_Bells_and_Whistles_Tracktor_++_ICCV_2019_paper.html), DOI [10.1109/ICCV.2019.00141](https://doi.org/10.1109/ICCV.2019.00141) | Detector box regression as tracker, motion model and Re-ID; MOT17 | Demonstrated that detector quality can be the main tracking bottleneck | Repeated detector/regression cost and older design; useful historical proof, not a final edge choice |
| 16 | [Lite DETR, 2023, CVPR](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Lite_DETR_An_Interleaved_Multi-Scale_Encoder_for_Efficient_DETR_CVPR_2023_paper.html), DOI [10.1109/CVPR52729.2023.00435](https://doi.org/10.1109/CVPR52729.2023.00435) | Interleaved multi-scale encoder and key-aware deformable attention; COCO | Reports 60% reduction in detection-head GFLOPs while retaining about 99% of original performance | Efficient on GPU does not automatically mean efficient on ARM; not a tracker and has memory-sensitive attention operations |
| 17 | [RT-DETR, 2024, CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html), DOI [10.1109/CVPR52733.2024.00066](https://doi.org/10.1109/CVPR52733.2024.00066) | Efficient hybrid encoder, IoU-aware query selection, adjustable decoder depth; COCO | RT-DETR-R50/R101 report 53.1/54.3 AP and 108/74 FPS on T4; scaled models compete with light YOLOs | T4 results are not edge results; decoder and export support can be less mature than YOLO; tracking integration is an extra experiment |
| 18 | [YOLOv8, 2023, Ultralytics technical release](https://github.com/ultralytics/ultralytics) | Anchor-free split head and modern augmentation/training; COCO and user datasets | Widely used small/nano models with broad ONNX/TensorRT support | Not a conventional peer-reviewed paper; version drift and licensing/export details must be recorded; benchmark claims depend on implementation |
| 19 | [YOLOv10, 2024, NeurIPS](https://arxiv.org/abs/2405.14458) | Consistent dual assignment, NMS-free inference, holistic efficiency design; COCO | YOLOv10-S is reported 1.8x faster than RT-DETR-R18 at similar AP and 2.8x smaller in parameters/FLOPs; YOLOv10-B reports 46% lower latency than YOLOv9-C at similar performance | Desktop GPU results; NMS-free detector still needs a tracker, and edge runtime kernels may erase the theoretical gain |
| 20 | [RTMDet, 2022, arXiv / OpenMMLab](https://arxiv.org/abs/2212.07784) | Real-time detector series with CSPNeXt and training refinements; COCO | Strong accuracy-speed trade-offs across model sizes | Primarily detector benchmark; deployment overhead and MOT identity quality are not the focus |
| 21 | [NanoDet-Plus, 2021, GitHub/preprint](https://github.com/RangiLyu/nanodet) | Tiny anchor-free detector and lightweight head; COCO | Designed for mobile CPU and low memory | Peer-review status is unclear and headline FPS is implementation-dependent; use as an engineering baseline, not novelty evidence |
| 22 | [VideoMAE, 2022, NeurIPS](https://arxiv.org/abs/2203.12602) | Masked video autoencoder pretraining; Kinetics and downstream video tasks | Strong video representation learning with high masking ratio | Pretraining and transformer inference are far beyond a typical edge MOT budget; useful only as a contrast to lightweight CNN pipelines |

### What is already solved, and what is not

| Idea | Status | Reason |
|---|---|---|
| Run a small YOLO detector on an edge GPU | Largely solved engineering problem | Mature export stacks and many public examples exist |
| Use SORT/DeepSORT with YOLO | Solved baseline, not novelty | The combination is standard and easy to reproduce |
| Replace SORT with ByteTrack or OC-SORT | Strong modern baseline | Both are established and should be included in comparison, not presented as the contribution |
| Quantize a detector to INT8 and report FPS only | Insufficient research claim | Accuracy, identity metrics, power, and calibration details are missing |
| Build a tiny Re-ID CNN | Incremental unless tied to a clear budget and occlusion protocol | Deep OC-SORT/BoT-SORT already establish appearance-aware tracking |
| Jointly optimize HOTA, p95 latency, RAM, and energy on real edge hardware | Underexplored and defensible | The literature commonly separates benchmark accuracy from deployment measurements |
| Adaptive detector skipping/model switching using track uncertainty, with power measurement | Underexplored at undergraduate scale | Known ingredients exist, but the constrained controller and reproducible cross-device Pareto study are a credible contribution |
| Cross-device benchmark with thermal throttling and frame-drop stress | Underexplored and useful | It tests deployment reality rather than only a new architecture |

---

## 5. Research-gap matrix

| Work family | Dataset(s) | Model/hardware reported | Main metrics | Strength | Limitation / gap exposed |
|---|---|---|---|---|---|
| MobileNetV3/EfficientNet | ImageNet, COCO transfer | Mobile CPU-oriented design; usually GPU for detector experiments | Top-1, AP, latency | Strong backbone foundations | No persistent identity, power, or MOT evaluation |
| YOLOv7/YOLOX/YOLOv8 | COCO | V100/T4/GPU and export runtimes | AP, FPS | Mature and fast | Detector FPS is not end-to-end MOT FPS |
| RT-DETR/Lite DETR | COCO | T4/GPU | AP, FPS, GFLOPs | NMS-free or efficient multi-scale encoding | Edge memory/kernel behavior and MOT integration are weakly documented |
| ByteTrack | MOT17/20, KITTI, DanceTrack, BDD100K | Detector plus CPU association; published system often GPU detector | MOTA, IDF1, HOTA, FPS | Excellent simple association and low tracker cost | Detector recall, threshold, and energy sensitivity not fully characterized |
| OC-SORT | MOT17/20, KITTI, DanceTrack | CPU tracker with detector input | HOTA, MOTA, IDF1, tracker FPS | Very cheap and robust to nonlinear motion | Appearance-free association fails on similar objects and long occlusions |
| BoT-SORT/StrongSORT | MOT17/20 | GPU detector plus Re-ID and camera compensation | MOTA, IDF1, HOTA | Strong identities | Appearance cost and power are rarely reported |
| Deep OC-SORT | MOT17/20, DanceTrack | GPU/desktop benchmark | HOTA and identity metrics | Adaptive appearance use | Not a low-power study; extra inference cost is a direct edge trade-off |
| MOTR/MOTRv2 | MOT17/20, DanceTrack | Transformer GPU training/inference | HOTA, MOTA, IDF1 | Joint temporal modeling | High memory and implementation complexity |
| CenterTrack/FairMOT | MOT17, KITTI, nuScenes | Joint CNN or shared backbone | MOTA, IDF1, AP | Joint detection/tracking alternatives | Higher-resolution shared features and older training assumptions |
| Edge deployment reports in industry/tutorials | Custom streams, COCO subsets | Jetson, Android, Raspberry Pi, NPU | FPS, sometimes power | Practical runtime knowledge | Often no public, controlled HOTA/IDF1 comparison |
| Proposed study | MOT17/MOT20 or UA-DETRAC plus one local sequence | Jetson Orin Nano and Raspberry Pi 5; TensorRT/ONNX/TFLite | HOTA, IDF1, MOTA, ID switches, p95 latency, RAM, watts, joules/frame, temperature | Reproducible systems-level Pareto analysis | Requires careful instrumentation and fair detector/tracker synchronization |

---

## 6. Novelty opportunities feasible for undergraduates

| Opportunity | Implementation | Novelty strength | Risk |
|---|---|---:|---:|
| 1. Uncertainty-driven detector skipping | Run detector every k frames when tracks are stable; force a full detection on covariance/confidence growth | High for a measured edge systems study | Skipping can cause drift and ID switches; use a safe fallback |
| 2. Tiny adaptive model ladder | Export nano/small detector variants or 320/512/640 inputs; select using object count, size, and track uncertainty | High | Need fair calibration and enough device measurements |
| 3. Energy-aware association policy | Compare SORT, ByteTrack, OC-SORT, and a tiny Re-ID branch under a joules/frame constraint | Medium-high | Base trackers are known; contribution is the Pareto analysis/policy |
| 4. Thermal-aware frame-rate controller | Reduce resolution or detector frequency when temperature/power crosses thresholds | High practical value | Needs repeatable thermal protocol and long runs |
| 5. Robustness under frame drops and compression | Inject 5-30% drops, H.264 compression, blur, and lighting changes; adapt thresholds | Medium-high | Evaluation design must avoid arbitrary corruption |
| 6. Lightweight occlusion recovery | Add a small MobileNetV3-Small embedding only after an occlusion trigger, not for every detection | High if measured carefully | Re-ID training/data preparation is time-consuming |
| 7. Two-camera edge partitioning | Compare fully local, detector-on-Jetson/tracker-on-Pi, and cloud-like split | Medium-high | Synchronization and networking complicate scope |
| 8. Pareto benchmark and reproducibility kit | Fixed containers/configs, power traces, p95 latency, and standardized device/runtime table | Medium, high placement value | More systems research than architecture novelty |

Avoid claiming novelty for "YOLOv8 + DeepSORT", "YOLOv5 + ByteTrack", or "INT8 quantization improves FPS" without a new constraint, protocol, or finding.

---

## 7. Three strongest project directions

Scores are qualitative: 5 is strongest. They assume a 6-9 month undergraduate project with access to one Jetson and one ARM board.

| Rank | Direction | Novelty | Feasibility | Dataset availability | Compute need | Evaluation potential | Placement value | Decision |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **Energy-aware adaptive detector scheduling with ByteTrack/OC-SORT** | 4.5 | 4.5 | 5 | 4 | 5 | 5 | Best overall: clear controller, strong baselines, measurable device contribution |
| 2 | **Thermal- and latency-aware model/resolution switching for crowded MOT** | 4.5 | 4 | 4.5 | 3.5 | 5 | 5 | Strong systems story; needs long controlled runs and careful thermal instrumentation |
| 3 | **Triggered tiny Re-ID for occlusion recovery** | 4 | 3.5 | 4 | 3 | 4.5 | 4.5 | Good research depth; riskier because Re-ID data/training and false associations add complexity |

### Recommended baseline stack

- **Detector baseline:** YOLOv8n or YOLOv10n at 320/416/640 input; include MobileNet-SSD or NanoDet if export is stable.
- **Tracker baselines:** SORT, ByteTrack, and OC-SORT. Add DeepSORT only if its Re-ID model can be measured fairly.
- **Datasets:** MOT17 for the main pedestrian benchmark; MOT20 for crowded scenes; UA-DETRAC for traffic; VisDrone for small objects; one short self-recorded sequence only as an external sanity check, never as the sole test set.
- **Metrics:** HOTA, IDF1, MOTA, ID switches, mostly-tracked/mostly-lost, detector AP/recall, end-to-end p50/p95 latency, FPS, dropped frames, peak RAM, model size, average power, joules/frame, and temperature.
- **Protocol:** fixed device power mode, warm-up, batch 1, same input resolution, synchronized timestamps, at least three repeated runs, and separate detector/tracker/end-to-end timings.

---

## 8. Final problem statement

### Title
**Energy-Aware Adaptive Multi-Object Tracking with Lightweight CNN Detectors on Resource-Constrained Edge Devices**

### Formal problem
Given a video stream $V = \{I_t\}_{t=1}^{T}$ and an edge device with a power, memory, and latency budget, design an online MOT pipeline that selects a detector configuration $c_t$ and detection interval $k_t$ from the current track state. The system must produce object boxes and persistent identities while minimizing energy and latency subject to an identity-quality constraint.

One useful constrained objective is:

$$
\max_{\pi} \; \mathrm{HOTA}(\pi) - \lambda_E E_{frame}(\pi) - \lambda_L L_{p95}(\pi)
$$

subject to:

$$
L_{p95} \le 150\,\mathrm{ms}, \quad E_{frame} \le E_{max}, \quad RAM_{peak} \le 1.5\,\mathrm{GB}, \quad \Delta \mathrm{HOTA} \le 10\%.
$$

Here $\pi$ is the adaptive policy, $E_{frame}$ is measured energy per processed frame, $L_{p95}$ is end-to-end 95th-percentile latency, and the HOTA loss is measured against the same detector running on every frame.

### Objective
Develop and evaluate a lightweight CNN detector plus ByteTrack or OC-SORT with an adaptive controller that changes detector frequency and/or input resolution using track confidence, Kalman uncertainty, object count, object scale, and camera-motion indicators.

### Measurable constraints

- Device targets: Raspberry Pi 5 CPU and Jetson Orin Nano GPU; optionally an x86 edge GPU for a reference.
- Batch size: 1; fixed camera stream; no offline batching.
- At least 15 end-to-end FPS on the selected target or an explicitly justified lower target.
- p95 latency <=150 ms and dropped frames reported.
- Peak RAM <=1.5 GB on the small-device experiment.
- HOTA loss <=10% relative to the every-frame baseline on the main benchmark.
- Report average power, joules/frame, temperature, and behavior before and after thermal stabilization.
- Report three seeds or repeated runtime trials and confidence intervals where appropriate.

### Expected contribution

1. A reproducible edge MOT implementation with detector, tracker, controller, and instrumentation.
2. A controlled comparison of always-on, detector-skipping, resolution-switching, and combined policies.
3. A Pareto frontier showing identity quality against p95 latency, RAM, power, and energy.
4. An evidence-based conclusion about when adaptive computation is beneficial and when it causes identity failure, especially in occlusion and crowded scenes.

This is a stronger contribution than claiming a new CNN block because it addresses a real deployment constraint and makes a falsifiable systems claim.

---

## 9. Datasets and hardware

### Datasets

| Dataset | Use | Notes |
|---|---|---|
| [MOT17](https://motchallenge.net/data/MOT17/) | Primary pedestrian MOT | Standard train/test, varied detectors and crowded scenes; use official split and public detections carefully |
| [MOT20](https://motchallenge.net/data/MOT20/) | Stress test for crowded pedestrian scenes | More severe occlusion and density; useful for measuring identity failures |
| [DanceTrack](https://github.com/DanceTrack/DanceTrack) | Nonlinear motion and similar appearance | Excellent for testing OC-SORT and occlusion, but not a general edge deployment dataset |
| [KITTI Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) | Road vehicles and camera motion | Good domain shift from pedestrians; licensing and download rules must be followed |
| [UA-DETRAC](https://detrac-db.rit.albany.edu/) | Traffic-camera vehicle MOT | Useful for fixed-camera edge analytics and vehicle counts |
| [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) | Small objects and aerial view | Tests resolution and small-object failure; compute cost can rise substantially |
| Self-recorded sequence | Deployment sanity check | Keep it separate from training and do not use it to make broad generalization claims |

### Hardware

| Hardware | Suggested role | Practical note |
|---|---|---|
| Raspberry Pi 5, 8 GB | CPU-only constrained target | Excellent for showing memory/latency limits; use TFLite/ONNX Runtime and low resolution |
| Jetson Orin Nano 8 GB | Main edge GPU | TensorRT FP16/INT8 makes a realistic robotics/surveillance target; log power mode and temperature |
| Jetson Xavier NX/Nano | Optional lower-power comparison | Useful only if available; older TensorRT support can complicate reproducibility |
| Intel N100 mini PC or laptop GPU | Reference edge-class system | Helps separate algorithm cost from ARM-specific runtime limitations |
| USB power meter or INA219/INA226 | Energy measurement | Sample at a known rate and subtract idle power if reporting dynamic energy |

**Hardware reporting minimum:** device model, OS, runtime version, driver/CUDA/TensorRT version, power mode, clock settings, input resolution, precision, batch size, warm-up frames, ambient temperature, and whether display/rendering is included.

---

## 10. Evaluation plan

1. Train or obtain the same detector variants and export them to the chosen runtime.
2. Run SORT, ByteTrack, and OC-SORT with the same detections and synchronized timestamps.
3. Establish an every-frame baseline on each device.
4. Evaluate detector skipping at $k \in \{1,2,3,5\}$ and resolution switching at 320/416/640 where feasible.
5. Add the controller only after fixed policies are measured; this isolates the controller's benefit.
6. Repeat on MOT17 and a stress set such as MOT20 or DanceTrack. Add UA-DETRAC or VisDrone for domain diversity if time permits.
7. Stress with frame drops, blur, compression, lighting shift, and camera motion.
8. Report mean and standard deviation over repeated runs, but keep test identities and frames fixed across policies.
9. Use paired bootstrap confidence intervals for HOTA/IDF1 differences and a Pareto plot for energy versus HOTA.
10. Release configuration files, measurement scripts, exported model checksums, and a limitations section.

### Minimum ablation table

| Variant | Detector interval | Resolution policy | Appearance branch | Purpose |
|---|---:|---|---|---|
| A | 1 | Fixed high | No | Accuracy upper baseline |
| B | 2/3/5 | Fixed high | No | Detector-skipping effect |
| C | 1 | Fixed low | No | Resolution effect |
| D | Adaptive | Adaptive | No | Main controller |
| E | Adaptive | Adaptive | Triggered tiny Re-ID | Occlusion extension |

---

## 11. Risks and scope control

- Do not train a detector from scratch. Fine-tune or use public weights, then focus the research effort on controlled deployment and evaluation.
- Do not make Raspberry Pi and Jetson FPS directly comparable without stating runtime and precision.
- Do not use only accuracy or only FPS. A tracker can have high FPS and unusable identity continuity.
- Do not let rendering, video decoding, or disk writes silently dominate timing; report both pipeline and model-only numbers.
- Do not use a private or self-recorded dataset as the main evidence.
- If power instrumentation is unavailable, state the limitation and use a repeatable external meter; do not infer energy from FLOPs.
- If the adaptive controller fails to preserve HOTA, that is a valid result. The study should explain the failure regime rather than tune until the negative result disappears.

## Bottom line

The strongest undergraduate problem is: **Can a lightweight CNN-based MOT pipeline dynamically trade detector computation for energy and latency while preserving identity quality under occlusion and crowding on real edge hardware?** The answer is measurable with public datasets, inexpensive hardware, established baselines, and a modest controller. It is novel enough to support a serious final-year report because the contribution is the constrained, end-to-end, cross-device evaluation and adaptive policy, not an unsupported claim of inventing another YOLO variant.

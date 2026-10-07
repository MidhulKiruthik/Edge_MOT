# RACE-MOT: Risk-Calibrated Adaptive Compute for Edge Multi-Object Tracking

**Comprehensive Academic Project Proposal & Faculty Presentation Report**  
**Document Type:** Formal Research Proposal, Technical Specification, and Presentation Pack  
**Project Outcome:** A proposed local edge-computing prototype, with implementation and research results gated by feasibility and measurement evidence; peer-reviewed conference paper optional.
**Target Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super with 7.6 GiB RAM, `p3768-0000+p3767-0005-super` compatibility, JetPack 7.2.1, CUDA 13.2, TensorRT 10.16.2, 25 W mode, and active fan telemetry recorded in private G1 inventory. Exact physical cooling and the final measurement boundary remain open.
**Author / Presenter:** Midhul Kiruthik M .
**Faculty / Department:** BalaMurugan .
**Date / Version:** October 2026 | Revision v1.0 (Comprehensive Expansion)  
**Project Repositories & Working Files:**
- Technical Research & Prior-Art Review: [deep_research_mot_edge_merged.md](../deep_research_mot_edge_merged.md)
- Product Documentation Pack: [product_docs/README.md](README.md)

**Current execution status:** Local MOT17, FP32 detector/output decoding,
OpenCV-reference parity, and pinned ByteTrack have bounded feasibility evidence.
Acceptable recall, terms/roles, the baseline, and G4 remain open. Further phone
work is temporarily deferred under D-33.
- Implementation Workspace: [race_mot/README.md](../race_mot/README.md)

---

## Executive Abstract

Multi-Object Tracking (MOT) in edge computing requires simultaneously localizing dynamic targets and maintaining trajectory continuity under strict compute, memory bandwidth, energy, and thermal budgets. Tracking-by-Detection (TBD) pipelines can spend a substantial share of their work on detector inference, but the share and device impact are workload- and hardware-dependent and are not project measurements. While heuristics (e.g., periodic skipping, confidence gating) and recent surrogate models (e.g., HSFSO, 2026) attempt to skip detection frames, this project investigates whether an individual track's avoidable identity failure can be predicted before the next detector decision.

**RACE-MOT (Risk-Calibrated Adaptive Compute for Edge Multi-Object Tracking)** proposes a product-first edge tracking system whose implementation and measurements remain to be completed. Its primary methodological claim is an online, per-track **counterfactual avoidable identity failure formulation ($Y_{g,t}^{(K,M)}$)**, predicting whether skipping the detector on the upcoming frame $t+1$ will cause an identity error that executing the detector would prevent. A candidate causal 1D Temporal Convolutional Network (TCN) processes causal kinematic, Kalman innovation, and association histories. Its parameter count, latency, calibration quality, and policy value will be measured after the model definition and device path are frozen. Frame-level aggregation is intended to guide a binary compute policy (`DETECT` vs. `SKIP`), backed by a low-cost scene-discovery guard.

The proposed pipeline comprises RTSP video ingestion, candidate YOLOX-Tiny with a runtime precision pending the G1 compatibility decision, ByteTrack, a risk head, binary policy scheduling, decision logging, and a local web dashboard hosted on an NVIDIA Jetson Orin Nano. It is intended to receive live video from a stationary smartphone running IP Webcam over a private USB-tethered local link. Evaluation is planned on MOT17 with sequence-grouped roles and MOT20 as a locked crowd/generalization set. Complete-pipeline energy, when a suitable external Jetson-input meter is available, will be measured per input frame with onboard rail telemetry reported separately as a diagnostic.

---

## 1. Introduction & Engineering Motivation

### 1.1 Application Context & Market Drivers (2024–2026)
Video analytics at the edge is undergoing a structural transition driven by bandwidth constraints, latency requirements, and privacy regulations. Market estimates vary by the definition of edge AI, hardware, and video analytics; they are motivation only and are not used as project requirements. Relevant application contexts include:
1. **Smart Urban Infrastructure & Occupancy Monitoring:** Monitoring pedestrian density and spatial distribution in public concourses, transit hubs, and commercial spaces without uploading raw video streams to public cloud endpoints.
2. **Edge-computing research:** constrained compute, energy, and latency can motivate adaptive tracking research, without extending this prototype to safety-critical use.
3. **Authorized academic demonstrations:** local, temporary tracks and operational diagnostics can support controlled demonstrations, without operational security or enforcement use.

### 1.2 The Edge Compute Bottleneck in Tracking-by-Detection
In modern Tracking-by-Detection (TBD) pipelines (e.g., YOLO + ByteTrack), the detector component dominates the computational footprint:
- **Detector Workload:** Detector inference can dominate a TBD pipeline, but timing and power depend on the exact model, runtime, device, and workload. RACE-MOT will measure them on the selected Jetson configuration.
- **Tracker Workload:** Kalman filtering and association are expected to be lower-cost than detector inference, but their timing and power will also be measured on the target device.

Running the detector on every frame is the reference policy for tracking comparison. In sustained low-power edge deployment it may affect temperature, clocks, memory pressure, and energy use; those effects require measurement on this device.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    THE EDGE TRACKING TRADEOFF DILEMMA                        │
├───────────────────────────────────────┬───────────────────────────────────────┤
│    DETECTOR ON EVERY FRAME (D=1)      │     AGGRESSIVE FRAME SKIPPING (D>1)   │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ • Reference tracking-quality condition│ • May lower detector work              │
│ • Baseline energy/latency measurement │ • Risk of drift and association errors │
│ • Sustained device effects TBD        │ • New entrants can be delayed          │
│ • Measured on the selected board      │ • Requires matched evaluation          │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

Skipping detector calls reduces average power draw, but leaves the tracker to propagate states blindly via linear motion models. In dynamic scenes with occlusions, direction changes, and intersecting trajectories, unguided skips result in **track drift, false termination, and identity swaps**.

### 1.3 Operational Failure Modes on Embedded Hardware

| Operating Condition | Physical & Algorithmic Failure Mode | What RACE-MOT Measures & Mitigates |
|---|---|---|
| **Continuous Per-Frame Detection** | Sustained load may change clocks, temperature, memory pressure, and deadline behavior; the effect is unmeasured on this device. | Measures true whole-board Joules per frame ($E_{\text{run}}/N_{\text{input}}$) and establishes the reference quality frontier. |
| **Unguided / Periodic Skipping** | Linear Kalman filter diverges during non-linear turns or crossing paths; identity is swapped upon detector re-acquisition. | Formulates an action-conditioned risk metric that prevents skips when tracks enter vulnerable geometric or crowding states. |
| **New Entrant During Skip Interval** | A new person enters the frame while existing tracks are deemed safe; a track-only skip scheduler fails to detect the entrant. | Integrates an asymmetric, low-overhead Scene-Discovery Guard that overrides `SKIP` $\to$ `DETECT` upon peripheral activity. |
| **Sustained Thermal Operation** | Prolonged execution may cause clock or thermal changes; the onset and latency impact are TBD on the selected board and cooling setup. | Monitors real-time SoC temperature, clocks, and throttling flags, logging deadline miss rates ($\text{DMR}$) under sustained runs. |

---

## 2. Problem Statement & Mathematical Formulation

### 2.1 Research Question
> **Can a lightweight, causal temporal model predict a statistically calibrated, per-track probability of avoidable identity failure conditioned on a proposed detector-skip action, and can this prediction be utilized to minimize whole-pipeline energy while strictly satisfying tracking quality and real-time deadline constraints on a physical Jetson Orin Nano?**

### 2.2 Formal Counterfactual Target Formulation
Let $t \in \{1, 2, \dots, T\}$ denote the discrete video frame index. Let $\mathcal{T}_t = \{1, \dots, N_t\}$ be the set of active tracks maintained by the tracker at frame $t$.

At anchor frame $t$, consider an active predicted track $i \in \mathcal{T}_t$ matched to an eligible, visible ground-truth identity $g$. We define an offline counterfactual simulation by forking the identical tracker internal state $S_t$ into two forward rollouts over a prediction horizon $K$:

1. **Skip Rollout ($a = \text{skip}$):** Omit detector inference at frame $t+1$; propagate active tracks using the frozen motion model. For frames $t+2$ through $t+K$, resume standard detector-every-frame updates.
2. **Detect Rollout ($a = \text{detect}$):** Execute detector inference at frame $t+1$; update tracker states via standard association. For frames $t+2$ through $t+K$, continue standard detector-every-frame updates.

```mermaid
flowchart TD
    S0["Tracker State S(t) at Frame t\n(Matched to GT Identity g)"]
    S0 -->|Action: SKIP at t+1| SK["Skip Branch\nPropagate State at t+1"]
    S0 -->|Action: DETECT at t+1| DT["Detect Branch\nRun Detector at t+1"]
    SK --> R1["Resume Detector Updates for t+2 ... t+K"]
    DT --> R2["Resume Detector Updates for t+2 ... t+K"]
    R1 --> E1{"Identity Failure\nF_skip = 1?"}
    R2 --> E2{"Identity Failure\nF_detect = 1?"}
    E1 -- Yes --> J{"F_skip=1 AND F_detect=0?"}
    E1 -- No --> N0["Y = 0 (No Failure under Skip)"]
    E2 -- Yes --> N1["Y = 0 (Unavoidable by Compute)"]
    E2 -- No --> J
    J -- True --> P["Y = 1 (Avoidable Identity Failure)"]
    J -- False --> N1
```

Let $F_{g,t}^{a, (K,M)} \in \{0, 1\}$ denote the binary failure indicator for branch $a \in \{\text{skip}, \text{detect}\}$. A failure $F = 1$ is declared if either of the following events occurs within the future horizon of $K$ frames:
1. **Identity Misassignment (Swap):** Track $i$ is assigned to a different ground-truth identity $g' \neq g$ while person $g$ remains visible.
2. **Persistent Track Loss:** Ground-truth identity $g$ remains unassigned to any active predicted track for at least $M$ consecutive visible frames ($M \le K$).

The ground-truth training label $Y_{g,t}^{(K,M)}$ is defined as:
$$Y_{g,t}^{(K,M)} = F_{g,t}^{\text{skip},(K,M)} \cdot \left(1 - F_{g,t}^{\text{detect},(K,M)}\right)$$

### 2.3 Interpretation and Label Censorship Rules
- **$Y = 1$ (Avoidable Failure):** Skipping frame $t+1$ directly causes an identity error or persistent loss that executing the detector at $t+1$ completely avoids. Compute expenditure is justified.
- **$Y = 0$ (Benign Skip):** Neither branch fails ($F^{\text{skip}} = 0, F^{\text{detect}} = 0$). Compute expenditure is redundant.
- **$Y = 0$ (Unavoidable Failure):** Both branches fail ($F^{\text{skip}} = 1, F^{\text{detect}} = 1$). The failure is driven by extreme occlusion or severe sensor noise that an immediate detector call cannot resolve. Spending compute is wasteful.
- **Censored Instances:** Ground-truth tracks that exit the field of view within $K$ frames, ambiguous boundary boxes, or tracks with $<3$ frames of historical trajectory are excluded from optimization to prevent training corruption.

---

## 3. Measurable Objectives

The project enforces a strict boundary between **functional prototype delivery** and **formal research verification**:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     RACE-MOT DUAL-TRACK OBJECTIVES                            │
├───────────────────────────────────────┬───────────────────────────────────────┤
│    TRACK A: PROTOTYPE DELIVERABLE     │     TRACK B: RESEARCH VERIFICATION    │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ 1. Autonomous execution on Jetson     │ 5. Counterfactual label generation    │
│ 2. Live H.264/RTSP stream ingestion   │ 6. Causal 1D TCN risk modeling        │
│ 3. Anonymous tracking & proxy count   │ 7. Statistical probability calibration│
│ 4. Local LAN dashboard & audit logs   │ 8. Physical Joules/frame benchmarking │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

1. **Autonomous Edge Ingestion & Execution:** Ingest an H.264/RTSP live stream from a stationary smartphone running IP Webcam over a private USB-tethered local link and execute on an NVIDIA Jetson Orin Nano with zero cloud dependencies.
2. **Local data-minimization design:** Provide temporary IDs and an active-track occupancy proxy without default raw-frame retention or biometric identities; verify this behavior during acceptance.
3. **Counterfactual Label Engine:** Implement an automated paired-rollout engine on MOT17 to extract causal failure labels ($Y_{g,t}^{(K,M)}$) without temporal leakage.
4. **Lightweight Risk Architecture:** Train and ablate a candidate causal 1D Temporal Convolutional Network (TCN) against compact GRU and temporal MLP baselines. The final parameter count will be derived from the frozen model definition.
5. **Statistical Probability Calibration:** Evaluate calibration on natural-prevalence sequences using Brier Score, ECE, and reliability plots. Acceptance thresholds remain TBD until the measurement contract is frozen.
6. **Bounded Decision Scheduling:** Construct an online runtime scheduler implementing a binary action space (`DETECT` vs. `SKIP`), enforced with hard limits on maximum consecutive skips ($S_{\max}$) and empty-track fallbacks.
7. **Multi-Faceted Comparative Evaluation:** Benchmark RACE-MOT against 7 distinct baseline classes: detector-every-frame, periodic skipping, confidence gating, Kalman uncertainty gating (ALBIREO-style), HSFSO-style surrogate scheduling, and EMO/RT-MOT variants.
8. **Empirical Edge Benchmarking:** Report complete-pipeline Joules per input frame, end-to-end $p50/p95$ latency, deadline-miss rates ($\text{DMR}$), peak memory, and sustained thermal profiles on the Jetson Orin Nano.

---

## 4. Literature Survey & Master Prior-Art Matrix

### 4.1 Master 9-Column Comparative Prior-Art Matrix
The following matrix evaluates RACE-MOT against all mandated and contemporary (2022–2026) prior works across 9 technical criteria:

| Method | What does it predict? | When does it predict it? | What input does it use? | What action does it trigger? | Per-track or frame-level? | Explicitly predicts avoidable identity failure? | Is it calibrated? | Used for next-frame detector scheduling? | Evaluated on physical edge hardware? |
|---|---|---|---|---|---|---|---|---|---|
| **RACE-MOT** *(Proposed)* | Proposed near-future score/probability $q_{i,t}$ of an avoidable ID failure over horizon $K$ | Intended online use after processing frame $t$, before frame $t+1$ | Bounding box kinematics, Kalman innovation covariance, track age/miss history, spatial density, low-cost frame cues | Intended binary gating: `DETECT` or `SKIP` | Proposed per-track score, aggregated to frame ($Q_t = \max_i q_{i,t}$) | **Proposed** via counterfactual paired rollout label $Y_{g,t}^{(K,M)}$ | **To be evaluated** with Platt/Isotonic calibration, Brier, and ECE | **Proposed core objective** | **Planned** Jetson evaluation; physical energy evidence is TBD |
| **ALBIREO** *(ACM/IEEE SEC 2026)* [10] | 10D kinematic state & covariance uncertainty ($P_t$) of object positions | Online, at frame $t$, prior to detector invocation | Past 2D box coordinates, aspect ratios, and estimated velocities/accelerations | Binary: Invoke detector vs. skip detector and extrapolate boxes via 10D Kalman Filter | Per-object covariance, aggregated via heuristic threshold | **NO** (Predicts spatial bounding box drift for *detection*; ignores identity switches & association) | **NO** (Uncalibrated Kalman covariance trace heuristic) | **YES** | **YES** (Jetson AGX Orin & Thor on BDD100K) |
| **HSFSO** *(Inf. Sciences 2026)* [9] | Scalar tracking score / aggregate MOTA degradation under candidate skip ratios | Offline / Segment-level (hierarchical global + local) | Optical flow vectors, inter-frame pixel differences, scene motion into an MLP surrogate | Selects global skip ratio $R$, then optical-flow frame selection for which frames to skip | **Scene/Sequence-level** (global surrogate; local selection uses motion energy) | **NO** (Predicts scalar dataset MOTA; no per-track counterfactual identity error target) | **NO** (Standard uncalibrated regression loss) | **NO** (Segment-level schedule allocation, not per-track causal online gating) | **NO** (Desktop GPU evaluation using YOLOv8-s + ByteTrack on MOT17/20) |
| **EMO** *(arXiv 2023)* [5] | Does not predict failure; measures inter-frame visual change & track variance | Online, per-frame | Inter-frame feature similarity, bounding box overlap, sliding window timers | Periodic skip with similarity-triggered detector re-invocation | Frame & sliding-window heuristics | **NO** (Heuristic similarity thresholds) | **NO** | **YES** (Heuristic frame skipping) | **YES** (Jetson boards; latency/throughput measurement study) |
| **RT-MOT** *(RTSS 2022)* [6] | Future tracking confidence & expected utility degradation under deadline limits | Online, before scheduling pipeline tasks | Detector confidence history, Kalman residual state, task execution deadlines | Workload adjustment: selects model branches or drops tasks to satisfy hard real-time deadlines | Frame & task queue workload level | **NO** (Tracks heuristic confidence decay, not identity swap causality) | **NO** (Heuristic confidence estimates) | **Partly** (Schedules tasks under RT constraints; no binary skip-action counterfactuals) | **YES** (Embedded RT platforms, CPU/GPU timing) |
| **Split and Connect** *(IEEE TMM 2022)* [8] | Temporal locations of potential ID switches along existing trajectories | **Offline / Post-processing** (on completed/windowed tracklets) | Historical tracklet trajectories (box coordinates & appearance) via dilated 1D temporal convolutions | **Tracklet Repair**: Splits trajectory at predicted switch point, reconnects via attention | **Per-tracklet** | **NO** (Predicts that an error *already occurred* in past frames to repair it) | **NO** (Uncalibrated classification logits) | **NO** (Zero scheduling role; downstream tracklet booster) | **NO** (Server GPUs on MOT17/MOT20) |
| **GLoMOT** *(AAAI 2026)* [11] | Bipartite graph edge affinities (association likelihood) under Low Frame Rates | Online, during association step at frame $t$ | Dynamic node buffer (long-term memory), context-aware spatial/appearance weights, pseudo-depth | Solves graph association across large temporal gaps ($\Delta t \gg 1$) | Pairwise track-to-detection edge level | **NO** (Solves association; does not predict whether future skips will induce error) | **NO** (Raw softmax/sigmoid edge affinities) | **NO** (Passive tracker operating on an externally fixed low FPS) | **NO** (Server GPUs on MOT17, DanceTrack, VisDrone) |
| **LUKF-Track** *(Entropy 2026)* [12] | Manifold state estimation (position, velocity) via Unscented Kalman Filter | Online, at every time-step during state filtering | 2D box detections and Lie-group manifold $SE(2) \times \mathbb{R}^2$ geometry | Propagates kinematic states; feeds Mahalanobis/IoU association matrices | Per-track kinematic filtering | **NO** (Kinematic state filter; does not predict failure or evaluate skip decisions) | **NO** (Uncalibrated Kalman covariance propagation) | **NO** (Always runs on every frame; tracker-only with no detector control) | **NO** (Server GPU benchmark on dense occlusion scenes) |

### 4.2 Comprehensive Synthesis of Prior-Art Limitations

#### 1. ALBIREO (ACM/IEEE SEC 2026) [10]
*Focus:* Video Object Detection acceleration on Jetson AGX Orin/Thor.  
*Limitation for MOT:* Albireo models bounding box spatial drift via a 10D Kalman filter. It does not perform multi-object data association across occlusions and has no concept of trajectory IDs, identity swaps, or track fragmentation. Its rescue trigger is an uncalibrated covariance heuristic. When two pedestrians cross, Albireo's linear filter continues straight, causing an identity swap upon the next detector run that its detection-only metric ($AP@50$) completely ignores.

#### 2. HSFSO (Information Sciences 2026) [9]
*Focus:* Hierarchical surrogate-based frame skip optimization for MOT17/MOT20.  
*Limitation for MOT:* HSFSO formulates frame skipping as an offline sequence-level optimization problem. An MLP surrogate predicts aggregate MOTA from global video features to choose a fixed skip ratio $R$, followed by optical-flow-based frame selection. This cannot handle local, per-track transient risks (e.g., eight stable pedestrians and one entering an occlusion). Furthermore, dense optical flow is prohibitively expensive on low-power edge SoCs, and HSFSO's experiments are confined to desktop GPUs.

#### 3. RT-MOT (IEEE RTSS 2022) [6] & EMO (arXiv 2023) [5]
*Focus:* Workload adaptation and real-time scheduling on edge devices.  
*Limitation for MOT:* RT-MOT schedules execution branches based on heuristic tracker confidence and worst-case execution time (WCET). EMO uses sliding-window timers and perceptual hashing. Neither models the causal, counterfactual outcome of a skip decision. A high-confidence track heading toward a pillar will be skipped by RT-MOT, resulting in an unrecoverable identity swap.

#### 4. Split and Connect (IEEE TMM 2022) [8]
*Focus:* Deep temporal prediction of identity switch positions.  
*Limitation for MOT:* While Split and Connect demonstrates that temporal 1D convolutions can detect ID switch locations, it operates strictly as an **offline or sliding-window tracklet repair mechanism**. It executes *after* tracking is finished to split corrupted trajectories. It cannot make forward-looking, real-time compute allocation decisions to prevent failures before they occur.

#### 5. GLoMOT (AAAI 2026) [11] & LUKF-Track (Entropy 2026) [12]
*Focus:* Robust tracking under low-frame-rate or non-linear motion conditions.  
*Limitation for MOT:* Both represent passive association models designed to tolerate externally downsampled video streams. They possess no control over the upstream detector and do not optimize system energy consumption.

---

## 5. Research Gap & Formal Hypotheses

### 5.1 The Unoccupied Research Gap
The intersection of multi-object tracking, causal machine learning, and edge systems reveals a critical unaddressed question:
> **No existing method predicts an online, per-track, skip-action-conditioned counterfactual failure probability to dynamically govern edge detector invocations under physical whole-board energy ($\text{Joules/frame}$) and thermal constraints.**

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                     THE UNMET RESEARCH GAP MATRIX                             │
├────────────────────────────────┬──────────────────────────────────────────────┤
│ Prior Works (ALBIREO, HSFSO)   │ Focus on Detection AP or Aggregate MOTA      │
│ Prior Works (EMO, RT-MOT)      │ Rely on Uncalibrated Heuristic Confidence    │
│ Prior Works (Split & Connect)  │ Perform Post-Hoc Offline Tracklet Repair     │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ RACE-MOT UNIFIED SOLUTION      │ Calibrated, Per-Track Counterfactual Failure │
│                                │ Prediction for Online Edge Detector Gating   │
└────────────────────────────────┴──────────────────────────────────────────────┘
```

### 5.2 Formal Hypotheses to be Tested
- **Hypothesis 1 ($H_1$ — Predictive Validity & Calibration):** A causal temporal network (1D TCN) ingesting multi-frame kinematic, Kalman residual, and association features achieves statistically superior PR-AUC and Brier scores compared to static classifiers, raw confidence thresholds, and Kalman covariance heuristics when predicting avoidable identity failures.
- **Hypothesis 2 ($H_2$ — Pareto Energy Superiority):** Conditioning next-frame detector invocations on calibrated risk $Q_t \ge \tau$ achieves a superior empirical Pareto frontier (lower Joules per input frame at non-inferior HOTA/IDF1) compared to periodic skipping, ALBIREO-style uncertainty triggers, and HSFSO surrogates on an NVIDIA Jetson Orin Nano.
- **Hypothesis 3 ($H_3$ — Discovery Guard Efficacy):** A peripheral activity guard may reduce new-entrant acquisition delay during skip sequences, but its false-trigger rate, overhead, and full-pipeline value must be measured against risk-only scheduling.
- **Hypothesis 4 ($H_4$ — Thermal & Deadline Robustness):** Adaptive scheduling may reduce redundant detector calls and change sustained thermal/deadline behavior; detector reduction, throttling, run duration, and DMR limits are all to be established by G3 and tested on the target device.

---

## 6. Proposed Methodology & System Architecture

### 6.1 End-to-End System Architecture
The RACE-MOT runtime architecture decouples video acquisition, edge inference, adaptive scheduling, and telemetry logging:

```mermaid
flowchart TD
    subgraph Input_Layer ["Input & Ingestion Layer"]
        CAM["Stationary Phone Camera\n(provisional 1280x720 @ 15fps H.264)"] -->|RTSP over USB-tethered private link| ADAPT["Input Adapter &\nTimestamp Validator"]
        FILE["Offline MOT17 / MOT20\nBenchmark Replay Files"] --> ADAPT
    end

    subgraph Decision_Control ["Decision & Scheduling Layer"]
        ADAPT -->|Frame t| SCHED_CHECK{"Controller Command\nfrom Frame t-1"}
        SCHED_CHECK -->|DETECT| DET["YOLOX-Tiny Engine\n(runtime precision pending G1 decision)"]
        SCHED_CHECK -->|SKIP| GUARD_EVAL{"Discovery Guard:\nPeripheral Motion?"}
        GUARD_EVAL -- Yes --> DET
        GUARD_EVAL -- No --> PROP["ByteTrack Predictor\n(Kalman Propagation)"]
        DET --> ASSOC["ByteTrack Association\n(High/Low Score Matching)"]
        PROP --> ASSOC
    end

    subgraph Risk_Inference ["Causal Risk Modeling Layer"]
        ASSOC --> FEAT["Feature Extractor:\nKinematics + Covariance + Cues"]
        FEAT --> TCN["Causal 1D TCN Head\n(Receptive Field H=8)"]
        TCN --> CALIB["Probability Calibration\n(Platt / Isotonic)"]
        CALIB --> AGG["Conservative Aggregator:\nQ(t) = max_i q(i,t)"]
        AGG --> DECIDE["Action Decider for t+1:\nQ(t) >= tau OR S >= S_max"]
        DECIDE -->|Command for t+1| SCHED_CHECK
    end

    subgraph Output_Telemetry ["Output & Telemetry Layer"]
        ASSOC --> UI["Local Web Dashboard\n(Anonymous Overlays & Counts)"]
        DECIDE --> LOG["XAI Decision Logger\n(Reason Code + Hardware Telemetry)"]
        LOG --> JTS["Tegrastats / Meter\n(Power, Clocks, Joules)"]
    end
```

### 6.2 Per-Frame Execution Pipeline Protocol
For every incoming frame $t$, the execution engine executes the following sequential lifecycle:

```
Algorithm 1: RACE-MOT Per-Frame Execution Cycle
Input : Input frame I_t with monotonic timestamp T_t
        Planned action A_t in {DETECT, SKIP} scheduled at frame t-1
        Active track collection Tracks_{t-1}
Output: Updated tracks Tracks_t, Action A_{t+1} scheduled for frame t+1, Telemetry log L_t

1. Ingest frame I_t; validate dimensions and record arrival timestamp;
2. if A_t == SKIP and Tracks_{t-1} is not empty then
3.     Compute low-cost peripheral difference D_periph outside bounding boxes;
4.     if D_periph > Delta_guard then
5.         A_t <- DETECT;  // One-way safety override
6.         ReasonCode <- GUARD_OVERRIDE;
7.     end
8. end
9. if A_t == DETECT then
10.    Detections_t <- YOLOX_Inference(I_t);  // TensorRT FP16
11.    Tracks_t <- ByteTrack_Associate(Tracks_{t-1}, Detections_t);
12.    ConsecutiveSkips <- 0;
13. else
14.    Tracks_t <- ByteTrack_Propagate(Tracks_{t-1});  // Kalman state update only
15.    ConsecutiveSkips <- ConsecutiveSkips + 1;
16. end
17. if Tracks_t is empty then
18.    A_{t+1} <- DETECT;
19.    ReasonCode <- NO_ACTIVE_TRACKS;
20. else if ConsecutiveSkips >= S_max then
21.    A_{t+1} <- DETECT;
22.    ReasonCode <- MAX_SKIP_EXCEEDED;
23. else
24.    Extract causal feature vectors X_{i,t} for each track i in Tracks_t;
25.    Compute raw failure logit z_{i,t} <- TCN(X_{i,t});
26.    Calibrate probability q_{i,t} <- Calibrate(z_{i,t});
27.    FrameRisk Q_t <- max_i (q_{i,t});
28.    if Q_t >= tau then
29.        A_{t+1} <- DETECT;
30.        ReasonCode <- RISK_EXCEEDED;
31.    else
32.        A_{t+1} <- SKIP;
33.        ReasonCode <- RISK_ACCEPTABLE;
34.    end
35. end
36. Record hardware telemetry (tegrastats: mW, deg C, clocks);
37. Emit render overlay to Local Web Dashboard and append L_t to audit log;
```

### 6.3 Constrained Mathematical Optimization Formulation
RACE-MOT formulates detector invocation scheduling as a constrained optimization problem:
$$\min_{\pi} \; E_{\text{total/input frame}}(\pi) = \frac{1}{N_{\text{input}}} \int_{0}^{T_{\text{run}}} P_{\text{system}}(t) \, dt$$
subject to:
$$\begin{aligned}
\text{HOTA}(\pi) &\ge \text{HOTA}_{\text{base}} - \delta_H \\
\text{IDF1}(\pi) &\ge \text{IDF1}_{\text{base}} - \delta_I \\
p95(L_{\pi}) &\le D \\
\text{DMR}(\pi) &= \frac{1}{N_{\text{input}}} \sum_{t=1}^{N_{\text{input}}} \mathbb{I}(L_t > D) \le d_{\max} \\
\text{RAM}_{\text{peak}}(\pi) &\le R_{\max} \\
T_{\text{sustained}}(\pi) &\le T_{\max}
\end{aligned}$$
where $P_{\text{system}}(t)$ represents the declared Jetson input-power boundary; $\delta_H$ and $\delta_I$ represent non-inferiority margins; $D$ is the end-to-end deadline derived from the selected cadence and frozen at G3; and $T_{\max}$ is the sustained thermal limit selected from device evidence and the use case at G3. These values are currently TBD.

---

## 7. Dataset Protocol & Leakage Controls

### 7.1 Dataset Profiles & Allocations
To eliminate data snooping and ensure unbiased evaluation, datasets are strictly partitioned into functional tiers:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                      DATASET PARTITIONING & ROLES                             │
├───────────────────────────────────────┬───────────────────────────────────────┤
│ MOT17 BENCHMARK (Grouped Folds)       │ • Model Training, Probability         │
│ 7 Unique Source Videos (5,316 frames) │   Calibration & Threshold Tuning      │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ MOT20 BENCHMARK (Locked Stress Set)   │ • Out-of-Distribution Dense Crowd     │
│ 4 Sequences (149.7 boxes/frame)       │   Generalization Evaluation           │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ LIVE PHONE RTSP STREAM                │ • End-to-End System Demo, Latency &   │
│ 1280x720 @ 15fps (Private LAN)        │   Hardware Telemetry Verification     │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

| Dataset / Input Stream | Scale & Characteristics | Dedicated Role | Leakage & Integrity Controls |
|---|---|---|---|
| **MOT17 Training Set** [3, 13] | 7 unique source video sequences (5,316 unique video frames; 112,297 ground-truth bounding boxes). Distributed across 21 subfolders corresponding to DPM, Faster R-CNN, and SDP detection files. | Model fitting, probability calibration, policy validation, and grouped evaluation roles selected before final scoring. | All detector variants of any source video remain in the same role. The exact grouped/cross-fitted allocation is TBD until G2, with no frame-level or tracklet-level random splitting. |
| **MOT20 Training Set** [4, 14] | 4 extremely crowded sequences (8,931 frames, 1,336,920 annotations; average density of $149.7\text{ pedestrians/frame}$). | Frozen out-of-distribution dense-crowd stress and generalization test. | Kept strictly locked. Zero feature selection, threshold tuning, or calibration parameter fitting on MOT20. Evaluated post-freeze. |
| **Live Smartphone RTSP Stream** | Authorized stationary smartphone feed over private USB tethering; 1280×720 at 15 FPS remains a proposed profile. Current G1 decode evidence observed 1920×1080 with backend cadence metadata deemed unreliable. | Hardware integration, live demonstration, and system fault-injection testing. | Unannotated. Explicitly designated as a demonstration vehicle; benchmark scores are never reported from this unannotated feed. |

### 7.2 Sequence-Grouped Cross-Fitting Protocol
Because MOT17 contains only seven independent source scenes, random frame or
tracklet splitting would leak scene information. The project will use grouped
or cross-fitted roles so that training, natural-prevalence calibration, policy
validation, and final evaluation are separated at the source-scene level.
The exact role permutations and any fixed outer evaluation fold will be frozen
at G2/G3 before model or policy selection. All detector variants of one source
scene remain together, and MOT20 remains locked until final transfer scoring.

---

## 8. AI/ML Model Design & Feature Engineering

### 8.1 Model Architecture: Causal 1D Temporal Convolutional Network (TCN)
The primary risk predictor is a compact causal 1D TCN candidate. Its parameter count and execution cost remain TBD until the model definition and selected device runtime are frozen.

```
Input History: X_{i,t} (Dimension: 8 time steps x 18 features)
  │
  ├──> Conv1D (Filters=32, Kernel=3, Dilation=1, Causal Padding) + LeakyReLU
  │      │
  │      └──> Residual Block 1:
  │             Conv1D (Filters=32, Kernel=3, Dilation=2, Causal) + LeakyReLU
  │             Conv1D (Filters=32, Kernel=3, Dilation=2, Causal)
  │             [Skip Connection: + Input]
  │
  ├──> Global Average Pooling across Time Dimension
  │
  ├──> Fully Connected Layer (32 -> 16) + LeakyReLU + Dropout(0.1)
  │
  └──> Output Dense Layer (16 -> 1) -> Raw Failure Logit z_{i,t}
         │
         └──> Post-Hoc Calibrator (Platt / Isotonic) -> Calibrated Risk q_{i,t}
```

- **Receptive Field:** $H = 8$ historical time steps.
- **Parameter count:** TBD until this exact model definition is implemented and counted.
- **Execution latency:** TBD until measured per track batch on the selected Jetson configuration; no sub-millisecond target is assumed.

### 8.2 Comprehensive Feature Dictionary
For every active track $i$ at frame $t$, the feature extractor compiles an 18-dimensional feature vector $x_{i,t}$:

| Feature Index | Group | Feature Name | Description & Mathematical Normalization |
|---|---|---|---|
| $f_1, f_2$ | Kinematics | Normalized Center $(c_x, c_y)$ | Bounding box center normalized by frame width $W$ and height $H$. |
| $f_3, f_4$ | Kinematics | Normalized Dimensions $(w, h)$ | Bounding box width and height normalized by frame dimensions. |
| $f_5, f_6$ | Kinematics | Instantaneous Velocity $(v_x, v_y)$ | $\Delta c / \Delta t$, normalized by actual inter-frame elapsed timestamp. |
| $f_7$ | Kinematics | Velocity Change (Acceleration) | $\|\mathbf{v}_t - \mathbf{v}_{t-1}\|_2 / \Delta t$. |
| $f_8$ | Motion Uncertainty | Kalman Innovation Residual | Normalized distance between detector measurement and Kalman prior. |
| $f_9$ | Motion Uncertainty | Covariance Trace $\text{Tr}(P_t)$ | Trace of the spatial position submatrix of the Kalman covariance $P_t$. |
| $f_{10}$ | Association | Detection Confidence History | Score of matched bounding box; zero if extrapolated during skip. |
| $f_{11}$ | Association | Association Margin | Distance margin between best match and second-best candidate in Hungarian cost matrix. |
| $f_{12}$ | Lifecycle | Track Age | Total frames elapsed since track initialization. |
| $f_{13}$ | Lifecycle | Miss Count | Consecutive frames where track received no matched detection. |
| $f_{14}$ | Lifecycle | Local Spatial Overlap | Maximum IoU between track $i$ and any neighboring active track $j \neq i$. |
| $f_{15}$ | Lifecycle | Density Index | Number of active tracks within a $2.5\times$ bounding box radius. |
| $f_{16}$ | Scheduling History | Wall Time Elapsed | True elapsed seconds since track $i$ last received a detector update. |
| $f_{17}$ | Scheduling History | Consecutive Skips | Count of consecutive frame skips executed by the scheduler. |
| $f_{18}$ | Scheduling History | Network Frame Drop Indicator | Binary flag indicating if preceding frame arrival suffered network/decode drop. |

### 8.3 Statistical Probability Calibration Protocol
Raw classification logits $z_{i,t}$ produced by neural networks are uncalibrated and cannot be interpreted as true probabilities. RACE-MOT integrates post-hoc calibration:
1. **Platt Scaling:** Logistic regression fit on validation logits:
   $$q_{i,t} = \sigma(a \cdot z_{i,t} + b)$$
2. **Isotonic Regression:** Non-parametric monotonic step calibration.

Calibration fidelity is evaluated via:
- **Brier Score:** Mean squared error between predicted probability $q$ and binary outcome $Y$:
  $$\text{BS} = \frac{1}{N} \sum_{k=1}^N (q_k - Y_k)^2$$
- **Expected Calibration Error (ECE):** Grouping predictions into $M=10$ equal-width bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 9. Hardware Manifest & Experimental Protocol

### 9.1 Edge Hardware & Software Manifest
The following inventory is captured for G1; benchmark configuration remains unfrozen:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                    JETSON ORIN NANO HARDWARE MANIFEST                         │
├───────────────────────────────────────────────────────────────────────────────┤
│ SoC: NVIDIA Orin (Architecture: Ampere GPU, 1024 CUDA Cores, 32 Tensor Cores) │
│ CPU: 6-core ARM Cortex-A78AE v8.2 64-bit @ 1.5 GHz                            │
│ Memory: 7.6 GiB; storage and carrier/cooling detail remain to record         │
│ Operating system / JetPack: Ubuntu 24.04.5 / JetPack 7.2.1                   │
│ CUDA / TensorRT / Python: CUDA 13.2 / TensorRT 10.16.2 / Python 3.12.3       │
│ Power mode: 25 W recorded; clocks/cooling frozen before comparison runs       │
│ Energy: external Jetson-input meter preferred; onboard rails diagnostic only │
└───────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Proposal-stage sequence

These proposal milestones do not replace the governing G0–G4 gates in the decision log and roadmap.

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                   RACE-MOT 6-STAGE VERIFICATION GATES                         │
├───────┬───────────────────────────────┬───────────────────────────────────────┤
│ GATE  │ WORK PRODUCT                  │ PASS CRITERIA BEFORE PROCEEDING       │
├───────┼───────────────────────────────┼───────────────────────────────────────┤
│ **M1**│ Device Manifest & RTSP Probe  │ Exact Jetson inventory captured;      │
│       │                               │ phone RTSP behavior and cadence      │
│       │                               │ recorded; profile remains provisional.│
├───────┼───────────────────────────────┼───────────────────────────────────────┤
│ **M2**│ Baseline Pipeline Feasibility │ Selected YOLOX-Tiny runtime + ByteTrack│
│       │                               │ runs cleanly; zero memory leakage.    │
├───────┼───────────────────────────────┼───────────────────────────────────────┤
│ **M3**│ Reference Measurement Freeze  │ Detector-every-frame baseline frozen; │
│       │                               │ HOTA, IDF1, and Joules/frame logged.  │
├───────┼───────────────────────────────┼───────────────────────────────────────┤
│ **M4**│ Counterfactual Label & TCN    │ Grouped MOT17 labels generated;       │
│       │                               │ TCN trained and calibration evaluated │
│       │                               │ against predeclared G3 criteria.      │
├───────┼───────────────────────────────┼───────────────────────────────────────┤
│ **M5**│ Closed-Loop Adaptive Scheduler│ Thresholds tau and S_max selected;    │
│       │                               │ full comparative baselines evaluated. │
├───────┼───────────────────────────────┼───────────────────────────────────────┤
│ **M6**│ Sustained Thermal & Demo Run  │ Sustained run and live demo completed │
│       │                               │ under predeclared device criteria.    │
└───────┴───────────────────────────────┴───────────────────────────────────────┘
```

---

## 10. Novelty Claims & Contributions (Mapped to University Rubric)

For faculty review, RACE-MOT presents one primary methodological claim and two supporting evaluation dimensions. Whether the method is a contribution beyond prior work remains to be established by direct comparison and measured evidence:

```
┌───────────────────────────────────────────────────────────────────────────────┐
│              ACADEMIC RUBRIC CONTRIBUTIONS MAPPING                            │
├─────────────────────────────────┬─────────────────────────────────────────────┤
│ UNIVERSITY RUBRIC CRITERIA      │ OUR CLAIMABLE NOVELTY CONTRIBUTION          │
├─────────────────────────────────┼─────────────────────────────────────────────┤
│ 1. Novel Problem Formulation &  │ Counterfactual Skip-Action-Conditioned      │
│    Enhanced Data Preprocessing  │ Avoidable Identity Failure Formulation      │
│    Technique                    │ ($Y_{g,t}^{(K,M)}$) and Paired Rollouts     │
├─────────────────────────────────┼─────────────────────────────────────────────┤
│ 2. Improved AI Model & Hybrid   │ Causal 1D TCN Meta-Controller with          │
│    Approach / Optimization      │ Calibrated Probabilities and Asymmetric     │
│                                 │ Kinematic-Covariance Fusion                 │
├─────────────────────────────────┼─────────────────────────────────────────────┤
│ 3. Explainable AI (XAI) &       │ Fully Local Jetson Orin Deployment with     │
│    Practical Edge Deployment    │ Real-Time Decision Auditing and Physical    │
│    Implementation               │ Whole-Board Energy/Thermal Benchmarking     │
└─────────────────────────────────┴─────────────────────────────────────────────┘
```

### Contribution 1: Novel Problem Formulation & Enhanced Preprocessing
*(Category: Novel problem formulation or application & Enhanced data preprocessing)*
- **The Proposed Method Claim:** Existing frame schedulers use varied heuristics, surrogates, or association objectives. We propose an **online, causal, per-track avoidable identity failure target ($Y_{g,t}^{(K,M)}$)** that isolates failures caused by skipping that a detector invocation would avoid. The claim is not described as first until the literature review and direct comparisons support that wording.
- **The Data Pipeline Innovation:** An automated, counterfactual paired-rollout data generation engine that forks tracker state on MOT17 annotations offline, creating an unbiased, leak-free training corpus.

### Contribution 2: Improved AI Model & Feature Engineering
*(Category: An improved AI model or hybrid approach & Feature engineering optimization)*
- **Model and Feature Evaluation:** We will compare a compact causal 1D TCN with matched temporal and static baselines. Parameter count, latency, calibration, and policy value will be measured rather than promised.
- **Algorithmic Evaluation:** The candidate model fuses causal kinematic, Kalman covariance, and lifecycle features into a score that is calibrated only if held-out Brier/ECE tests support probability interpretation.

### Contribution 3: Explainable Edge Deployment & Physical Benchmarking
*(Category: Explainable AI component, deployment framework, or practical real-world implementation)*
- **Deployment and Evaluation Evidence:** The product target is a local Jetson prototype receiving authorized IP Webcam video over USB tethering without cloud runtime dependencies. Completion, latency, thermal behavior, and energy availability will be reported from the actual device.
- **Auditability:** Each planned and executed action is intended to log a reason code (for example `RISK_EXCEEDED`, `MAX_SKIP_EXCEEDED`, or `GUARD_OVERRIDE`) together with calibration status and timing. Logging is an audit feature, not a claim of causal explainability or total transparency.
- **Physical Measurement:** If a suitable external meter is available, report complete-pipeline Joules per input frame with onboard rail telemetry separately. Do not claim energy savings until a matched baseline and constraints are measured.

---

## 11. Bibliographic References

```
[1] J. Luiten et al., "HOTA: A Higher Order Metric for Evaluating Multi-object Tracking," International Journal of Computer Vision (IJCV), vol. 129, pp. 548–578, 2021. https://doi.org/10.1007/s11263-020-01375-2

[2] Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box," European Conference on Computer Vision (ECCV), 2022. https://doi.org/10.1007/978-3-031-20047-2_1

[3] A. Dendorfer et al., "MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking," International Journal of Computer Vision (IJCV), vol. 129, pp. 845–881, 2021. https://doi.org/10.1007/s11263-020-01393-0

[4] P. Dendorfer et al., "MOT20: A Benchmark for Multi Object Tracking in Crowded Scenes," arXiv:2003.09003, 2020. https://arxiv.org/abs/2003.09003

[5] S. V. Ganesh et al., "Fast and Resource-Efficient Object Tracking on Edge Devices: A Measurement Study," arXiv:2309.02666, 2023. https://arxiv.org/abs/2309.02666

[6] D. Kang et al., "RT-MOT: Confidence-Aware Real-Time Scheduling Framework for Multi-Object Tracking Tasks," in Proceedings of the 43rd IEEE Real-Time Systems Symposium (RTSS), 2022, pp. 318–330. https://doi.org/10.1109/RTSS55097.2022.00035

[7] K. Nishimura et al., "SDOF-Tracker: Fast and Accurate Multiple Human Tracking by Skipped-Detection and Optical-Flow," IEICE Transactions on Information and Systems, vol. E105-D, no. 11, pp. 1994–2003, 2022. https://doi.org/10.1587/transinf.2022EDP7022

[8] G. Wang et al., "Split and Connect: A Universal Tracklet Booster for Multi-Object Tracking," IEEE Transactions on Multimedia (TMM), vol. 25, pp. 6757–6769, 2022. https://doi.org/10.1109/TMM.2022.3140919

[9] H. Han et al., "Hierarchical Surrogate-Based Frame Skip Optimization for Multi-Object Tracking," Information Sciences, vol. 753, article 123610, 2026. https://doi.org/10.1016/j.ins.2026.123610

[10] A. Taherin, J. Cano, B. Ren, Y. Wang, and D. Kaeli, "Albireo: Adaptive, Energy-Efficient Inference Framework for Video Object Detection on the Edge," accepted at the ACM/IEEE Symposium on Edge Computing (SEC), 2026; arXiv:2609.29648. https://arxiv.org/abs/2609.29648

[11] Y. Hu, J. Hua, G. Wu, Y. Yang, A. Suzuki, and Z. Wang, "GLoMOT: Efficient Online GNN-based Low-Frame-Rate Multi-Object Tracker," in Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-26), vol. 40, no. 6, pp. 4959–4967, 2026. https://doi.org/10.1609/aaai.v40i6.42500

[12] X. Wang, L. Liu, and F. Li, "A Multi-Object Tracking Method with an Unscented Kalman Filter on a Lie Group Manifold," Entropy, vol. 28, no. 1, article 103, 2026. https://doi.org/10.3390/e28010103

[13] J. Luiten, "TrackEval: Standardized Tracking Evaluation Tool," 2020. https://github.com/JonathonLuiten/TrackEval

[14] MOTChallenge, "MOT20 Benchmark Official Distribution." https://motchallenge.net/data/MOT20/

[15] M. Liu, X. Ding, and W. Du, "Continuous, Real-Time Object Detection on Mobile Devices without Offloading," in Proceedings of the 40th IEEE International Conference on Distributed Computing Systems (ICDCS), 2020, pp. 976–986. https://doi.org/10.1109/ICDCS47774.2020.00085

[16] K. Nalaie, R. Xu, and R. Zheng, "DeepScale: Online Frame Size Adaptation for Multi-object Tracking on Smart Cameras and Edge Servers," in Proceedings of the 7th IEEE/ACM International Conference on Internet-of-Things Design and Implementation (IoTDI), 2022. https://doi.org/10.1109/IoTDI54339.2022.00010

[17] X. Chi et al., "Edge-Assisted Video Object Detection With Adaptive Resource Allocation and Model Selection," IEEE Transactions on Mobile Computing (TMC), Early Access, 2026. https://doi.org/10.1109/TMC.2026.3719713

[18] Z. Ge, S. Liu, F. Wang, Z. Li, and J. Sun, "YOLOX: Exceeding YOLO Series in 2021," arXiv:2107.08430, 2021. https://arxiv.org/abs/2107.08430

[19] NVIDIA Corporation, "NVIDIA Jetson Orin Nano Technical Reference Manual & Developer Guide," 2024. https://developer.nvidia.com/embedded/jetson-modules
```

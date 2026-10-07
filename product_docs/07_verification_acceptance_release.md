# RACE-MOT Verification, Acceptance, and Release Plan

**Status:** Draft v0.2 — this is a plan for later verification; no tests have been run by creating this document.

## 1. Verification layers

| Layer | What to verify | Evidence to retain |
|---|---|---|
| Data/label | Dataset parsing, sequence grouping, counterfactual labels, timestamps, no leakage | Dataset manifest, sampled hand audits, label statistics, split manifest, checksums |
| Model | Predictor output, calibration, sequence generalization, feature ablation | Config, weights/hash, seeds, PR-AUC, Brier, ECE/reliability plots, sequence-level confidence intervals |
| Policy | Binary decisions, threshold, max-skip fallback, no future inputs | Trace review, hand-calculated scenarios, decision reason counts |
| Pipeline | Phone RTSP input, MOT-file replay, detector, tracker, risk model, UI, exports, stream-loss handling, stop/restart | Run logs, sample output, phone-network configuration, failure/recovery records |
| Device | End-to-end latency, memory, power, sustained temperature/throttling | Hardware manifest, synchronized traces, repeated run summaries |
| Responsible use | Local-only default, data deletion, access, export controls | Data-flow checklist, configuration review, demo-media review |

## 2. Required policy baselines

Compare under matched detector, tracker, sequence, hardware, input cadence, and measurement boundary wherever feasible:

1. Detector every frame + ByteTrack.
2. Fixed detector intervals.
3. Confidence-only threshold.
4. Kalman-uncertainty-only threshold.
5. Static logistic regression/MLP risk baseline.
6. EMO-like and RT-MOT-like policies; document adaptations.
7. HSFSO direct reproduction or a clearly labelled faithful approximation; do not claim superiority from unmatched published tables.
8. ALBIREO-like per-object uncertainty scheduler adapted to the fixed detector/ByteTrack path; label deviations as an approximation and compare HOTA/IDF1 and full-pipeline system metrics.
9. Temporal risk model: compare GRU/TCN/temporal MLP on track-history-only input, then compare the primary TCN with track-only, simple context concatenation, and optional shared-context gated fusion. Report calibration and policy behavior by actual detector-gap bins.
10. Risk-only versus risk-plus-scene-discovery guard, and both against ALBIREO-like scheduling with its empty-scene/rescue behavior where reproduced. The guard may upgrade SKIP to full-frame DETECT; compare new-track discovery delay/recall while other tracks are active, false-trigger rate, MOT metrics, latency, and energy.
11. SDOF-Tracker as direct related work/skip-plus-flow comparison if reproducible without conflating the primary policy.

## 3. Metrics

### Product/tracking

HOTA, DetA/AssA, IDF1, MOTA, ID switches, fragmentation, detector precision/recall or AP, failure rate by predeclared visibility/crowding condition, new-track discovery delay/recall, and scene-guard false-trigger rate.

### System

Full-pipeline Jetson joules per input frame at a declared power-input boundary, end-to-end p50/p95 latency, sustained throughput, deadline-miss rate, dropped frames, peak RAM, temperature, and throttling. Include scene-context extraction/fusion, risk inference, policy, tracking, decode, and required outputs. State separately which attached devices are outside the boundary; do not treat onboard rail telemetry as an exact substitute for a whole-device energy meter.

### Predictor/calibration

Failure prevalence, PR-AUC, operating-point precision/recall, Brier score, stated ECE binning, reliability plots, per-sequence calibration, predictor latency/memory/energy, and uncertainty intervals. Do not treat frames as independent statistical samples.

## 4. Repetition and statistics

Use repeated training seeds and repeated hardware runs. Compare paired policy outcomes at sequence/run level; use sequence-aware resampling or grouped folds. MOT17 has few independent source sequences, so report wide uncertainty honestly. Freeze choices before held-out MOT20 and stress evaluation.

## 5. Acceptance gates

### Product prototype passes when

- It runs the agreed input on the selected physical device from start to clean stop.
- It receives the stationary phone's local H.264/RTSP stream on the Jetson, reports a disconnect clearly, and can replay recorded MOT sequences for repeatable evaluation.
- It emits understandable temporary tracks/counts and the correct action/status record.
- It produces the run summary and can recover from documented input/model/runtime faults.
- It records separate wall-clock detector gap, source-frame gap, and consecutive detector-skip count. The scene-discovery guard may only upgrade a planned skip to full-frame detection; its action and reason are visible in the run record.
- It reports complete-pipeline metrics with device/configuration provenance.
- Privacy defaults and export controls pass the responsible-use checklist.

### Adaptive policy passes its performance claim only when

- It meets predeclared HOTA/IDF1 non-inferiority, p95 latency, deadline-miss, RAM, and sustained-temperature constraints.
- It reduces measured complete-pipeline energy relative to the detector-every-frame reference at acceptable quality, with sequence/run uncertainty reported.
- It is compared against closest prior and simple baselines under matched conditions.
- Predictor calibration is evaluated on natural-prevalence held-out sequences.
- The guard is retained only if held-out discovery benefit justifies its false triggers, latency, and energy cost; otherwise ship the risk-only scheduler and record the negative result.

If no policy meets constraints, do not call the policy successful or energy-saving. The product demo may still be a valid prototype, with the negative result stated clearly.

## 6. Optional conference-paper gate

Conference publication is not an MVP requirement. Consider it only after the prototype is reproducible, the closest-work review is complete, results are statistically defensible, and the claimed distinction survives direct comparison. A product demonstration alone is not evidence of methodological novelty. If the method does not outperform closest baselines, a paper may still be possible as a careful systems/evaluation study, but do not assume acceptance.

## 7. Release checklist

- [ ] Device/runtime/model versions and licenses recorded.
- [ ] Setup is reproducible on the declared device.
- [ ] No restricted dataset/video is included in the repository/package.
- [ ] No secret, credential, or sensitive full path is in config/logs.
- [ ] Default data retention and export behavior are documented.
- [ ] Known failure modes and measured limits are included in the product notes.
- [ ] Results are linked to configs, seeds, checksums, and raw logs.
- [ ] Demo uses authorized, reviewed footage.

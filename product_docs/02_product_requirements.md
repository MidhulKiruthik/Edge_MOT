# RACE-MOT Product Requirements and Acceptance Criteria

**Status:** Draft v0.4, updated 8 October 2026. D-48 closes the bounded deterministic local-MOT replay implementation, D-49 records the full Phase 5 baseline/official TrackEval metrics/sustained telemetry, D-50 records the audited training-role Phase 6 dataset, and D-51 records transient phone live integration; requirements marked **TBD** cannot be treated as passed and whole-device external energy is still open.
**Scope:** MVP requirements for one local video stream and one edge device.

**Current execution note:** MOT17 replay is the repeatable development input and D-48 verifies its detector/tracker runtime. D-51 verifies the authorized transient phone path, controlled reconnect, and private-route reachability; phone remains excluded from benchmark labels and raw-video persistence.

## 1. User-facing functional requirements

| ID | Requirement | Acceptance evidence |
|---|---|---|
| FR-01 | The user can start, pause, stop, and restart a run from one stationary mobile-phone camera using IP Webcam H.264/RTSP over a direct USB-tethered local link to the Jetson. An authorized local video file is a fallback and MOT17/MOT20 files are used for repeatable evaluation. | Demonstration of USB-tethered phone-to-Jetson stream and recorded-file replay; state transitions and errors appear in the UI/log. |
| FR-02 | The application validates stream/file input and reports frame dimensions, source cadence/timestamps, and decode/network errors, including reconnect events. | Exercise a valid stream, interrupted USB network, invalid RTSP URL, valid file, and damaged file. |
| FR-03 | The system displays per-frame detections/tracks with temporary IDs and active-track count as an occupancy proxy. It does not claim line-crossing or validated people-counting accuracy. | Compare rendered samples and exported records against pipeline outputs. |
| FR-04 | The application supports the fixed primary pipeline: one detector, ByteTrack, temporal predictor, scene-discovery guard, and binary detect/skip policy. The guard may upgrade a planned skip to full-frame detection, never cancel a planned detection. | Configuration identifies exact model/checkpoint/version; a run log records every planned and executed action and its trigger. |
| FR-05 | The user can select baseline mode or adaptive mode. Baseline mode invokes the detector on every frame. | Run configuration and logs show the selected mode; output is repeatable for frozen settings. |
| FR-06 | The application records policy evidence: risk estimate, threshold, wall-clock/source-frame detector gap, consecutive skip count, scene-guard result, action, trigger/reason, and timestamps. | One log record per input frame; source-frame loss, detector skips, guard overrides, and detector invocations are distinguishable. |
| FR-07 | The application reports tracking quality and system-resource metrics after an evaluation run, including discovery delay/recall and guard false-trigger rate when the guard is enabled. | Summary contains the metrics specified in the evaluation plan, or an explicit unavailable reason. |
| FR-08 | The user can export a run summary and decision log. Annotated video export is opt-in. | Export files open, include configuration/run IDs, and follow the privacy defaults. |
| FR-09 | Errors are surfaced with an actionable message; the run can stop cleanly on stream loss, decode, model-load, or device failure. | Fault-injection checklist completed; disconnect/reconnect behavior is documented; no orphaned process or locked output remains. |
| FR-10 | The scene-discovery guard evaluates incoming-frame activity outside padded active-track boxes and can only upgrade a planned SKIP to full-frame DETECT. | Trace scenarios show override on configured activity, no override on quiet regions, no cancellation of planned detection, and recorded guard cost. Retain only if held-out discovery benefit justifies cost. |

## 2. Quality and operational requirements

| ID | Requirement | Acceptance evidence / target status |
|---|---|---|
| NFR-01 | Runtime measurements use the actual selected edge hardware and a declared software/power configuration. | Hardware manifest complete; laptop-only timing does not pass. |
| NFR-02 | Latency includes frame arrival/decode through emitted tracks; p50, p95, deadline-miss rate, and dropped frames are reported. | Deadline $D$ is derived from the agreed stream cadence/use case before final testing. **TBD.** |
| NFR-03 | Energy includes decode, preprocessing, detector, predictor, policy, tracker, rendering/logging required by the product. | Declared system boundary and meter/telemetry protocol; report joules per all input frames. |
| NFR-04 | Tracking quality is not inferred from detector AP alone. | Report HOTA, IDF1, MOTA, ID switches, fragmentation, and defined detector metrics. |
| NFR-05 | The adaptive policy cannot skip indefinitely. | Maximum consecutive skip count and no-active-track fallback are configured and logged. |
| NFR-06 | The application does not require internet or cloud services at runtime; the phone-to-Jetson demo requires only the private USB-tethered local link. | Demonstrate local RTSP operation with internet/cloud relay disabled and replay a benchmark file. |
| NFR-07 | Raw video is not uploaded or retained by default. | Data-flow review and file-system inspection during acceptance. |
| NFR-08 | The product does not perform face recognition or identify a person across sequences. | Feature review and product copy review. |
| NFR-09 | Performance claims state exact device, model, resolution, precision, batch size, runtime, power mode, and thermal conditions. | Reproducible run manifest accompanies every published number. |

## 3. Acceptance thresholds that must be set after baseline

Do not invent these values. Record them in a signed run configuration after a pilot on the actual device:

- Application input cadence and per-frame deadline $D$.
- HOTA non-inferiority margin $\epsilon_H$ and IDF1 margin $\epsilon_I$ relative to detector-every-frame.
- Maximum allowable deadline-miss rate $\rho$.
- Memory limit and sustained temperature limit supported by device/application requirements.
- Minimum energy reduction that would count as useful after predictor overhead.
- Minimum duration and repeat count for sustained thermal/power evaluation.
- Guard false-trigger and new-track discovery thresholds; derive them from validation runs and declare before held-out evaluation.

The adaptive policy passes product-quality acceptance only if the predeclared quality and service constraints are met on held-out evaluation data. If no policy passes, the result is a valid negative finding; the product must not label itself energy-saving.

## 4. User interface minimum

Required visible items: source name (not full RTSP credentials/path), run state, processed/input frames, temporary track overlay and active-track count, current detect/skip action, risk/trigger reason, current FPS or processing delay, error/health state, and stop control. The dashboard is served locally by the Jetson and accessed only on the trusted LAN. Avoid presenting an uncalibrated risk value as a probability; if calibration fails, label it as a score.

## 5. Requirement traceability

Each requirement must map to at least one architecture component and one verification case in [the verification plan](07_verification_acceptance_release.md). Any requirement without a feasible acceptance test is not ready for G4.

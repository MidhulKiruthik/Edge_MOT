# RACE-MOT Product Brief

**Status:** Updated 8 October 2026 — product direction agreed; bounded device, three-scene MOT17, expanded FP32 detector/reference-parity, paired-rollout contracts, ByteTrack API, deterministic local replay evidence, full Phase 5 baseline metrics, sustained telemetry, audited Phase 6 training-role dataset, and transient phone integration exist. D-47 records supervisor G4 sign-off, D-48 records Phase 4 implementation, D-49 records official TrackEval measurement, D-50 records the dataset, and D-51 records the phone path; G3 is partial only because external-meter energy is unavailable.
**Product type:** Local edge-computing prototype.  
**Primary outcome:** A demonstrable software prototype; conference publication is optional and evidence-dependent.

## 1. Product concept

RACE-MOT is a local video analytics prototype that detects and tracks pedestrians on a single fixed-camera stream. It uses a risk-estimation policy to decide whether to run the detector on the next frame or skip that detector call and propagate active tracks. The product should make its current processing state understandable and provide enough logs to inspect errors and resource use.

The MVP does not identify people across cameras or sessions. Track IDs are temporary algorithmic labels within a video sequence; they are not names or biometric identities.

Current execution is local-MOT17-first for repeatable metrics, with transient live phone integration measured under D-51. The phone endpoint has no-save decode, FP32/tracker, controlled reconnect, and private-route evidence; it remains excluded from labeled benchmark claims.

## 2. Intended user and job

**Primary user:** the student/faculty demonstrator evaluating a local edge MOT prototype.

**User job:** connect one authorized pedestrian video stream, view anonymous detections/tracks and the per-frame active-track count (an occupancy proxy), see whether the system is keeping up, and export a report of tracking and compute behavior.

**Deployment setting:** a single mobile phone mounted as a stationary camera in an authorized, controlled pedestrian scene (such as a staged corridor demo). The phone streams to the Jetson over a private local link supplied by direct USB tethering. This remains an academic demonstration, not an approved operational deployment. Obtain required authorization before capturing people. The active-track count is not claimed as validated pedestrian counting or directional flow analytics.

## 3. User problem

Running a detector on every frame can consume significant edge-device compute. Skipping detector calls may reduce resource use, but can also make tracks drift, fragment, or switch identities. The prototype tests whether a temporal risk estimate can decide when a detector call is worth its cost while preserving acceptable tracking quality and deadline behavior.

## 4. MVP workflow

1. For the live demonstration, the user starts an IP Webcam H.264/RTSP feed from the stationary phone camera over the direct USB-tethered local link. Recorded MOT17/MOT20 files are used for repeatable evaluation. A local authorized video file is a fallback input.
2. The application validates the stream/file and reports dimensions, source cadence, timestamps, and decoding/network errors.
3. The pipeline processes frames, showing anonymous bounding boxes, temporary track IDs, and the active-track count.
4. When calibration passes its held-out checks, the calibrated risk controller schedules either **detect** or **skip detector and propagate tracks** for the next input frame. If calibration is not valid, the system must expose the output as a score and remain in a non-accepting adaptive state. When a new frame arrives, a low-cost scene-discovery guard may upgrade a planned skip to full-frame detection for untracked activity or an abrupt scene change. It cannot cancel a planned detection. The UI/log exposes the action and trigger.
5. The user can pause, stop, and restart processing without leaving a stuck camera, model, or output file open.
6. The user exports an annotated video only when explicitly requested, plus a compact CSV/JSON run summary and decision log.

## 5. Product goals

- Demonstrate a complete detector–tracker–risk-policy pipeline on the available Jetson Orin Nano.
- Keep processing local by default and make data handling visible.
- Make system failures and tracking degradation observable rather than hiding them behind a single FPS number.
- Measure the predictor/controller overhead as part of the full pipeline.
- Provide a repeatable evaluation on MOT17 and a held-out MOT20 stress/generalization set.

## 6. Non-goals for the MVP

- Face recognition, person naming, biometric identification, or cross-camera identity matching.
- Multi-camera, edge-mesh, cloud, or RTSP fleet management.
- A custom detector architecture, Re-ID branch, learned Mamba/SSM motion replacement, ROI/patch detector action, pseudo-depth association, variable-resolution action, optical-flow propagation, or thermal-aware mode-switching policy.
- Claims of production reliability, safety certification, legal compliance, or suitability for enforcement/employment decisions.
- A new public dataset from simple cropping, augmentation, or a few unvalidated clips.

## 7. MVP deliverables

- A runnable local prototype with documented setup and supported device/runtime.
- A single configuration for detector, ByteTrack, risk model, scene-discovery guard, and two-action policy.
- A local web dashboard with tracks/active-track count, throughput/latency state, current action, and basic health/errors. It is served by the Jetson and available only on the trusted local network.
- A reproducible evaluation command/configuration and an exportable run summary.
- Evidence package: source/data provenance, model checksums, configuration, raw metric logs, and limitations.

## 8. Product success definition

The prototype is successful only if it completes an authorized video run, outputs valid tracks and counts, exposes policy decisions, logs full-pipeline metrics, and can be restarted after a controlled failure. The adaptive mode must be compared to detector-every-frame and simple scheduling baselines. Numeric quality, latency, and energy thresholds remain **TBD until the chosen hardware baseline and application cadence are measured**.

## 9. Product risks

- The project may optimize a generic benchmark rather than solve a user-validated problem.
- Inaccurate tracks can make counts misleading; counts must not be presented as ground truth.
- A local video prototype can still create privacy risks through recording, export, or unintended access.
- Limited MOT17 sequence count can make model selection and confidence estimates unstable.
- The selected device may not support the expected model/runtime or power measurements.

See the decision log for the remaining external-energy and adaptive-policy gates. Device/input, model, data, Phase 4, Phase 5, Phase 6, and transient phone integration evidence are recorded under D-40 through D-51.

# Phone Live Integration Record

**Date:** 8 October 2026  
**Decision:** D-51  
**Scope:** authorized, transient phone-camera integration for Phases 0–6

The phone stream was exercised on the target Jetson through the redacted
endpoint `http://10.152.75.6:8080/video`. The application retained only
aggregate reports, run metadata, detections, and temporary tracks. No decoded
frame, image, annotated video, face crop, embedding, credential, or persistent
identity was written.

## Evidence

| Check | Result | Evidence |
|---|---|---|
| 30-second no-save decode | **PASS** — 904 frames, 1920×1080, reported 25 FPS, measured 30.116 FPS, zero read errors, FFmpeg backend | `race_mot/reports/2026-10-08-g1-phone-live/probe.json` |
| FP32 detector smoke | **PASS** — 20/20 frames, finite TensorRT output, 0.10 confidence, 0.45 NMS, engine SHA-256 `c045bfd4…3d7` | `race_mot/reports/2026-10-08-g1-phone-live/detector_smoke.json` |
| Detector/tracker live run | **PASS** — 100 frames completed, 51 person detections and 51 temporary track snapshots; no raw frames | `race_mot/runs/2026-10-08-phone-live-100/` |
| Controlled reconnect | **PASS** — 3 independent open/read/release cycles, 30 frames each, zero read errors, 1920×1080 each cycle | `race_mot/reports/2026-10-08-g1-phone-live/reconnect.json` |
| Trusted private-route reachability | **PASS** — HTTP 200 MJPEG headers over route `10.152.75.6 dev wlP1p1s0 src 10.152.75.87`; response prefix discarded | `race_mot/reports/2026-10-08-g1-phone-live/trusted_lan.json` |

The route audit also observed a USB-network interface (`enx8ec4ed3d07cc`,
`10.106.147.79`), but the supplied endpoint selected the Wi-Fi interface for
`10.152.75.6`. Therefore this record closes live private-endpoint integration
and controlled reconnect, while preserving the limitation that a physical
USB-cable/network fault was not injected and the endpoint's path was not the
observed USB interface.

## Phase impact

- **Phase 0/G0:** the agreed stationary-phone demo workflow is executable and remains bounded to an authorized controlled scene.
- **Phase 1/G1:** input feasibility and FP32 detector feasibility are measured on the target board; live integration is demonstrated without making an accuracy claim from unannotated phone frames.
- **Phase 2/G2:** transient processing, redacted logs, temporary IDs, and no-frame persistence were exercised. MOT17 remains the repeatable labeled source; phone frames are not used for training, calibration, or evaluation labels.
- **Phase 3/G4:** the live source, clean stop/restart behavior, reconnect cycle, and trusted private route are covered by executable evidence. Supervisor authorization is already recorded under D-47 and D-51.
- **Phase 4:** `phone-run` executes the same TensorRT FP32 plus ByteTrack-compatible runtime used by local replay, with the source marked as live phone and raw-video persistence disabled.
- **Phase 5:** the phone run is an integration demonstration only. It is not included in MOT17/TrackEval quality, latency, thermal, or energy benchmark claims because it has no ground truth and uses a different unannotated scene.
- **Phase 6:** phone data are explicitly excluded from the training-role paired-rollout dataset and all label/statistics generation.

## Operational commands

```bash
PYTHONPATH=race_mot/src python3 -m race_mot.cli phone-reconnect \
  --input 'http://10.152.75.6:8080/video' --cycles 3 \
  --frames-per-cycle 30 --pause-sec 1 \
  --output race_mot/reports/2026-10-08-g1-phone-live/reconnect.json

PYTHONPATH=race_mot/src python3 -m race_mot.cli phone-run \
  --input 'http://10.152.75.6:8080/video' \
  --engine race_mot/models/provisional/yolox_tiny_fp32_diagnostic.engine \
  --config race_mot/configs/baseline.json \
  --output race_mot/runs/2026-10-08-phone-live-100 --frames 100
```

The supplied URL must stay out of committed logs and shell history in future
runs; use `--input-env` for a credential-bearing deployment.


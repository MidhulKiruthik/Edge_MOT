# Phase 4 deterministic implementation record

**Date:** 8 October 2026  
**Decision:** D-48  
**Scope at implementation freeze:** local MOT17 image-sequence replay only. The later D-51 record reopens the authorized phone path for a separate transient integration run; it does not alter the deterministic replay evidence or benchmark scope.

Phase 4 is implemented as a single-worker, detector-every-frame baseline. The source keeps MOTChallenge's one-based frame index and nominal sequence timestamp. A source read/decode exception becomes a failed run summary; normal local sequence exhaustion is a clean terminal state. No raw image is written by the runtime.

The detector adapter in [`race_mot/src/race_mot/detectors/yolox_tensorrt.py`](../race_mot/src/race_mot/detectors/yolox_tensorrt.py) uses the frozen YOLOX-Tiny TensorRT FP32 engine. It measures preprocessing, host/device inference, YOLOX decode/NMS, coordinate restoration, and total detector time separately. The run manifest records the ONNX checkpoint hash `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` and engine hash `c045bfd4ce2baa4ea8ae75850802df89ef1d71c2af3200d3bda6970dd3f0d3d7`.

The tracker adapter in [`race_mot/src/race_mot/trackers/bytetrack.py`](../race_mot/src/race_mot/trackers/bytetrack.py) exposes `initialize`, `update`, `skip`, and `reset`. `update([])` records a detector-run/no-person result and does not emit predictions; `skip()` advances motion and emits predicted snapshots. IDs reset at each run. The adapter records the pinned ByteTrack commit metadata and the current dependency-free deterministic IoU compatibility backend; replacing that backend with the upstream implementation requires repeating the lifecycle and replay checks.

The orchestrator in [`race_mot/src/race_mot/application.py`](../race_mot/src/race_mot/application.py) writes an immutable manifest, append-only redacted `frame_log.jsonl`, and a summary containing stable hashes. The queue is bounded to one sequential frame. `stop()` is idempotent, and a completed runner can be restarted with reset tracker state. Logs contain no image payload, URL credentials, face data, or persistent identity.

The OpenCV reference golden fixture is [`race_mot/reports/2026-10-08-phase4-runtime/golden_reference_fixture.json`](../race_mot/reports/2026-10-08-phase4-runtime/golden_reference_fixture.json). It stores the first-frame input/output hashes, tensor shape, preprocessing, postprocessing settings, and restored boxes without storing the source frame.

## Replay evidence

The repeatability check was run with:

```bash
PYTHONPATH=race_mot/src python3 -m race_mot.cli replay-check \
  --sequence race_mot/data/mot17/MOT17-02-FRCNN \
  --engine race_mot/models/provisional/yolox_tiny_fp32_diagnostic.engine \
  --config race_mot/configs/baseline.json \
  --output-root race_mot/runs/2026-10-08-phase4-replay-100-v2 \
  --frames 100 --repeats 2
```

The result is [`replay_check.json`](../race_mot/runs/2026-10-08-phase4-replay-100-v2/replay_check.json): both repeats completed 100 frames, emitted 1,449 detections and 1,449 tracks, and matched these stable hashes:

| Record | SHA-256 |
|---|---|
| Stable frame records | `9cf01d5150543e8a950dba9a828722c3b02593a1c6d65a7092c9fb299bf51133` |
| Detection records | `1ed5c87d757e4611af5f3bccce28b214c4490ec52fa72a6bf200b23e11f2e4a4` |
| Track records | `0f553ce09b704064e105ff35a3690440c58aa643c4141c7fe73f78270f4cfaf1` |

The same 100-frame runtime pass completed on `MOT17-04-FRCNN` and `MOT17-05-FRCNN`. Their summaries are [`scene04`](../race_mot/runs/2026-10-08-phase4-scene04/summary.json) and [`scene05`](../race_mot/runs/2026-10-08-phase4-scene05/summary.json), with 2,485 and 825 detections/tracks respectively.

Contract tests cover source index/drop preservation, detector postprocessing, tracker empty-update versus skip, restart, idempotent stop, injected source failure, paired rollouts, and configuration validation. The Phase 4 test snapshot was 29 tests passing. This record closes the bounded deterministic Phase 4 implementation; it does not claim HOTA/IDF1, full-sequence latency limits, external-meter energy, thermal limits for the complete pipeline, dashboard acceptance, or adaptive-policy benefit. D-49 subsequently records the Phase 5 metrics, sustained telemetry, and minimal dashboard; external-meter energy and adaptive-policy benefit remain open.

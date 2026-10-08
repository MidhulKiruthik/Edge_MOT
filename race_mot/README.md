# RACE-MOT implementation

This directory is the implementation workspace for the product-first RACE-MOT project. Bounded G1 device, three-scene local MOT17 development coverage, expanded TensorRT FP32/OpenCV parity, paired-rollout contracts, pinned ByteTrack feasibility, the audited Phase 6 training-role dataset, and transient no-save phone integration exist on the actual Jetson Orin Nano. D-47 records supervisor G4 sign-off, D-48 records the deterministic Phase 4 local replay implementation, D-50 records the Phase 6 dataset, and D-51 records live phone integration; G3 external energy plus the adaptive risk policy remain gated.

## G1 feasibility utilities

The utilities below collect device/software metadata, probe local/authorized inputs, and run bounded detector diagnostics without saving images. The Phase 4 runtime includes a TensorRT FP32 detector adapter, deterministic ByteTrack-compatible lifecycle adapter, redacted run logger, and local replay commands. These are implementation/replay results, not final performance or tracking-quality claims.

Install the package in the project environment on the Jetson:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The inventory command uses the Python standard library. The stream probe requires the Jetson's OpenCV Python module with a decoder backend that supports the supplied input. Prefer the OpenCV package already supplied by the selected JetPack image; avoid replacing Jetson system OpenCV with an unrelated pip wheel.

```bash
race-mot inventory --output reports/device_inventory.json
race-mot probe --input "rtsp://PHONE_LAN_ADDRESS:PORT/STREAM" --duration-sec 30 --output reports/phone_stream_probe.json
```

Keep credentials out of command history. Enable USB tethering and IP Webcam, identify the Jetson's USB-network address, and use the resulting private local link. The probe reports a redacted input identifier, dimensions, reported source FPS, measured read rate, read errors, and interval percentiles. It does not guarantee real-time MOT performance. The initial camera profile is 1280x720, H.264, 15 fps if the phone app supports it.

## Local implementation slice

The first hardware-independent contracts are now available under `src/race_mot/`:

- `domain.py` defines frame, detection, track, action, and policy records.
- `sources/mot_sequence.py` reads MOTChallenge sequences in source order and emits in-memory `FramePacket`s with original frame indices and timestamps.
- `config.py` validates the JSON form of the documented run configuration.
- `policy.py` implements bounded binary `DETECT`/`SKIP` scheduling and the one-way scene-guard upgrade.
- `evaluation/mot.py` reads MOTChallenge person annotations without extra dependencies.
- `evaluation/labels.py` implements the paired-rollout avoidable-failure label primitive.
- `detectors/yolox_tensorrt.py` runs the frozen FP32 engine with stage timings and original-frame boxes.
- `trackers/bytetrack.py` exposes initialize/update/skip/reset with explicit empty-update versus skip semantics.
- `application.py` runs a sequential detector-every-frame replay and writes manifest, JSONL frame log, and summary hashes.
- `evaluation/tracking_metrics.py` evaluates detector diagnostics and runtime/identity metrics; official TrackEval 1.3.0 is available through the optional evaluation environment.
- `dashboard.py` serves a local redacted replay dashboard with temporary track geometry, action/reason, latency, queue, drop, and error state; it never serves source frames.

Run and repeat a bounded local replay:

```bash
race-mot replay-check --sequence data/mot17/MOT17-02-FRCNN \
  --engine models/provisional/yolox_tiny_fp32_diagnostic.engine \
  --config configs/baseline.json --output-root runs/phase4-replay \
  --frames 100 --repeats 2
```

The Phase 5 baseline command accepts repeated full local sequences and writes per-run metrics. Use `pip install -e '.[evaluation]'` in an isolated environment for the official TrackEval path, then run `scripts/run_trackeval.py` on the generated MOT logs. Whole-device energy remains unavailable until an external Jetson-input meter is connected.

Phase 6 generates the frozen training-role paired-rollout dataset from the Phase 5 logs:

```bash
PYTHONPATH=src python scripts/generate_phase6_dataset.py \
  --sequence data/mot17/MOT17-02-FRCNN runs/2026-10-08-phase5-baseline/repeat-01-MOT17-02-FRCNN \
  --sequence data/mot17/MOT17-04-FRCNN runs/2026-10-08-phase5-baseline/repeat-01-MOT17-04-FRCNN \
  --sequence data/mot17/MOT17-05-FRCNN runs/2026-10-08-phase5-baseline/repeat-01-MOT17-05-FRCNN \
  --roles data/roles.json --protocol data/rollout_protocol.json \
  --output-root runs/2026-10-08-phase6-dataset
```

The generator writes unique serialized anchor states, paired branch labels, training-only statistics, and leakage/prevalence audits. It reads only the training role; phone, MOT20, calibration, policy-validation, and final-evaluation labels are excluded.

To inspect a completed local run, serve the dashboard on loopback:

```bash
PYTHONPATH=src python -m race_mot.cli dashboard \
  --run-dir runs/2026-10-08-phase5-baseline/repeat-01-MOT17-02-FRCNN
```

Use an explicitly authorized trusted-LAN bind address only when needed; phone capture is not part of this Phase 5 path.

For the authorized live demonstration, use the transient phone checks. They retain
only redacted metadata, detections, and temporary tracks:

```bash
race-mot phone-reconnect --input-env RACE_MOT_PHONE_URL --cycles 3 --frames-per-cycle 30 \
  --output reports/NEW_RUN/reconnect.json
race-mot phone-run --input-env RACE_MOT_PHONE_URL \
  --engine models/provisional/yolox_tiny_fp32_diagnostic.engine \
  --config configs/baseline.json --output runs/NEW_RUN --frames 100
```

Validate the local baseline configuration and inspect an annotation file before using Jetson-specific tools:

```bash
race-mot validate-config --config configs/baseline.json
race-mot inspect-mot --gt /path/to/MOT17-02-FRCNN/gt/gt.txt
race-mot mot-probe --sequence /path/to/MOT17-02-FRCNN --frames 100 \
  --output reports/NEW_RUN/mot_decode.json
python tests/test_core.py -v
```

These commands validate configuration, labels, and local decode behavior. They do not claim tracking quality, calibration, or energy savings. Detector feasibility is recorded only by the bounded smoke evidence below; full performance remains gated by the actual-device baseline and G3 measurement contract.

## G1 FP32 detector smoke check

The current JetPack/TensorRT stack builds YOLOX-Tiny only through the documented
FP32 diagnostic path; FP16 engine generation is blocked and remains a separate
compatibility issue. The smoke command reads a bounded number of authorized
frames, letterboxes them in memory, executes the engine, checks output values
for finiteness, and writes no decoded, preprocessed, or annotated frame.

```bash
race-mot detector-smoke --input-env RACE_MOT_RTSP_URL \
  --engine models/provisional/yolox_tiny_fp32_diagnostic.engine \
  --frames 10 --output reports/NEW_RUN/detector_smoke.json
race-mot detector-smoke --mot-sequence data/mot17/MOT17-02-FRCNN \
  --engine models/provisional/yolox_tiny_fp32_diagnostic.engine \
  --reference-onnx models/provisional/yolox_tiny.onnx \
  --frames 10 --output reports/NEW_RUN/mot_detector_smoke.json
```

This validates the engine and I/O path, decodes the official export's raw
stride-grid output, restores original-frame coordinates, and performs YOLOX
class-zero confidence filtering and NMS. For a MOT sequence with local ground
truth it also reports a bounded IoU-overlap recall diagnostic. It records only
aggregate counts and scores. It does not establish detector accuracy, freeze
thresholds, run ByteTrack, or measure latency, throughput, tracking, energy,
or thermal behavior.

The expanded D-42 reports compare 200 identical MOT17 frames through OpenCV
DNN and TensorRT FP32, including restored-box coordinate/IoU agreement and
precision/recall diagnostics. They are retained under the ignored dated
directory `reports/2026-10-08-g3-detector-acceptance/`; they do not approve the
checkpoint for final evaluation.

`mot-probe` is a separate bounded JPEG-decoding check for an extracted
MOTChallenge image sequence. It verifies metadata, sequential image presence,
and decoded dimensions without retaining frames. Its decode rate is not a
detector, tracker, or end-to-end result.

## Next implementation milestone

Next, resolve acceptable detector recall, checkpoint/data terms, grouped scene roles, exact physical cooling, and phone reconnect/LAN acceptance, then pass G4. After G4, implement the deterministic every-frame YOLOX-Tiny + ByteTrack baseline, result log, and local dashboard. Only after the measured baseline and G3 contract should the calibrated risk predictor and binary detect/skip policy be implemented.

See [the product documentation](../product_docs/README.md), [the implementation gate](../product_docs/08_decisions_and_pre_code_gate.md), and the [Software Design Document](SDD.md) for scope, acceptance criteria, and planned module contracts. The working technical plan is [the research report](../deep_research_mot_edge_merged.md).

## G1 probe evidence and private input

Use `--input-env RACE_MOT_RTSP_URL` after setting that variable privately in the
launching shell; this keeps the URL out of the CLI argument list. Decoder backend
errors may still expose source details, so keep raw diagnostics private.
Reports refuse existing output paths. Use a fresh dated directory for each run.

```bash
race-mot probe --input-env RACE_MOT_RTSP_URL --duration-sec 30 --output reports/NEW_RUN/phone.json
race-mot probe --input /authorized/clip.avi --duration-sec 30 --output reports/NEW_RUN/file.json
```

Reports include `stop_reason`, reported frame count, and decoder position/timestamp
sample counts, first/last values, repeats, and regressions. `expected_file_end`
means a local file reached its reported frame count; this is metadata-based and
not a corruption guarantee. Unknown-length and early failures remain
`read_failure`. Decoder positions do not prove camera frame order or source drops;
constant timestamps may mean unsupported metadata. Reconnect is not implemented,
and default-backend reads may exceed the requested duration. The probe saves no
frames and supplies no tracking or energy result.

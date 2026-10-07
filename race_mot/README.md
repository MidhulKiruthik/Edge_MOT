# RACE-MOT implementation

This directory is the implementation workspace for the product-first RACE-MOT project. Bounded G1 device, local MOT17, TensorRT FP32/OpenCV parity, and upstream ByteTrack feasibility checks exist on the actual Jetson Orin Nano. Full adapters and the adaptive risk policy remain gated by G4. Further phone-camera work is deferred under D-33.

## G1 feasibility utilities

The utilities below collect device/software metadata, probe local/authorized inputs, and run bounded detector diagnostics without saving images. A separate private smoke check exercised pinned upstream ByteTrack; no project tracker adapter exists yet. These are not performance or tracking-quality results.

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
- `config.py` validates the JSON form of the documented run configuration.
- `policy.py` implements bounded binary `DETECT`/`SKIP` scheduling and the one-way scene-guard upgrade.
- `evaluation/mot.py` reads MOTChallenge person annotations without extra dependencies.
- `evaluation/labels.py` implements the paired-rollout avoidable-failure label primitive.

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

`mot-probe` is a separate bounded JPEG-decoding check for an extracted
MOTChallenge image sequence. It verifies metadata, sequential image presence,
and decoded dimensions without retaining frames. Its decode rate is not a
detector, tracker, or end-to-end result.

## Next implementation milestone

Next, resolve acceptable detector recall, checkpoint/data terms, grouped scene roles, exact physical cooling, and the D-33 phone deferral, then pass G4. After G4, implement the deterministic every-frame YOLOX-Tiny + ByteTrack baseline, result log, and local dashboard. Only after the measured baseline and G3 contract should the calibrated risk predictor and binary detect/skip policy be implemented.

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

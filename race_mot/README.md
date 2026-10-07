# RACE-MOT implementation

This directory is the implementation workspace for the product-first RACE-MOT project. The first milestone is device and camera-input feasibility on the actual Jetson Orin Nano; the adaptive risk policy is not part of this milestone.

## Milestone 0: inventory and input probe

The utilities below collect device/software metadata and measure whether a phone camera feed can be read locally. They do not save video, transmit it, or perform tracking. They are not performance results for the MOT pipeline.

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

These commands do not claim detector feasibility, tracking quality, calibration, or energy savings. Those remain gated by the actual-device baseline and G3 measurement contract.

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

After the device and stream are verified, add the every-frame YOLOX-Tiny + ByteTrack baseline, then the result log and local dashboard. Freeze the exact detector checkpoint, its terms, TensorRT export, tracker version, and end-to-end measurement boundary before claiming a baseline. Only after that should the calibrated risk predictor and binary detect/skip policy be implemented. Preserve actual timestamps and source-frame gaps; compare an ALBIREO-like object-wise uncertainty scheduler; then add and measure the scene-discovery guard, which may upgrade a planned skip to a full-frame detector call. The guard is retained only if its new-track discovery benefit justifies its false triggers and system cost.

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

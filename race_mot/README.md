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

Keep credentials out of command history. Use the phone and Jetson on the same private Wi-Fi/hotspot. The probe reports a redacted input identifier, dimensions, reported source FPS, measured read rate, read errors, and interval percentiles. It does not guarantee real-time MOT performance. The initial camera profile is 1280x720, H.264, 15 fps if the phone app supports it.

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
python tests/test_core.py -v
```

These commands do not claim detector feasibility, tracking quality, calibration, or energy savings. Those remain gated by the actual-device baseline and G3 measurement contract.

## Next implementation milestone

After the device and stream are verified, add the every-frame YOLOX-Tiny + ByteTrack baseline, then the result log and local dashboard. Freeze the exact detector checkpoint, its terms, TensorRT export, tracker version, and end-to-end measurement boundary before claiming a baseline. Only after that should the calibrated risk predictor and binary detect/skip policy be implemented. Preserve actual timestamps and source-frame gaps; compare an ALBIREO-like object-wise uncertainty scheduler; then add and measure the scene-discovery guard, which may upgrade a planned skip to a full-frame detector call. The guard is retained only if its new-track discovery benefit justifies its false triggers and system cost.

See [the product documentation](../product_docs/README.md), [the implementation gate](../product_docs/08_decisions_and_pre_code_gate.md), and the [Software Design Document](SDD.md) for scope, acceptance criteria, and planned module contracts. The working technical plan is [the research report](../deep_research_mot_edge_merged.md).

# RACE-MOT Edge

RACE-MOT is a local, single-camera pedestrian-tracking prototype intended for the Jetson Orin Nano. The repository contains the product and research documentation, the initial hardware/input feasibility utilities, and the small deterministic contracts that can be tested without Jetson hardware.

**Current status (8 October 2026):** bounded local MOT17 decode, restored three-scene development coverage, TensorRT FP32 YOLOX output decoding, expanded OpenCV-reference parity, paired-rollout/label contracts, pinned ByteTrack feasibility checks, the deterministic local detector/tracker runtime, full two-repeat baseline measurements, complete-pipeline sustained telemetry, official TrackEval 1.3.0 metrics, the audited Phase 6 training-role dataset, transient no-save phone integration, and a 30-minute FP32 thermal diagnostic have run on the target Jetson. D-47 records supervisor G4 sign-off, D-48 records Phase 4 implementation, D-49 records Phase 5 baseline measurement, D-50 records Phase 6 dataset generation, and D-51 records phone integration; the detector's measured 52.43% recall is accepted by waiver. G3 is partial only because external-meter energy is unavailable; adaptive scheduling remains gated.

This is an academic prototype. It does not provide face recognition, persistent identity, cloud processing, or production surveillance functionality. No performance, tracking-quality, calibration, latency, or energy result is claimed until it is measured on the target device.

## Repository layout

- `product_docs/` - product requirements, architecture, data, hardware, privacy, verification, and decision documents.
- `race_mot/` - the Python package, configuration, implementation gate, and tests.
- `deep_research_mot_edge_merged.md` - the current research and prior-art plan; the other research/proposal files are retained as historical inputs where they conflict.
- `todo.md` - the evidence-gated implementation roadmap.
- `energytrack_project_proposal.md` - historical proposal material; it is not the current specification where the documents differ.

Read `product_docs/README.md` first. Deterministic Phase 4 implementation is evidenced under D-47/D-48 and the transient phone path under D-51; the adaptive scheduler must wait for the documented baseline and G3 measurement gates.

## Clone on the Jetson

From the Jetson, clone the repository into a workspace directory:

```bash
mkdir -p ~/work
cd ~/work
git clone <REPOSITORY_URL> edge
cd edge/race_mot
```

Replace `<REPOSITORY_URL>` with the URL of the GitHub/GitLab repository created from this folder. Keep the repository private if the documentation or future run metadata is not intended for public release.

## Install and run the local checks

The package has no third-party Python dependency for its current inventory/configuration/test slice:

```bash
cd ~/work/edge/race_mot
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
race-mot validate-config --config configs/baseline.json
```

On the Jetson, keep the OpenCV module supplied by the selected JetPack image. Do not replace system OpenCV with an unrelated pip wheel before checking the installed environment.

## First Jetson checks

Run these only on the physical device and keep generated output out of Git:

```bash
race-mot inventory --output reports/device_inventory.json
race-mot probe --input "rtsp://PHONE_LAN_ADDRESS:PORT/STREAM" --duration-sec 30 --output reports/phone_stream_probe.json
```

The stream probe is intended to measure local input feasibility without saving video. Keep RTSP credentials out of shell history and logs. See `race_mot/README.md` and `todo.md` for the gate requirements and evidence that must be recorded before full implementation.

# Repository takeover notes

Inspection date: 7 October 2026. This records observed repository state, not a gate approval or a performance result.

> **Historical snapshot:** Sections describing missing MOT input, inactive USB,
> absent detector/tracker execution, and the original test count record the
> takeover state only. They are superseded by D-28 through D-36 and the current
> status below; retain them for audit history rather than treating them as the
> latest plan.

## Current handoff status — 8 October 2026

- Phone-camera work is temporarily deferred under D-33; local MOT17 is active.
- `MOT17-02-FRCNN` local decode, TensorRT FP32 YOLOX grid decoding and coordinate
  restoration, 100-frame overlap diagnostics, and ten-frame OpenCV DNN parity
  have private evidence.
- Pinned ByteTrack commit `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`
  runs on real detector outputs in the isolated G1 environment; its project
  adapter and explicit skip semantics remain post-G4 work.
- Twenty-three unit tests and baseline configuration validation pass.
- MOT17 manifests preserve source-scene grouping, but explicit data terms,
  role assignments, acceptable detector recall, checkpoint terms, exact
  physical cooling, and G4 approval remain open.

## Working environment and access

The user has SSH access to the Jetson and supplied `jetson` as the SSH alias. The current execution environment is already on the Jetson: `aarch64`, device-tree model `NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super`, and an SSH session environment is present. The working repository is `/home/madhan/projects/edge`. A nested `ssh jetson` attempt could not resolve the alias; this device has no user SSH config. Use the existing device shell here. If moving to another execution environment, verify where the alias is configured rather than assuming it resolves everywhere. No credentials were requested or stored.

The repository is on `main`, with existing uncommitted documentation edits, `AGENTS.md`, and requirements traceability work. Preserve these changes. No runtime code or device configuration was changed during inspection.

## Implemented and checked

- Python package under `race_mot/src/race_mot`, requiring Python 3.10 or newer, with no declared third-party dependencies for the core slice.
- CLI commands: `inventory`, `probe`, `validate-config`, and `inspect-mot`.
- Frame/detection/track records, config validation, a standalone binary scheduler, MOT annotation parsing, and a paired-outcome label primitive. These are scaffolding, not an integrated tracking or training pipeline.
- Seven existing unit tests passed on the current device. Baseline config validation passed. Existing diff passed `git diff --check`.
- OpenCV and TensorRT modules are discoverable. Package queries report JetPack `7.2.1-b49` and `libnvinfer10` `10.16.2.10-1+cuda13.2`. PyTorch, ONNX, YOLOX, and `yolox.tracker` are not discoverable in the checked system Python environment. This is not an exhaustive search of other environments.
- Expected root/package `models/` and `data/` directories are absent. No detector or tracker inference was run.

Reproduce the local checks without installing packages:

```bash
cd /home/madhan/projects/edge/race_mot
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m race_mot.cli validate-config --config configs/baseline.json
```

## Evidence and reconciliation backlog

1. Private, ignored reports exist at `race_mot/reports/device_inventory.json` and `race_mot/reports/phone_stream_probe.json`. The roadmap still says the utilities have not run; the decision log records partial G1 progress. Reconcile status without marking G1 complete.
2. The saved phone probe is HTTP input: 1920x1080, reported 25 FPS, 452 frames in approximately 15.021 seconds, 30.0916 reads/s, zero read errors. D-25 instead records 454 frames and 30.26 reads/s against the same report path. The reason for the discrepancy is unknown; preserve the report and investigate before changing historical measurements. The decision log identifies Wi-Fi transport; the report alone does not establish network interface or codec. USB-tethered H.264/RTSP, timestamps, reconnect, and local MOT decode remain unverified. Read throughput is not pipeline FPS.
3. The SDD calls config/domain/scheduler/evaluation helpers planned and lists only two implemented CLI commands, despite the source containing the above scaffold. Update implementation status without presenting these primitives as a finished adaptive runtime.
4. The pitch section 9.2 redefines G2 and G4, conflicting with the governing decision log (G2 is data/responsible use; G4 is pre-code approval). The pitch also retains unsupported absolute novelty/privacy/thermal language and a sub-millisecond design statement despite its latency-TBD note. Its feature table omits the separate source-frame detector gap and its policy algorithm omits frame-level recalibration and invalid-risk handling required elsewhere. Phase 0 reconciliation still has outstanding issues despite its completion header.
5. The roadmap and some acceptance/privacy text still refer to Wi-Fi as the intended path while D-09 selects USB tethering. Keep the observed Wi-Fi probe separate from the selected deployment requirement.
6. `.gitignore` excludes reports but does not yet exclude `runs/`, `models/`, or `data/`. Reconcile artifact exclusions before those directories are populated.
7. The existing tests cover only a small core slice; inventory/probe lifecycle, stream interruption, and complete runtime behavior have no test coverage here. The probe does not provide source frame-order/timestamp evidence or reconnect behavior and counts any failed read, including file EOF, as a read failure. Its default backend fallback has no guaranteed hard timeout.

## Next work

Follow [AGENTS.md](../AGENTS.md), the [decision log](08_decisions_and_pre_code_gate.md), and [roadmap](../todo.md). Preserve the installed JetPack/OpenCV environment.

1. Reconcile the documentation issues and evidence references above, retaining uncertain measurements as uncertain.
2. Complete the bounded G1 input checks using an authorized source: USB-tethered RTSP, timestamps, bounded interruption/recovery, and local MOT frame-order decoding. Save each new report under a unique dated path; do not overwrite previous reports.
3. Select and record YOLOX-Tiny checkpoint provenance/terms and a pinned ByteTrack source before dependency installation or detector/tracker feasibility checks, as required by D-26.
4. Close data/permission and pre-code requirements. Implement the deterministic detector-every-frame pipeline when G4 permits it; freeze G3 measurements before adaptive runtime work or model training.

The user has not provided a new authorized input file or confirmed a current camera endpoint during this inspection. No camera was opened or footage captured.

## Continuation completed on 7 October 2026

The earlier sections are the initial inspection snapshot. Subsequent work added
probe EOF/decoder observations, finite-duration validation, capture cleanup on
metadata errors, CLI environment input, and exclusive report writes. The roadmap
and SDD implementation status were corrected; `runs/`, `models/`, and `data/` are
now ignored. Sixteen tests and baseline validation pass. Synthetic decoder and
fresh inventory evidence are under `race_mot/reports/2026-10-07-g1-131122/`.

The user confirms a USB-connected phone and has no MOT video yet. USB enumeration
sees the vivo phone, but no active USB network link was observed. Tethering and a
current authorized stream endpoint are pending. No live camera was opened during
this continuation. The pitch audit, real MOT decode, and detector/tracker
provenance/feasibility remain open. See D-27 and its validation note.

## Live USB input follow-up

The user subsequently enabled tethering and authorized the live phone check.
Both HTTP and H.264/RTSP decode succeeded over the verified USB network route;
D-28 supersedes the earlier USB-link blocker. The 30-second RTSP probe read
923 frames with zero read errors and saved no frames. Private summary:
`race_mot/reports/2026-10-07-g1-rtsp-usb-131401/summary.md`.
OpenCV reported an unreliable 90000 FPS value; measured read throughput was
30.729 frames/s. Timestamps had one repeat and no regressions. No tracking,
energy, source-loss, or reconnect claim follows from this probe.

## Detector feasibility follow-up

The official YOLOX-Tiny 0.1.1rc0 ONNX artifact was downloaded locally only for
the bounded Jetson feasibility check. Its provenance, SHA-256, and TensorRT
10.16.2 build evidence are recorded in D-29. The parser accepts the model, but
FP16 engine creation failed through an internal TensorRT convolution
timing-model shader assertion; both attempts produced zero-byte engines. The
default build was stopped after extended tactic selection. No detector ran, and
no model/tracker runtime is yet feasible. Do not delete or use the zero-byte
engine files; they remain ignored evidence artifacts.

The final two sentences above are also historical: D-30 through D-36 record the
subsequent FP32 detector, reference-parity, MOT17, and ByteTrack feasibility
results. FP16 remains blocked and the zero-byte engines remain invalid.

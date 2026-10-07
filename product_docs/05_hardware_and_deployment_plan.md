# RACE-MOT Hardware, Deployment, and Measurement Plan

**Status:** Updated 6 October 2026 — Jetson Orin Nano confirmed; exact configuration and software image still need inventory.  
**Rule:** Use the available board; do not buy hardware to satisfy an unmeasured performance target.

## 1. Device decision card

Complete before G1. If the physical device is unavailable, do not make an edge-performance claim.

| Field | Selected value |
|---|---|
| Device/module and RAM | Jetson Orin Nano confirmed; record exact 4 GB/8 GB SKU and module marking during inventory |
| Carrier board / accelerator | Record developer-kit/carrier revision and any attached accelerator; do not assume an external accelerator |
| OS/image/kernel | Read from the existing installation; preserve it for the initial feasibility spike |
| Driver/runtime/compiler | Record JetPack, CUDA, TensorRT, Python, and PyTorch versions; verify model support before environment changes |
| Supported detector formats/precision | Candidate: YOLOX-Tiny exported to TensorRT FP16; verify conversion and output parity on the installed image |
| Power mode and clocks | Record active mode/clocks for every run; use one declared mode for policy comparisons |
| Cooling, fan, enclosure | Record fan/heatsink/enclosure and keep configuration fixed during comparative runs |
| Power measurement source and sample rate | Use an external meter at the Jetson power input as the primary whole-device measure when available; report onboard rail telemetry separately as diagnostic/cross-check |
| Ambient test condition | Measure and record at each sustained run |
| Available storage/network/input | Record free storage; phone camera via IP Webcam H.264/RTSP over direct USB tethering; no cloud relay |

The device family is selected. Before installing anything, inventory the board in place. NVIDIA's current product documentation lists Orin Nano configurations with different RAM and performance/power modes; do not assume that the specific unit is the 8 GB developer kit or the Super configuration. The currently published NVIDIA JetPack download page lists JetPack 7.2.1 / Jetson Linux 39.2.1 for the Orin family, while JetPack 6.2.3 is also listed in the archive for Orin Nano. Preserve a working installed image; if a reflash becomes necessary, choose an officially supported release only after checking the YOLOX/TensorRT dependencies. [NVIDIA JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads), [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive), [Jetson Orin Nano modules](https://developer.nvidia.com/embedded/jetson-modules).

Do not compare device datasheet TOPS to measured FPS or estimate joules from FLOPs. Record the board's actual configuration and measurements.

## 2. Frozen deployment profile

Record phone camera resolution/cadence, codec, stream URL with credentials redacted, detector checkpoint, precision, batch size (MVP batch size 1), preprocessing, decode method, tracker version/parameters, predictor/model hash, policy threshold, maximum skip count, rendering state, and logging state. Start phone demonstrations at 1280×720, H.264, 15 input frames/s if the phone app supports it; treat this as a reproducible starting profile, then freeze a profile after the G1/G3 baseline. Use recorded benchmark frames for repeatable policy comparisons. Use identical settings across compared policies except for the policy/model change under study.

### Device inventory commands (run on the Jetson)

Save the output in a private hardware manifest before setting up Python/model dependencies:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
cat /etc/os-release
uname -a
dpkg-query -W nvidia-jetpack
sudo nvpmodel -q
tegrastats --interval 1000
free -h
df -h
```

If an optional package query is unavailable, record that rather than changing the system to make the query succeed. Capture `tegrastats` during an idle period and during the later sustained run; do not use a one-second sample as a complete energy measurement.

## 3. End-to-end timing boundary

Primary latency is from frame arrival/read to emitted tracks and required status/log output. Record component timings as diagnostics: decode, preprocessing, detector, temporal features, predictor, policy, ByteTrack, rendering/export, and queueing. Report p50/p95, throughput, frame queue depth, dropped frames, and deadline misses against a deadline $D$ derived from the agreed input cadence.

Do not report detector-only timing as product latency. State whether video rendering, decode, local UI, and power metering are included in each measurement.

## 4. Energy boundary

Measure energy from frame arrival through emitted tracks for the complete Jetson pipeline. Use an external meter at the Jetson power input as the primary whole-device measure when available, and declare whether the board, carrier, fan, and attached peripherals fall inside the measured boundary. Include decoding, scene feature extraction/context encoder/fusion, detector, predictor, calibration, policy, tracker, and required rendering/logging. Integrate energy over the complete run and divide by **all input frames**, including detector skips. State that the phone, access point, and browser are outside the Jetson boundary unless separately metered. Report total Jetson joules/input-frame as primary; any idle-subtracted result is secondary and must show the subtraction method.

Report onboard rail telemetry as a separate diagnostic/cross-check, with its rail definition and sampling limitations; do not combine it with external-meter readings as a single exact value. Record meter sampling rate, calibration/check procedure, idle baseline, warm-up, run length, repeat count, and synchronization between power and frame logs. A one-second tegrastats sample is not a complete energy measurement.

## 5. Thermal and sustained operation

Run long enough to reach sustained operating behavior, not just a short benchmark. Log temperature, clocks, power mode, fan state, ambient temperature, and throttling indicators. Define the sustained thermal limit from the selected device documentation and use case before final testing. Report when throttling occurs and whether quality/deadline behavior changes afterward.

## 6. Deployment constraints

- One stream in the MVP. Multi-stream performance is out of scope.
- Local inference by default; no cloud upload in the MVP.
- Model files and logs have explicit storage paths and cleanup behavior.
- The phone is the live demo source. Keep it stationary on a stable mount, enable USB tethering and IP Webcam, use RTSP over the resulting private local link, disable any cloud relay, and specify reconnect, buffering, timestamp, and frame-drop semantics separately from offline MOT file evaluation.
- On device or runtime failure, stop safely, report the fault, and preserve only authorized summary diagnostics.

## 7. Device feasibility gate

Before implementing the learned controller, demonstrate that the candidate detector + ByteTrack can load and process both the phone RTSP stream and MOT video files on the Jetson, with stable memory and observable timing. This is a limited feasibility milestone, not a performance claim. Then measure the detector-every-frame baseline. Only after its quality/service thresholds are frozen should adaptive-policy implementation proceed.

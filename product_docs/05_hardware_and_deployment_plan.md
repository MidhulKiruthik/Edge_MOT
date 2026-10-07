# RACE-MOT Hardware, Deployment, and Measurement Plan

**Status:** Updated 8 October 2026 — target Jetson/software, carrier identity, power mode, and active fan telemetry are recorded; exact physical cooling assembly and sustained measurement setup remain open.
**Rule:** Use the available board; do not buy hardware to satisfy an unmeasured performance target.

## 1. Device decision card

Complete before G1 closure. Existing observations are feasibility evidence, not edge-performance claims.

| Field | Selected value |
|---|---|
| Device/module and RAM | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super; 7.6 GiB observed RAM; module compatibility `p3767-0005-super` |
| Carrier board / accelerator | NVIDIA `p3768-0000` carrier compatibility; no external accelerator recorded |
| OS/image/kernel | Ubuntu 24.04.5, Jetson Linux R39.2.1; preserve the installed image |
| Driver/runtime/compiler | JetPack 7.2.1, CUDA 13.2, TensorRT 10.16.2.10, Python 3.12.3, OpenCV 4.8.0; isolated CPU PyTorch is G1-only |
| Supported detector formats/precision | YOLOX-Tiny TensorRT FP32 is the current verified path; FP16 engine creation is blocked by a TensorRT builder assertion |
| Power mode and clocks | 25 W mode observed; record clocks and mode for every comparative run |
| Cooling, fan, enclosure | Enabled PWM fan and tachometer observed; exact physical heatsink/fan/enclosure still requires visual recording |
| Power measurement source and sample rate | Use an external meter at the Jetson power input as the primary whole-device measure when available; report onboard rail telemetry separately as diagnostic/cross-check |
| Ambient test condition | Measure and record at each sustained run |
| Available storage/network/input | Local MOT17 replay active; phone H.264/RTSP basic USB decode previously verified but further phone work deferred under D-33 |

The actual device inventory above now controls this project. Preserve the working installed image; if a reflash becomes necessary, choose an officially supported release only after checking YOLOX/TensorRT dependencies. [NVIDIA JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads), [JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive), [Jetson Orin Nano modules](https://developer.nvidia.com/embedded/jetson-modules).

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
- The phone remains the eventual live demo source, but additional phone work is deferred under D-33. When resumed, keep it stationary, use the private local link, disable cloud relay, and specify reconnect, buffering, timestamp, and frame-drop semantics separately from MOT evaluation.
- On device or runtime failure, stop safely, report the fault, and preserve only authorized summary diagnostics.

## 7. Device feasibility gate

Bounded local evidence shows the FP32 detector and pinned ByteTrack can process MOT17 inputs; this is not a performance or quality claim. Phone acceptance remains deferred under D-33. Before the learned controller, resolve or explicitly defer the remaining G1/G2 issues, pass G4, then measure the detector-every-frame baseline and freeze its quality/service thresholds.

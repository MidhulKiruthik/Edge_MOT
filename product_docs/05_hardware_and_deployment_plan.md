# RACE-MOT Hardware, Deployment, and Measurement Plan

**Status:** Updated 8 October 2026 — target Jetson/software, exact physical cooling details, power mode, ambient condition, a 30-minute FP32 detector thermal diagnostic, deterministic local detector/tracker replay, full-sequence baseline metrics, and sustained complete-pipeline telemetry are recorded. Whole-pipeline external-meter energy and final G3 limit acceptance remain open.
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
| Cooling, fan, enclosure | Stock integrated active aluminum heatsink with 4-pin PWM fan; fan PWM enable `1`, duty `42`; open benchtop; private photo reference recorded in `race_mot/data/device_manifest.json` |
| Power measurement source and sample rate | Use an external meter at the Jetson power input as the primary whole-device measure when available; report onboard rail telemetry separately as diagnostic/cross-check |
| Ambient test condition | 27°C air-conditioned room for the 8 October 2026 sustained diagnostic |
| Available storage/network/input | Local MOT17 replay active; phone H.264/RTSP basic USB decode and current endpoint probe verified under D-28/D-39 |

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

### 2.1 Sustained FP32 detector diagnostic (8 October 2026)

The private report at `race_mot/reports/2026-10-08-g3-sustained-fp32/` ran the
FP32 TensorRT diagnostic engine against repeated `MOT17-02-FRCNN` prefixes for
1,800 requested seconds and 1,801.026 observed seconds. It processed 34,180
detector frames in every-frame mode; all outputs were finite and the median
20-frame batch rate was 19.04 frames/s. `tegrastats` recorded 1,789 samples:

| Quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| `tj` temperature | 40.94°C | 46.59°C | 49.25°C |
| RAM used | 2,981 MB | 3,732 MB | 4,393 MB of 7,485 MB |
| `GR3D_FREQ` | 0% | 19% | 99% |
| `VDD_IN` reported fields | 4.06 W | 6.135 W | 9.176 W |

This is a detector and thermal diagnostic. It repeats the same local MOT17
prefix, excludes ByteTrack and the product logger/dashboard, and has no
external power meter. It therefore does not close G3 latency, tracking-quality,
or whole-pipeline energy acceptance.

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
- The phone remains the eventual live demo source. Keep it stationary, use the private local link, disable cloud relay, and specify reconnect, buffering, timestamp, and frame-drop semantics separately from MOT evaluation; the current D-39 probe does not close those acceptance items.
- On device or runtime failure, stop safely, report the fault, and preserve only authorized summary diagnostics.

## 7. Device feasibility gate

Bounded local evidence shows the FP32 detector and pinned ByteTrack can process MOT17 inputs, and the 30-minute FP32 diagnostic remained below 49.25°C at the reported junction peak in the open benchtop setup. This is not a complete performance or quality claim. Phone acceptance has only bounded decode evidence under D-39. Before the learned controller, resolve or explicitly defer the remaining G1/G2 issues, pass G4, then measure the detector-plus-tracker every-frame baseline and freeze its quality/service thresholds.

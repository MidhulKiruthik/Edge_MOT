# RACE-MOT Agent Instructions

## Project Mission

RACE-MOT is a product-first, local edge-computing prototype for anonymous,
single-camera pedestrian multi-object tracking. The target deployment is one
stationary phone camera running IP Webcam and streaming H.264/RTSP over a
private USB-tethered local link to
one NVIDIA Jetson Orin Nano. Recorded MOT17 and MOT20 inputs provide repeatable
evaluation.

The MVP contains:

- One person detector: YOLOX-Tiny at 416-pixel input, pending device and
  checkpoint feasibility verification.
- One pinned ByteTrack implementation.
- A detector-every-frame baseline and a binary `DETECT`/`SKIP` policy.
- A causal, per-track predictor for avoidable identity failure after a
  proposed detector skip, with calibration evaluated separately from raw
  scores.
- A low-cost scene-discovery guard that may upgrade `SKIP` to `DETECT`, never
  the reverse.
- Temporary per-run track IDs, an active-track count used only as an occupancy
  proxy, a local dashboard, decision logs, and device measurements.

This is not a people-identification, face-recognition, validated people-
counting, cloud, multi-camera, Re-ID, or consequential-decision system.

## Source Of Truth

Read these documents before making a scope or architecture change:

1. `todo.md` for the phase-by-phase execution roadmap.
2. `product_docs/08_decisions_and_pre_code_gate.md` for accepted decisions,
   open blockers, and G4 approval conditions.
3. `product_docs/01_product_brief.md` through
   `product_docs/07_verification_acceptance_release.md` for product,
   requirements, architecture, data, hardware, privacy, and verification
   contracts.
4. `race_mot/SDD.md` for implementation interfaces and runtime behavior.
5. `product_docs/09_faculty_pitch.md` for research framing and prior art.

The faculty pitch is a proposal and presentation document, not evidence that
performance, calibration, energy savings, thermal stability, or novelty have
been demonstrated. The research report and decision log control technical
claims. The older `energytrack_project_proposal.md` is historical where it
conflicts with the current binary-action scope.

## Current Phase And Gate Rules

Phases 0–6 are complete for the approved deterministic/local-data scope. G4 is
recorded under D-47, Phase 5 baseline measurement is recorded under D-49, the
Phase 6 training-role dataset is recorded under D-50, and transient phone
integration is recorded under D-51. Do not claim a trained/calibrated risk
model, adaptive-policy benefit, or whole-device energy result until the later
measurement and validation gates are complete.

### Phase 0: Reconcile And Freeze The Working Scope

Start here. Reconcile the documents and record changes in
`product_docs/08_decisions_and_pre_code_gate.md` before changing code. In
particular:

- Mark every quantitative statement in the faculty pitch as `cited`,
  `measured`, or `TBD`.
- Remove or qualify unsupported detector timings, compute shares, energy
  outcomes, thermal claims, parameter counts, latency guarantees, and words
  such as `first`, `breakthrough`, and `proves`.
- Resolve the conflicting TCN size and latency claims by deriving them from a
  frozen model definition or keeping them as `TBD`.
- Reconcile Jetson SKU, RAM, JetPack, CUDA, TensorRT, power mode, and thermal
  details with actual inventory and official documentation.
- Keep requirements, architecture, data plan, hardware plan, privacy plan,
  verification plan, SDD, roadmap, and pitch consistent.
- Build a requirements-to-component-to-verification map and record unresolved
  decisions with owner, due phase, and consequence.

### Phase 1 / G1: Device And Input Feasibility

The feasibility evidence is complete and retained as the current baseline:

- Run `race-mot inventory` on the actual Jetson and save a private manifest.
- Probe the authorized phone RTSP stream without saving frames.
- Decode an authorized local MOT file and record frame-order behavior.
- Check candidate YOLOX-Tiny, TensorRT FP16, and ByteTrack feasibility on the
  exact board without assuming another Jetson configuration.

Do not reflash or replace Jetson system OpenCV as the first troubleshooting
step. Keep secrets out of command history and reports.

### Phase 2 / G2: Data, Privacy, And Labels

Before training or collecting local footage, freeze dataset terms, sequence
groups, permissions, retention, privacy defaults, and the paired-rollout label
protocol. Group all MOT17 detector variants from one source scene together.
Keep MOT20 locked for transfer evaluation. Runtime features must be causal;
ground truth and future frames are offline-only.

### Phase 3 / G4: Pre-Code Approval

G4 and the deterministic implementation gate are closed. Continue to preserve
the frozen contracts, dependency/model provenance, verification mapping, and
measurement boundary when working on later phases.

### Later Phases

Follow `todo.md` in order: deterministic every-frame runtime, G3 baseline and
measurement contract, paired labels, calibrated model, risk-only scheduler,
guard and ablations, final evaluation, product hardening, and optional paper
decision. Do not skip the every-frame baseline before adaptive-policy work.

## Engineering Constraints

- Preserve the exact two-action scope: `DETECT` and `SKIP`.
- First frame, no active tracks, invalid risk, and maximum consecutive skips
  must force `DETECT`.
- Record planned and executed actions, reason codes, calibrated validity,
  detector gap in seconds and source frames, consecutive skips, input drops,
  and timestamps separately.
- Distinguish a detector call returning no detections from an intentional
  detector skip in the tracker and logs.
- Never use future frames, annotations, or labels in runtime decisions.
- Use sequence-grouped splits; never randomly split MOT frames or tracklets.
- Fit normalization on training data, calibration on natural-prevalence
  calibration data, and policy thresholds on separate validation data.
- Keep MOT20 and held-out stress conditions untouched until evaluation.
- Measure complete-pipeline cost, including decode, preprocessing, detector,
  tracker, predictor, policy, guard/context, rendering, and required logging.
- Divide energy by all input frames, including frames where detection is
  skipped. Keep external whole-device meter readings separate from onboard
  rail telemetry.
- Never report a detector-only timing as end-to-end latency or infer energy
  from TOPS/FLOPs.
- Keep raw video transient by default. Do not log frames, face crops,
  embeddings, persistent identities, credentials, or full secret-bearing URLs.
- Do not add comments, abstractions, dependencies, or features unrelated to
  the current phase. Do not add Re-ID, optical flow, ROI inference, multiple
  cameras, cloud processing, thermal control, or a new dataset without a
  recorded scope decision.

## Development Workflow

Before editing, identify the owning module and state one falsifiable local
hypothesis about the behavior. Make the smallest change that tests it. After
the first edit, run the narrowest relevant executable check before broadening
the work.

For the current scaffold in `race_mot/`:

```bash
cd race_mot
python -m unittest discover -s tests -v
race-mot validate-config --config configs/baseline.json
```

On the Jetson, after the environment is authorized and installed:

```bash
race-mot inventory --output reports/device_inventory.json
race-mot probe --input "rtsp://REDACTED" --duration-sec 30 --output reports/phone_stream_probe.json
```

Do not put real RTSP credentials in shell history, source files, or logs. A
probe is feasibility evidence only; it is not tracking, latency, calibration,
thermal, or energy evidence.

## Reporting Rules

Every result must identify the device/module, RAM, software image, model and
checkpoint, input resolution/cadence, precision, batch size, tracker version,
power mode, cooling, measurement boundary, and run configuration. If a metric
is unavailable, say why. If a policy fails a frozen constraint, report the
negative result and do not call it energy-saving. Keep product acceptance and
research-method acceptance separate.

Use links to the governing local documents in implementation notes and update
the decision log when a decision changes scope, data, hardware, metrics, or
privacy behavior.

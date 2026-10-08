# RACE-MOT End-to-End Implementation Roadmap

**Purpose:** Convert the approved product and research design into a sequenced, evidence-gated implementation plan. Completed items cite their implementation or measurement evidence; open items remain explicitly gated.

**Project outcome:** A local, single-camera pedestrian-tracking prototype on the available Jetson Orin Nano, accompanied by a reproducible evaluation package. A conference paper is optional and depends on the resulting evidence.

**Status on 8 October 2026:** Device inventory, carrier identity, active fan telemetry, USB-tethered H.264/RTSP decode, full three-scene MOT17 replay, FP32 YOLOX-Tiny execution/output decoding, expanded OpenCV-reference parity, paired-rollout/label contracts, pinned ByteTrack feasibility, the deterministic local detector/tracker runtime, repeated baseline measurements, complete-pipeline sustained telemetry, official TrackEval 1.3.0 metrics, the audited Phase 6 training-role dataset, transient no-save phone integration, and a 30-minute FP32 thermal diagnostic have evidence (D-23, D-26, D-28 through D-51). D-47 records supervisor G4 sign-off, D-48 records Phase 4 implementation, D-49 records Phase 5 baseline measurement, D-50 records Phase 6 dataset generation, and D-51 records phone integration. The detector's measured three-scene recall is 52.43%, below the earlier 65% proposal, and is documented as an explicit acceptance waiver. G3 is partial only because external-meter energy is unavailable; adaptive policy remains gated.

## Current gate status and next actions

| Gate | Current state | What closes it |
|---|---|---|
| G0 - Product definition | Accepted in the current product documents | Reopen only through a recorded scope decision |
| G1 - Feasibility | Substantially evidenced; device/cooling, three-scene MOT17 replay, FP32 decoding/parity, pinned ByteTrack feasibility, and machine-generated transient phone integration are recorded; FP16 is explicitly deferred | Use G3 for full quality/resource claims; the phone is not a labeled benchmark source |
| G2 - Data and responsible use | Passed for local MOT17 implementation and transient phone processing; terms, grouped roles, MOT20 manifest, rollout protocol, paired-rollout contract, privacy boundary, and label audit fixture are recorded | Keep raw phone frames transient and excluded from training/calibration/evaluation labels |
| G3 - Measurement contract | Partial; full baseline, official TrackEval metrics, sustained telemetry, and frozen limits are recorded under D-49, but external-meter energy is unavailable | Connect an external Jetson-input meter and approve the recorded final quality/energy limits |
| G4 - Pre-code approval | Passed for deterministic Phase 4 and authorized transient phone integration | Required documents, verification map, acceptance matrix, supervisor sign-off, and D-51 phone evidence are recorded |

**Implementation order update (D-38/D-39/D-47/D-48/D-50/D-51, 8 October 2026):** Supervisor sign-off authorized deterministic Phase 4 implementation, Phase 5 measurement is recorded, the frozen training-role Phase 6 dataset now has evidence, and transient phone integration is measured. External-meter G3 closure, model training, and adaptive scheduling remain gated. Keep MOT17 and derived labels local and unredistributed under D-35/D-40; keep phone frames transient and unlabeled under D-51.

**Do next, in order:**

1. Connect an external meter at the Jetson input boundary and repeat the frozen baseline long enough to integrate Joules per input frame.
2. Record the meter calibration/sampling boundary, synchronize it with the frame log, and accept or revise the frozen G3 limits.
3. Keep phone frames transient and excluded from labels; do not enable adaptive scheduling until the external energy boundary and limits are accepted.

Phases 0–6 are complete for the approved deterministic/local-data scope. Keep the frozen Phase 6 artifact unchanged while proceeding to Phase 7 model fitting/calibration; the adaptive scheduler remains gated by the external-energy G3 boundary and later validation phases.

## Approved MVP boundary

- One stationary phone camera for the live demonstration, using H.264/RTSP over a private local network; recorded benchmark files are used for repeatable evaluation.
- One candidate person detector: YOLOX-Tiny at 416-pixel input, batch size one. This remains provisional until checkpoint terms, actual device compatibility, and reference-output parity are checked. Use TensorRT FP16 only if supported and validated.
- One fixed primary tracker: ByteTrack. Pin its implementation/version and parameters for all matched policy comparisons.
- One physical device: the available Jetson Orin Nano. Record the exact module/SKU, RAM, carrier, OS/JetPack, CUDA/TensorRT, power mode, cooling, clocks, and measurement setup.
- MOT17 source scenes are the main grouped development/evaluation data. MOT20 is held out for dense-crowd transfer evaluation; it is not used for feature, model, calibrator, corruption, or threshold tuning.
- One primary methodological claim: calibrated per-track prediction of avoidable identity failure under a specified next-frame detector-skip action, used to schedule the next detector call.
- The runtime has two actions, `DETECT` and `SKIP`. A scene-discovery guard is a secondary product-safety component and may only upgrade a planned skip to full-frame detection.
- Deliver a local prototype and evidence package. Multi-camera tracking, Re-ID, face identity, cloud processing, variable resolution, ROI/patch actions, learned motion replacement, pseudo-depth, Mamba/SSM, and thermal-feedback control are outside the MVP.

## Source-of-truth and claim rules

- Use the decision log for approved decisions; requirements, architecture, data/hardware plans, verification plan, and SDD for product contracts and implementation behavior.
- Use the research report for the bounded novelty claim, related work, target definition, and comparison plan. The claim is the proposed research framing until results establish its measured value.
- Treat `energytrack_project_proposal.md` as historical where it conflicts with the current binary detect/skip scope.
- Treat the faculty pitch as a presentation document that must be reconciled with the above sources. A number in a pitch is not a requirement or a result by itself.
- Do not describe the offline paired rollout as proof of causal effects in every deployment. Do not claim calibration, savings, real-time performance, thermal stability, or superiority before measurement.
- Record scope changes in the decision log first, then update affected documents before implementation follows the change.

### Pitch claim reconciliation required in Phase 0

Earlier pitch revisions contained conflicting parameter and latency claims. The current pitch still needs a complete claim audit, consistent gate numbering, and reconciliation of feature/calibration contracts. Before faculty submission:

- Replace unmeasured timings, compute shares, and energy outcomes with `TBD` or clearly marked hypotheses; do not retain them as promised specifications.
- Reconcile hardware and power-mode details against the exact Orin Nano SKU, installed JetPack, and official NVIDIA documentation. Do not copy AGX Orin or other-module figures onto this board.
- Either derive the TCN parameter count from the frozen model definition or remove it until that definition exists. Remove latency targets unless they are explicitly proposal targets and are measured on the target device before any result claim.
- Replace unsupported terms such as "first", "breakthrough", and "proves" with the bounded method claim and direct-comparison plan. Keep application/market claims only when they are sourced and relevant to the prototype.

## Terms used in this roadmap

- `K`: future-frame horizon of the offline paired rollout.
- `M`: persistence length used by the frozen identity-failure rule.
- `alpha`: IoU matching threshold for anchor/identity assignment.
- `tau`: policy-validation threshold for calibrated frame-level risk.
- `S_max`: maximum consecutive detector skips.
- `D`: end-to-end deadline derived from the selected input cadence/use case and recorded at G3.
- `DMR`: deadline-miss rate. A source frame that never reaches the pipeline is tracked separately as an input drop.

## How to execute this roadmap in this project

This section converts the phase checklist into an implementation procedure for the current repository. The commands assume a shell on the Jetson unless marked `Windows`. Run them from `edge/race_mot`. Keep `reports/`, `runs/`, model files, credentials, and private footage local; commit only source code, non-secret configs, manifests that contain no private identifiers, and documentation.

### Working rules

1. Create one dated working directory for each gate or experiment, for example `reports/2026-10-07-g1/` and `runs/2026-10-07-baseline-mot17-02/`. Never overwrite a report or run.
2. Before every hardware or benchmark run, record the Git revision, config path, model hash, tracker version, device manifest, power mode, input source, warm-up duration, and operator initials in the run manifest.
3. Use the existing dependency-free tests as the first check after every code change:

    ```bash
    cd edge/race_mot
    python -m unittest discover -s tests -v
    python -m race_mot.cli validate-config --config configs/baseline.json
    ```

    The installed console command `race-mot` is equivalent after `python -m pip install -e .`. If a command does not exist yet, add it only when its phase begins; do not create a fake command that produces incomplete evidence.
4. Every experiment has four files: `manifest.json` (inputs and hashes), `config.json` (frozen settings), `frames.jsonl` or `metrics.csv` (row-level evidence), and `summary.json`/`summary.md` (derived results and failures). Store plots beside the summary and keep the script that generated them.
5. Treat `TBD` as a blocked value. A run may measure a provisional value, but it may not be used to close a gate until the corresponding decision is recorded and the value is frozen.
6. Separate three clocks in every runtime record: source timestamp, local frame-arrival time, and detector/pipeline completion time. Separately count source input drops, detector skips, and frames that miss the deadline.

### Repository build order

Implement the runtime in this order. Each item must pass its local tests before the next item is started.

| Order | Location | What to implement | Local proof |
|---|---|---|---|
| 1 | `src/race_mot/sources/` | File and RTSP adapters that emit `FramePacket` and explicit end/loss/error states | Replay a short file and assert source indices, timestamps, and drop counters |
| 2 | `src/race_mot/detectors/` | YOLOX reference/TensorRT adapter with original-frame coordinate restoration | Golden detections on a fixed clip; reference versus engine tolerance report |
| 3 | `src/race_mot/trackers/` | Pinned ByteTrack wrapper with `initialize`, `update`, `skip`, and `reset` | Same detections produce stable IDs; empty detection and skip remain distinct |
| 4 | `src/race_mot/logging/` and `telemetry/` | Manifest, append-only frame records, summaries, Jetson telemetry | Restart and disk/error tests; no image or credential fields in logs |
| 5 | `src/race_mot/application.py` | Sequential baseline orchestrator and clean lifecycle | Detector-every-frame file replay reaches a complete summary |
| 6 | `src/race_mot/web/` | Local dashboard backed by a non-blocking snapshot | Dashboard cannot block inference; bind address is explicit and tested |
| 7 | `src/race_mot/evaluation/` | Rollouts, grouped splits, metrics, calibration, statistics | Tiny synthetic fixture plus one permitted MOT sequence |
| 8 | `src/race_mot/risk/` and `policy/` | Causal features, predictor, calibrators, scheduler, guard | Unit tests for causality, invalid risk, max skips, and guard upgrade |

Do not start with the TCN or dashboard. The first complete implementation target is a deterministic detector-every-frame file replay. The adaptive path is enabled only after G3.

### Phase-by-phase execution cards

#### Phase 0: reconcile and freeze the plan

1. Read the nine product documents and SDD into a one-page traceability table with columns `ID`, `requirement/decision`, `owner module`, `verification command or test`, `evidence path`, and `status`.
2. Search the faculty pitch for every number, result verb, and superlative. Mark each as `cited`, `measured`, `proposal target`, or `TBD`; replace unsupported results with proposal wording.
3. Record unresolved choices in `product_docs/08_decisions_and_pre_code_gate.md` before editing downstream documents. At minimum resolve detector checkpoint, ByteTrack source, MOT version/terms, exact split roles, dashboard bind address, and raw-footage handling.
4. Add only non-secret baseline defaults to `configs/baseline.json`. Put private RTSP values in an environment variable, never in JSON or shell history.

**Phase output:** `reports/phase0/traceability.csv`, reconciled pitch, updated decision log, and a signed or explicitly deferred G4 checklist. The phase is complete when each G4 checkbox has a verifiable owner and evidence location.

#### Phase 1: verify the Jetson and inputs

1. On the Jetson, create the environment with system packages visible, install this package editable, and run the existing inventory command. Save the output under a dated report directory. Capture the exact module/SKU, RAM, JetPack/L4T, CUDA, TensorRT, Python, OpenCV backend, power mode, clocks, cooling, storage, and telemetry availability.
2. Run the existing no-save probe against the phone stream for at least 30 seconds. Redact the source identifier before sharing the report. Repeat once after a controlled USB-network interruption and record reconnect behavior separately.
3. Run `inspect-mot` against an authorized local MOT annotation and decode a short local video without writing frames. Compare frame count/order and timestamps with the source metadata.
4. Perform a bounded detector smoke check: load the candidate checkpoint, run batch-one inference on ten permitted frames, and record load time, per-stage time, output shape, person detections, memory, and errors. This is feasibility evidence, not an FPS or quality claim.
5. If TensorRT conversion fails, save the full error and installed versions, diagnose an operator/version mismatch, and update the decision log before changing detector families.

```bash
race-mot inventory --output reports/2026-10-07-g1/device_inventory.json
race-mot probe --input-env RACE_MOT_RTSP_URL --duration-sec 30 --output reports/2026-10-07-g1/phone_stream_probe.json
race-mot inspect-mot --gt /authorized/MOT17-02-FRCNN/gt/gt.txt
```

**Phase output:** device manifest, stream report, local-file decode report, detector/tracker smoke note, and a list of blockers. Do not report tracking quality, real-time performance, or energy at this phase.

#### Phase 2: freeze data, privacy, and labels

1. Create `data/manifest.json` locally with source name/version, terms location, checksum, sequence path, source-scene group, and assigned role. Group MOT17 detector variants by source scene before splitting.
2. Create `data/roles.json` with train, calibration, policy-validation, and final-evaluation roles. Keep MOT20 outside all fitting and selection operations.
3. Implement the paired rollout as a pure offline function. Clone the tracker state at anchor frame `t`; branch A skips detector at `t+1`, branch B detects; then run detector-every-frame for the next `K-1` frames. Assert that both branches begin from identical serialized state and use the same future frames.
4. Implement the label predicate in `evaluation/labels.py` using frozen `alpha`, `M`, visibility, persistence, identity matching, and boundary-censor rules. Return `positive`, `safe_negative`, `ineligible`, or `censored`; never coerce the last two to negative.
5. Write a small audit file with hand-checked ordinary motion, crossing, occlusion, entry, exit, re-entry, and boundary cases. Compare generated labels to the expected outcomes in tests.
6. Record local-footage authorization before using the phone around people. Keep raw video transient, use temporary IDs, disable face crops/embeddings, and test that default logs contain no frame payload.

**Phase output:** source/terms manifest, grouped split manifest, frozen label protocol and pseudocode, audited fixture results, privacy decision, and exclusion/prevalence report.

#### Phase 3: close G4 and lock contracts

1. Walk the traceability table with the supervisor. For each open item choose `closed`, `deferred with owner/date`, or `removed from MVP`; do not leave an unqualified TBD.
2. Freeze the JSON configuration schema and add validation for model hash, tracker version, source kind, output path, policy fields, measurement fields, and privacy/export flags.
3. Freeze the runtime records from the SDD: `FramePacket`, `Detection`, `TrackSnapshot`, schedule state, decision, frame log, and run manifest. Add schema-version fields so later analysis can reject incompatible records.
4. Write the acceptance test matrix before implementation: input, expected action/state, evidence file, and pass/fail rule. Include clean stop, restart, stream loss, invalid input, detector failure, disk-full, and credential-redaction cases.
5. Sign and date the gate in the decision log. Until this exists, only G1 feasibility code and hardware-independent contract tests are permitted.

**Phase output:** signed G4 checklist, frozen config/schema, traceability matrix, acceptance matrix, dependency/license record, and implementation branch/tag.

#### Phase 4: build the deterministic baseline

1. Implement the source adapters first and test them on files. Preserve original source indices; never renumber after a drop. RTSP reconnect must emit a visible event and never silently turn a missing frame into a normal frame.
2. Implement the detector adapter with separate preprocessing, inference, postprocessing, and coordinate-mapping timers. Save a golden input/output fixture from the reference runtime before TensorRT conversion.
3. Wrap ByteTrack behind the SDD API. Verify `update([])` means detector ran with no accepted person and `skip()` means detector did not run. Reset all state at sequence boundaries and run starts.
4. Implement the sequential orchestrator. In baseline mode every received frame executes DETECT, then tracker update, history update, logging, and output. Keep queue limits bounded and make stop/restart idempotent.
5. Add manifest and JSONL logging before the dashboard. Log actions, reasons, timestamps, counters, timings, track state, errors, and telemetry; never log raw frames, secrets, or persistent identities.
6. Replay a short file repeatedly and compare hashes/counts of detections, tracks, and frame records. Only after this is stable, connect the phone source.

**Phase output:** repeatable file replay, fixed detector/tracker evidence, run manifest, frame log, summary, and failure/restart test results. No adaptive policy.

#### Phase 5: measure baseline and close G3

1. [x] Warm up the Jetson, fix 25 W/open-benchtop cooling and 30 FPS input cadence, and run MOT17-02/04/05 in the same order for two repetitions. Manifests record the detector/tracker/configuration and environment.
2. [x] Evaluate the full local sequences at IoU 0.50 and visibility >=0.20. Official TrackEval 1.3.0 reports HOTA/DetA/AssA, IDF1, MOTA, ID switches, fragmentation, and detector counts; the dependency-free evaluator remains a cross-check.
3. [x] Measure arrival-to-track pipeline latency, preprocessing, inference, postprocessing, coordinate mapping, queue depth, throughput, DMR, and source drops. Warm-up-excluded p50/p95 values are retained.
4. [ ] Measure whole-pipeline energy at the Jetson input boundary with an external meter. Onboard VDD_IN is retained only as a diagnostic; no Joules/input-frame claim is made.
5. [x] Run a 60-second complete detector/tracker sustained test with 60 telemetry samples; record RAM, junction temperature, GR3D utilization, onboard VDD_IN, power mode, cooling, and explicit unavailable throttling flags. The earlier 30-minute detector-only thermal record remains separate.
6. [x] Freeze the 30 FPS deadline, warm-up/repetition protocol, quality margins, service limits, RAM/thermal limits, and 10% minimum useful-energy target in `configs/g3_limits.json`; official TrackEval is now recorded, while external-meter energy remains the open prerequisite for policy validation.

**Phase output:** baseline report, raw timing/telemetry traces, official metric summary, repetition protocol, frozen G3 limits, and an explicit external-energy gap. The Phase 5 implementation is measured but G3 is not fully closed until the external meter is connected and the limits are accepted.

#### Phase 6: generate paired learning data

1. [x] Run the frozen detector/tracker on the selected MOT17 training-role sequences and save versioned anchor states, detections, tracker state, timestamps, and hashes.
2. [x] Generate paired rollouts only from eligible anchors. Serialize branch configuration, unique anchor states, and initial-state hashes so labels are reproducible without hidden runtime state.
3. [x] Run duplicate-frame, sequence-leakage, boundary-censor, and label-prevalence checks. Produce per-source-scene counts and a deterministic audit sample.
4. [x] Fit training-only geometry normalization statistics and label class weights. Calibration and policy-validation roles remain untouched; varied prior skip histories are included as generated runtime states.
5. [x] Freeze detector-gap bins (`0`, `1`, `2`, `3`) and keep source-frame subsampling, network/source drops, and policy skips as separate conditions; the generated local runs have zero source drops and no subsampling.

**Phase output:** reproducible rollout dataset, label manifest, audit report, grouped roles, and leakage checks. No model selection on MOT20.

#### Phase 7: train and calibrate the risk model

1. Build a feature extractor that accepts only current/past state: box geometry/velocity, elapsed seconds, source-frame gap, skip count, Kalman residual/covariance, confidence/association history, age/misses, and local overlap.
2. Assert causality with a test that changing a future frame cannot change features at anchor `t`. Normalize using training statistics saved with a hash.
3. Train the causal TCN first, then matched GRU, temporal MLP, static, confidence-only, and Kalman-only baselines. Use grouped roles and at least three seeds for final comparisons.
4. Fit per-track Platt/isotonic calibration on the calibration role. Aggregate with `max_i(q_i)` and fit a separate frame-level calibrator; never reuse the per-track map for frame scores.
5. Select model/configuration using policy-validation results and predictor overhead, not test results. Measure parameter count from the actual frozen model and measure latency/memory/energy on the Jetson.

**Phase output:** model/checkpoint hashes, seed/config manifests, calibration files, prediction metrics, reliability plots, device overhead report, and a model-selection decision.

#### Phase 8: implement and compare the risk-only scheduler

1. Implement the scheduler as a pure state transition around the existing `DetectorAction`, `DecisionReason`, and policy contracts. First frame, no tracks, invalid risk, threshold crossing, and `S_max` always force DETECT.
2. Plan the next action only after the current frame is processed. At the next frame, record planned action, executed action, and any guard override as separate fields.
3. Choose `tau` and `S_max` on policy-validation sequences under frozen G3 limits. Include predictor, calibration, tracker, logging, and UI costs in energy and latency.
4. Implement matched baselines with the same detector, tracker, source frames, device, power mode, output, and measurement boundary: every-frame, fixed interval, confidence-only, Kalman uncertainty, static learned risk, EMO-like, RT-MOT-like, ALBIREO-like, and HSFSO where reproducible.
5. Label each comparator as reproduction, adaptation, or approximation. Record departures and do not compare unmatched published FPS/AP numbers to this project result.

**Phase output:** frozen policy config, action/reason logs, comparator matrix, policy-validation report, and risk-only held-out result. Do not add the guard until this report exists.

#### Phase 9: add only justified guard/ablations/stress

1. Reuse one per-frame low-resolution scene descriptor for context and discovery guard where possible; time it independently. The guard may only upgrade SKIP.
2. Define entrant cases before evaluation. Measure acquisition delay/recall, censored events, false overrides per 1,000 frames, tracking metrics, latency, and complete-pipeline energy.
3. Compare risk-only and risk-plus-guard. Remove the guard if the incremental discovery value does not justify cost or false triggers.
4. Run feature-group, context-fusion, MOT20 transfer, brightness/contrast, blur, compression, and timestamp-preserving subsampling experiments from predeclared configs. Keep input drops distinct from policy skips.
5. Run quantization last, recalibrate each variant, and compare detector diagnostics, tracking, calibration, latency, energy, RAM, and thermal behavior.

**Phase output:** ablation/stress matrix, guard decision, held-out transfer report, and removed-feature/guard record where applicable.

#### Phase 10: final statistics and claims

1. Freeze all code/config/model hashes and execute the planned repeated runs: at least three learned-model seeds and five warm-up-separated hardware runs per finalist when resources permit.
2. Pair policies by sequence/run condition. Use sequence-aware resampling or grouped confidence intervals; never treat frames as independent observations.
3. Generate the results package from raw traces, not manually edited spreadsheets: tracking, calibration, energy per all input frames, latency, DMR, drops, RAM, temperature, clocks, throttling, and Pareto plots.
4. Write a claim table with columns `claim`, `evidence`, `hardware/data scope`, `uncertainty`, and `allowed wording`. Explicitly state when no policy meets every frozen constraint.

**Phase output:** reproducible tables/plots, uncertainty estimates, claim/limitation record, and raw-trace archive.

#### Phase 11: product acceptance and handoff

1. Execute the acceptance matrix on the Jetson: start/pause/stop/restart, RTSP loss/reconnect, invalid input, decoder/model failure, disk/log failure, clean shutdown, dashboard access, and opt-in export.
2. Check every release run for manifest linkage, temporary IDs, no raw video by default, redacted credentials, no face/embedding data, and correct retention cleanup.
3. Prepare installation steps, a live phone demo, a recorded-file fallback, known failure cases, limitations, architecture figure, and results slides. Test the backup path before the demonstration.
4. Package source/configuration without restricted data and include model/code/data terms and cleanup instructions.

**Phase output:** signed product acceptance, release/evidence package, demo script, setup guide, known-limitations list, and independent method-acceptance result.

### Definition of done for a phase

A phase is done only when its code (if any), tests, configuration, raw evidence, derived report, and decision-log update exist together. A passing unit test does not close a hardware or research gate; conversely, a negative policy result is valid evidence and should be recorded rather than hidden by changing thresholds after the fact.

## Phase 0 - Reconcile documents and freeze the working scope

- [x] Record product-first outcome, one-camera/device boundary, candidate detector/tracker, two-action policy, and primary method claim in the decision log.
- [x] Maintain the research report and faculty pitch as research framing and prior-art records.
- [x] Complete the pitch claim reconciliation above: quantitative statements are marked as cited context, measured bounded evidence, proposal targets, or `TBD`; unsupported outcome language is qualified (D-37).
- [x] Confirm the brief, requirements, architecture, data plan, hardware plan, privacy plan, verification plan, decision log, SDD, research report, and pitch agree on model, actions, datasets, device, outputs, and metrics (D-37, D-39).
- [x] Create a requirements-to-component-to-verification map in `product_docs/10_requirements_traceability.md` (D-21).
- [x] Record unresolved choices with an owner, due phase, and consequence in the decision log. `TBD` values remain blocked until resolved.

**Exit evidence:** One internally consistent scope, reconciled pitch, current decision log, and traceable requirements. The high-level scope is already agreed; this phase closes remaining contradictions and unsupported specifications.

## Phase 1 - G1: Verify the Jetson and both input paths

The existing inventory and stream-probe utilities are scaffolding, not evidence. This limited feasibility work is the explicit pre-MVP exception; it is not permission to build the full application.

- [x] Device inventory, exact physical cooling, open-benchtop setup, 27°C ambient, current 25W mode, and sustained FP32 thermal diagnostic are recorded (D-23/D-24/D-41). The detector-plus-tracker baseline is closed under D-49; whole-pipeline external energy remains open.
- [x] Preserve the installed image for the initial feasibility check. No reflash or system OpenCV replacement was performed.
- [x] Current phone endpoint `http://10.152.75.6:8080/video` decoded without errors in a five-second no-save probe (D-39); prior USB H.264 evidence remains in D-28.
- [x] Phone interruption/reconnect and trusted-LAN acceptance is measured for controlled client reconnect/private-route reachability under D-51; no physical cable/network fault is claimed.
- [x] Decode an authorized local MOT17 sequence and verify bounded frame order/dimensions (D-31).
- [x] Establish expanded YOLOX-Tiny FP32/OpenCV parity and pinned ByteTrack plausibility on the exact board (D-29 through D-36, D-42); the measured recall deviation and checkpoint candidate are accepted for implementation planning under D-43.
- [x] Explicitly defer TensorRT FP16 after the builder failure; revisit only through an official compatible JetPack/TensorRT path (D-45).
- [x] Restore and bounded-decode the planned MOT17-04 and MOT17-05 development scenes (D-44); three-scene detector recall is recorded under D-46.
- [x] Live-demo network exposure and reconnect verification is user-attested under D-43; a dated machine-generated report remains recommended for audit.

**Exit evidence (G1):** Device manifest, phone-stream and file-decode reports, candidate stack feasibility note, and any deferred issue recorded. Do not claim tracking quality, detector FPS, or energy savings at this gate.

## Phase 2 - G2: Approve data, privacy, and the failure-label protocol

- [x] Private MOT17 manifest with official terms (CC BY-NC-SA 3.0), locally computed archive SHA-256 hashes, permitted-distribution record, and final grouped scene roles (D-40/GC-01, 8 October 2026).
- [x] Create the MOT20 manifest and freeze the paired-rollout label protocol before any label generation or training (`data/mot20_manifest.json`, `data/rollout_protocol.json`, D-45).
- [x] Record the grouping rule that all MOT17 detector variants of one source scene remain in the same role (D-35).
- [x] Record MOT20 as locked held-out dense-crowd transfer evaluation; do not use it for fitting or selection.
- [x] Before label generation, freeze `alpha`, visibility/occlusion rules, `K`, `M`, sequence-boundary censoring, detector/tracker versions, tracker initialization, matching implementation, and branch semantics (`data/rollout_protocol.json`, D-45).
- [x] Implement the paired offline rollout contract from the same tracker state at frame `t`: one branch skips detection at `t+1`, the other detects at `t+1`; both then use detector-every-frame updates for the next `K-1` frames (`evaluation/paired_rollout.py`).
- [x] Use the research report's frozen target: `Y = F_skip * (1 - F_detect)`. Positive, safe-negative, ineligible, and boundary-censored outcomes are represented by the protocol contract.
- [x] Write label pseudocode and manually audit examples for ordinary motion, crossings, occlusion, entries/exits, re-entry, and sequence boundaries (`data/label_audit.json` and tests); report exclusions and prevalence in the fixture.
- [x] Record the Phase 2 freeze boundary; D-51 later authorizes transient phone processing with no raw-frame retention. Local MOT17 remains private and unredistributed.
- [x] Review and record local-network access, credential redaction, log allowlists, and the prohibition on face crops, appearance embeddings, and persistent identity in the Phase 2 freeze record.

**Exit evidence (G2):** Data/footage permissions and handling record, sequence/split manifest, frozen label protocol, pseudocode, and audited examples. These are required before training or selecting a policy threshold.

## Phase 3 - G4: Close the full-MVP pre-code gate

- [x] Close G1 and G2, or record each explicit deferral, owner, risk, and supervisor acceptance in the decision log (D-43, D-45, D-47, D-51); phone integration is now evidenced for the transient demo scope.
- [x] Complete requirement-to-component-to-verification traceability and agree which acceptance items are MVP blockers.
- [x] Confirm SDD contracts: frame packet and timestamps, detector output, tracker update versus skip, scheduler action/reason, run record, queue limits, restart, and error behavior; runtime execution remains Phase 4 work.
- [x] Verify detector/checkpoint provenance and terms, TensorRT compatibility, ByteTrack source/license, and dependency pinning plan; FP16 is explicitly deferred under D-45.
- [x] Approve the measurement boundary, metric definitions, repetition plan, and method for deriving quality margins, deadline, deadline-miss bound, memory/temperature limits, and minimum useful energy reduction after the every-frame pilot.
- [x] Reserve time for integration, target-device measurements, failure recovery, analysis, and demo preparation.
- [x] Obtain formal dated student/supervisor G4 sign-off; recorded under D-47.

**Gate:** G4 is signed for deterministic Phase 4 implementation. The bounded tracker/runtime slice is complete under D-48; do not enable training or the adaptive scheduler until their G2/G3 requirements are met, and do not claim a useful compute policy until full every-frame measurement closes G3.

## Phase 4 - Build the deterministic, every-frame runtime

- [x] Implement and replay-verify the local MOT image-sequence source adapter; preserve one-based frame indices, timestamps, and visible decode failures (D-38).
- [x] Keep raw phone persistence disabled while retaining source index/drop semantics in the adapter; D-51 records the transient live run.
- [x] Add config validation, run ID/manifest, clean start/stop/restart, bounded queue metadata, redacted logs, and visible source/detector failure handling (D-48).
- [x] Freeze the FP32 detector checkpoint, preprocessing/coordinate mapping, batch size, and OpenCV reference fixture. FP16 remains deferred under D-45 (D-42, D-48).
- [x] Pin the ByteTrack API/version and parameters; test initialization, `update([])`, `skip()`, reset, and stable IDs (D-48).
- [x] Add frame-stage timing and retain source timestamp, arrival clock, detector action, and input-drop fields (D-48).
- [x] Replay the three restored development scenes in detector-every-frame mode; repeat `MOT17-02-FRCNN` for hash equality, and pass failure/restart tests (D-48).

**Exit evidence:** Repeatable local file replay, fixed FP32 detector/tracker outputs, golden reference fixture, immutable manifest, redacted frame log, summary hashes, clean failure/stop/restart behavior, and the separate D-51 phone integration record. No adaptive policy is enabled.

## Phase 5 - Measure the every-frame baseline and close G3

- [x] Run the frozen detector + ByteTrack on every input frame on the physical Orin Nano: two full repetitions of MOT17-02/04/05 ran in frozen order, and D-51 records a separate 100-frame phone integration pass. Phone frames are not benchmark labels.
- [x] Add the minimum local replay dashboard after the core path: source/run health, temporary-track geometry overlay, temporary IDs, active-track count, action/status, latency, queue/drop state, and errors. It binds to loopback by default; a trusted-LAN bind is explicit and authorized.
- [x] Compute HOTA, DetA/AssA, IDF1, MOTA, ID switches, fragmentation, and detector diagnostics using pinned official TrackEval 1.3.0; the dependency-free evaluator remains a cross-check.
- [x] Measure end-to-end latency from frame arrival/read through emitted tracks and required product log output. Component times, queue depth, drops, p50/p95, sustained throughput, and DMR are recorded.
- [ ] Measure complete-pipeline energy at a declared Jetson input boundary with an external meter as the primary measure. The run records the unavailable state and does not claim Joules/input-frame.
- [x] Keep onboard rail telemetry separate from whole-device meter readings. The report labels VDD_IN as diagnostic only while the external meter is unavailable.
- [x] Record peak RAM, temperature, clocks, power mode, fan/cooling, ambient condition, warm-up, run duration, and throttling availability during sustained operation. The 60-second complete-pipeline run and earlier 30-minute detector-only thermal record are separate.
- [x] Derive and freeze `D`, HOTA/IDF1 non-inferiority margins, maximum DMR, RAM/thermal limits, minimum useful energy reduction, and sustained-run duration in `configs/g3_limits.json`; guard discovery/false-trigger criteria remain Phase 9 work.

**Exit evidence (G3):** Frozen every-frame baseline configs/results, declared measurement boundary, repeat protocol, official metric summary, sustained telemetry, and frozen numeric limits are recorded under D-49. G3 remains partial only for the external-energy measurement; risk-policy optimization depends on closing that boundary.

## Phase 6 - Build and audit the paired-rollout learning dataset

- [x] Run the fixed detector/tracker over selected MOT17 training-role source sequences; preserve outputs, annotations, timestamps, configs, hashes, and software versions.
- [x] Generate the paired skip/detect rollouts from identical anchor state according to the frozen `K=5`, `M=2`, IoU `0.50`, visibility `0.20`, and future-frame schedule.
- [x] Implement the avoidable-failure label. Boundary, no-visible-observation, and other ambiguous anchors are censored instead of silently labelled safe.
- [x] Include varied prior detector-skip histories (`1`, `2`, and `3` previous skips) in sampled training-role rollout states.
- [x] Compare generated labels to the frozen hand-audit protocol fixture. Report positive prevalence, exclusions, per-scene counts, duplicate-frame checks, and sequence-leakage checks.
- [x] Freeze sequence-grouped training, calibration, policy-validation, and final-evaluation roles from `data/roles.json`; only the training role is generated in this phase and no random frame/tracklet split is used.
- [x] Fit geometry normalization and label class weights on training data only. Preserve calibration and policy-validation natural prevalence by generating no labels for those roles.
- [x] Freeze detector-gap bins before held-out scoring. Source-frame subsampling, network/source drops, and policy detector skips remain separate fields.

**Exit evidence:** Reproducible data/label manifest, unique serialized anchor states, audited rollout implementation, sequence-role manifest, training-only statistics, and an explicit proof that test labels were not used for model or policy selection are recorded in `product_docs/18_phase_6_dataset_record.md`.

## Phase 7 - Train and calibrate the temporal risk predictor

- [ ] Construct only causal per-track features: normalized box geometry/velocity, elapsed-time-normalized motion, Kalman residual/covariance, confidence and association-margin history, age/misses, local overlap/crowding, elapsed seconds, source-frame gap, consecutive policy skips, and source-input-drop count.
- [ ] Train a small causal TCN as the primary architecture. Compare compact GRU, temporal MLP, static logistic/MLP, confidence-only, and Kalman-uncertainty-only baselines using identical labels and grouped sequence roles.
- [ ] Apply weighting/mining only in training. Fit Platt or isotonic calibration on separate natural-prevalence calibration sequences.
- [ ] Evaluate per-track calibration and separately fit/evaluate the frame-level calibration after max-risk aggregation. Max aggregation changes the score distribution; it cannot inherit the per-track calibrator.
- [ ] Report prevalence, PR-AUC, operating-point precision/recall, Brier score, ECE with stated bins, reliability curves, per-scene outcomes, and detector-gap strata.
- [ ] Measure parameter count from the frozen model definition and measure predictor latency, memory, and energy on the Orin Nano. Do not impose or repeat unmeasured sub-millisecond claims.
- [ ] Only after the history-only model is stable, test context concatenation and optional shared-context gated fusion as matched ablations. Retain extra context computation only if held-out policy value justifies its cost.

**Exit evidence:** Model/calibrator hashes, configs, seeds, calibration results, predictor overhead on device, and documented model-selection decision.

## Phase 8 - Implement the risk-only scheduler and matched policy comparisons

- [ ] Implement the bounded risk-only scheduler first: first frame detects; force `DETECT` if no tracks are active, risk is invalid, calibrated frame risk reaches `tau`, or `S_max` skips have accumulated; otherwise plan `SKIP`.
- [ ] Fit frame risk from `max_i(q_i)` using frame labels `max_i(Y_i)` on calibration data. Select `tau` and `S_max` only on policy-validation sequences under frozen G3 limits.
- [ ] Log planned and executed actions, calibration-validity status, risk/threshold, max-risk temporary track, trigger/reason, actual detector gaps, skip count, and component times.
- [ ] Compare under matched detector, tracker, sequence, device, input cadence, power mode, output settings, and energy boundary: every-frame; fixed intervals; confidence-only; Kalman uncertainty; static learned risk; EMO-like; RT-MOT-like; and ALBIREO-like per-object uncertainty scheduling.
- [ ] Reproduce HSFSO if feasible; otherwise document the faithful approximation, inputs, and departures. Never treat unmatched published FPS/AP values as a direct win.
- [ ] Address temporal ID-switch prediction such as Split and Connect in the comparison record. Implement an online-input baseline only if it can be adapted faithfully; otherwise explain why its offline repair output is not a matched detector scheduler and compare its prediction target/timing explicitly.
- [ ] State for every comparator whether it is a direct reproduction, an adapted implementation, or a simplified approximation. Report the gap rather than presenting an approximation as the original method.
- [ ] Choose `tau` and `S_max` to minimize measured complete-pipeline energy subject to the frozen HOTA/IDF1, p95 latency, DMR, RAM, and sustained-temperature limits. Count predictor/policy cost; choose only on policy-validation sequences.
- [ ] Confirm that the risk-only model improves useful held-out policy behavior over simple rules and the closest reproducible learned scheduler before adding optional complexity.

**Exit evidence:** Trace-checked bounded policy, frozen validation choices, complete action/reason logs, matched comparator matrix, and risk-only results including predictor overhead.

## Phase 9 - Test the discovery guard, ablations, stress, and quantization

- [ ] Add the scene-discovery guard only after risk-only evaluation. It can upgrade `SKIP` to full-frame `DETECT`, never downgrade planned detection.
- [ ] Define new-entry scoring before held-out evaluation. Test entrants while other tracks are active and compare risk-only, risk-plus-guard, and ALBIREO-like empty-scene/rescue behavior. Report acquisition delay/recall, censored/undiscovered events, guard precision, false overrides per 1,000 input frames, HOTA/IDF1, latency, and complete-pipeline energy.
- [ ] Retain the guard only if its incremental discovery benefit justifies false triggers and full-system cost. Otherwise ship risk-only mode and record the negative finding.
- [ ] Compare track-history-only against context concatenation and, optionally, shared-context gated fusion with labels, folds, model family, calibration, and policy held fixed.
- [ ] Run predeclared feature-group ablations for the cues that motivate the method (especially actual detector-gap history, motion/Kalman residuals, and local crowding/overlap). Report changes in prediction calibration and policy-level quality/energy; treat logs as audit evidence, not causal explanations.
- [ ] Run predeclared MOT20 dense-crowd transfer evaluation with the frozen model and policy; report crowd/occlusion and detector-gap strata without tuning on MOT20.
- [ ] Run secondary brightness/contrast, blur, compression, and timestamp-preserving source-frame-subsampling conditions separately. Do not conflate input-frame loss with policy skips or tune on held-out stress conditions.
- [ ] Run quantization last and only if supported by the installed runtime. Recalibrate each variant and compare detector diagnostics, HOTA, IDF1, ID switches, fragmentation, calibration, latency, energy, and thermal behavior.

**Exit evidence:** Frozen configs and complete results for each ablation/stress condition, including incremental compute cost and any removed feature/guard.

## Phase 10 - Final evaluation and statistical analysis

- [ ] Use at least three independent training seeds for learned-model comparisons and at least five warm-up-separated hardware runs for each finalist policy/dataset/configuration. Treat these as planned minima; if a documented resource constraint prevents them, record the shortfall before final claims and downgrade the strength of conclusions.
- [ ] Compare policies in paired sequence/run conditions. Use sequence-aware resampling or grouped confidence intervals; do not treat frames as independent samples.
- [ ] Report HOTA/IDF1 and differences from the same-device every-frame baseline, MOTA, DetA/AssA, IDSW, fragmentation, calibration, Joules per all input frames, p50/p95 latency, DMR, input drops, peak RAM, temperature, clocks, and throttling.
- [ ] Plot the tracking-quality/energy Pareto frontier and mark the feasible region under frozen quality and service constraints. Include predictor, context, guard, and required logging/UI overhead.
- [ ] Report per-sequence and stress-stratum results so failures on dense or long-gap scenes remain visible.
- [ ] State explicitly whether any policy satisfies every constraint. If none does, report that outcome plainly and do not describe the policy as energy-saving.
- [ ] Preserve raw traces and enough metadata to regenerate tables/plots from frozen configs, hashes, and evaluation outputs.

**Exit evidence:** Reproducible traces/summaries, paired uncertainty estimates, Pareto plots, and an evidence-bounded result/limitation statement.

## Phase 11 - Product hardening, acceptance, and handoff

- [ ] Complete start/pause/stop/restart, stream interruption/reconnect, invalid input, decoder/model failure, disk/log failure, and clean-shutdown behavior.
- [ ] Verify dashboard access is limited to the trusted LAN and RTSP credentials are never exposed. Raw video retention remains off by default; annotated export is opt-in.
- [ ] Produce a run manifest linked to device, model, tracker, policy, data split, seed, hashes, and measurement configuration.
- [ ] Verify decision logs contain only approved audit evidence and no frame payload, ground-truth labels, identity embedding, persistent identity, secret, or credential.
- [ ] Complete each applicable case in `07_verification_acceptance_release.md`; review demo footage for authorization and identifiable bystanders.
- [ ] Prepare installation/setup steps, demo script, architecture figure, results slides, known failure cases, limitations, and a backup recorded-file demonstration.
- [ ] Package code/configuration without restricted benchmark data; include software/model license provenance and cleanup/retention instructions.

**Product acceptance is separate from method acceptance.** A repeatable local prototype may pass product acceptance even if the adaptive policy fails its quality/energy hypothesis. The report must state that distinction accurately.

**Exit evidence:** Repeatable local demo and release/evidence package satisfying product requirements, with the adaptive-policy result reported independently.

## Phase 12 - Optional conference-paper decision

- [ ] Consider a paper only after the prototype, closest-work comparisons, repeatable statistics, and systematic novelty review are complete.
- [ ] Center any paper on the measured HOTA/IDF1-energy Pareto frontier, calibrated avoidable-failure prediction, and exact hardware/data conditions.
- [ ] Cite borrowed ideas, code, data, and figures; report reproductions/approximations and negative results. Avoid “first”, guaranteed calibration, or performance claims beyond the evidence.
- [ ] Make a supervisor-approved go/no-go decision. Publication is optional and is not a product acceptance gate.

## No-go and fallback rules

| Finding | Required response |
|---|---|
| Phone RTSP is unreliable but local decode works | Keep the approved phone-stream target visible; record the issue and demonstrate/evaluate via local files until the stream is repaired. Do not silently claim live-stream acceptance. |
| Candidate detector/checkpoint or TensorRT path is not viable on the exact board | Return to the decision log and approve a replacement before changing the frozen stack; update affected requirements, SDD, data generation, and comparisons. |
| Data/footage terms or permissions are unresolved | Do not use/distribute the affected data or footage; resolve terms or remove that source from the plan. |
| Suitable whole-device energy measurement is unavailable | Continue product and tracking evaluation, mark whole-pipeline Joules/frame unavailable, and make no measured energy-saving claim. |
| Paired labels are too rare, ambiguous, or unstable across source scenes | Revisit the target/label protocol with the supervisor before model selection. Do not rebrand an uncalibrated score as failure probability. |
| Calibration fails on held-out sequences | Report the output as a score, not a probability; the calibrated-risk novelty/performance claim has not passed. |
| No policy meets frozen G3 quality/service/energy constraints | Report the negative result; ship a valid prototype if product acceptance passes, but do not label it energy-saving. |
| Guard/context/quantization adds more cost than held-out value | Remove it from the product path; retain the result as an ablation if useful. |

## Critical path

```text
G1 actual-device and input probe ----\
                                     +--> G4 signed pre-code gate --> deterministic every-frame runtime
G2 data/privacy/label protocol ------/                                  |
                                                                        v
                                                        G3 every-frame baseline + limits
                                                                        |
                                                                        v
                                                paired labels --> model + calibration
                                                                        |
                                                                        v
                                           risk-only policy + matched comparators
                                                                        |
                                                                        v
                                   guard/ablations/stress/quantization (only if justified)
                                                                        |
                                                                        v
                                            statistical evaluation --> product acceptance
                                                                        |
                                                                        v
                                                       optional paper decision
```

## Source documents

- [Research proposal and prior-art review](deep_research_mot_edge_merged.md)
- [Earlier energy-tracking proposal (historical; superseded where scope conflicts)](energytrack_project_proposal.md)
- [Product documentation index and gate definitions](product_docs/README.md)
- [Product brief](product_docs/01_product_brief.md)
- [Product requirements and acceptance criteria](product_docs/02_product_requirements.md)
- [System architecture](product_docs/03_system_architecture.md)
- [Data, labels, and model plan](product_docs/04_data_and_model_plan.md)
- [Hardware and deployment plan](product_docs/05_hardware_and_deployment_plan.md)
- [Privacy, security, and responsible-use plan](product_docs/06_privacy_security_responsible_use.md)
- [Verification, acceptance, and release plan](product_docs/07_verification_acceptance_release.md)
- [Decision log and pre-code gate](product_docs/08_decisions_and_pre_code_gate.md)
- [Faculty pitch](product_docs/09_faculty_pitch.md)
- [Software Design Document](race_mot/SDD.md)
- [Implementation workspace status](race_mot/README.md)

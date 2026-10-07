# RACE-MOT Requirements Traceability

**Status:** Phase 0 working baseline, updated 8 October 2026. Some G1
feasibility evidence exists, but no row by itself is evidence of product acceptance.

This matrix links each product requirement to its owning component and a
specific verification case. A requirement remains open until its case has
retained evidence from the declared device, input, configuration, and run.

| ID | Requirement | Owning component | Verification case | Expected result | Evidence to retain |
|---|---|---|---|---|---|
| FR-01 | Start, pause, stop, restart phone RTSP or authorized file input | Application/source adapter/dashboard | V-PIPE-01 | State transitions complete cleanly for RTSP and file replay | Run manifest, state log, clean-stop record |
| FR-02 | Validate dimensions, cadence, timestamps, decode errors, reconnects | Source adapter | V-INPUT-01 | Valid input reports metadata; invalid/interrupted input reports actionable failure and separate drops | Probe report, fault-injection log |
| FR-03 | Display temporary tracks and active-track count as occupancy proxy | Tracker/dashboard | V-UI-01 | Rendered IDs reset per run and count matches emitted active snapshots | Screenshot/sample records, summary |
| FR-04 | Support one detector, ByteTrack, risk model, guard, and binary actions | Orchestrator/detector/tracker/policy | V-POLICY-01 | Configuration selects only `DETECT` or `SKIP`; guard only upgrades skip | Config hash, action trace |
| FR-05 | Provide detector-every-frame baseline and adaptive mode | Orchestrator | V-BASE-01 | Baseline invokes detector on every processed source frame | Baseline action log, detector-call trace |
| FR-06 | Record risk, gaps, skips, guard, action, reason, and timestamps | Run logger/telemetry | V-LOG-01 | One bounded record per processed frame with source drops distinct from policy skips | JSONL sample, schema check, log allowlist |
| FR-07 | Report tracking and resource metrics or unavailable reasons | Evaluation/telemetry | V-METRIC-01 | Summary contains defined metrics for labeled replay and explicit unavailable fields for unlabeled live input | Summary, evaluator manifest |
| FR-08 | Export summary/log; annotated video opt-in | Export/storage | V-PRIV-01 | Default run has no video; explicit authorized export is visible and removable | Filesystem review, export record |
| FR-09 | Surface errors and stop cleanly | Application/error handling | V-ERR-01 | Model, decoder, stream, disk, and stop faults produce failed/clean terminal state without orphaned resources | Fault matrix, process/resource check |
| FR-10 | Guard checks activity outside tracks and only upgrades skip | Scene-discovery guard/policy | V-GUARD-01 | Planned detect is never cancelled; validated activity can override skip and is logged | Scenario traces, guard cost/discovery results |
| NFR-01 | Measure on exact selected edge hardware/software | Inventory/run manifest | V-DEV-01 | Manifest records module/RAM, carrier, image, runtimes, cooling, power mode, and input | Inventory JSON, immutable manifest |
| NFR-02 | End-to-end latency, p50/p95, DMR, and drops | Timing/telemetry | V-MEASURE-01 | Arrival/read through emitted output is timed; deadline is declared at G3 | Synchronized timing trace, G3 config |
| NFR-03 | Complete-pipeline energy per all input frames | Energy telemetry | V-MEASURE-02 | External Jetson-input meter is primary when available; onboard rails remain separate diagnostics | Meter trace, boundary note, frame count |
| NFR-04 | Report HOTA, IDF1, MOTA, IDSW, fragmentation, detector metrics | Evaluation runner | V-EVAL-01 | Pinned evaluator produces metrics with sequence-aware uncertainty | Evaluator version, result tables, configs |
| NFR-05 | Prevent indefinite skipping | Binary scheduler | V-POLICY-02 | First frame, no tracks, invalid risk, and `S_max` force `DETECT` | Hand traces and unit tests |
| NFR-06 | No cloud dependency; private LAN only | Source/dashboard/deployment | V-SEC-01 | Runtime works with internet disabled; exposed interfaces are loopback or trusted LAN only | Network check, deployment config |
| NFR-07 | No raw video retention by default | Storage/export | V-PRIV-02 | No frame payload is written unless authorized export is enabled | Filesystem/log inspection |
| NFR-08 | No face recognition or persistent identity | Detector/features/logging | V-PRIV-03 | No face crops, embeddings, names, or cross-run IDs in code/config/logs | Feature and artifact review |
| NFR-09 | Every published number has complete provenance | Run manifest/reporting | V-REPORT-01 | Device, model, resolution, precision, batch, tracker, power, thermal, and boundary accompany results | Manifest-linked report |

## Gate Ownership

| Gate | Required traceability closure |
|---|---|
| G1 | V-INPUT-01, V-DEV-01, and candidate detector/tracker feasibility evidence |
| G2 | Dataset/footage authorization, grouped split manifest, label audit, V-PRIV-01/02/03 |
| G3 | V-BASE-01, V-MEASURE-01/02, V-EVAL-01, and signed limits derived from the baseline |
| G4 | All MVP rows have an owner, executable case, and accepted deferral or evidence |

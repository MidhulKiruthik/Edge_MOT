# RACE-MOT Product-First Pre-Code Documentation Pack

**Status:** Draft v0.4 — updated with deterministic Phase 4 local-replay, Phase 5 measurement, Phase 6 training-role dataset, and transient phone live-integration evidence; no adaptive-policy result or G3 external-energy acceptance is claimed.
**Date:** 8 October 2026
**Purpose:** Define the product prototype, its boundaries, evidence plan, and acceptance gates before coding begins.

## Product direction

The working MVP is a **local, single-camera pedestrian-tracking prototype** running on the available Jetson Orin Nano. A stationary mobile phone running IP Webcam supplies the live H.264/RTSP demo feed over a private USB-tethered local link; recorded MOT17/MOT20 files provide repeatable evaluation. The system detects and tracks people with anonymous short-lived IDs, displays tracks and an active-track count (an occupancy proxy), and records compute decisions and device measurements. It is an academic prototype, not a validated people-counting or production surveillance system.

The student/faculty demo workflow and phone-to-Jetson direction remain agreed. The target board/software, three-scene local MOT17 replay, TensorRT FP32 detector path, bounded reference parity, paired-rollout contracts, pinned ByteTrack API, deterministic local replay runtime, full Phase 5 baseline with official TrackEval metrics, sustained telemetry, the audited Phase 6 training-role dataset, and the transient phone integration checks have evidence. D-47 records supervisor G4 sign-off, D-48 records the local Phase 4 implementation with a documented detector recall waiver, D-49 records Phase 5 measurement, D-50 records the Phase 6 dataset, and D-51 records live phone integration; G3 is partial only because external-meter energy is unavailable.

## Read these documents in order

1. [Product brief](01_product_brief.md) — who the prototype serves, what problem it addresses, and what the MVP includes.
2. [Product requirements](02_product_requirements.md) — user-visible behavior and acceptance criteria.
3. [System architecture](03_system_architecture.md) — components, data flow, module boundaries, and failure behavior.
4. [Data and model plan](04_data_and_model_plan.md) — data sources, labels, splits, model comparisons, and leakage controls.
5. [Hardware and deployment plan](05_hardware_and_deployment_plan.md) — device decision, software environment, telemetry, and end-to-end energy boundary.
6. [Privacy, security, and responsible-use plan](06_privacy_security_responsible_use.md) — footage handling, access, retention, and use limits.
7. [Verification and acceptance plan](07_verification_acceptance_release.md) — what must be verified after implementation and what counts as a successful prototype.
8. [Decision log and pre-code gate](08_decisions_and_pre_code_gate.md) — unresolved assumptions, owner decisions, and the conditions that must be met before code starts.
9. [Faculty pitch](09_faculty_pitch.md) — concise research framing, prior art, gap, objectives, and proposed method for supervisor discussion.
10. [Software Design Document](../race_mot/SDD.md) — implementation structure, data/API contracts, run artifacts, runtime behavior, and development order.
11. [Requirements traceability](10_requirements_traceability.md) — requirement owners, verification cases, expected results, and retained evidence.
12. [Phase 4 implementation record](16_phase_4_implementation_record.md) — deterministic local replay runtime, hashes, tests, and remaining G3 limits.
13. [Phase 5 baseline record](17_phase_5_baseline_record.md) — full local baseline metrics, latency, telemetry, frozen limits, and remaining measurement gaps.
14. [Phase 6 dataset record](18_phase_6_dataset_record.md) — training-role paired rollouts, serialized anchor states, label audits, grouped-role protection, and training-only statistics.
15. [Phone live integration record](19_phone_live_integration_record.md) — no-save decode, FP32/tracker pass, controlled reconnect, and private-route evidence.
16. [Phase 0–6 completion status](20_phase_0_to_6_completion_status.md) — current phase-by-phase closure and remaining post-Phase-6 boundaries.

## Project source of truth

- The decision log is the authority for approved scope and decisions. The product documents are the authority for requirements, architecture, data, hardware, privacy, and acceptance contracts. The SDD is the authority for implementation interfaces and runtime behavior. The current technical direction and prior-art review in [deep_research_mot_edge_merged.md](../deep_research_mot_edge_merged.md) is research framing and evidence review; it does not override an approved product decision.
- [energytrack_project_proposal.md](../energytrack_project_proposal.md) is an earlier proposal draft. It contains broader compute modes and optional dataset ideas that conflict with the current binary skip/detect scope. Treat it as historical material, not an approved product specification.
- Keep new product decisions in the decision log. If a decision changes model actions, datasets, hardware, or metrics, update the technical report too.

## Document lifecycle and gates

| Gate | Required outcome | Coding allowed? |
|---|---|---|
| G0 — Product definition | A target user, usage setting, and MVP workflow are accepted; open assumptions are resolved or explicitly deferred. | No |
| G1 — Feasibility | The actual available edge device and video input are identified; a detector/tracker path is technically plausible on that device. | No feature implementation. A minimal feasibility spike may begin only after G0/G1 sign-off. |
| G2 — Data and responsible use | Dataset access/terms, evaluation split, footage permissions, privacy defaults, and retention are documented. | No |
| G3 — Measurement contract | Baseline metrics, quality margins, latency deadline, energy boundary, and run protocol are frozen after a pilot baseline. | No adaptive-policy implementation until baseline is recorded. |
| G4 — Pre-code approval | All required documents have owners/status, no blocking decision remains, and this checklist is signed. | Yes, scoped MVP implementation may begin. |

## Change control

Each document is a living draft. Record the date and reason for changes in [the decision log](08_decisions_and_pre_code_gate.md). Do not silently broaden the product to multiple cameras, Re-ID, face identity, cloud processing, variable resolution, or hardware-state-driven modes. Those require a new product requirement, privacy review, direct prior-art check, architecture update, and schedule approval.

## Current readiness

**Not ready to start adaptive-policy implementation.** D-47 authorizes deterministic Phase 4 implementation, D-48 records the bounded local replay, D-49 records the repeated full-sequence baseline, official TrackEval metrics, sustained telemetry, and frozen service limits, D-50 records the training-role dataset, and D-51 records the transient phone integration path. The detector recall deviation is explicitly recorded. External-meter energy and final G3 limit acceptance remain open. Set thresholds from the repeated baseline measurement protocol; do not report planned results as measured results.

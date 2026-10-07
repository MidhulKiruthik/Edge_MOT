# RACE-MOT Product-First Pre-Code Documentation Pack

**Status:** Draft v0.2 — planning only; device and live input direction now confirmed; no implementation or results are claimed.  
**Date:** 6 October 2026  
**Purpose:** Define the product prototype, its boundaries, evidence plan, and acceptance gates before coding begins.

## Product direction

The working MVP is a **local, single-camera pedestrian-tracking prototype** running on the available Jetson Orin Nano. A stationary mobile phone running IP Webcam supplies the live H.264/RTSP demo feed over a private USB-tethered local link; recorded MOT17/MOT20 files provide repeatable evaluation. The system detects and tracks people with anonymous short-lived IDs, displays tracks and an active-track count (an occupancy proxy), and records compute decisions and device measurements. It is an academic prototype, not a validated people-counting or production surveillance system.

The student/faculty demo workflow, Orin Nano device family, and phone-to-Jetson live input direction are agreed. Exact board/RAM/software inventory and stream feasibility still need a short G1 check. The project prioritizes a functioning prototype; a conference paper is a later option only if the work and evidence support it.

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

**Not ready to start adaptive-policy implementation.** Product user, demonstration scene, device family, and input direction are now selected. First inventory the exact Orin Nano/software, verify phone RTSP and candidate detector/ByteTrack feasibility, confirm data/footage terms, and record the detector-every-frame baseline. Set quality, deadline, and energy thresholds only after that baseline. A limited setup/feasibility spike is the next authorized step; do not report planned results as measured results.

# Incomplete Work: Phases 0–3

**Status date:** 8 October 2026  
**Source of truth:** [`todo.md`](../todo.md), [`08_decisions_and_pre_code_gate.md`](08_decisions_and_pre_code_gate.md), [`13_phase_2_freeze_record.md`](13_phase_2_freeze_record.md), and [`14_phase_3_acceptance_matrix.md`](14_phase_3_acceptance_matrix.md).

Phone capture is now reopened for authorized transient integration under D-51. Raw phone frames remain excluded from training, calibration, evaluation labels, and persistent storage.

## Phase 0 — Reconcile and freeze the plan

- **Final faculty claim pass:** complete under [`15_phase_0_claim_audit.md`](15_phase_0_claim_audit.md); repeat only after a future pitch change.

## Phase 1 — G1: Verify the Jetson and inputs

- **Three-scene detector acceptance:** complete under D-46; its 52.43% recall is accepted by supervisor waiver and remains below the earlier 65% proposal.
- **FP16 disposition is closed by D-45.** FP16 is explicitly deferred for this project version and may only be revisited through an official compatible JetPack/TensorRT path.
- **Integrated runtime evidence:** closed for the deterministic local-MOT Phase 4 implementation under D-48 and live phone integration under D-51. The 100-frame repeated replay, tracker lifecycle tests, restart test, injected source-failure report, 100-frame phone run, and controlled reconnect report are recorded.

## Phase 2 — G2: Freeze data, privacy, and labels

- **No remaining non-phone label-protocol blocker.** The frozen protocol and six-case audit fixture are recorded and tested.
- **Local-footage authorization:** supervisor authorization for transient phone processing is recorded under D-51; no raw phone footage is retained and no phone labels are generated.
- **Privacy implementation audit:** verify the running application enforces credential redaction, no frame payloads by default, no face crops/embeddings, temporary IDs, opt-in export, and retention/deletion behavior.
- **Phone data boundary is closed for the transient demo scope.** Terms, roles, MOT20 manifest, rollout protocol, audit fixture, privacy rules, and no-frame phone runtime evidence are recorded. A physical cable/network fault remains outside the measured reconnect check.

## Phase 3 — G4: Close contracts and evidence

- **G4 sign-off is recorded under D-47.** D-51 closes transient phone integration and controlled reconnect for the live demo; phone remains excluded from labeled benchmark claims.
- **Runtime contract implementation:** complete for the deterministic baseline under D-48 and live phone behavior under D-51; adaptive scheduler remains outside this implementation slice.
- **Configuration contract:** schema, model/engine hashes, tracker version, source kind, measurement/privacy fields, and schema version validation are implemented; G3 acceptance limits remain to be measured.
- **Acceptance execution:** local clean stop/restart, detector/source failure handling, stable replay, and default no-frame logging are executed under D-48. Disk-full and dashboard behavior remain deployment checks if those components are added.
- **Dependency freeze:** FP16 deferment and detector/checkpoint provenance are recorded. The ByteTrack API uses the pinned commit metadata with a dependency-free deterministic IoU compatibility backend for this baseline.
- **G3 readiness:** run the detector-plus-tracker every-frame baseline and freeze the measurement boundary, repetitions, latency/deadline definitions, DMR, RAM/temperature limits, energy method, and quality margins.

## Immediate order

1. Run the detector-plus-tracker every-frame G3 baseline with the full measurement protocol.
2. Keep the external-meter energy boundary open before enabling adaptive scheduling; the live phone integration record is complete for the authorized demo scope.

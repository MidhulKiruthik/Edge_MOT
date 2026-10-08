# Phase 2 Freeze Record

**Date:** 8 October 2026  
**Decision:** Phase 2 local MOT17 data and label protocol are frozen for implementation. Phone-footage authorization remains explicitly deferred by user instruction.

## Data and roles

- MOT17 terms: CC BY-NC-SA 3.0; artifacts remain private and unredistributed.
- Grouping: DPM, FRCNN, and SDP variants of one source scene stay in the same role.
- Training: scenes 02, 04, 05.
- Calibration: scene 09.
- Policy validation: scene 10.
- Final local evaluation: scenes 11, 13.
- MOT20: manifest frozen as locked zero-shot transfer/stress evaluation; not downloaded and not used for fitting or selection.
- Local development scenes restored and decoded: MOT17-02-FRCNN, MOT17-04-FRCNN, MOT17-05-FRCNN.

## Paired-rollout protocol

The versioned machine-readable record is [`race_mot/data/rollout_protocol.json`](../race_mot/data/rollout_protocol.json).

| Setting | Frozen value |
|---|---|
| Anchor matching IoU (`alpha`) | 0.50 |
| Minimum visibility | 0.20 |
| Future horizon (`K`) | 5 frames |
| Persistent failure length (`M`) | 2 consecutive visible frames |
| Matching | Greedy one-to-one IoU matching at the declared threshold |
| Sequence boundary | Censor anchors whose future horizon crosses the sequence boundary |
| Tracker initialization | Reset at sequence start; clone the serialized anchor state for both branches |
| Branch A | Skip detector at `t+1`; propagate tracker state |
| Branch B | Run detector at `t+1`; update tracker state |
| Future frames | Both branches use detector-every-frame updates for `t+2` through `t+K` |
| Target | `Y = F_skip * (1 - F_detect)` |
| Eligible outcomes | `positive` or `safe_negative` only |
| Excluded outcomes | `ineligible`, `ambiguous`, and `boundary_censored` |

The executable contract is implemented in [`paired_rollout.py`](../race_mot/src/race_mot/evaluation/paired_rollout.py), with tests for identical anchor state, identical future-frame indices, positive avoidable failure, and boundary censoring.

## Privacy and authorization

- Local MOT17 processing is authorized under the recorded dataset terms and remains private.
- No phone footage is captured, retained, or used in this Phase 2 freeze.
- Phone-footage consent, live-scene authorization, and any annotated export decision remain deferred until the phone work is resumed.

**Superseding note (D-51, 8 October 2026):** the supervisor later authorized
transient live phone processing for the demonstration. No raw phone frames are
retained and no phone labels are used for training, calibration, or evaluation.
- Default processing retains summaries and logs only; no face crops, embeddings, persistent identities, or raw frame payloads are allowed.

## Phase 2 exit evidence

- [`race_mot/data/manifest.json`](../race_mot/data/manifest.json)
- [`race_mot/data/roles.json`](../race_mot/data/roles.json)
- [`race_mot/data/mot20_manifest.json`](../race_mot/data/mot20_manifest.json)
- [`race_mot/data/rollout_protocol.json`](../race_mot/data/rollout_protocol.json)
- [`race_mot/src/race_mot/evaluation/paired_rollout.py`](../race_mot/src/race_mot/evaluation/paired_rollout.py)
- `race_mot/tests/test_core.py` paired-rollout contract tests
- D-40, D-43, and D-44 in the decision log

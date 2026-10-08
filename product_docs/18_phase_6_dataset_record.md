# Phase 6 paired-rollout dataset record

**Date:** 8 October 2026  
**Scope:** local MOT17 training-role scenes only; phone, MOT20, calibration, policy-validation, and final-evaluation data are excluded.

Phase 6 is implemented by [`phase6_dataset.py`](../race_mot/src/race_mot/evaluation/phase6_dataset.py) and executed through [`generate_phase6_dataset.py`](../race_mot/scripts/generate_phase6_dataset.py). The generator consumes the frozen detector-every-frame Phase 5 logs, reconstructs the deterministic tracker, serializes unique tracker states once, and runs matched SKIP/DETECT branches over the same five future frames.

The frozen protocol is [`rollout_protocol.json`](../race_mot/data/rollout_protocol.json): anchor IoU `0.50`, visibility minimum `0.20`, horizon `K=5`, persistence `M=2`, one-to-one IoU matching, detector skip only at `t+1`, and detector-every-frame updates thereafter. The grouped roles are read from [`roles.json`](../race_mot/data/roles.json); only representative FRCNN folders for scenes 02, 04, and 05 are generated because DPM/FRCNN/SDP variants share frames and ground truth and remain in one role.

## Generated artifact

The private artifact directory is [`runs/2026-10-08-phase6-dataset`](../race_mot/runs/2026-10-08-phase6-dataset):

- `anchor_states.jsonl` stores each unique serialized tracker state and its SHA-256 state hash.
- `anchors.jsonl` stores target identity/anchor matches and references a unique state by hash and key.
- `rollouts.jsonl` stores both branch observations, failure flags, labels, exclusions, future indices, and detector-gap bin.
- `training_statistics.json` stores geometry normalization statistics and class weights fitted only from the training role.
- `audit.json` stores prevalence, exclusions, duplicate-frame checks, sequence-role leakage checks, gap bins, and the audit sample.
- `manifest.json` records protocol/role hashes, source-run hashes, software backend, artifact hashes, and selection boundaries.

The run generated **31,971** paired rollouts across the three training scenes. There are **31,534** eligible labels: **31,507** safe negatives and **27** avoidable failures (positive prevalence `0.0856%`). **285** boundary-censored and **152** no-visible-observation pairs were excluded. Source drops were zero. The prior-history sample includes real one-, two-, and three-skip states; detector-gap bins `0`, `1`, `2`, and `3` are present.

The audit found no duplicate image frames within the selected sequences and no scene-role conflicts. The role boundary explicitly records that calibration, policy-validation, final-evaluation, official-test, MOT20, phone, and other source labels were not generated or used for selection. The existing six-case hand-audit fixture remains the protocol reference for ordinary motion, crossings, occlusion, entries/exits, re-entry, and sequence-boundary censoring.

Phase 6 is complete for the approved local training-role dataset. Model fitting, calibration, policy threshold selection, and any use of held-out roles remain Phase 7/8 work and must consume this frozen artifact without changing its labels or role boundaries.


# Phase 3 Acceptance Matrix

**Date:** 8 October 2026  
**Scope:** deterministic implementation gate; transient phone rows are closed under D-51, while phone frames remain excluded from labeled benchmark claims.

| ID | Case | Expected result | Evidence/status |
|---|---|---|---|
| G4-01 | Start local MOT sequence | Source emits ordered frame packets and run ID | MOT source adapter and D-38 replay evidence; closed |
| G4-02 | Clean stop | Capture/resources release and summary remains valid | D-48 `SequentialBaseline.stop()` is idempotent; 100-frame summaries complete; closed for local replay |
| G4-03 | Restart same sequence | State resets; no track/run state leaks across runs | D-48 lifecycle test runs the same runner twice with reset IDs; closed for local replay |
| G4-04 | Phone interruption/reconnect | Visible loss event, reconnect, source drops counted | D-51 controlled client reconnect: 3/3 cycles, 30/30 frames each, zero read errors; physical network fault not injected |
| G4-05 | Invalid input | Fail before a misleading successful report | CLI/probe validation; closed for current commands |
| G4-06 | Detector/model failure | Stop visibly with error and preserve diagnostic | D-48 exceptions are recorded as failed summaries; closed for local replay |
| G4-07 | Disk/log failure | Bounded failure path; no raw-frame fallback | Runtime integration pending |
| G4-08 | Credential redaction | Reports/logs contain no URL credentials or secrets | Existing CLI test; closed for current commands |
| G4-09 | Local-network exposure | Bind only to declared trusted LAN | D-51 private-route HTTP 200 response; endpoint reached via `wlP1p1s0`, body discarded |
| G4-10 | Privacy/export | No frame payload by default; export requires opt-in | D-48 manifest/log schema contains no images, URLs, secrets, or persistent IDs; closed for local replay |
| G4-11 | Detector provenance | Checkpoint/engine hashes and preprocessing are recorded | D-42/GC-02 and `checkpoint_provenance.json`; closed for implementation planning |
| G4-12 | Data split leakage | Detector variants of one scene remain in one role | D-40/GC-01 and `roles.json`; closed |
| G4-13 | Paired rollout integrity | Both branches share anchor hash and future-frame indices | `paired_rollout.py` tests; closed |
| G4-14 | G3 baseline readiness | Measurement boundary, repetitions, and metrics are defined | G3 pilot plan; actual every-frame run pending |

## Gate result

The Phase 3 contracts and evidence package are prepared. D-47 records supervisor sign-off, D-48 records deterministic Phase 4 implementation evidence, and D-51 records no-save phone decode, FP32/tracker integration, controlled reconnect, and private-route reachability. The repeated local baseline is complete for the bounded replay slice; full-sequence quality, latency, energy, and thermal limits remain G3 measurements. Adaptive-policy implementation remains blocked until G3 closes.

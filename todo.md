# RACE-MOT End-to-End Implementation Roadmap

**Purpose:** Convert the approved product and research design into a sequenced, evidence-gated implementation plan. This document plans the work; it does not claim that implementation or evaluation has happened.

**Project outcome:** A local, single-camera pedestrian-tracking prototype on the available Jetson Orin Nano, accompanied by a reproducible evaluation package. A conference paper is optional and depends on the resulting evidence.

**Status on 6 October 2026:** The product direction and documentation scaffold are in place. The scaffold's inventory and stream-probe utilities have not yet been run on the target Jetson. The detector/tracker baseline, counterfactual labels, risk model, adaptive policy, and measured results are pending. G4 pre-code approval has **not** been passed.

## Current gate status and next actions

| Gate | Current state | What closes it |
|---|---|---|
| G0 - Product definition | Accepted in the current product documents | Reopen only through a recorded scope decision |
| G1 - Feasibility | Open; Orin Nano family and phone-stream direction are selected, exact device and stream behavior are unverified | Actual-device inventory, no-save phone stream probe, local-file decode check, and candidate detector/tracker feasibility evidence |
| G2 - Data and responsible use | Open | Dataset terms/manifest, sequence split plan, phone-footage authorization, privacy and retention decisions |
| G3 - Measurement contract | Open; no baseline or numeric acceptance limits | Same-device every-frame baseline, declared measurement boundary, and approved limits derived from baseline evidence |
| G4 - Pre-code approval | Not passed | Required documents consistent, blockers resolved or explicitly deferred, verification mapped, and required sign-off recorded |

**Do next, in order:**

1. Run the existing inventory utility on the physical Jetson and save its device/software manifest.
2. Run the no-save RTSP probe with the phone on the private local network; separately verify local MOT-file decoding.
3. Confirm data-use and footage permissions, and reconcile the faculty pitch's unmeasured numeric claims before presenting it as a specification.

These are feasibility and planning tasks. Full MVP implementation starts only after G4. Adaptive scheduling and energy-saving claims additionally wait for G3.

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

The current pitch includes TCN size/latency statements that disagree (`<80k` parameters and `<0.4 ms` versus `<40k` and `<0.35 ms`), a separate parameter count, unmeasured detector timing/compute-share figures, specific Orin power-mode statements, and wording that says energy savings are already proven. Before faculty submission:

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

## Phase 0 - Reconcile documents and freeze the working scope

- [x] Record product-first outcome, one-camera/device boundary, candidate detector/tracker, two-action policy, and primary method claim in the decision log.
- [x] Maintain the research report and faculty pitch as research framing and prior-art records.
- [ ] Complete the pitch claim reconciliation above: mark each quantitative statement as `cited`, `measured`, or `TBD`; remove contradictions and unsupported outcome language.
- [ ] Confirm the brief, requirements, architecture, data plan, hardware plan, privacy plan, verification plan, decision log, SDD, research report, and pitch agree on model, actions, datasets, device, outputs, and metrics.
- [ ] Create a requirements-to-component-to-verification map. Every requirement needed for G4 must have a feasible acceptance case.
- [ ] Record unresolved choices with an owner, due phase, and consequence if unresolved. Do not let a TBD silently become a default.

**Exit evidence:** One internally consistent scope, reconciled pitch, current decision log, and traceable requirements. The high-level scope is already agreed; this phase closes remaining contradictions and unsupported specifications.

## Phase 1 - G1: Verify the Jetson and both input paths

The existing inventory and stream-probe utilities are scaffolding, not evidence. This limited feasibility work is the explicit pre-MVP exception; it is not permission to build the full application.

- [ ] Run device inventory on the physical Orin Nano. Record exact model/SKU, RAM, carrier, OS/JetPack, Jetson Linux, CUDA/TensorRT, Python/PyTorch, power mode, clocks, cooling, storage, and available telemetry.
- [ ] Preserve the installed image for the initial feasibility check. Do not reflash or replace system OpenCV as the first troubleshooting step.
- [ ] Connect the stationary phone and Jetson to the same private Wi-Fi/hotspot. Probe H.264/RTSP; 1280x720 at 15 input frames/s is a starting profile only if the phone supports it.
- [ ] Run the no-save probe. Record dimensions, reported/measured cadence, timestamp behavior, read interval, drops, reconnect behavior, and a redacted source identifier. Keep credentials out of command history and logs.
- [ ] Decode an authorized MOT file locally and verify frame order and timestamps.
- [ ] Establish whether the candidate detector/checkpoint and ByteTrack can plausibly run on the exact board. Record evidence and any blockers; do not infer feasibility from a different Jetson model.
- [ ] Verify the demo has no cloud relay or public-network exposure.

**Exit evidence (G1):** Device manifest, phone-stream and file-decode reports, candidate stack feasibility note, and any deferred issue recorded. Do not claim tracking quality, detector FPS, or energy savings at this gate.

## Phase 2 - G2: Approve data, privacy, and the failure-label protocol

- [ ] Record MOT17/MOT20 source/version, terms, checksums, annotation format, permitted storage/distribution, and the exact sequence manifest.
- [ ] Group MOT17 detector variants by source scene: all variants of a source video stay in the same fold. The 21 MOT17 folders correspond to seven source scenes with three detector variants; do not count the detector variants as independent scenes.
- [ ] Freeze MOT20 as held-out dense-crowd transfer evaluation. No model fitting, feature selection, calibration, corruption selection, or policy-threshold tuning on MOT20.
- [ ] Before label generation, freeze `alpha`, visibility/occlusion rules, `K`, `M`, sequence-boundary censoring, detector/tracker versions, tracker initialization, matching implementation, and branch semantics.
- [ ] Implement paired offline rollouts from the same tracker state at frame `t`: one branch skips detection at `t+1`, the other detects at `t+1`; both then use detector-every-frame updates for the next `K-1` frames. All other settings are matched.
- [ ] Use the research report's frozen target: `Y = F_skip * (1 - F_detect)`. A positive label means the skip branch has the defined identity failure within the horizon and the matched detect branch does not. Keep ineligible, ambiguous, and boundary-censored anchors out of the safe-negative class.
- [ ] Write label pseudocode and manually audit examples for ordinary motion, crossings, occlusion, entries/exits, re-entry, and sequence boundaries. Report exclusions and event prevalence by source scene.
- [ ] Create a local-footage authorization/consent decision before capturing people. Keep raw video transient by default, make annotated export opt-in, use temporary IDs only, and apply the documented retention/deletion policy.
- [ ] Review local-network access, credential redaction, log allowlists, and the prohibition on face crops, appearance embeddings, and persistent identity.

**Exit evidence (G2):** Data/footage permissions and handling record, sequence/split manifest, frozen label protocol, pseudocode, and audited examples. These are required before training or selecting a policy threshold.

## Phase 3 - G4: Close the full-MVP pre-code gate

- [ ] Close G1 and G2, or record each explicit deferral, owner, risk, and supervisor acceptance in the decision log.
- [ ] Complete requirement-to-component-to-verification traceability and agree which acceptance items are MVP blockers.
- [ ] Confirm SDD contracts: frame packet and timestamps, detector output, tracker update versus skip, scheduler action/reason, run record, queue limits, restart, and error behavior.
- [ ] Verify detector/checkpoint provenance and terms, TensorRT compatibility, ByteTrack source/license, and dependency pinning plan.
- [ ] Approve the measurement boundary, metric definitions, repetition plan, and method for deriving quality margins, deadline, deadline-miss bound, memory/temperature limits, and minimum useful energy reduction after the every-frame pilot.
- [ ] Reserve time for integration, target-device measurements, failure recovery, analysis, and demo preparation.
- [ ] Obtain required student/supervisor G4 sign-off and save the dated checklist.

**Gate:** Start full MVP implementation only after G4 is signed. The G1 inventory/input feasibility spike is the limited exception. Do not implement the adaptive scheduler or claim a useful compute policy until the every-frame baseline closes G3.

## Phase 4 - Build the deterministic, every-frame runtime

- [ ] Implement source adapters for MOT files and the verified phone stream. Preserve source index, source timestamp, monotonic arrival time, and known input drops as separate fields.
- [ ] Add config validation, run ID/manifest, clean start/stop/restart, bounded queues, redacted logs, and visible source/decode failure handling.
- [ ] Freeze one detector checkpoint, preprocessing/coordinate mapping, batch size, and reference runtime. Convert to TensorRT FP16 only if supported; compare outputs with the reference path using declared tolerances.
- [ ] Pin ByteTrack source/version and parameters. Test initialization, `update([])` when detection ran but found no person, and `skip()` when detection did not run. Verify lifecycle and time advancement over detector skips and source timestamp gaps.
- [ ] Add frame-stage timing and separately track elapsed seconds since detector, source frames since detector, consecutive policy skips, and source-input drops.
- [ ] Replay a short file and phone stream in detector-every-frame mode; verify output records and graceful failure/stop before adding adaptation.

**Exit evidence:** Repeatable local file replay and phone ingest; fixed detector/tracker outputs; run manifest and logs; clean failure/stop behavior. No adaptive policy yet.

## Phase 5 - Measure the every-frame baseline and close G3

- [ ] Run the frozen detector + ByteTrack on every input frame on the physical Orin Nano: repeatable MOT files first, then the phone feed for integration.
- [ ] Add the minimum local dashboard only after the core path works: source/run health, overlay, temporary IDs, active-track count as an occupancy proxy, action/status, latency, and errors. Bind only to the trusted LAN.
- [ ] Compute HOTA, DetA/AssA, IDF1, MOTA, ID switches, fragmentation, and detector recall/AP diagnostics using a pinned TrackEval/evaluation path.
- [ ] Measure end-to-end latency from frame arrival/read through emitted tracks and required product status/log output. Record component times, queue depth, drops, p50/p95, sustained throughput, and DMR.
- [ ] Measure complete-pipeline energy at a declared Jetson input boundary with an external meter as the primary measure when available. Include decode, preprocessing, detector, tracker, predictor, policy, guard/context if enabled, and product-required UI/logging. Divide by every input frame, including frames where detection is skipped.
- [ ] Keep onboard rail telemetry separate from whole-device meter readings. If no suitable external meter exists, mark whole-pipeline Joules/input-frame unavailable; do not present onboard telemetry as an exact substitute.
- [ ] Record peak RAM, temperature, clocks, power mode, fan/cooling, ambient condition, warm-up, run duration, and throttling during sustained operation. A brief `tegrastats` sample is not a sustained thermal result.
- [ ] Derive and freeze `D`, HOTA/IDF1 non-inferiority margins, maximum DMR, RAM/thermal limits, minimum useful energy reduction, sustained-run duration, and guard discovery/false-trigger criteria. Justify each using the product cadence, baseline variation, and user need; do not choose limits after seeing final test results.

**Exit evidence (G3):** Frozen every-frame baseline configs/results, declared measurement boundary, repeat protocol, and approved numeric acceptance limits with rationale. Risk-policy optimization depends on this gate.

## Phase 6 - Build and audit the paired-rollout learning dataset

- [ ] Run the fixed detector/tracker over selected MOT17 source sequences; preserve outputs, annotations, timestamps, configs, hashes, and software versions.
- [ ] Generate the paired skip/detect rollouts from identical anchor state according to the frozen `K`, `M`, `alpha`, visibility/matching rules, and future schedule.
- [ ] Implement the avoidable-failure label. Censor ambiguous or ineligible anchors instead of silently labelling them as safe.
- [ ] Include varied prior detector-skip histories in training rollouts to reduce mismatch between training and adaptive runtime states.
- [ ] Compare generated labels to hand-reviewed examples. Report positive prevalence, exclusions, per-scene counts, duplicate-frame checks, and scene-leakage checks.
- [ ] Freeze sequence-grouped training, calibration, policy-validation, and final evaluation roles. With only seven MOT17 source scenes, use grouped/cross-fitted roles if a fixed split is unstable; never use random frame or tracklet splits.
- [ ] Fit normalization and any class weighting/mining on training data only. Preserve natural event prevalence for calibration.
- [ ] Freeze detector-gap bins before held-out scoring. Keep source-frame subsampling, network/source drops, and policy detector skips as separate conditions.

**Exit evidence:** Reproducible data/label manifest, audited rollout implementation, sequence-role manifest, and proof that test labels were not used for model or policy selection.

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

# RACE-MOT Decision Log and Pre-Code Readiness Gate

**Status:** Updated 6 October 2026 — product direction and main deployment choices agreed; limited feasibility and measurement gates remain.  
**Owner:** Student, with supervisor confirmation where indicated.

## 1. Accepted working decisions

| ID | Decision | Basis | Status |
|---|---|---|---|
| D-01 | Product-first outcome is a functional local prototype; conference paper is optional. | Faculty guidance reported by student. | Accepted |
| D-02 | MVP is one fixed-camera pedestrian stream, one detector, ByteTrack, one physical edge device, and MOT17/MOT20 evaluation. | Current consolidated technical proposal. | Working scope |
| D-03 | Compute policy has two actions: run detector or skip detector and propagate tracks. | Keeps the risk-policy experiment interpretable; multi-resolution is deferred. | Working scope |
| D-04 | Keep scene-quality/activity descriptors as an ablation. Add a separate low-cost scene-discovery guard that may upgrade SKIP to full-frame DETECT for untracked activity; measure discovery benefit and cost, and do not claim novelty. | Per-track risk cannot score an object before a track exists. This is a product safety inference, not a result established by prior work. | Accepted for plan; threshold/effect require validation |
| D-05 | Quantization is a late deployment ablation, not the main contribution. | Existing detector-compression prior art. | Accepted for plan |
| D-06 | No performance target is stated before a baseline on selected hardware. | Device/runtime-dependent performance. | Accepted |
| D-07 | Local processing, anonymous temporary tracks, no face recognition, no video persistence by default, and opt-in export are project defaults. | Product privacy and scope; institutional rules still govern whether any local capture is permitted. | Accepted default; check authorization before capture |
| D-08 | The available edge computer is a Jetson Orin Nano. Record exact module RAM, carrier-board revision, cooling, power supply, and installed software before the feasibility run. | Student confirms device availability; Orin Nano modules/dev kits vary. | Device family confirmed; inventory pending |
| D-09 | Use the mobile phone as the single camera source, mounted stationary for the demo, streaming H.264/RTSP over the same private local network to the Jetson. Use recorded MOT sequences for repeatable benchmark runs. | Student confirms mobile camera feed; RTSP provides a direct one-camera network input and recorded sequences make policy comparisons repeatable. | Agreed default; verify phone app/stream at G1 |
| D-10 | Treat the product user as the student/faculty demonstrator and the use case as anonymous pedestrian tracking with a per-frame active-track count in an authorized controlled scene. This is an occupancy proxy, not a validated people-counting system. No person identification or consequential decisions. | Existing product scope and student confirmation that remaining requirements are agreed. | Agreed working definition |
| D-11 | Candidate detector is YOLOX-Tiny at its published 416-pixel input as a person-only detector, exported to TensorRT FP16 if the selected software image supports it. Freeze exact checkpoint, preprocessing, and license after the feasibility spike. | The [official YOLOX repository](https://github.com/Megvii-BaseDetection/YOLOX) lists Tiny/Nano models and TensorRT deployment support under Apache-2.0 source-code licensing; that does not automatically settle the separate checkpoint/weights terms. | Provisional technical choice |
| D-12 | Preserve the installed NVIDIA JetPack image if it supports the board and chosen TensorRT deployment. Record its exact version and do not reflash as a first step. If a clean reimage is necessary, target the latest official JetPack release that supports Orin Nano only after checking YOLOX/TensorRT compatibility. | Avoids an unnecessary system change; NVIDIA's current JetPack releases and supported libraries change over time. | Accepted |
| D-13 | Do not save phone footage by default. Keep only authorized run summaries and decision logs in the project workspace until assessment completion, then delete within 90 days unless institutional rules require earlier deletion. Annotated video export remains opt-in and must be deleted after the demo. | Data minimization default for a classroom prototype. | Agreed default; stricter institutional rule takes precedence |
| D-14 | Primary project outcome is the working prototype and its reproducible evidence package. A conference paper is optional and considered only after the prototype succeeds and closest-prior-art comparisons support a defensible claim. | Faculty guidance reported by student. | Accepted |
| D-15 | Provide a minimal local web dashboard served by the Jetson and opened from a browser on the trusted local network; no cloud account or public access. Dashboard shows stream/run status, overlay, active-track count, current action, and errors. | Keeps the demo usable while the Orin Nano runs headless and requires no dedicated app. | Accepted product choice; choose lightweight framework during implementation |
| D-16 | Begin development with a bounded G1 milestone: read-only device inventory and no-save camera/video input probe. | Verifies the actual Jetson software and phone stream before selecting/deploying the detector engine; avoids reporting unmeasured model performance. | Implemented as initial scaffold in `edge/race_mot/`; device execution pending |
| D-17 | Define the project's single novelty claim as calibrated, per-track prediction of avoidable identity failure conditioned on an explicit detector-skip action, used to schedule the next detector call. Treat fusion as an ablation and Orin deployment/energy as product evidence. | Paired skip/detect labels specify the target; TCN/gating/context descriptors are established ingredients. No fixed latency/parameter guarantee, calibration guarantee before measurement, zero-leakage claim, thermal-control novelty, or Re-ID cost claim is accepted. The comparative matrix in the research report distinguishes ALBIREO, HSFSO, RT-MOT, EMO, Split and Connect, GLoMOT, and LUKF-Track. | Accepted as the proposal's novelty claim; practical gain and calibration are evaluated experimentally |
| D-18 | Make elapsed seconds, source-frame gap, and consecutive detector skips separate risk-model inputs/log fields; keep ByteTrack's motion filter unchanged. Add an ALBIREO-like per-object uncertainty scheduler as a mandatory matched comparator. | [GLoMOT](https://doi.org/10.1609/aaai.v40i6.42500) and [LUKF-Track](https://doi.org/10.3390/e28010103) address low-frame-rate/nonlinear motion; [ALBIREO](https://arxiv.org/abs/2609.29648) schedules object-wise detector calls. This defines comparator boundaries and input requirements for the claimed target. | Adopt for model schema and evaluation |
| D-19 | Keep a scene-discovery override that only upgrades a scheduled SKIP to DETECT as a product-safety branch. Add risk-only versus risk-plus-guard ablation with new-track delay/recall, false-trigger rate, and full cost. | A per-track model has no active track for new entrants. This safeguard aims to protect discovery and must be removed if held-out benefit does not justify its cost. | Adopt for design; measure before product acceptance |

## 2. Blocking open decisions

| ID | Question | Needed before | Owner/status |
|---|---|---|---|
| O-01 | Primary user/workflow | G0 | **Closed:** student/faculty demo; inspect anonymous tracks and per-frame active-track count for one authorized pedestrian scene. |
| O-02 | Product scene | G0 | **Closed:** one stationary phone camera for a controlled pedestrian-tracking demo; display active-track count as an occupancy proxy; no consequential use. |
| O-03 | Device-specific feasibility | G1 | **Partly closed:** Jetson Orin Nano confirmed. Inventory exact RAM/SKU, carrier, cooling, installed JetPack/TensorRT, and power/temperature telemetry. |
| O-04 | Product input | G0/G1 | **Closed as default:** live H.264/RTSP from the phone over private local Wi-Fi; retain video-file replay for reproducible evaluation. Verify phone streaming app and reconnect behavior at G1. |
| O-05 | Detector and runtime | G1 | **Candidate selected:** YOLOX-Tiny + TensorRT FP16. G1 must confirm export/inference, tracker integration, checkpoint provenance/license, and acceptable person recall before freezing. |
| O-06 | Data terms and footage authorization | G2 | **Open before collecting phone footage or distributing artifacts:** record MOT17/MOT20 source/terms; use only authorized/staged phone footage; confirm institution's review/consent expectations. |
| O-07 | Deadline and tracking-quality margins | G3, after baseline | **Intentionally deferred:** set from phone-stream cadence and detector-every-frame baseline; use held-out validation data before final evaluation. |
| O-08 | Minimum useful energy reduction | G3, after baseline | **Intentionally deferred:** include predictor, decode, tracker, and logging overhead; set before adaptive-policy evaluation. |
| O-09 | Retention and deletion | G2 | **Closed as project default:** no video persistence by default; summaries/logs until assessment completion, delete within 90 days, or earlier if institutional policy requires. |

## 3. Decision record template

| ID | Date | Question/decision | Options considered | Evidence/impact | Decision and rationale | Documents updated |
|---|---|---|---|---|---|---|
| D-___ | YYYY-MM-DD |  |  |  |  |  |

## 4. Ordered next steps before adaptive-policy coding

1. **Inventory the board.** Record exact model/RAM, carrier, cooling, storage, JetPack/Jetson Linux, CUDA/TensorRT, power mode, and available telemetry using the checklist in [the hardware plan](05_hardware_and_deployment_plan.md). The initial CLI is in [the implementation workspace](../race_mot/README.md). Preserve the current image for the first feasibility check.
2. **Verify the live input.** Place the phone on a stable mount; connect phone and Jetson to the same private Wi-Fi/hotspot; configure H.264/RTSP at 1280×720 and 15 fps as the initial profile if supported; verify timestamps, reconnect, and that no internet/cloud relay is needed.
3. **Run the bounded pipeline feasibility spike.** Test YOLOX-Tiny person detections, ByteTrack, and batch-1 TensorRT FP16 on the Jetson. Inspect model/checkpoint provenance and terms. Use a small permitted clip and the phone stream. If TensorRT export fails, first diagnose installed JetPack/operator compatibility; do not silently switch detector families.
4. **Build/verify the minimum product shell.** Serve the dashboard on the Jetson, accessible only on the trusted local network; show live overlay, active-track count, run state, action, latency, and errors. Keep raw video persistence off.
5. **Freeze and run detector-every-frame baselines.** Use the selected board and recorded MOT evaluation input. Report HOTA/IDF1 and system/energy/thermal measures; derive quality margins and deadline/energy targets from the application cadence and repeatability. Use the phone feed as the demo, not the source of benchmark claims.
6. **Defend and operationalize the novelty claim.** The proposal claims action-conditioned, calibrated per-track identity-failure prediction for next-frame detector scheduling. Inspect ALBIREO and its code, HSFSO full text/code, GLoMOT, LUKF-Track, and other close work; freeze matched baselines including ALBIREO-like uncertainty scheduling. Implement and evaluate the claim against these methods; do not present a measured gain until results exist.

## 5. Pre-code gate checklist (G4)

Full MVP coding may begin only when all blocking items are resolved or explicitly marked deferred by the student and supervisor.

- [x] Product brief reviewed; user and deployment scenario accepted as the controlled single-camera pedestrian-flow demo.
- [x] MVP workflow and non-goals accepted; one phone stream, one detector, ByteTrack, and detect/skip action only, with a discovery guard that can upgrade SKIP to DETECT.
- [ ] Physical device is available and device feasibility plan is complete; record exact RAM/SKU, cooling, image, and telemetry.
- [ ] Candidate detector/runtime is verified on the device; freeze checkpoint, tracker version, runtime, and license provenance.
- [ ] Dataset terms, sequence-level split plan, labels, and leakage controls reviewed.
- [ ] Product requirements have testable acceptance evidence.
- [ ] Architecture and module boundaries reviewed.
- [x] Local data flow, opt-in video export, default no-video-retention, and 90-day post-assessment log cleanup policy recorded (subject to stricter institutional rules).
- [ ] Detector-every-frame feasibility baseline has an agreed measurement plan; no numeric targets invented.
- [ ] Metrics, quality margins, latency deadline, energy boundary, run repetitions, and analysis plan are specified or scheduled as an explicit pilot gate.
- [x] Focused novelty claim and closest cited comparisons are stated in the proposal; matched reproduction/evaluation remains a required project milestone.
- [ ] Project schedule reserves time for integration, error handling, sustained-device runs, and demo preparation.
- [ ] The technical report and product documents agree on action space, datasets, device count, and scope.

**Gate decision:** ☒ Not ready for full MVP implementation  
**Remaining gates:** Orin inventory/software capture; RTSP phone-to-Jetson feasibility; detector/checkpoint/license smoke check; dataset/footage terms and institutional review; detector-every-frame baseline before numeric targets and adaptive policy. A minimal G1 feasibility spike is allowed once G0 is accepted.  
**Student/date:** ____________________  **Supervisor/date:** ____________________

## 6. Scope-change rule

Any proposal to add learned motion compensation/Mamba, ROI or patch inference, pseudo-depth association, thermal-state control, Re-ID, optical-flow propagation, variable resolution, multiple cameras, custom detector training, a new local dataset, or cloud services must include: user value, prior-art review, data/privacy impact, architecture change, acceptance tests, schedule cost, and a written decision. Do not add it informally during implementation.

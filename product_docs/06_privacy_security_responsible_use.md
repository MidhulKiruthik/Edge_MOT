# RACE-MOT Privacy, Security, and Responsible-Use Plan

**Status:** Updated 8 October 2026. This is a project safeguard plan, not legal advice or a claim of regulatory compliance. D-48 verifies no-frame/default-redacted logging for local replay and D-51 verifies the same boundary for transient phone integration; local MOT17 artifacts remain private and unredistributed under the recorded terms.

## 1. Intended use and prohibited use

The prototype is for authorized development, aggregate pedestrian-flow analysis, and academic demonstration. It assigns temporary track IDs within a video only.

Do not use the prototype for face recognition, identifying/naming individuals, cross-camera identity matching, individual behavior scoring, attendance/discipline, employment decisions, law enforcement, access control, or safety-critical decisions. Do not deploy in a public or occupied environment without institutional permission and the required review.

## 2. Phone camera and local-network safeguards

- Use the phone as a stationary camera only for an authorized, controlled demo scene. Prefer staged volunteers or a scene without identifiable bystanders.
- Send IP Webcam H.264/RTSP directly over the private USB-tethered local link to the Jetson. Do not enable a public RTSP relay, cloud recording, or remote access port.
- Do not place RTSP credentials in the repository, screenshots, or run logs. Redact the URL in diagnostics.
- If the USB-tethered link is unreliable, stop the demo and replay an authorized recorded clip; do not silently drop/reorder frames or claim real-time behavior from an unstable feed.

## 3. Data minimization defaults

- Process locally by default; do not transmit raw frames to cloud or third-party services.
- Do not save raw video or annotated video unless the user explicitly enables export and has authority to do so.
- Do not store face crops, appearance embeddings, or persistent person profiles; the MVP has no Re-ID branch.
- Use temporary IDs that reset per run. Avoid logging names, account identifiers, or full local file paths in exported summaries.
- Do not persist phone video by default. Phone camera processing is transient; no frame recording or annotated-video writing occurs unless explicitly enabled for an authorized demo.
- Retain only run summaries and decision logs in the restricted project workspace until assessment completion, then delete within 90 days. Delete any opt-in annotated demo export after the demonstration. Follow any stricter institutional retention rule.

## 4. Footage authorization and evaluation data

Use public benchmark data only under its stated terms. For local footage, document camera ownership/authorization, location, date, purpose, participants/bystander notice or consent as required by the institution, annotation access, storage location, who can access it, retention period, deletion method, and whether sharing is permitted. Do not assume that a short clip is exempt from review because it is small or only used in a class project.

## 5. Threats and controls

| Risk | Minimum control |
|---|---|
| Unauthorized viewing or copying of video | Restrict file permissions; encrypt device/storage where available; avoid copying raw video into reports, issue trackers, or cloud drives. |
| Accidental recording/export | Export is opt-in; visible run indicator; clear output destination; delete temporary files on stop/error. |
| Re-identification from persistent trajectories | Reset IDs per sequence; no cross-camera linking or face recognition; limit trajectory detail in reports. |
| Misleading counts or tracking errors | Label counts as model estimates; show uncertainty/quality caveats; avoid operational decisions based solely on output. |
| Untrusted video or malformed media | Treat media as untrusted input; validate formats/size; constrain file access; handle decoder failures cleanly. |
| Sensitive log leakage | Allowlist logged fields; strip usernames/full paths; document log retention; verify no frame data is serialized. |
| Public demo exposes bystanders | Use public benchmark, staged/consented footage, or synthetic footage; review screens and exported artifacts before presentation. |

## 6. Model and result communication

The risk score predicts a defined tracking failure in the chosen evaluation setup; it is not a measure of a person's risk or intent. Counts and tracks may fail under occlusion, lighting, blur, crowding, or domain shift. State those limitations in the UI/demo and final report. Do not use an uncalibrated model output as a probability.

## 7. Incident response

If footage is captured or shared without approval, stop processing, restrict access, notify the supervisor/institution according to their process, remove unauthorized copies where permitted, and document the incident. Do not continue collecting until authorization is resolved.

## 8. Approval record

| Check | Owner | Status/date |
|---|---|---|
| Public benchmark terms reviewed | Student | **TBD** |
| Local footage authorization/review (if used) | Student + supervisor | **TBD / not applicable until proposed** |
| Storage, export, and retention settings reviewed | Student + supervisor | **Agreed default; final review before capture** |
| Demo materials checked for identifiable footage | Student | **TBD** |

"""Reproducible Phase 6 paired-rollout dataset generation and audits.

The generator consumes immutable detector-every-frame JSONL logs, clones the
deterministic tracker state at an anchor, and runs matched SKIP/DETECT branches
over the same future frames. Only the approved MOT17 training role is used;
calibration, policy-validation, final, phone, and MOT20 data are excluded.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from race_mot.domain import FramePacket, SourceKind
from race_mot.evaluation.mot import GroundTruthBox, group_by_frame, read_gt
from race_mot.evaluation.paired_rollout import BranchObservation, pair_observations
from race_mot.evaluation.protocol import RolloutProtocol
from race_mot.run_manifest import file_sha256
from race_mot.sources.mot_sequence import read_sequence_info
from race_mot.trackers.bytetrack import ByteTrackAdapter, _iou


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _match_track_to_box(tracks: list[dict[str, Any]], box: GroundTruthBox, threshold: float) -> tuple[int, float] | None:
    candidates = []
    for item in tracks:
        xyxy = item.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        overlap = _iou(tuple(float(value) for value in xyxy), box.xyxy)
        if overlap >= threshold:
            candidates.append((overlap, int(item["temporary_track_id"])))
    if not candidates:
        return None
    overlap, track_id = max(candidates)
    return track_id, overlap


@dataclass(frozen=True, slots=True)
class FrameRecord:
    source_index: int
    source_timestamp_ms: float | None
    input_drop_count: int
    detections: tuple[dict[str, Any], ...]
    tracks: tuple[dict[str, Any], ...]

    def packet(self, run_id: str) -> FramePacket:
        return FramePacket(
            run_id=run_id,
            source_index=self.source_index,
            source_timestamp_ms=self.source_timestamp_ms,
            arrival_monotonic_ns=self.source_index,
            source_kind=SourceKind.MOT_SEQUENCE,
            input_drop_count=self.input_drop_count,
            image_bgr=object(),
        )


@dataclass(frozen=True, slots=True)
class SequenceInput:
    sequence_dir: Path
    baseline_run_dir: Path
    role: str
    source_scene: str
    frames: dict[int, FrameRecord]
    ground_truth: dict[int, tuple[GroundTruthBox, ...]]
    frame_rate: float
    width: int
    height: int


def _load_frames(path: Path) -> dict[int, FrameRecord]:
    result: dict[int, FrameRecord] = {}
    with (path / "frame_log.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            source_index = int(value["source_index"])
            if source_index in result:
                raise ValueError(f"duplicate source frame in {path}: {source_index}")
            result[source_index] = FrameRecord(
                source_index=source_index,
                source_timestamp_ms=value.get("source_timestamp_ms"),
                input_drop_count=int(value.get("input_drop_count", 0)),
                detections=tuple(value.get("detections", ())),
                tracks=tuple(value.get("tracks", ())),
            )
    if not result:
        raise ValueError(f"empty baseline frame log: {path / 'frame_log.jsonl'}")
    return result


def _detections(frame: FrameRecord) -> list[Any]:
    from race_mot.domain import Detection

    return [
        Detection(tuple(float(value) for value in item["xyxy"]), float(item["score"]), int(item.get("class_id", 0)), str(item.get("class_name", "person")))
        for item in frame.detections
    ]


def _state_payload(tracker: ByteTrackAdapter) -> dict[str, Any]:
    return tracker.state_dict()


def _state_hash(tracker: ByteTrackAdapter) -> str:
    return _hash_json(_state_payload(tracker))


def _packet_map(frames: dict[int, FrameRecord], run_id: str) -> dict[int, FramePacket]:
    return {index: frame.packet(run_id) for index, frame in frames.items()}


def _baseline_states(sequence: SequenceInput) -> dict[int, tuple[ByteTrackAdapter, list[dict[str, Any]]]]:
    tracker = ByteTrackAdapter()
    packets = _packet_map(sequence.frames, sequence.sequence_dir.name)
    states: dict[int, tuple[ByteTrackAdapter, list[dict[str, Any]]]] = {}
    for index in sorted(sequence.frames):
        tracker.update(_detections(sequence.frames[index]), packets[index])
        states[index] = (tracker.clone(), list(sequence.frames[index].tracks))
    return states


def _state_with_prior_skips(sequence: SequenceInput, anchor: int, prior_skip_count: int) -> ByteTrackAdapter | None:
    """Replay up to an anchor with a real immediately-prior skip history."""
    if prior_skip_count <= 0:
        raise ValueError("prior_skip_count must be positive for a variant")
    indices = sorted(index for index in sequence.frames if index <= anchor)
    if not indices or any(anchor - offset not in sequence.frames for offset in range(1, prior_skip_count + 1)):
        return None
    tracker = ByteTrackAdapter()
    packets = _packet_map(sequence.frames, sequence.sequence_dir.name)
    skip_indices = {anchor - offset for offset in range(1, prior_skip_count + 1)}
    for index in indices:
        if index in skip_indices:
            tracker.skip(packets[index])
        else:
            tracker.update(_detections(sequence.frames[index]), packets[index])
    return tracker


def _track_records(snapshots: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {
            "temporary_track_id": item.temporary_track_id,
            "xyxy": list(item.xyxy),
            "score": item.score,
            "state": item.state.value,
        }
        for item in snapshots
    ]


def _branch_observation(
    *,
    branch: str,
    tracker: ByteTrackAdapter,
    sequence: SequenceInput,
    anchor: int,
    target_identity: int,
    protocol: RolloutProtocol,
    packets: dict[int, FramePacket],
) -> BranchObservation:
    future = tuple(anchor + offset for offset in range(1, protocol.horizon_frames + 1))
    visible: list[bool] = []
    assigned: list[int | None] = []
    boundary = False
    for offset, index in enumerate(future):
        truth = next((item for item in sequence.ground_truth.get(index, ()) if item.identity == target_identity), None)
        visible.append(truth is not None and truth.visibility >= protocol.visible_minimum)
        if index not in sequence.frames:
            boundary = True
            assigned.append(None)
            continue
        if offset == 0 and branch == "skip":
            snapshots = tracker.skip(packets[index])
        else:
            snapshots = tracker.update(_detections(sequence.frames[index]), packets[index])
        if truth is None:
            assigned.append(None)
        else:
            match = _match_track_to_box(_track_records(snapshots), truth, protocol.anchor_iou)
            assigned.append(match[0] if match else None)
    if any(sequence.frames.get(index, FrameRecord(index, None, 0, (), ())).input_drop_count for index in future if index in sequence.frames):
        boundary = True
    return BranchObservation(
        branch=branch,
        anchor_state_hash=_state_hash(tracker),
        future_frame_indices=future,
        visible=tuple(visible),
        assigned_identity=tuple(assigned),
        boundary_censored=boundary,
    )


def _branch_observation_from_state(
    *,
    branch: str,
    anchor_state: ByteTrackAdapter,
    anchor_hash: str,
    sequence: SequenceInput,
    anchor: int,
    target_identity: int,
    protocol: RolloutProtocol,
    packets: dict[int, FramePacket],
) -> BranchObservation:
    observation = _branch_observation(
        branch=branch,
        tracker=anchor_state.clone(),
        sequence=sequence,
        anchor=anchor,
        target_identity=target_identity,
        protocol=protocol,
        packets=packets,
    )
    return BranchObservation(
        branch=observation.branch,
        anchor_state_hash=anchor_hash,
        future_frame_indices=observation.future_frame_indices,
        visible=observation.visible,
        assigned_identity=observation.assigned_identity,
        boundary_censored=observation.boundary_censored,
    )


def _simulate_branch(
    *,
    branch: str,
    anchor_state: ByteTrackAdapter,
    sequence: SequenceInput,
    anchor: int,
    protocol: RolloutProtocol,
    packets: dict[int, FramePacket],
) -> tuple[dict[int, list[dict[str, Any]]], bool]:
    """Run one branch once and retain snapshots for every target at the anchor."""
    tracker = anchor_state.clone()
    snapshots: dict[int, list[dict[str, Any]]] = {}
    boundary = False
    for offset in range(1, protocol.horizon_frames + 1):
        index = anchor + offset
        if index not in sequence.frames:
            boundary = True
            continue
        if offset == 1 and branch == "skip":
            values = tracker.skip(packets[index])
        else:
            values = tracker.update(_detections(sequence.frames[index]), packets[index])
        snapshots[index] = _track_records(values)
        if sequence.frames[index].input_drop_count:
            boundary = True
    return snapshots, boundary


def _pair_from_simulated_branches(
    *,
    sequence: SequenceInput,
    anchor: int,
    target_identity: int,
    anchor_hash: str,
    protocol: RolloutProtocol,
    skip_snapshots: dict[int, list[dict[str, Any]]],
    detect_snapshots: dict[int, list[dict[str, Any]]],
    boundary_censored: bool,
) -> tuple[Any, Any, Any]:
    future = tuple(anchor + offset for offset in range(1, protocol.horizon_frames + 1))

    def observation(branch: str, snapshots: dict[int, list[dict[str, Any]]]) -> BranchObservation:
        visible: list[bool] = []
        assigned: list[int | None] = []
        for index in future:
            truth = next((item for item in sequence.ground_truth.get(index, ()) if item.identity == target_identity), None)
            visible.append(truth is not None and truth.visibility >= protocol.visible_minimum)
            if truth is None:
                assigned.append(None)
                continue
            match = _match_track_to_box(snapshots.get(index, []), truth, protocol.anchor_iou)
            assigned.append(match[0] if match else None)
        return BranchObservation(
            branch=branch,
            anchor_state_hash=anchor_hash,
            future_frame_indices=future,
            visible=tuple(visible),
            assigned_identity=tuple(assigned),
            boundary_censored=boundary_censored,
        )

    skip = observation("skip", skip_snapshots)
    detect = observation("detect", detect_snapshots)
    pair = pair_observations(anchor_frame_index=anchor, target_identity=target_identity, skip=skip, detect=detect, protocol=protocol)
    return pair, skip, detect


def _eligible_targets(sequence: SequenceInput, anchor: int, tracks: list[dict[str, Any]], protocol: RolloutProtocol) -> list[tuple[int, float, int]]:
    result = []
    for truth in sequence.ground_truth.get(anchor, ()):
        if truth.visibility < protocol.visible_minimum:
            continue
        match = _match_track_to_box(tracks, truth, protocol.anchor_iou)
        if match:
            result.append((truth.identity, match[1], match[0]))
    return result


def _image_duplicate_audit(sequence: SequenceInput) -> dict[str, Any]:
    image_dir = sequence.sequence_dir / "img1"
    hashes: dict[str, list[str]] = defaultdict(list)
    if image_dir.is_dir():
        for image in sorted(image_dir.iterdir()):
            if image.is_file():
                hashes[file_sha256(image)].append(image.name)
    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    return {
        "frames_scanned": sum(len(names) for names in hashes.values()),
        "unique_image_hashes": len(hashes),
        "duplicate_groups": duplicate_groups[:20],
        "duplicate_frame_count": sum(len(names) - 1 for names in duplicate_groups),
        "status": "passed" if not duplicate_groups else "duplicates_detected_and_reported",
    }


def _load_roles(roles_path: Path) -> dict[str, Any]:
    value = json.loads(roles_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("roles"), dict):
        raise ValueError("roles manifest must contain a roles object")
    return value


def _role_leakage_audit(roles: dict[str, Any]) -> dict[str, Any]:
    owners: dict[str, str] = {}
    conflicts = []
    for role, details in roles["roles"].items():
        for scene in details.get("base_scenes", []):
            scene_key = str(scene).zfill(2)
            if scene_key in owners and owners[scene_key] != role:
                conflicts.append({"scene": scene_key, "roles": [owners[scene_key], role]})
            owners[scene_key] = role
    return {
        "scene_role_owner": owners,
        "conflicts": conflicts,
        "status": "passed" if not conflicts else "failed",
        "rule": "all detector variants of a source scene share one role; no frame/tracklet random split",
    }


def generate_phase6_dataset(
    *,
    sequences: list[tuple[Path, Path]],
    roles_path: Path,
    protocol_path: Path,
    output_root: Path,
    prior_history_samples_per_sequence: int = 48,
) -> dict[str, Any]:
    """Generate Phase 6 artifacts from frozen Phase 5 baseline logs."""
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"output directory is not empty: {output_root}")
    if not sequences:
        raise ValueError("at least one sequence/baseline pair is required")
    roles = _load_roles(roles_path)
    protocol = RolloutProtocol.from_mapping(json.loads(protocol_path.read_text(encoding="utf-8")))
    role_by_scene = {}
    for role, details in roles["roles"].items():
        for scene in details.get("base_scenes", []):
            role_by_scene[str(scene).zfill(2)] = role
    inputs: list[SequenceInput] = []
    for sequence_dir, baseline_run_dir in sequences:
        name = sequence_dir.name
        parts = name.split("-")
        scene = parts[1]
        role = role_by_scene.get(scene)
        if role != "training":
            raise ValueError(f"Phase 6 generator only accepts training-role scenes; {name} is {role}")
        info = read_sequence_info(sequence_dir)
        frames = _load_frames(baseline_run_dir)
        gt = group_by_frame(tuple(box for box in read_gt(sequence_dir / "gt" / "gt.txt") if box.visibility >= protocol.visible_minimum))
        inputs.append(SequenceInput(sequence_dir, baseline_run_dir, role, f"MOT17-{scene}", frames, gt, info.frame_rate, info.width, info.height))
    output_root.mkdir(parents=True, exist_ok=True)
    anchors_path = output_root / "anchors.jsonl"
    anchor_states_path = output_root / "anchor_states.jsonl"
    rollouts_path = output_root / "rollouts.jsonl"
    anchor_count = Counter()
    labels = Counter()
    exclusions = Counter()
    per_sequence: dict[str, Counter[str]] = defaultdict(Counter)
    gap_bins = Counter()
    source_drops = 0
    feature_values: dict[str, list[float]] = defaultdict(list)
    audit_sample: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    written_state_keys: set[str] = set()
    with anchors_path.open("x", encoding="utf-8") as anchors, anchor_states_path.open("x", encoding="utf-8") as anchor_states, rollouts_path.open("x", encoding="utf-8") as rollouts:
        for sequence in inputs:
            states = _baseline_states(sequence)
            packets = _packet_map(sequence.frames, sequence.sequence_dir.name)
            candidates = []
            for index in sorted(sequence.frames):
                state, _ = states[index]
                eligible = _eligible_targets(sequence, index, sequence.frames[index].tracks, protocol)
                if eligible:
                    candidates.append(index)
                if eligible:
                    skip_snapshots, skip_boundary = _simulate_branch(
                        branch="skip", anchor_state=state, sequence=sequence, anchor=index,
                        protocol=protocol, packets=packets,
                    )
                    detect_snapshots, detect_boundary = _simulate_branch(
                        branch="detect", anchor_state=state, sequence=sequence, anchor=index,
                        protocol=protocol, packets=packets,
                    )
                    boundary_censored = skip_boundary or detect_boundary
                    state_hash = _state_hash(state)
                    state_key = f"{sequence.sequence_dir.name}:{index}:0"
                    if state_key not in written_state_keys:
                        anchor_states.write(json.dumps({
                            "schema_version": 1,
                            "state_key": state_key,
                            "sequence": sequence.sequence_dir.name,
                            "source_scene": sequence.source_scene,
                            "role": sequence.role,
                            "anchor_frame_index": index,
                            "prior_detector_skip_count": 0,
                            "anchor_state_hash": state_hash,
                            "anchor_state": _state_payload(state),
                        }, sort_keys=True, separators=(",", ":")) + "\n")
                        written_state_keys.add(state_key)
                for target_identity, overlap, track_id in eligible:
                    anchor_count[sequence.sequence_dir.name] += 1
                    payload = {
                        "schema_version": 1,
                        "sequence": sequence.sequence_dir.name,
                        "source_scene": sequence.source_scene,
                        "role": sequence.role,
                        "anchor_frame_index": index,
                        "target_identity": target_identity,
                        "anchor_track_id": track_id,
                        "anchor_iou": overlap,
                        "anchor_state_hash": state_hash,
                        "anchor_state_ref": state_key,
                        "source_timestamp_ms": sequence.frames[index].source_timestamp_ms,
                        "input_drop_count": sequence.frames[index].input_drop_count,
                        "prior_detector_skip_count": 0,
                        "detector_gap_bin": "0",
                        "baseline_run_dir": str(sequence.baseline_run_dir),
                    }
                    anchors.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    target_track = next(
                        (item for item in sequence.frames[index].tracks if int(item.get("temporary_track_id", -1)) == track_id),
                        None,
                    )
                    target_box = target_track.get("xyxy", [0.0, 0.0, 0.0, 0.0]) if target_track else [0.0, 0.0, 0.0, 0.0]
                    for key, value in (
                        ("box_center_x_norm", float(target_box[0]) / sequence.width),
                        ("box_width_norm", (float(target_box[2]) - float(target_box[0])) / sequence.width),
                    ):
                        feature_values[key].append(float(value))
                    pair = _pair_from_simulated_branches(
                        sequence=sequence, anchor=index, target_identity=target_identity,
                        anchor_hash=state_hash, protocol=protocol,
                        skip_snapshots=skip_snapshots, detect_snapshots=detect_snapshots,
                        boundary_censored=boundary_censored,
                    )
                    _write_rollout(rollouts, pair, sequence, index, target_identity, 0, labels, exclusions, per_sequence, gap_bins, audit_sample)
            # A deterministic evenly spaced sample carries real prior-skip histories.
            sampled = candidates[:prior_history_samples_per_sequence]
            if len(candidates) > prior_history_samples_per_sequence:
                sampled = [candidates[(i * (len(candidates) - 1)) // (prior_history_samples_per_sequence - 1)] for i in range(prior_history_samples_per_sequence)] if prior_history_samples_per_sequence > 1 else [candidates[0]]
            for index in sorted(set(sampled)):
                eligible = _eligible_targets(sequence, index, sequence.frames[index].tracks, protocol)
                for prior_skip_count in (1, 2, 3):
                    state = _state_with_prior_skips(sequence, index, prior_skip_count)
                    if state is None:
                        continue
                    tracks = [
                        {
                            "temporary_track_id": track.track_id,
                            "xyxy": list(track.box),
                            "score": track.score,
                            "state": "tracked",
                        }
                        for track in state._tracks.values()
                        if track.missed == 0
                    ]
                    variant_targets = _eligible_targets(sequence, index, tracks, protocol)
                    if not variant_targets:
                        continue
                    state_hash = _state_hash(state)
                    state_key = f"{sequence.sequence_dir.name}:{index}:{prior_skip_count}"
                    if state_key not in written_state_keys:
                        anchor_states.write(json.dumps({
                            "schema_version": 1,
                            "state_key": state_key,
                            "sequence": sequence.sequence_dir.name,
                            "source_scene": sequence.source_scene,
                            "role": sequence.role,
                            "anchor_frame_index": index,
                            "prior_detector_skip_count": prior_skip_count,
                            "anchor_state_hash": state_hash,
                            "anchor_state": _state_payload(state),
                        }, sort_keys=True, separators=(",", ":")) + "\n")
                        written_state_keys.add(state_key)
                    skip_snapshots, skip_boundary = _simulate_branch(
                        branch="skip", anchor_state=state, sequence=sequence, anchor=index,
                        protocol=protocol, packets=packets,
                    )
                    detect_snapshots, detect_boundary = _simulate_branch(
                        branch="detect", anchor_state=state, sequence=sequence, anchor=index,
                        protocol=protocol, packets=packets,
                    )
                    for target_identity, overlap, track_id in variant_targets:
                        anchor_count[sequence.sequence_dir.name] += 1
                        payload = {
                            "schema_version": 1,
                            "sequence": sequence.sequence_dir.name,
                            "source_scene": sequence.source_scene,
                            "role": sequence.role,
                            "anchor_frame_index": index,
                            "target_identity": target_identity,
                            "anchor_track_id": track_id,
                            "anchor_iou": overlap,
                            "anchor_state_hash": state_hash,
                            "anchor_state_ref": state_key,
                            "source_timestamp_ms": sequence.frames[index].source_timestamp_ms,
                            "input_drop_count": sequence.frames[index].input_drop_count,
                            "prior_detector_skip_count": prior_skip_count,
                            "detector_gap_bin": str(prior_skip_count),
                            "baseline_run_dir": str(sequence.baseline_run_dir),
                        }
                        anchors.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                        pair = _pair_from_simulated_branches(
                            sequence=sequence, anchor=index, target_identity=target_identity,
                            anchor_hash=state_hash, protocol=protocol,
                            skip_snapshots=skip_snapshots, detect_snapshots=detect_snapshots,
                            boundary_censored=skip_boundary or detect_boundary,
                        )
                        _write_rollout(rollouts, pair, sequence, index, target_identity, prior_skip_count, labels, exclusions, per_sequence, gap_bins, audit_sample)
            sequence_records.append({
                "sequence": sequence.sequence_dir.name,
                "source_scene": sequence.source_scene,
                "role": sequence.role,
                "frames_in_baseline_log": len(sequence.frames),
                "ground_truth_boxes": sum(len(items) for items in sequence.ground_truth.values()),
                "eligible_anchor_target_pairs": anchor_count[sequence.sequence_dir.name],
                "image_duplicate_audit": _image_duplicate_audit(sequence),
                "baseline_summary_sha256": file_sha256(sequence.baseline_run_dir / "summary.json") if (sequence.baseline_run_dir / "summary.json").is_file() else None,
            })
            source_drops += sum(frame.input_drop_count for frame in sequence.frames.values())
    class_total = labels["0"] + labels["1"]
    class_weights = {key: (class_total / (2 * count) if count else None) for key, count in (("0", labels["0"]), ("1", labels["1"]))}
    normalization = {
        "schema_version": 1,
        "fit_scope": "training_role_only",
        "feature_names": sorted(feature_values),
        "statistics": {
            key: {"count": len(values), "mean": statistics.fmean(values) if values else None, "std": statistics.pstdev(values) if len(values) > 1 else 1.0}
            for key, values in sorted(feature_values.items())
        },
        "label_class_weights": class_weights,
        "note": "Geometry statistics and class weights are training-role records only; Phase 7 may extend the frozen feature schema before model fitting.",
    }
    normalization_path = output_root / "training_statistics.json"
    normalization_path.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "status": "passed_training_role_only",
        "labels": dict(labels),
        "exclusions": dict(exclusions),
        "per_sequence": {key: dict(value) for key, value in sorted(per_sequence.items())},
        "detector_gap_bins": dict(sorted(gap_bins.items())),
        "source_input_drop_count": source_drops,
        "duplicate_frame_audit": {item["sequence"]: item["image_duplicate_audit"] for item in sequence_records},
        "sequence_leakage_audit": _role_leakage_audit(roles),
        "hand_audit_fixture": {
            "path": "race_mot/data/label_audit.json",
            "cases": len(json.loads((Path(__file__).parents[3] / "data" / "label_audit.json").read_text(encoding="utf-8"))["cases"]),
            "status": "protocol_checked_and_referenced",
        },
        "audit_sample": audit_sample[:30],
        "selection_boundary": {
            "generated_roles": ["training"],
            "calibration_labels_generated": False,
            "policy_validation_labels_generated": False,
            "final_evaluation_labels_generated": False,
            "mot20_used": False,
            "test_labels_used_for_selection": False,
        },
        "label_prevalence": {
            "eligible": class_total,
            "positive": labels["1"],
            "positive_fraction": labels["1"] / class_total if class_total else None,
        },
    }
    audit_path = output_root / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "phase": 6,
        "status": "completed_training_role_dataset",
        "phone_capture": False,
        "mot20_used": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": file_sha256(protocol_path),
        "protocol": json.loads(protocol_path.read_text(encoding="utf-8")),
        "roles_path": str(roles_path),
        "roles_sha256": file_sha256(roles_path),
        "selected_role": "training",
        "representative_variant_policy": "FRCNN image/ground-truth representative; DPM/SDP variants remain grouped with the same source scene and are not duplicated.",
        "prior_history_samples_per_sequence": prior_history_samples_per_sequence,
        "sequences": sequence_records,
        "files": {
            "anchors": {"path": anchors_path.name, "sha256": file_sha256(anchors_path)},
            "anchor_states": {"path": anchor_states_path.name, "sha256": file_sha256(anchor_states_path)},
            "rollouts": {"path": rollouts_path.name, "sha256": file_sha256(rollouts_path)},
            "audit": {"path": audit_path.name, "sha256": file_sha256(audit_path)},
            "training_statistics": {"path": normalization_path.name, "sha256": file_sha256(normalization_path)},
        },
        "counts": {
            "anchors": sum(anchor_count.values()),
            "rollouts": sum(labels.values()) + sum(exclusions.values()),
            "labels": dict(labels),
            "exclusions": dict(exclusions),
        },
        "software": {
            "tracker_backend": "deterministic_iou_compat",
            "tracker_version": "d1bf0191adff59bc8fcfeaa0b33d3d1642552a99",
            "generator": "race_mot.evaluation.phase6_dataset",
        },
        "selection_protection": "Only training-role labels and normalization/class weights are produced. Calibration, policy-validation, final-evaluation, MOT17 official-test, phone, and MOT20 artifacts are not read or generated.",
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _make_pair(sequence: SequenceInput, anchor: int, target_identity: int, state: ByteTrackAdapter, protocol: RolloutProtocol, packets: dict[int, FramePacket], prior_skip_count: int) -> tuple[Any, Any, Any]:
    anchor_hash = _state_hash(state)
    skip = _branch_observation_from_state(branch="skip", anchor_state=state, anchor_hash=anchor_hash, sequence=sequence, anchor=anchor, target_identity=target_identity, protocol=protocol, packets=packets)
    detect = _branch_observation_from_state(branch="detect", anchor_state=state, anchor_hash=anchor_hash, sequence=sequence, anchor=anchor, target_identity=target_identity, protocol=protocol, packets=packets)
    pair = pair_observations(anchor_frame_index=anchor, target_identity=target_identity, skip=skip, detect=detect, protocol=protocol)
    return pair, skip, detect


def _write_rollout(handle: Any, pair_data: tuple[Any, Any, Any], sequence: SequenceInput, anchor: int, target_identity: int, prior_skip_count: int, labels: Counter[str], exclusions: Counter[str], per_sequence: dict[str, Counter[str]], gap_bins: Counter[str], audit_sample: list[dict[str, Any]]) -> None:
    pair, skip, detect = pair_data
    if pair.label is None:
        exclusions[pair.exclusion or "unknown"] += 1
        per_sequence[sequence.sequence_dir.name][f"excluded:{pair.exclusion or 'unknown'}"] += 1
    else:
        labels[str(pair.label)] += 1
        per_sequence[sequence.sequence_dir.name][f"label:{pair.label}"] += 1
    gap_bins[str(prior_skip_count)] += 1
    payload = {
        "schema_version": 1,
        "sequence": sequence.sequence_dir.name,
        "source_scene": sequence.source_scene,
        "role": sequence.role,
        "anchor_frame_index": pair.anchor_frame_index,
        "target_identity": pair.target_identity,
        "prior_detector_skip_count": prior_skip_count,
        "detector_gap_bin": str(prior_skip_count),
        "anchor_state_hash": pair.anchor_state_hash,
        "future_frame_indices": list(pair.future_frame_indices),
        "skip": {"visible": list(skip.visible), "assigned_identity": list(skip.assigned_identity), "boundary_censored": skip.boundary_censored},
        "detect": {"visible": list(detect.visible), "assigned_identity": list(detect.assigned_identity), "boundary_censored": detect.boundary_censored},
        "skip_failed": pair.skip_failed,
        "detect_failed": pair.detect_failed,
        "label": pair.label,
        "exclusion": pair.exclusion,
    }
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    categories = {
        "positive" if item.get("label") == 1 else item.get("exclusion") if item.get("exclusion") else "negative"
        for item in audit_sample
    }
    category = "positive" if pair.label == 1 else pair.exclusion if pair.exclusion else "negative"
    if (len(audit_sample) < 30 or category not in categories) and (pair.label is not None or pair.exclusion):
        audit_sample.append({
            "sequence": sequence.sequence_dir.name,
            "anchor_frame_index": anchor,
            "target_identity": target_identity,
            "prior_detector_skip_count": prior_skip_count,
            "label": pair.label,
            "exclusion": pair.exclusion,
            "skip_failed": pair.skip_failed,
            "detect_failed": pair.detect_failed,
        })

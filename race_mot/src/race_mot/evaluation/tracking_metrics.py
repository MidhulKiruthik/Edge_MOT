"""Dependency-free MOT metrics and baseline-log measurement summaries.

The implementation follows the usual IoU-0.5 matching conventions and is
intended as a reproducible fallback when the pinned TrackEval package is not
installed.  Reports identify the evaluator explicitly; they must not be
described as official TrackEval output unless TrackEval is available and used.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from race_mot.evaluation.mot import group_by_frame, read_gt


def _iou(left: Iterable[float], right: Iterable[float]) -> float:
    a = tuple(float(value) for value in left)
    b = tuple(float(value) for value in right)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _match(predictions: list[dict[str, object]], truths: list[object], threshold: float) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(predictions):
        for truth_index, truth in enumerate(truths):
            overlap = _iou(prediction["xyxy"], truth.xyxy)
            if overlap >= threshold:
                candidates.append((overlap, prediction_index, truth_index))
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for overlap, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        pairs.append((prediction_index, truth_index, overlap))
    return pairs


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, min(len(values) - 1, math.ceil(len(values) * percentile) - 1))]


def _trackeval_info() -> dict[str, object]:
    try:
        import trackeval  # type: ignore[import-not-found]
    except ImportError:
        return {"available": False, "version": None, "evaluator": "dependency_free_mot_metrics"}
    return {
        "available": True,
        "version": getattr(trackeval, "__version__", "unknown"),
        "evaluator": "trackeval_available_but_fallback_not_invoked",
    }


def evaluate_tracking(
    sequence_dir: Path,
    frame_log_path: Path,
    *,
    visibility_min: float = 0.2,
    iou_threshold: float = 0.5,
    warmup_frames: int = 5,
) -> dict[str, object]:
    """Evaluate one deterministic baseline frame log against MOT ground truth."""
    if not sequence_dir.is_dir():
        raise ValueError("sequence_dir must be a directory")
    if not frame_log_path.is_file():
        raise ValueError("frame_log_path must name an existing JSONL file")
    gt_path = sequence_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise ValueError("sequence is missing gt/gt.txt")
    if not 0.0 <= visibility_min <= 1.0:
        raise ValueError("visibility_min must be in [0, 1]")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    truths_by_frame = group_by_frame(tuple(box for box in read_gt(gt_path) if box.visibility >= visibility_min))
    frame_count = 0
    gt_count = 0
    prediction_count = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    fragments = 0
    last_assignment: dict[int, int | None] = {}
    ever_matched: set[int] = set()
    previously_unmatched: set[int] = set()
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    latency_ms: list[float] = []
    steady_latency_ms: list[float] = []
    detector_ms: list[float] = []
    steady_detector_ms: list[float] = []
    preprocess_ms: list[float] = []
    postprocess_ms: list[float] = []
    coordinate_ms: list[float] = []
    source_drops = 0
    deadline_misses = 0
    deadline_ms: float | None = None
    with frame_log_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_count += 1
            frame_index = int(record["source_index"])
            truths = list(truths_by_frame.get(frame_index, ()))
            predictions = list(record.get("tracks", ()))
            gt_count += len(truths)
            prediction_count += len(predictions)
            pairs = _match(predictions, truths, iou_threshold)
            true_positives += len(pairs)
            false_positives += len(predictions) - len(pairs)
            false_negatives += len(truths) - len(pairs)
            matched_truths = {truth_index for _, truth_index, _ in pairs}
            current_assignment: dict[int, int | None] = {
                truth.identity: None for truth in truths
            }
            for prediction_index, truth_index, _ in pairs:
                truth_id = truths[truth_index].identity
                track_id = int(predictions[prediction_index]["temporary_track_id"])
                current_assignment[truth_id] = track_id
                pair_counts[(truth_id, track_id)] += 1
            for truth in truths:
                truth_id = truth.identity
                current = current_assignment[truth_id]
                previous = last_assignment.get(truth_id)
                if current is not None and previous is not None and current != previous:
                    id_switches += 1
                if current is not None:
                    if truth_id in previously_unmatched and truth_id in ever_matched:
                        fragments += 1
                    ever_matched.add(truth_id)
                    previously_unmatched.discard(truth_id)
                elif truth_id in ever_matched:
                    previously_unmatched.add(truth_id)
            last_assignment = current_assignment
            source_drops += int(record.get("input_drop_count", 0))
            value = record.get("pipeline_latency_ms")
            if isinstance(value, (int, float)) and math.isfinite(value):
                latency_ms.append(float(value))
                if frame_count > warmup_frames:
                    steady_latency_ms.append(float(value))
            timing = record.get("detector_timing")
            if isinstance(timing, dict):
                for key, target in (
                    ("inference_ms", detector_ms),
                    ("preprocessing_ms", preprocess_ms),
                    ("postprocessing_ms", postprocess_ms),
                    ("coordinate_mapping_ms", coordinate_ms),
                ):
                    value = timing.get(key)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        target.append(float(value))
                        if key == "inference_ms" and frame_count > warmup_frames:
                            steady_detector_ms.append(float(value))
            value = record.get("deadline_ms")
            if isinstance(value, (int, float)) and value > 0:
                deadline_ms = float(value)
            if deadline_ms is not None and latency_ms and latency_ms[-1] > deadline_ms:
                deadline_misses += 1
    idtp = 0
    used_truths: set[int] = set()
    used_tracks: set[int] = set()
    for (truth_id, track_id), count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
        if truth_id in used_truths or track_id in used_tracks:
            continue
        used_truths.add(truth_id)
        used_tracks.add(track_id)
        idtp += count
    idfn = gt_count - idtp
    idfp = prediction_count - idtp
    det_a = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) else None
    ass_a = idtp / true_positives if true_positives else None
    hota = math.sqrt(det_a * ass_a) if det_a is not None and ass_a is not None else None
    mota = 1.0 - (false_negatives + false_positives + id_switches) / gt_count if gt_count else None
    idf1 = (2.0 * idtp) / (2.0 * idtp + idfp + idfn) if (2.0 * idtp + idfp + idfn) else None
    return {
        "schema_version": 1,
        "sequence": sequence_dir.name,
        "frame_log": frame_log_path.name,
        "frames_evaluated": frame_count,
        "visibility_min": visibility_min,
        "iou_threshold": iou_threshold,
        "trackeval": _trackeval_info(),
        "evaluator_note": "DetA/AssA/HOTA are dependency-free IoU/global-assignment approximations; install and invoke pinned TrackEval before publishing official TrackEval metrics.",
        "detector_diagnostics": {
            "ground_truth_boxes": gt_count,
            "predicted_tracks": prediction_count,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": true_positives / prediction_count if prediction_count else None,
            "recall": true_positives / gt_count if gt_count else None,
        },
        "tracking_metrics": {
            "HOTA_approx": hota,
            "DetA_approx": det_a,
            "AssA_approx": ass_a,
            "IDF1": idf1,
            "MOTA": mota,
            "IDTP": idtp,
            "IDFP": idfp,
            "IDFN": idfn,
            "ID_switches": id_switches,
            "fragmentation": fragments,
        },
        "runtime_metrics": {
            "latency_ms": {
                "count": len(latency_ms),
                "median": statistics.median(latency_ms) if latency_ms else None,
                "p95": _percentile(latency_ms, 0.95),
                "max": max(latency_ms) if latency_ms else None,
            },
            "detector_inference_ms": {"median": statistics.median(detector_ms) if detector_ms else None, "p95": _percentile(detector_ms, 0.95)},
            "preprocessing_ms": {"median": statistics.median(preprocess_ms) if preprocess_ms else None, "p95": _percentile(preprocess_ms, 0.95)},
            "postprocessing_ms": {"median": statistics.median(postprocess_ms) if postprocess_ms else None, "p95": _percentile(postprocess_ms, 0.95)},
            "coordinate_mapping_ms": {"median": statistics.median(coordinate_ms) if coordinate_ms else None, "p95": _percentile(coordinate_ms, 0.95)},
            "throughput_fps": frame_count / (sum(latency_ms) / 1000.0) if latency_ms and sum(latency_ms) > 0 else None,
            "source_input_drops": source_drops,
            "deadline_ms": deadline_ms,
            "deadline_misses": deadline_misses,
                "deadline_miss_rate": deadline_misses / frame_count if frame_count else None,
                "queue_depth_max": 0,
            "warmup_frames_excluded": warmup_frames,
            "steady_state_latency_ms": {
                "count": len(steady_latency_ms),
                "median": statistics.median(steady_latency_ms) if steady_latency_ms else None,
                "p95": _percentile(steady_latency_ms, 0.95),
                "max": max(steady_latency_ms) if steady_latency_ms else None,
            },
            "steady_state_detector_inference_ms": {
                "count": len(steady_detector_ms),
                "median": statistics.median(steady_detector_ms) if steady_detector_ms else None,
                "p95": _percentile(steady_detector_ms, 0.95),
            },
        },
    }

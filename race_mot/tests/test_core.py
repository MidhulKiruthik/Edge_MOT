import json
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from race_mot.config import load_config
from race_mot.dashboard import load_dashboard_state
from race_mot.application import SequentialBaseline
from race_mot.detectors.yolox_tensorrt import FixedDetectionsDetector
from race_mot.domain import (
    DetectorAction,
    Detection,
    FramePacket,
    SourceKind,
    TrackSnapshot,
    TrackState,
    aggregate_frame_risk,
)
from race_mot.evaluation.labels import RolloutFailure, avoidable_failure_label
from race_mot.evaluation.mot import group_by_frame, read_gt
from race_mot.evaluation.paired_rollout import BranchObservation, pair_observations
from race_mot.evaluation.protocol import RolloutProtocol
from race_mot.evaluation.tracking_metrics import evaluate_tracking
from race_mot.detector_smoke import (
    DetectorSmokeConfig,
    _decode_yolox_output,
    _match_count,
    _person_detection_count,
    _restore_boxes,
)
from race_mot.inventory import _hwmon_devices, _thermal_zones
from race_mot.policy import BinaryScheduler, SchedulerConfig
from race_mot.trackers.bytetrack import ByteTrackAdapter
from race_mot.sources.video import VideoSource


def track(track_id: int = 1) -> TrackSnapshot:
    return TrackSnapshot(track_id, (0.0, 0.0, 10.0, 10.0), 0.9, TrackState.TRACKED, 3)


class CoreContractsTests(unittest.TestCase):
    def test_tracker_update_empty_is_distinct_from_skip(self) -> None:
        tracker = ByteTrackAdapter()
        packet = FramePacket("run", 1, 0.0, 1, SourceKind.MOT_SEQUENCE, image_bgr=object())
        detections = [Detection((0.0, 0.0, 10.0, 10.0), 0.9, 0)]
        self.assertEqual(len(tracker.initialize(detections, packet)), 1)
        empty = tracker.update([], FramePacket("run", 2, 33.0, 2, SourceKind.MOT_SEQUENCE, image_bgr=object()))
        self.assertEqual(empty, [])
        predicted = tracker.skip(FramePacket("run", 3, 66.0, 3, SourceKind.MOT_SEQUENCE, image_bgr=object()))
        self.assertEqual(len(predicted), 1)
        self.assertEqual(tracker.last_event, "skip")

    def test_tracker_state_clone_is_independent_and_hashable(self) -> None:
        tracker = ByteTrackAdapter()
        packet = FramePacket("run", 1, 0.0, 1, SourceKind.MOT_SEQUENCE, image_bgr=object())
        tracker.update([Detection((0.0, 0.0, 10.0, 10.0), 0.9, 0)], packet)
        clone = tracker.clone()
        self.assertEqual(clone.state_dict(), tracker.state_dict())
        clone.skip(FramePacket("run", 2, 33.0, 2, SourceKind.MOT_SEQUENCE, image_bgr=object()))
        self.assertNotEqual(clone.state_dict(), tracker.state_dict())

    def test_baseline_restart_and_failure_are_recorded(self) -> None:
        config_path = Path(__file__).parents[1] / "configs" / "baseline.json"
        def source(run_id: str):
            yield FramePacket(run_id, 1, 0.0, 1, SourceKind.MOT_SEQUENCE, image_bgr=object())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = SequentialBaseline(
                FixedDetectionsDetector([Detection((0.0, 0.0, 10.0, 10.0), 0.9, 0)]),
                ByteTrackAdapter(),
            )
            first = runner.run(source("one"), output_dir=root / "one", config_path=config_path, source_name="local", source_kind="mot_sequence")
            second = runner.run(source("two"), output_dir=root / "two", config_path=config_path, source_name="local", source_kind="mot_sequence")
            self.assertEqual(first.status, "completed")
            self.assertEqual(second.status, "completed")
            runner.stop()
            runner.stop()

            def broken_source():
                yield FramePacket("broken", 1, 0.0, 1, SourceKind.MOT_SEQUENCE, image_bgr=object())
                raise RuntimeError("synthetic source loss")
            failed = SequentialBaseline(FixedDetectionsDetector(), ByteTrackAdapter()).run(
                broken_source(), output_dir=root / "failed", config_path=config_path,
                source_name="local", source_kind="mot_sequence"
            )
            self.assertEqual(failed.status, "failed")
            self.assertIn("synthetic source loss", failed.error or "")

    def test_video_source_preserves_decoder_indices_and_drops(self) -> None:
        class Capture:
            def __init__(self):
                self.items = [(1, 0.0), (3, 66.0)]
                self.current = (0, 0.0)
                self.released = False
            def isOpened(self): return True
            def read(self):
                if not self.items: return False, None
                return True, SimpleNamespace(shape=(8, 8, 3))
            def get(self, prop):
                if prop == 1: return 0.0
                if prop == 5: return self.current[0]
                if prop == 6: return self.current[1]
                return 0.0
            def release(self): self.released = True
        capture = Capture()
        original_read = capture.read
        def read_with_progress():
            if not capture.items: return False, None
            capture.current = capture.items.pop(0)
            return True, SimpleNamespace(shape=(8, 8, 3))
        capture.read = read_with_progress
        with patch.dict("sys.modules", {"cv2": SimpleNamespace(CAP_PROP_BUFFERSIZE=38, CAP_PROP_POS_FRAMES=5, CAP_PROP_POS_MSEC=6)}), patch(
            "race_mot.sources.video._open_capture", return_value=(capture, "fake")
        ):
            packets = list(VideoSource("clip.avi", run_id="video"))
        self.assertEqual([packet.source_index for packet in packets], [1, 3])
        self.assertEqual([packet.input_drop_count for packet in packets], [0, 1])
        self.assertTrue(capture.released)

    def test_tracking_metrics_report_detector_and_identity_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "MOT17-test"
            (sequence / "gt").mkdir(parents=True)
            (sequence / "gt" / "gt.txt").write_text(
                "1,7,0,0,10,10,1,1,1\n2,7,1,0,10,10,1,1,1\n", encoding="utf-8"
            )
            frame_log = root / "frame_log.jsonl"
            frame_log.write_text(
                json.dumps({"source_index": 1, "tracks": [{"temporary_track_id": 3, "xyxy": [0, 0, 10, 10]}], "detections": [], "input_drop_count": 0, "pipeline_latency_ms": 10.0, "detector_timing": {"inference_ms": 5.0}}) + "\n"
                + json.dumps({"source_index": 2, "tracks": [{"temporary_track_id": 3, "xyxy": [1, 0, 11, 10]}], "detections": [], "input_drop_count": 0, "pipeline_latency_ms": 20.0, "detector_timing": {"inference_ms": 6.0}}) + "\n",
                encoding="utf-8",
            )
            report = evaluate_tracking(sequence, frame_log)
        self.assertEqual(report["detector_diagnostics"]["true_positives"], 2)
        self.assertEqual(report["tracking_metrics"]["ID_switches"], 0)
        self.assertEqual(report["tracking_metrics"]["MOTA"], 1.0)
        self.assertEqual(report["runtime_metrics"]["latency_ms"]["median"], 15.0)
    def test_baseline_config_loads_without_jetson(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "baseline.json")
        self.assertEqual(config.mode, "baseline")
        self.assertEqual(config.input_size, (416, 416))

    def test_dashboard_state_is_redacted_and_tracks_latest_frame(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "manifest.json").write_text(json.dumps({
            "run_id": "dashboard-test", "source": "MOT17-02-FRCNN",
            "source_kind": "mot_sequence", "runtime": {"action": "DETECT", "deadline_ms": 33.3},
        }), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps({
            "run_id": "dashboard-test", "status": "completed", "frames_processed": 2,
            "detections": 3, "tracks": 3, "error": None,
        }), encoding="utf-8")
        (root / "frame_log.jsonl").write_text(json.dumps({
            "source_index": 2, "source_timestamp_ms": 66.7, "executed_action": "DETECT",
            "planned_action": "DETECT", "reason": "BASELINE_EVERY_FRAME", "pipeline_latency_ms": 12.5,
            "processing_ms": 10.0, "active_track_count": 1, "queue_depth": 0, "input_drop_count": 0,
            "tracks": [{"temporary_track_id": 4, "xyxy": [1, 2, 30, 40], "score": 0.9, "state": "tracked"}],
        }) + "\n", encoding="utf-8")
        state = load_dashboard_state(root)
        self.assertFalse(state["phone_capture"])
        self.assertEqual(state["latest"]["source_index"], 2)
        self.assertEqual(state["latest"]["tracks"][0]["temporary_track_id"], 4)
        self.assertNotIn("image_bgr", json.dumps(state))

    def test_scheduler_forces_detect_without_tracks_or_risk(self) -> None:
        scheduler = BinaryScheduler(SchedulerConfig(0.5, 2))
        no_tracks = scheduler.decide_next(next_frame_index=1, tracks=[], track_risks={})
        invalid = scheduler.decide_next(
            next_frame_index=2,
            tracks=[track()],
            track_risks=None,
            predictor_valid=False,
        )
        self.assertEqual(no_tracks.reason.value, "NO_TRACKS")
        self.assertEqual(invalid.executed_action, DetectorAction.DETECT)

    def test_scheduler_guard_only_upgrades_skip(self) -> None:
        scheduler = BinaryScheduler(SchedulerConfig(0.5, 2))
        planned = scheduler.decide_next(
            next_frame_index=1, tracks=[track()], track_risks={1: 0.1}
        )
        executed = scheduler.apply_execution(
            planned, scene_activity_score=0.9, guard_threshold=0.5
        )
        self.assertEqual(planned.planned_action, DetectorAction.SKIP)
        self.assertEqual(executed.executed_action, DetectorAction.DETECT)
        self.assertTrue(executed.scene_guard_triggered)

    def test_scheduler_ignores_risk_for_removed_track(self) -> None:
        scheduler = BinaryScheduler(SchedulerConfig(0.5, 2))
        decision = scheduler.decide_next(
            next_frame_index=1, tracks=[track()], track_risks={1: 0.1, 99: 1.0}
        )
        self.assertEqual(decision.executed_action, DetectorAction.SKIP)

    def test_avoidable_failure_requires_detect_branch_success(self) -> None:
        skip = RolloutFailure(identity_failure=True, persistent_loss=False)
        detect = RolloutFailure(identity_failure=False, persistent_loss=False)
        self.assertEqual(avoidable_failure_label(skip, detect), 1)
        self.assertEqual(avoidable_failure_label(skip, RolloutFailure(True, False)), 0)

    def test_mot_ground_truth_parser_groups_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.txt"
            path.write_text(
                "1,7,10,20,30,40,1,1,0.9\n2,7,11,20,30,40,1,1,0.8\n", encoding="utf-8"
            )
            boxes = read_gt(path)
        grouped = group_by_frame(boxes)
        self.assertEqual(tuple(grouped), (1, 2))
        self.assertEqual(grouped[1][0].xyxy, (10.0, 20.0, 40.0, 60.0))

    def test_frame_risk_uses_max_and_preserves_empty_state(self) -> None:
        self.assertEqual(aggregate_frame_risk({1: 0.2, 2: 0.7}), 0.7)
        self.assertIsNone(aggregate_frame_risk({}))

    def test_inventory_optional_sysfs_sources_are_safe_when_absent(self) -> None:
        self.assertIsInstance(_thermal_zones(), dict)
        self.assertIsInstance(_hwmon_devices(), dict)

    def test_detector_smoke_requires_at_least_one_frame(self) -> None:
        with self.assertRaises(ValueError):
            DetectorSmokeConfig(Path("engine.plan"), 0)

    def test_detector_smoke_rejects_missing_reference_model(self) -> None:
        with self.assertRaises(ValueError):
            DetectorSmokeConfig(
                Path("engine.plan"), reference_onnx_path=Path("missing.onnx")
            )

    def test_person_postprocessing_filters_and_suppresses_boxes(self) -> None:
        import numpy as np

        output = np.array(
            [[[50, 50, 20, 20, 0.9, 0.9],
              [51, 51, 20, 20, 0.8, 0.9],
              [100, 100, 10, 10, 0.9, 0.8],
              [200, 200, 10, 10, 0.2, 0.9]]],
            dtype=np.float32,
        )
        count, maximum_score = _person_detection_count(output, 0.3, 0.45)
        self.assertEqual(count, 2)
        self.assertAlmostEqual(maximum_score, 0.81, places=6)

    def test_yolox_raw_output_decodes_expected_grid(self) -> None:
        import numpy as np

        output = np.zeros((1, 3549, 85), dtype=np.float32)
        decoded = _decode_yolox_output(output, 416, 416)
        self.assertTrue(np.allclose(decoded[0, 0, :4], [0, 0, 8, 8]))
        self.assertTrue(np.allclose(decoded[0, 1, :4], [8, 0, 8, 8]))

    def test_detector_boxes_restore_and_match_original_coordinates(self) -> None:
        import numpy as np

        restored = _restore_boxes(
            np.array([[20.8, 10.4, 41.6, 31.2]], dtype=np.float32),
            1920,
            1080,
            416,
            416,
        )
        self.assertTrue(np.allclose(restored[0], [96, 48, 192, 144], atol=1e-4))
        self.assertEqual(_match_count(restored, [[95, 47, 193, 145]], 0.5), 1)

    def test_paired_rollout_requires_identical_anchor_and_future_frames(self) -> None:
        protocol = RolloutProtocol(0.5, 0.2, 2, 1, "detector", "tracker")
        skip = BranchObservation("skip", "anchor", (2, 3), (True, True), (7, None))
        detect = BranchObservation("detect", "anchor", (2, 3), (True, True), (7, 7))
        result = pair_observations(
            anchor_frame_index=1,
            target_identity=7,
            skip=skip,
            detect=detect,
            protocol=protocol,
        )
        self.assertEqual(result.label, 1)
        self.assertTrue(result.skip_failed)
        self.assertFalse(result.detect_failed)

    def test_paired_rollout_boundary_is_excluded(self) -> None:
        protocol = RolloutProtocol(0.5, 0.2, 2, 1, "detector", "tracker")
        skip = BranchObservation("skip", "anchor", (2, 3), (True, True), (7, 7), True)
        detect = BranchObservation("detect", "anchor", (2, 3), (True, True), (7, 7), True)
        result = pair_observations(
            anchor_frame_index=1,
            target_identity=7,
            skip=skip,
            detect=detect,
            protocol=protocol,
        )
        self.assertIsNone(result.label)
        self.assertEqual(result.exclusion, "boundary_censored")

    def test_label_audit_fixture_matches_frozen_protocol(self) -> None:
        import json

        fixture = json.loads(
            (Path(__file__).parents[1] / "data" / "label_audit.json").read_text()
        )
        protocol = RolloutProtocol(0.5, 0.2, 5, 2, "detector", "tracker")
        observed = {"positive": 0, "safe_negative": 0, "excluded": 0}
        for case in fixture["cases"]:
            skip = BranchObservation(
                "skip", "audit", tuple(range(1, 6)), tuple(case["skip_visible"]), tuple(case["skip_identity"]), case["boundary_censored"]
            )
            detect = BranchObservation(
                "detect", "audit", tuple(range(1, 6)), tuple(case["detect_visible"]), tuple(case["detect_identity"]), case["boundary_censored"]
            )
            result = pair_observations(
                anchor_frame_index=0,
                target_identity=7,
                skip=skip,
                detect=detect,
                protocol=protocol,
            )
            self.assertEqual(result.label, case["expected_label"], case["name"])
            self.assertEqual(result.exclusion, case["expected_exclusion"], case["name"])
            if result.label == 1:
                observed["positive"] += 1
            elif result.label == 0:
                observed["safe_negative"] += 1
            else:
                observed["excluded"] += 1
        self.assertEqual(observed, {"positive": 2, "safe_negative": 3, "excluded": 1})


if __name__ == "__main__":
    unittest.main()

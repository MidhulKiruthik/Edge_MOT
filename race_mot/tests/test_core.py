import json
import tempfile
import unittest
from pathlib import Path

from race_mot.config import load_config
from race_mot.domain import (
    DetectorAction,
    TrackSnapshot,
    TrackState,
    aggregate_frame_risk,
)
from race_mot.evaluation.labels import RolloutFailure, avoidable_failure_label
from race_mot.evaluation.mot import group_by_frame, read_gt
from race_mot.detector_smoke import (
    DetectorSmokeConfig,
    _decode_yolox_output,
    _match_count,
    _person_detection_count,
    _restore_boxes,
)
from race_mot.inventory import _hwmon_devices, _thermal_zones
from race_mot.policy import BinaryScheduler, SchedulerConfig


def track(track_id: int = 1) -> TrackSnapshot:
    return TrackSnapshot(track_id, (0.0, 0.0, 10.0, 10.0), 0.9, TrackState.TRACKED, 3)


class CoreContractsTests(unittest.TestCase):
    def test_baseline_config_loads_without_jetson(self) -> None:
        config = load_config(Path(__file__).parents[1] / "configs" / "baseline.json")
        self.assertEqual(config.mode, "baseline")
        self.assertEqual(config.input_size, (416, 416))

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


if __name__ == "__main__":
    unittest.main()

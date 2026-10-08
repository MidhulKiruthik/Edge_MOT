"""Run the pinned official TrackEval package on Phase 5 MOT logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import trackeval
    from trackeval.datasets.mot_challenge_2d_box import MotChallenge2DBox
    from trackeval.eval import Evaluator
    from trackeval.metrics.clear import CLEAR
    from trackeval.metrics.hota import HOTA
    from trackeval.metrics.identity import Identity

    data_root = args.output.parent / "trackeval_data"
    gt_root = data_root / "gt" / "MOT17-train"
    tracker_root = data_root / "trackers" / "MOT17-train" / "race_mot" / "data"
    tracker_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    seq_info: dict[str, int] = {}
    for sequence in args.sequence:
        name = sequence.name
        info = sequence / "seqinfo.ini"
        import configparser
        parser_ini = configparser.ConfigParser(); parser_ini.read(info)
        seq_info[name] = int(parser_ini["Sequence"]["seqLength"])
        gt_link = gt_root / name / "gt"
        gt_link.parent.mkdir(parents=True, exist_ok=True)
        if gt_link.exists() or gt_link.is_symlink():
            gt_link.unlink()
        gt_link.symlink_to((sequence / "gt").resolve(), target_is_directory=True)
        frame_log = args.run_root / f"repeat-01-{name}" / "frame_log.jsonl"
        tracker_file = tracker_root / f"{name}.txt"
        with tracker_file.open("w", encoding="utf-8") as target:
            for line in frame_log.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                for track in record.get("tracks", ()):
                    x1, y1, x2, y2 = track["xyxy"]
                    score = track.get("score") or 1.0
                    target.write(
                        f"{record['source_index']},{track['temporary_track_id']},{x1},{y1},"
                        f"{x2-x1},{y2-y1},{score},-1,-1,-1\n"
                    )
    dataset_config = MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(data_root / "gt"),
        "TRACKERS_FOLDER": str(data_root / "trackers"),
        "TRACKERS_TO_EVAL": ["race_mot"],
        "SEQ_INFO": seq_info,
        "PRINT_CONFIG": False,
        "DO_PREPROC": False,
        "OUTPUT_FOLDER": str(args.output.parent / "trackeval_output"),
        "PLOT_CURVES": False,
    })
    dataset = MotChallenge2DBox(dataset_config)
    evaluator = Evaluator({
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    result, _messages = evaluator.evaluate([dataset], [HOTA(), CLEAR(), Identity()])
    payload = {
        "schema_version": 1,
        "trackeval": {"available": True, "version": getattr(trackeval, "__version__", "unknown")},
        "sequences": [sequence.name for sequence in args.sequence],
        "tracker": "race_mot",
        "preprocessing": "TrackEval DO_PREPROC=false; local MOT annotations are evaluated as supplied",
        "results": _jsonable(result["MotChallenge2DBox"]["race_mot"]),
        "source_logs": str(args.run_root),
        "phone_capture": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

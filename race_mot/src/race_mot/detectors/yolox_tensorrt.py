"""TensorRT FP32 YOLOX-Tiny adapter.

The adapter deliberately keeps preprocessing, engine execution, postprocessing,
and coordinate restoration as separate measured stages.  It returns only
canonical person detections in source-frame coordinates; frames and tensors are
never written by this module.
"""

from __future__ import annotations

import ctypes
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from race_mot.domain import Detection, FramePacket
from race_mot.detector_smoke import (
    _CudaRuntime,
    _decode_yolox_output,
    _letterbox_bgr,
    _person_detections,
    _restore_boxes,
)


@dataclass(frozen=True, slots=True)
class DetectorTiming:
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    coordinate_mapping_ms: float
    total_ms: float
    input_sha256: str
    output_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "preprocessing_ms": self.preprocessing_ms,
            "inference_ms": self.inference_ms,
            "postprocessing_ms": self.postprocessing_ms,
            "coordinate_mapping_ms": self.coordinate_mapping_ms,
            "total_ms": self.total_ms,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
        }


class TensorRTYoloXDetector:
    """Batch-one YOLOX TensorRT adapter for the frozen FP32 engine."""

    def __init__(
        self,
        engine_path: Path,
        *,
        checkpoint_path: Path | None = None,
        confidence_threshold: float = 0.10,
        nms_iou_threshold: float = 0.45,
        warmup_runs: int = 0,
    ) -> None:
        if not Path(engine_path).is_file() or Path(engine_path).stat().st_size == 0:
            raise ValueError("engine_path must name a non-empty TensorRT engine")
        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 < nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in (0, 1]")
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        self.engine_path = Path(engine_path)
        if checkpoint_path is not None and (not Path(checkpoint_path).is_file() or Path(checkpoint_path).stat().st_size == 0):
            raise ValueError("checkpoint_path must name a non-empty file when provided")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.warmup_runs = warmup_runs
        self.last_timing: DetectorTiming | None = None
        self._loaded = False
        self._trt = None
        self._np = None
        self._cuda: _CudaRuntime | None = None
        self._engine = None
        self._context = None
        self._input_name = None
        self._output_name = None
        self._input_shape = None
        self._output_shape = None
        self._input_host = None
        self._output_host = None
        self._input_device = None
        self._output_device = None
        self._stream = None

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "family": "yolox_tiny",
            "runtime": "TensorRT",
            "precision": "fp32",
            "engine_path": self.engine_path.name,
            "engine_sha256": hashlib.sha256(self.engine_path.read_bytes()).hexdigest(),
            "checkpoint_path": self.checkpoint_path.name if self.checkpoint_path else None,
            "checkpoint_sha256": hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest() if self.checkpoint_path else None,
            "confidence_threshold": self.confidence_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "class_id": 0,
            "input_size": [self._input_shape[3], self._input_shape[2]] if self._input_shape else [416, 416],
            "batch_size": 1,
        }

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            import numpy as np
            import tensorrt as trt
        except ImportError as exc:  # pragma: no cover - board dependency
            raise RuntimeError("NumPy and TensorRT Python are required for the detector") from exc
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if engine is None:
            raise RuntimeError("could not deserialize TensorRT engine")
        input_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        output_names = [
            engine.get_tensor_name(i)
            for i in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        if len(input_names) != 1 or len(output_names) != 1:
            raise RuntimeError("detector requires one static input and one output")
        context = engine.create_execution_context()
        input_shape = tuple(context.get_tensor_shape(input_names[0]))
        output_shape = tuple(context.get_tensor_shape(output_names[0]))
        if len(input_shape) != 4 or any(size < 1 for size in input_shape):
            raise RuntimeError(f"engine input shape must be static NCHW, got {input_shape}")
        if any(size < 1 for size in output_shape):
            raise RuntimeError(f"engine output shape must be static, got {output_shape}")
        input_dtype = trt.nptype(engine.get_tensor_dtype(input_names[0]))
        if input_dtype != np.float32:
            raise RuntimeError(f"expected FP32 input engine, got {input_dtype}")
        cuda = _CudaRuntime()
        input_host = np.empty(input_shape, dtype=input_dtype)
        output_host = np.empty(output_shape, dtype=trt.nptype(engine.get_tensor_dtype(output_names[0])))
        input_device = cuda.malloc(input_host.nbytes)
        output_device = cuda.malloc(output_host.nbytes)
        stream = cuda.stream()
        if not context.set_tensor_address(input_names[0], int(input_device.value)):
            raise RuntimeError("could not bind TensorRT input buffer")
        if not context.set_tensor_address(output_names[0], int(output_device.value)):
            raise RuntimeError("could not bind TensorRT output buffer")
        self._trt, self._np, self._cuda = trt, np, cuda
        self._engine, self._context = engine, context
        self._input_name, self._output_name = input_names[0], output_names[0]
        self._input_shape, self._output_shape = input_shape, output_shape
        self._input_host, self._output_host = input_host, output_host
        self._input_device, self._output_device, self._stream = input_device, output_device, stream
        self._loaded = True

    def _execute(self) -> None:
        assert self._cuda is not None
        assert self._context is not None and self._stream is not None
        assert self._input_host is not None and self._output_host is not None
        assert self._input_device is not None and self._output_device is not None
        self._cuda.copy(self._input_device, self._input_host.ctypes.data, self._input_host.nbytes, self._cuda._HOST_TO_DEVICE)
        if not self._context.execute_async_v3(int(self._stream.value)):
            raise RuntimeError("TensorRT execution failed")
        self._cuda.copy(
            ctypes.c_void_p(self._output_host.ctypes.data),
            int(self._output_device.value),
            self._output_host.nbytes,
            self._cuda._DEVICE_TO_HOST,
        )
        self._cuda.synchronize(self._stream)

    def detect(self, frame_packet: FramePacket) -> list[Detection]:
        if frame_packet.image_bgr is None:
            raise ValueError("detector requires an in-memory BGR frame")
        self._load()
        np = self._np
        assert np is not None and self._input_host is not None and self._output_host is not None
        input_height, input_width = int(self._input_shape[2]), int(self._input_shape[3])
        frame_height, frame_width = frame_packet.image_bgr.shape[:2]
        started = time.perf_counter_ns()
        prep_start = time.perf_counter_ns()
        self._input_host[0] = _letterbox_bgr(frame_packet.image_bgr, input_height, input_width)
        prep_end = time.perf_counter_ns()
        input_hash = hashlib.sha256(self._input_host.tobytes()).hexdigest()
        infer_start = time.perf_counter_ns()
        self._execute()
        infer_end = time.perf_counter_ns()
        output_hash = hashlib.sha256(self._output_host.tobytes()).hexdigest()
        post_start = time.perf_counter_ns()
        decoded = _decode_yolox_output(self._output_host, input_height, input_width)
        boxes, scores = _person_detections(decoded, self.confidence_threshold, self.nms_iou_threshold)
        post_end = time.perf_counter_ns()
        mapping_start = time.perf_counter_ns()
        restored = _restore_boxes(boxes, frame_width, frame_height, input_width, input_height)
        mapping_end = time.perf_counter_ns()
        self.last_timing = DetectorTiming(
            preprocessing_ms=(prep_end - prep_start) / 1_000_000,
            inference_ms=(infer_end - infer_start) / 1_000_000,
            postprocessing_ms=(post_end - post_start) / 1_000_000,
            coordinate_mapping_ms=(mapping_end - mapping_start) / 1_000_000,
            total_ms=(mapping_end - started) / 1_000_000,
            input_sha256=input_hash,
            output_sha256=output_hash,
        )
        return [
            Detection(tuple(float(value) for value in box), float(score), 0, "person")
            for box, score in zip(restored.tolist(), scores.tolist())
        ]

    def close(self) -> None:
        if self._cuda is not None:
            if self._stream is not None:
                self._cuda._library.cudaStreamDestroy(self._stream)
            self._cuda.free(self._output_device)
            self._cuda.free(self._input_device)
        self._stream = self._output_device = self._input_device = None
        self._loaded = False


class FixedDetectionsDetector:
    """Small test detector used by lifecycle tests; it never stores frames."""

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = list(detections or [])
        self.calls = 0
        self.last_timing = None

    @property
    def metadata(self) -> dict[str, object]:
        return {"family": "fixed_test_detector", "runtime": "test"}

    def detect(self, frame_packet: FramePacket) -> list[Detection]:
        self.calls += 1
        return list(self.detections)

    def close(self) -> None:
        return None

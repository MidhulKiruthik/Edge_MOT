"""Bounded, no-save TensorRT detector feasibility check for G1."""

from __future__ import annotations

import ctypes
import math
import statistics
import time
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

from race_mot.stream_probe import _open_capture, redact_source


@dataclass(frozen=True, slots=True)
class DetectorSmokeConfig:
    engine_path: Path
    frames: int = 10
    confidence_threshold: float = 0.3
    nms_iou_threshold: float = 0.45

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("frames must be positive")
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0.0 < self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in (0, 1]")


class _CudaRuntime:
    """Minimal CUDA runtime bindings; avoids installing a Python CUDA package."""

    _HOST_TO_DEVICE = 1
    _DEVICE_TO_HOST = 2

    def __init__(self) -> None:
        self._library = ctypes.CDLL("libcudart.so")
        self._library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self._library.cudaMalloc.restype = ctypes.c_int
        self._library.cudaFree.argtypes = [ctypes.c_void_p]
        self._library.cudaFree.restype = ctypes.c_int
        self._library.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self._library.cudaMemcpy.restype = ctypes.c_int
        self._library.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._library.cudaStreamCreate.restype = ctypes.c_int
        self._library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self._library.cudaStreamDestroy.restype = ctypes.c_int
        self._library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._library.cudaStreamSynchronize.restype = ctypes.c_int

    def _check(self, result: int, operation: str) -> None:
        if result:
            raise RuntimeError(f"CUDA {operation} failed with status {result}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self._check(self._library.cudaMalloc(ctypes.byref(pointer), size), "allocation")
        return pointer

    def free(self, pointer: ctypes.c_void_p | None) -> None:
        if pointer is not None:
            self._check(self._library.cudaFree(pointer), "free")

    def copy(self, destination: ctypes.c_void_p, source: int, size: int, kind: int) -> None:
        self._check(self._library.cudaMemcpy(destination, ctypes.c_void_p(source), size, kind), "copy")

    def stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self._library.cudaStreamCreate(ctypes.byref(stream)), "stream creation")
        return stream

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self._check(self._library.cudaStreamSynchronize(stream), "stream synchronization")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1))]


def _letterbox_bgr(frame: object, input_height: int, input_width: int) -> object:
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = min(input_width / width, input_height / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)


def _person_detection_count(
    output: object, confidence_threshold: float, nms_iou_threshold: float
) -> tuple[int, float | None]:
    """Count class-zero YOLOX proposals after confidence filtering and NMS."""
    import numpy as np

    values = np.asarray(output)
    predictions = values.reshape(-1, values.shape[-1])
    if predictions.shape[1] < 6:
        raise RuntimeError("YOLOX output must contain box, objectness, and class scores")
    scores = predictions[:, 4] * predictions[:, 5]
    valid = (
        np.isfinite(predictions[:, :6]).all(axis=1)
        & (predictions[:, 2] > 0)
        & (predictions[:, 3] > 0)
        & (scores >= confidence_threshold)
    )
    selected = predictions[valid]
    selected_scores = scores[valid]
    if not len(selected):
        return 0, None
    boxes = np.column_stack(
        (
            selected[:, 0] - selected[:, 2] / 2,
            selected[:, 1] - selected[:, 3] / 2,
            selected[:, 0] + selected[:, 2] / 2,
            selected[:, 1] + selected[:, 3] / 2,
        )
    )
    order = np.argsort(selected_scores)[::-1]
    kept: list[int] = []
    while len(order):
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        left = np.maximum(boxes[current, 0], boxes[remaining, 0])
        top = np.maximum(boxes[current, 1], boxes[remaining, 1])
        right = np.minimum(boxes[current, 2], boxes[remaining, 2])
        bottom = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
        current_area = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        remaining_area = np.maximum(0, boxes[remaining, 2] - boxes[remaining, 0]) * np.maximum(
            0, boxes[remaining, 3] - boxes[remaining, 1]
        )
        union = current_area + remaining_area - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= nms_iou_threshold]
    return len(kept), float(selected_scores[kept].max())


def run_detector_smoke(source: str | Path, config: DetectorSmokeConfig) -> dict[str, object]:
    """Run a fixed number of in-memory frames through a static TensorRT engine."""
    if not config.engine_path.is_file() or config.engine_path.stat().st_size == 0:
        raise ValueError("engine path must name a non-empty TensorRT engine")
    try:
        import cv2
        import numpy as np
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("OpenCV, NumPy, and TensorRT Python are required for detector smoke") from exc

    sequence_dir = Path(source)
    if sequence_dir.is_dir():
        info = ConfigParser()
        info.read(sequence_dir / "seqinfo.ini", encoding="utf-8")
        if not info.has_section("Sequence"):
            raise ValueError("MOT sequence is missing valid seqinfo.ini metadata")
        metadata = info["Sequence"]
        try:
            sequence_length = int(metadata["seqLength"])
        except (KeyError, ValueError) as exc:
            raise ValueError("MOT sequence has invalid seqLength metadata") from exc
        image_dir = sequence_dir / metadata.get("imDir", "img1")
        extension = metadata.get("imExt", ".jpg")
        if not image_dir.is_dir():
            raise ValueError("MOT sequence is missing its image directory")
        sequence_index = 0

        def read_frame() -> tuple[bool, object | None]:
            nonlocal sequence_index
            if sequence_index >= sequence_length:
                return False, None
            sequence_index += 1
            frame = cv2.imread(str(image_dir / f"{sequence_index:06d}{extension}"), cv2.IMREAD_COLOR)
            return frame is not None, frame

        release = lambda: None
        backend = "opencv_imread"
        source_name = sequence_dir.name
        source_kind = "mot_sequence"
    else:
        capture, backend = _open_capture(cv2, str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("Could not open input for detector smoke")

        def read_frame() -> tuple[bool, object | None]:
            return capture.read()

        release = capture.release
        source_name = redact_source(str(source))
        source_kind = "video_or_stream"
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(config.engine_path.read_bytes())
    if engine is None:
        release()
        raise RuntimeError("could not deserialize TensorRT engine")
    input_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT]
    output_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT]
    if len(input_names) != 1 or len(output_names) != 1:
        release()
        raise RuntimeError("detector smoke supports exactly one engine input and output")
    context = engine.create_execution_context()
    input_name, output_name = input_names[0], output_names[0]
    input_shape = tuple(context.get_tensor_shape(input_name))
    output_shape = tuple(context.get_tensor_shape(output_name))
    if len(input_shape) != 4 or any(size < 1 for size in input_shape):
        release()
        raise RuntimeError(f"engine input shape must be static NCHW, got {input_shape}")
    if any(size < 1 for size in output_shape):
        release()
        raise RuntimeError(f"engine output shape must be static, got {output_shape}")
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))
    if input_dtype != np.float32:
        release()
        raise RuntimeError(f"expected FP32 input engine, got {input_dtype}")
    cuda = _CudaRuntime()
    input_host = np.empty(input_shape, dtype=input_dtype)
    output_host = np.empty(output_shape, dtype=output_dtype)
    input_device = output_device = stream = None
    inference_ms: list[float] = []
    frames_processed = 0
    finite_output_frames = 0
    person_counts: list[int] = []
    maximum_person_scores: list[float] = []
    stop_reason = "frame_limit"
    dimensions: tuple[int, int] | None = None
    try:
        input_device = cuda.malloc(input_host.nbytes)
        output_device = cuda.malloc(output_host.nbytes)
        stream = cuda.stream()
        if not context.set_tensor_address(input_name, int(input_device.value)):
            raise RuntimeError("could not bind TensorRT input buffer")
        if not context.set_tensor_address(output_name, int(output_device.value)):
            raise RuntimeError("could not bind TensorRT output buffer")
        while frames_processed < config.frames:
            ok, frame = read_frame()
            if not ok or frame is None:
                stop_reason = "read_failure"
                break
            dimensions = (int(frame.shape[1]), int(frame.shape[0]))
            input_host[0] = _letterbox_bgr(frame, input_shape[2], input_shape[3])
            started = time.perf_counter_ns()
            cuda.copy(input_device, input_host.ctypes.data, input_host.nbytes, cuda._HOST_TO_DEVICE)
            if not context.execute_async_v3(int(stream.value)):
                raise RuntimeError("TensorRT execution failed")
            cuda.copy(ctypes.c_void_p(output_host.ctypes.data), int(output_device.value), output_host.nbytes, cuda._DEVICE_TO_HOST)
            cuda.synchronize(stream)
            inference_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            finite_output_frames += int(bool(np.isfinite(output_host).all()))
            count, maximum_score = _person_detection_count(
                output_host,
                config.confidence_threshold,
                config.nms_iou_threshold,
            )
            person_counts.append(count)
            if maximum_score is not None:
                maximum_person_scores.append(maximum_score)
            frames_processed += 1
    finally:
        release()
        if stream is not None:
            cuda._library.cudaStreamDestroy(stream)
        cuda.free(output_device)
        cuda.free(input_device)
    if frames_processed == 0:
        raise RuntimeError("no frames were processed")
    return {
        "source": source_name,
        "source_kind": source_kind,
        "opencv_backend": backend,
        "engine_path": config.engine_path.name,
        "frames_requested": config.frames,
        "frames_processed": frames_processed,
        "stop_reason": stop_reason,
        "input_resolution": {"width": dimensions[0], "height": dimensions[1]} if dimensions else None,
        "engine": {"input": {"name": input_name, "shape": list(input_shape), "dtype": str(input_dtype)}, "output": {"name": output_name, "shape": list(output_shape), "dtype": str(output_dtype)}},
        "finite_output_frames": finite_output_frames,
        "person_postprocessing": {
            "class_id": 0,
            "confidence_threshold": config.confidence_threshold,
            "nms_iou_threshold": config.nms_iou_threshold,
            "counts_by_frame": person_counts,
            "count_min": min(person_counts),
            "count_median": statistics.median(person_counts),
            "count_max": max(person_counts),
            "maximum_score": max(maximum_person_scores) if maximum_person_scores else None,
        },
        "in_memory_only": True,
        "frames_saved": False,
        "inference_plus_transfer_ms": {"median": statistics.median(inference_ms), "p95": _percentile(inference_ms, 0.95)},
        "limitations": [
            "This is a bounded feasibility smoke check, not a detector-accuracy, tracking, latency, or energy measurement.",
            "The timing includes host-device transfers and is too short and uncontrolled for a performance claim.",
            "Person proposals are decoded and filtered, but they are not matched to ground truth in this smoke check.",
            "Person counts assume the official YOLOX ONNX export layout and are not an accuracy or recall result.",
            "No decoded or preprocessed frames are saved.",
        ],
    }

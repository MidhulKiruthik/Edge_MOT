"""Collect a reproducible, read-only inventory from a Jetson device."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def _command_output(args: Sequence[str], timeout: float = 8.0) -> dict[str, object]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"available": False, "reason": "command not installed"}
    try:
        result = subprocess.run(
            [executable, *args[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "available": True,
            "return_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"available": True, "reason": f"timed out after {timeout:g}s"}
    except OSError as exc:
        return {"available": True, "reason": str(exc)}


def collect_inventory() -> dict[str, object]:
    """Return hardware and software details without changing device state."""
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "device_tree_model": _read_text("/proc/device-tree/model"),
            "os_release": _read_text("/etc/os-release"),
            "jetson_release": _read_text("/etc/nv_tegra_release"),
            "meminfo": _read_text("/proc/meminfo"),
            "thermal_zones": _thermal_zones(),
            "hwmon_devices": _hwmon_devices(),
        },
        "software": {
            "jetpack_package": _command_output(["dpkg-query", "-W", "nvidia-jetpack"]),
            "tensorrt_packages": _command_output(["dpkg-query", "-W", "libnvinfer*"]),
            "power_mode": _command_output(["nvpmodel", "-q"]),
            "cuda_compiler": _command_output(["nvcc", "--version"]),
            "tegrastats_path": shutil.which("tegrastats"),
            "opencv_python": _opencv_version(),
        },
        "environment": {
            "cwd": os.getcwd(),
            "note": "This file contains system metadata; review it before sharing publicly.",
        },
    }


def _thermal_zones() -> dict[str, str | None]:
    zones: dict[str, str | None] = {}
    for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/type")):
        zones[path.parent.name] = _read_text(str(path))
    return zones


def _hwmon_devices() -> dict[str, str | None]:
    devices: dict[str, str | None] = {}
    for path in sorted(Path("/sys/class/hwmon").glob("hwmon*/name")):
        devices[path.parent.name] = _read_text(str(path))
    return devices


def _opencv_version() -> str | None:
    try:
        import cv2  # type: ignore[import-not-found]

        return str(cv2.__version__)
    except ImportError:
        return None


def write_inventory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(collect_inventory(), indent=2, ensure_ascii=False) + "\n")

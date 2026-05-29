from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
ANNOTATION_EXTENSIONS = {".json", ".npz", ".pkl", ".pickle", ".txt"}

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]

SWIMXYZ_TO_COCO_ALIASES = {
    "nose": {"nose", "head", "head_top", "headtop"},
    "left_eye": {"left_eye", "leye", "l_eye"},
    "right_eye": {"right_eye", "reye", "r_eye"},
    "left_ear": {"left_ear", "lear", "l_ear"},
    "right_ear": {"right_ear", "rear", "r_ear"},
    "left_shoulder": {"left_shoulder", "lshoulder", "l_shoulder", "leftshoulder"},
    "right_shoulder": {"right_shoulder", "rshoulder", "r_shoulder", "rightshoulder"},
    "left_elbow": {"left_elbow", "lelbow", "l_elbow", "leftelbow"},
    "right_elbow": {"right_elbow", "relbow", "r_elbow", "rightelbow"},
    "left_wrist": {"left_wrist", "lwrist", "l_wrist", "leftwrist", "left_hand", "lhand"},
    "right_wrist": {"right_wrist", "rwrist", "r_wrist", "rightwrist", "right_hand", "rhand"},
    "left_hip": {"left_hip", "lhip", "l_hip", "lefthip"},
    "right_hip": {"right_hip", "rhip", "r_hip", "righthip"},
    "left_knee": {"left_knee", "lknee", "l_knee", "leftknee"},
    "right_knee": {"right_knee", "rknee", "r_knee", "rightknee"},
    "left_ankle": {"left_ankle", "lankle", "l_ankle", "leftankle", "left_foot", "lfoot"},
    "right_ankle": {"right_ankle", "rankle", "r_ankle", "rightankle", "right_foot", "rfoot"},
}


@dataclass
class PoseRecord:
    annotation_file: str
    video_key: str
    frame_index: int
    keypoints2d: np.ndarray
    keypoints3d: np.ndarray | None = None
    joint_names: list[str] | None = None
    bbox: list[float] | None = None
    timestamp_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalized_key(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).replace("\\", "/").lower()
    if "/" in text:
        parts = text.split("/")
        if parts and "." in parts[-1]:
            parts[-1] = Path(parts[-1]).stem
        text = "/".join(parts)
    elif "." in Path(text).name:
        text = Path(text).stem
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def safe_stem(path: str | Path) -> str:
    source = str(path).replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_")[:180]


def find_video_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def find_annotation_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    output = []
    for path in root_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in ANNOTATION_EXTENSIONS:
            continue
        if path.suffix.lower() == ".txt" and not _is_supported_text_label(path):
            continue
        output.append(path)
    return sorted(output)


def load_annotation_file(path: str | Path) -> Any:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        return json.loads(source.read_text(encoding="utf-8"))
    if suffix == ".npz":
        with np.load(source, allow_pickle=True) as payload:
            return {key: _np_to_python(payload[key]) for key in payload.files}
    if suffix in {".pkl", ".pickle"}:
        with source.open("rb") as handle:
            return pickle.load(handle)
    if suffix == ".txt":
        return _load_swimxyz_text_label(source)
    raise ValueError(f"Unsupported annotation format: {source}")


def infer_stroke_view_from_path(path: str | Path, metadata: dict[str, Any] | None = None) -> tuple[str | None, str | None]:
    text = f"{path} {json.dumps(metadata or {}, default=str)}".lower()
    strokes = ["freestyle", "frontcrawl", "front_crawl", "butterfly", "breaststroke", "backstroke"]
    views = ["side", "front", "back", "underwater", "top", "above"]
    stroke = next((item for item in strokes if item in text), None)
    if stroke in {"frontcrawl", "front_crawl"}:
        stroke = "freestyle"
    view = next((item for item in views if item in text), None)
    if view == "above":
        view = "top"
    return stroke, view


def describe_schema(value: Any, *, max_depth: int = 4, max_items: int = 8) -> str:
    lines: list[str] = []

    def visit(node: Any, name: str, depth: int) -> None:
        indent = "  " * depth
        if isinstance(node, np.ndarray):
            lines.append(f"{indent}- {name}: ndarray shape={tuple(node.shape)} dtype={node.dtype}")
            return
        if isinstance(node, dict):
            lines.append(f"{indent}- {name}: dict keys={len(node)}")
            if depth >= max_depth:
                return
            for key in list(node.keys())[:max_items]:
                visit(node[key], str(key), depth + 1)
            return
        if isinstance(node, (list, tuple)):
            lines.append(f"{indent}- {name}: {type(node).__name__} len={len(node)}")
            if depth >= max_depth or not node:
                return
            visit(node[0], "[0]", depth + 1)
            return
        lines.append(f"{indent}- {name}: {type(node).__name__} value={_short(node)}")

    visit(value, "root", 0)
    return "\n".join(lines)


def find_candidate_arrays(value: Any, *, name_hint: str = "") -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, np.ndarray):
            if node.ndim >= 2 and np.issubdtype(node.dtype, np.number):
                candidates.append((path, node))
            return
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(node, (list, tuple)):
            if node and _looks_numeric_nested(node):
                try:
                    arr = np.asarray(node, dtype=float)
                    if arr.ndim >= 2:
                        candidates.append((path or "list", arr))
                except (TypeError, ValueError):
                    pass
            for index, child in enumerate(node[:12]):
                visit(child, f"{path}[{index}]")

    visit(value, name_hint)
    return candidates


def detect_joint_shapes(value: Any) -> tuple[list[tuple[str, tuple[int, ...]]], list[tuple[str, tuple[int, ...]]]]:
    arrays = find_candidate_arrays(value)
    joints2d = []
    joints3d = []
    for path, arr in arrays:
        if arr.ndim >= 2 and arr.shape[-1] in {2, 3, 4}:
            path_lower = path.lower()
            if "2d" not in path_lower and (arr.shape[-1] >= 3 or "3d" in path_lower):
                joints3d.append((path, tuple(arr.shape)))
            if arr.shape[-1] in {2, 3, 4} and ("2d" in path_lower or "joint" in path_lower or "keypoint" in path_lower or "pose" in path_lower):
                joints2d.append((path, tuple(arr.shape)))
    return joints2d, joints3d


def extract_pose_records(annotation: Any, annotation_file: str | Path) -> list[PoseRecord]:
    """Best-effort SwimXYZ parser that adapts to JSON/NPZ/pickle dictionary layouts."""

    if _looks_like_coco(annotation):
        return _records_from_coco(annotation, annotation_file)

    records = _records_from_sequence_list(annotation, annotation_file)
    if records:
        return records

    arrays = find_candidate_arrays(annotation)
    keypoints_path, keypoints = _select_2d_array(arrays)
    if keypoints is None:
        return []
    joints3d = _select_3d_array(arrays, frame_count=_first_dim(keypoints))
    names = _extract_joint_names(annotation)
    frame_indices = _extract_frame_indices(annotation, _first_dim(keypoints))
    timestamps = _extract_timestamps(annotation, _first_dim(keypoints))
    bboxes = _extract_bboxes(annotation, _first_dim(keypoints))
    video_key = normalized_key(_extract_video_name(annotation) or annotation_file)
    frame_keypoints = _reshape_frame_joint_array(keypoints)
    frame_joints3d = _reshape_frame_joint_array(joints3d) if joints3d is not None else None

    output = []
    for index, points in enumerate(frame_keypoints):
        output.append(
            PoseRecord(
                annotation_file=str(annotation_file),
                video_key=video_key,
                frame_index=int(frame_indices[index] if index < len(frame_indices) else index),
                timestamp_sec=float(timestamps[index]) if timestamps and index < len(timestamps) else None,
                keypoints2d=points,
                keypoints3d=frame_joints3d[index] if frame_joints3d is not None and index < len(frame_joints3d) else None,
                joint_names=names,
                bbox=bboxes[index] if bboxes and index < len(bboxes) else None,
                metadata={
                    "keypoints_path": keypoints_path,
                    **(annotation.get("metadata", {}) if isinstance(annotation, dict) and isinstance(annotation.get("metadata"), dict) else {}),
                },
            )
        )
    return output


def map_to_coco_keypoints(points: np.ndarray, joint_names: list[str] | None = None) -> tuple[list[float], int, list[str]]:
    warnings: list[str] = []
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return [0.0, 0.0, 0.0] * len(COCO_KEYPOINT_NAMES), 0, ["invalid keypoint array"]

    if joint_names:
        source_lookup = {normalized_key(name): index for index, name in enumerate(joint_names)}
        mapped = []
        for coco_name in COCO_KEYPOINT_NAMES:
            source_index = None
            for alias in SWIMXYZ_TO_COCO_ALIASES[coco_name]:
                source_index = source_lookup.get(normalized_key(alias))
                if source_index is not None:
                    break
            if source_index is None:
                mapped.append(None)
            else:
                mapped.append(arr[source_index])
        return _flatten_coco(mapped), sum(_visible(point) for point in mapped), warnings

    if len(arr) == len(COCO_KEYPOINT_NAMES):
        mapped_points = [arr[index] for index in range(len(COCO_KEYPOINT_NAMES))]
    elif len(arr) > len(COCO_KEYPOINT_NAMES):
        mapped_points = [arr[index] for index in range(len(COCO_KEYPOINT_NAMES))]
        warnings.append(f"source has {len(arr)} joints and no joint names; first 17 were interpreted as COCO order")
    else:
        mapped_points = [arr[index] if index < len(arr) else None for index in range(len(COCO_KEYPOINT_NAMES))]
        warnings.append(f"source has only {len(arr)} joints; missing COCO joints were set invisible")
    return _flatten_coco(mapped_points), sum(_visible(point) for point in mapped_points), warnings


def bbox_from_coco_keypoints(keypoints: list[float], *, padding: float = 8.0) -> list[float] | None:
    xs = []
    ys = []
    for index in range(0, len(keypoints), 3):
        if keypoints[index + 2] <= 0:
            continue
        xs.append(float(keypoints[index]))
        ys.append(float(keypoints[index + 1]))
    if not xs or not ys:
        return None
    x1 = min(xs) - padding
    y1 = min(ys) - padding
    x2 = max(xs) + padding
    y2 = max(ys) + padding
    return [round(x1, 3), round(y1, 3), round(max(1.0, x2 - x1), 3), round(max(1.0, y2 - y1), 3)]


def load_frame_manifest(frames_root: str | Path) -> list[dict[str, Any]]:
    root = Path(frames_root)
    manifest = root / "frame_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        return list(payload.get("frames", []))
    frames = []
    for metadata_path in root.rglob("_frame_metadata.json"):
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames.extend(payload.get("frames", []))
    return frames


def write_markdown(path: str | Path, title: str, sections: Iterable[tuple[str, str]]) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip() or "_No data._", ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _np_to_python(value: np.ndarray) -> Any:
    if value.dtype == object and value.shape == ():
        return value.item()
    return value


def _short(value: Any) -> str:
    text = str(value).replace("\n", " ")
    return text[:120] + ("..." if len(text) > 120 else "")


def _looks_numeric_nested(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    first = value[0]
    if isinstance(first, (int, float)):
        return True
    return _looks_numeric_nested(first)


def _looks_like_coco(value: Any) -> bool:
    return isinstance(value, dict) and {"images", "annotations", "categories"}.issubset(value.keys())


def _records_from_coco(annotation: dict[str, Any], annotation_file: str | Path) -> list[PoseRecord]:
    images = {int(image["id"]): image for image in annotation.get("images", []) if "id" in image}
    output = []
    for ann in annotation.get("annotations", []):
        image = images.get(int(ann.get("image_id", -1)))
        keypoints = ann.get("keypoints")
        if image is None or not keypoints:
            continue
        points = np.asarray(keypoints, dtype=float).reshape(-1, 3)
        output.append(
            PoseRecord(
                annotation_file=str(annotation_file),
                video_key=normalized_key(image.get("video") or image.get("file_name") or annotation_file),
                frame_index=int(image.get("frame_index", image.get("id", 0))),
                keypoints2d=points,
                joint_names=COCO_KEYPOINT_NAMES,
                bbox=ann.get("bbox"),
                metadata={"source_schema": "coco"},
            )
        )
    return output


def _records_from_sequence_list(annotation: Any, annotation_file: str | Path) -> list[PoseRecord]:
    if isinstance(annotation, dict):
        sequence_key = next((key for key in ["sequences", "videos", "clips", "samples", "annotations"] if isinstance(annotation.get(key), list)), None)
        if sequence_key is None:
            return []
        items = annotation[sequence_key]
    elif isinstance(annotation, list):
        items = annotation
    else:
        return []

    output: list[PoseRecord] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        arrays = find_candidate_arrays(item)
        _, keypoints = _select_2d_array(arrays)
        if keypoints is None:
            continue
        frame_points = _reshape_frame_joint_array(keypoints)
        frame_indices = _extract_frame_indices(item, len(frame_points))
        timestamps = _extract_timestamps(item, len(frame_points))
        bboxes = _extract_bboxes(item, len(frame_points))
        names = _extract_joint_names(item)
        video_key = normalized_key(_extract_video_name(item) or item.get("sequence") or item.get("id") or f"{annotation_file}_{item_index}")
        for index, points in enumerate(frame_points):
            output.append(
                PoseRecord(
                    annotation_file=str(annotation_file),
                    video_key=video_key,
                    frame_index=int(frame_indices[index] if index < len(frame_indices) else index),
                    timestamp_sec=float(timestamps[index]) if timestamps and index < len(timestamps) else None,
                    keypoints2d=points,
                    joint_names=names,
                    bbox=bboxes[index] if bboxes and index < len(bboxes) else None,
                    metadata={
                        "source_item": item_index,
                        **(item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}),
                    },
                )
            )
    return output


def _select_2d_array(arrays: list[tuple[str, np.ndarray]]) -> tuple[str, np.ndarray | None]:
    scored = []
    for path, arr in arrays:
        if arr.ndim < 2 or arr.shape[-1] not in {2, 3, 4}:
            continue
        lower = path.lower()
        score = 0
        if "2d" in lower:
            score += 8
        if "keypoint" in lower or "joint" in lower:
            score += 5
        if "3d" in lower or "smpl" in lower:
            score -= 10
        if arr.shape[-1] in {2, 3, 4}:
            score += 2
        scored.append((score, path, arr))
    if not scored:
        return "", None
    _, path, arr = max(scored, key=lambda item: item[0])
    return path, np.asarray(arr, dtype=float)


def _select_3d_array(arrays: list[tuple[str, np.ndarray]], *, frame_count: int) -> np.ndarray | None:
    scored = []
    for path, arr in arrays:
        if arr.ndim < 2 or arr.shape[-1] not in {3, 4}:
            continue
        lower = path.lower()
        if "2d" in lower:
            continue
        score = 0
        if "3d" in lower:
            score += 8
        if "joint" in lower or "keypoint" in lower:
            score += 4
        if _first_dim(arr) == frame_count:
            score += 3
        scored.append((score, arr))
    return np.asarray(max(scored, key=lambda item: item[0])[1], dtype=float) if scored else None


def _reshape_frame_joint_array(arr: np.ndarray | None) -> list[np.ndarray]:
    if arr is None:
        return []
    value = np.asarray(arr, dtype=float)
    if value.ndim == 2:
        return [value]
    if value.ndim == 3:
        return [value[index] for index in range(value.shape[0])]
    if value.ndim >= 4:
        if value.shape[1] <= 4:
            value = value[:, 0, :, :]
        else:
            value = value.reshape((-1, value.shape[-2], value.shape[-1]))
        return [value[index] for index in range(value.shape[0])]
    return []


def _first_dim(arr: np.ndarray) -> int:
    return int(arr.shape[0]) if arr.ndim >= 3 else 1


def _extract_joint_names(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        for key in ["joint_names", "joints_name", "keypoint_names", "kp_names", "skeleton"]:
            names = value.get(key)
            if isinstance(names, (list, tuple)) and all(isinstance(item, str) for item in names):
                return list(names)
        for child in value.values():
            found = _extract_joint_names(child)
            if found:
                return found
    return None


def _extract_video_name(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ["video", "video_path", "video_name", "file_name", "filename", "sequence_name", "seq_name", "clip_name"]:
            candidate = value.get(key)
            if isinstance(candidate, (str, Path)):
                return str(candidate)
        for child in value.values():
            found = _extract_video_name(child)
            if found:
                return found
    return None


def _extract_frame_indices(value: Any, count: int) -> list[int]:
    arrays = _find_named_numeric_arrays(value, {"frame", "frame_idx", "frame_index", "indices", "image_id"})
    for arr in arrays:
        flat = np.asarray(arr).reshape(-1)
        if len(flat) >= count:
            return [int(x) for x in flat[:count]]
    return list(range(count))


def _extract_timestamps(value: Any, count: int) -> list[float] | None:
    arrays = _find_named_numeric_arrays(value, {"timestamp", "timestamps", "time", "times", "sec"})
    for arr in arrays:
        flat = np.asarray(arr, dtype=float).reshape(-1)
        if len(flat) >= count:
            return [float(x) for x in flat[:count]]
    return None


def _extract_bboxes(value: Any, count: int) -> list[list[float]] | None:
    arrays = _find_named_numeric_arrays(value, {"bbox", "bboxes", "box", "boxes"})
    for arr in arrays:
        value_arr = np.asarray(arr, dtype=float)
        if value_arr.ndim == 1 and value_arr.shape[0] >= 4:
            return [[float(x) for x in value_arr[:4]] for _ in range(count)]
        if value_arr.ndim >= 2 and value_arr.shape[-1] >= 4 and value_arr.shape[0] >= count:
            return [[float(x) for x in value_arr[index, :4]] for index in range(count)]
    return None


def _find_named_numeric_arrays(value: Any, keys: set[str]) -> list[np.ndarray]:
    output = []

    def visit(node: Any, name: str) -> None:
        normalized = normalized_key(name)
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, str(key))
            return
        if not any(key in normalized for key in keys):
            return
        try:
            arr = np.asarray(node)
        except (TypeError, ValueError):
            return
        if arr.size and np.issubdtype(arr.dtype, np.number):
            output.append(arr)

    visit(value, "")
    return output


def _flatten_coco(points: list[np.ndarray | None]) -> list[float]:
    output: list[float] = []
    for point in points:
        if not _visible(point):
            output.extend([0.0, 0.0, 0.0])
            continue
        assert point is not None
        visibility = _visibility_value(point)
        output.extend([round(float(point[0]), 3), round(float(point[1]), 3), float(visibility)])
    return output


def _visible(point: np.ndarray | None) -> bool:
    if point is None or len(point) < 2:
        return False
    x = float(point[0])
    y = float(point[1])
    if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
        return False
    if len(point) >= 3 and float(point[2]) <= 0:
        return False
    return True


def _visibility_value(point: np.ndarray) -> int:
    if len(point) < 3:
        return 2
    score = float(point[2])
    if score <= 0:
        return 0
    return 2 if score >= 0.5 else 1


def _is_supported_text_label(path: Path) -> bool:
    """Limit text parsing to SwimXYZ camera-space COCO 2D label files."""

    return path.name.lower() == "2d_cam.txt" and path.parent.name.lower() == "coco"


def _load_swimxyz_text_label(path: Path) -> dict[str, Any]:
    """Parse SwimXYZ semicolon-separated COCO text labels.

    SwimXYZ label files use headers such as ``Nose.x;Nose.y;Nose.z`` and decimal
    commas in the values. The third value is kept as metadata/visibility support,
    while COCO export later maps finite positive points to visible keypoints.
    """

    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError(f"empty text label file: {path}")
    headers = [part.strip() for part in lines[0].split(";") if part.strip()]
    if len(headers) < 6 or len(headers) % 3 != 0:
        raise ValueError(f"unexpected SwimXYZ text label header: {path}")

    joint_names = []
    for index in range(0, len(headers), 3):
        name = headers[index].split(".", 1)[0]
        joint_names.append(name)

    frames = []
    for row_index, line in enumerate(lines[1:]):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(";") if part.strip()]
        if len(parts) < 6:
            continue
        if len(parts) < len(headers):
            parts.extend(["nan"] * (len(headers) - len(parts)))
        values = [_parse_decimal_number(part) for part in parts[: len(headers)]]
        frames.append(np.asarray(values, dtype=float).reshape(len(joint_names), 3))
    if not frames:
        raise ValueError(f"no frame rows parsed from text label file: {path}")

    return {
        "video_name": _video_key_from_text_label_path(path),
        "joint_names": joint_names,
        "frame_indices": list(range(len(frames))),
        "keypoints2d": np.stack(frames, axis=0),
        "metadata": {
            "source_schema": "swimxyz_text_label",
            "source_file": str(path),
            "coordinate_origin": "bottom_left",
            "coordinate_space": "pixel",
        },
    }


def _parse_decimal_number(value: str) -> float:
    text = value.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _video_key_from_text_label_path(path: Path) -> str:
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    start = next((index for index, part in enumerate(lowered) if part.startswith("side_")), None)
    if start is None:
        start = next((index + 1 for index, part in enumerate(lowered) if part in {"freestyle", "backstroke", "breaststroke", "butterfly"}), None)
    end = next((index for index, part in enumerate(lowered) if part == "coco"), len(parts) - 1)
    if start is None or start >= end:
        return str(path.with_suffix(""))
    return "/".join(parts[start:end])

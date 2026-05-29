from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.swim_analysis.pose.base import (
    KEYPOINT_NAMES,
    Keypoint,
    PoseBackendUnavailable,
    PoseEstimator,
    PoseFrameResult,
    average_keypoint_confidence,
    bbox_from_keypoints,
    empty_pose_result,
)


logger = logging.getLogger(__name__)
_UNSET = object()


class RTMPoseBackend(PoseEstimator):
    name = "rtmpose"

    def __init__(
        self,
        config_path: str | None | object = _UNSET,
        checkpoint_path: str | None | object = _UNSET,
        device: str | None | object = _UNSET,
        confidence_threshold: float | None | object = _UNSET,
        keypoint_names: list[str] | None = None,
    ) -> None:
        settings = get_settings()
        self.config_path = settings.rtmpose_config_path if config_path is _UNSET else config_path
        self.checkpoint_path = settings.rtmpose_checkpoint_path if checkpoint_path is _UNSET else checkpoint_path
        self.device = settings.pose_device if device is _UNSET else device
        threshold = settings.pose_confidence_threshold if confidence_threshold is _UNSET else confidence_threshold
        self.confidence_threshold = max(0.0, min(1.0, float(threshold if threshold is not None else 0.05)))
        self.keypoint_names = keypoint_names or _keypoint_names_from_settings(settings.rtmpose_keypoint_names)
        self.model: Any | None = None
        self._inference_topdown: Any | None = None
        self.diagnostics: dict[str, Any] = {
            "requested_backend": self.name,
            "actual_backend": self.name,
            "confidence_threshold": self.confidence_threshold,
            "keypoint_names": self.keypoint_names,
            "loaded": False,
            "warnings": [],
        }

    def load(self) -> None:
        config, checkpoint = resolve_rtmpose_paths(self.config_path, self.checkpoint_path)
        self.diagnostics.update(
            {
                "config_path": str(config),
                "checkpoint_path": str(checkpoint),
                "config_exists": config.exists(),
                "checkpoint_exists": checkpoint.exists(),
                "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
                "device": self.device,
            }
        )
        if len(self.keypoint_names) != len(KEYPOINT_NAMES):
            raise PoseBackendUnavailable(
                f"RTMPose keypoint mapping must contain {len(KEYPOINT_NAMES)} names, got {len(self.keypoint_names)}."
            )

        try:
            from mmpose.apis import inference_topdown, init_model  # type: ignore
        except ImportError as exc:
            raise PoseBackendUnavailable(f"MMPose is not installed or not importable: {exc}") from exc

        try:
            self.model = init_model(str(config), str(checkpoint), device=str(self.device))
            self._inference_topdown = inference_topdown
        except Exception as exc:
            raise PoseBackendUnavailable(f"Could not initialize RTMPose on {self.device}: {exc}") from exc

        self._validate_loaded_model()
        self.diagnostics["loaded"] = True
        logger.info(
            "RTMPose loaded: config=%s checkpoint=%s device=%s threshold=%.3f keypoints=%s",
            config,
            checkpoint,
            self.device,
            self.confidence_threshold,
            len(self.keypoint_names),
        )

    def infer_frame(
        self,
        frame: Any,
        *,
        frame_index: int = 0,
        timestamp: float = 0.0,
        view_type: str = "side",
        quality_flags: list[str] | None = None,
    ) -> PoseFrameResult:
        if self.model is None or self._inference_topdown is None:
            self.load()

        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise PoseBackendUnavailable(f"NumPy is required for RTMPose inference: {exc}") from exc

        height, width = frame.shape[:2]
        full_frame_bbox = np.array([[0, 0, width, height]], dtype=np.float32)
        try:
            predictions = self._inference_topdown(self.model, frame, bboxes=full_frame_bbox)
        except TypeError:
            predictions = self._inference_topdown(self.model, frame, full_frame_bbox)
        except Exception as exc:
            logger.exception("RTMPose inference failed at frame=%s timestamp=%.3f", frame_index, timestamp)
            raise PoseBackendUnavailable(f"RTMPose inference failed at frame {frame_index}: {exc}") from exc

        result = self._prediction_to_result(
            predictions,
            frame_index=frame_index,
            timestamp=timestamp,
            view_type=view_type,
            quality_flags=quality_flags or [],
        )
        raw_count = len(result.raw_keypoints or result.keypoints)
        filtered_count = len(result.keypoints)
        logger.info(
            "pose_frame backend=rtmpose frame=%s time=%.3f raw_joints=%s filtered_joints=%s avg_conf=%.3f removed_by_threshold=%s",
            frame_index,
            timestamp,
            raw_count,
            filtered_count,
            result.confidence,
            max(0, raw_count - filtered_count),
        )
        return result

    def _prediction_to_result(
        self,
        predictions: Any,
        *,
        frame_index: int,
        timestamp: float,
        view_type: str,
        quality_flags: list[str],
    ) -> PoseFrameResult:
        if not predictions:
            return _empty_with_debug(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
                debug_info={"reason": "rtmpose_no_predictions"},
            )

        best = max(predictions, key=_prediction_area)
        pred_instances = getattr(best, "pred_instances", None)
        if pred_instances is None:
            return _empty_with_debug(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
                debug_info={"reason": "rtmpose_missing_pred_instances"},
            )

        keypoints_raw = getattr(pred_instances, "keypoints", None)
        scores_raw = getattr(pred_instances, "keypoint_scores", None)
        if keypoints_raw is None:
            return _empty_with_debug(
                frame_index=frame_index,
                timestamp=timestamp,
                view_type=view_type,
                quality_flags=quality_flags,
                backend=self.name,
                debug_info={"reason": "rtmpose_missing_keypoints"},
            )

        keypoints = _as_first_pose(keypoints_raw)
        scores = _as_first_score_vector(scores_raw) if scores_raw is not None else [1.0] * len(keypoints)
        raw_debug = raw_keypoint_debug(keypoints, scores, self.keypoint_names)
        raw_points = [
            Keypoint(
                name=item["mapped_joint"],
                x=float(item["x"]),
                y=float(item["y"]),
                confidence=float(item["score"]),
                source_backend=self.name,
                frame_index=frame_index,
            )
            for item in raw_debug
            if item["mapped_joint"]
        ]
        filtered = [
            point
            for point in raw_points
            if point.confidence >= self.confidence_threshold
        ]
        bbox = _extract_bbox(pred_instances) or bbox_from_keypoints(filtered, min_confidence=self.confidence_threshold) or bbox_from_keypoints(raw_points, min_confidence=0.0)
        removed = max(0, len(raw_points) - len(filtered))
        body_raw = sum(1 for point in raw_points if _is_body_joint(point.name))
        body_filtered = sum(1 for point in filtered if _is_body_joint(point.name))
        debug_info = {
            "backend": self.name,
            "raw_keypoints_by_index": raw_debug,
            "raw_joint_count": len(raw_points),
            "filtered_joint_count": len(filtered),
            "removed_by_threshold": removed,
            "confidence_threshold": self.confidence_threshold,
            "raw_body_joint_count": body_raw,
            "filtered_body_joint_count": body_filtered,
            "keypoint_count_matches_coco17": len(raw_points) == len(KEYPOINT_NAMES),
            "bbox": bbox,
        }
        events = []
        flags = list(quality_flags)
        if len(raw_points) != len(KEYPOINT_NAMES):
            events.append(f"RTMPose returned {len(raw_points)} mapped keypoints, expected {len(KEYPOINT_NAMES)}")
            flags.append("rtmpose_keypoint_count_mismatch")
        if removed >= 10:
            events.append(f"RTMPose threshold removed {removed}/{len(raw_points)} joints")
            flags.append("rtmpose_threshold_removed_many_joints")
        if body_raw and body_filtered / max(1, body_raw) < 0.35:
            events.append("RTMPose filtering removed most body joints")
            flags.append("rtmpose_body_joints_low_confidence")

        return PoseFrameResult(
            timestamp=timestamp,
            frame_index=frame_index,
            keypoints=filtered,
            raw_keypoints=raw_points,
            confidence=average_keypoint_confidence(filtered),
            bbox=bbox,
            view_type=view_type,
            quality_flags=flags,
            backend=self.name,
            debug_events=events,
            debug_info=debug_info,
        )

    def _validate_loaded_model(self) -> None:
        expected = len(self.keypoint_names)
        cfg = getattr(self.model, "cfg", None)
        model = getattr(self.model, "module", self.model)
        head = getattr(model, "head", None)
        out_channels = getattr(head, "out_channels", None)
        dataset_meta = getattr(self.model, "dataset_meta", None) or getattr(model, "dataset_meta", None)
        cfg_out_channels = _cfg_get(cfg, ["model", "head", "out_channels"])
        input_size = _cfg_get(cfg, ["codec", "input_size"]) or _cfg_get(cfg, ["model", "head", "input_size"])
        self.diagnostics.update(
            {
                "model_head_out_channels": out_channels,
                "config_head_out_channels": cfg_out_channels,
                "config_input_size": list(input_size) if isinstance(input_size, (list, tuple)) else input_size,
                "dataset_meta_keys": sorted(list(dataset_meta.keys())) if isinstance(dataset_meta, dict) else [],
            }
        )
        channel_count = _first_int(out_channels, cfg_out_channels)
        if channel_count is not None and channel_count != expected:
            raise PoseBackendUnavailable(f"RTMPose config/checkpoint keypoint count mismatch: model has {channel_count}, mapping expects {expected}.")
        meta_keypoints = _dataset_meta_keypoints(dataset_meta)
        if meta_keypoints:
            self.diagnostics["dataset_meta_keypoints"] = meta_keypoints
            if len(meta_keypoints) != expected:
                raise PoseBackendUnavailable(f"RTMPose dataset metadata has {len(meta_keypoints)} keypoints, expected {expected}.")
            if meta_keypoints != self.keypoint_names:
                message = "RTMPose dataset metadata keypoint order differs from configured mapping."
                self.diagnostics.setdefault("warnings", []).append(message)
                logger.warning("%s metadata=%s configured=%s", message, meta_keypoints, self.keypoint_names)


def resolve_rtmpose_paths(config_path: str | None | object, checkpoint_path: str | None | object) -> tuple[Path, Path]:
    if not config_path or not checkpoint_path:
        raise PoseBackendUnavailable(
            "RTMPose requires RTMPOSE_CONFIG_PATH and RTMPOSE_CHECKPOINT_PATH. "
            "Set both variables before running inference."
        )
    config = Path(str(config_path)).expanduser().resolve()
    checkpoint = Path(str(checkpoint_path)).expanduser().resolve()
    if not config.exists():
        raise PoseBackendUnavailable(f"RTMPose config file does not exist: {config}")
    if not checkpoint.exists():
        raise PoseBackendUnavailable(f"RTMPose checkpoint file does not exist: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise PoseBackendUnavailable(f"RTMPose checkpoint is empty: {checkpoint}")
    return config, checkpoint


def raw_keypoint_debug(keypoints: Any, scores: Any, keypoint_names: list[str] | None = None) -> list[dict[str, Any]]:
    names = keypoint_names or KEYPOINT_NAMES
    points = _as_first_pose(keypoints)
    score_values = _as_first_score_vector(scores) if scores is not None else [1.0] * len(points)
    output = []
    for index, point in enumerate(points):
        mapped = names[index] if index < len(names) else None
        x = point[0] if len(point) > 0 else None
        y = point[1] if len(point) > 1 else None
        output.append(
            {
                "raw_index": index,
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "score": _score_at(score_values, index),
                "mapped_joint": mapped,
            }
        )
    return output


def _keypoint_names_from_settings(value: str | None) -> list[str]:
    if not value:
        return list(KEYPOINT_NAMES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    return names or list(KEYPOINT_NAMES)


def _empty_with_debug(
    *,
    frame_index: int,
    timestamp: float,
    view_type: str,
    quality_flags: list[str],
    backend: str,
    debug_info: dict[str, Any],
) -> PoseFrameResult:
    result = empty_pose_result(frame_index=frame_index, timestamp=timestamp, view_type=view_type, quality_flags=quality_flags, backend=backend)
    return PoseFrameResult(**{**result.__dict__, "debug_info": debug_info, "debug_events": [str(debug_info.get("reason", "empty_pose"))]})


def _as_first_pose(value: Any) -> Any:
    value = _to_python_list(value)
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], (list, tuple)):
        return value[0]
    return value


def _as_first_score_vector(value: Any) -> list[float]:
    value = _to_python_list(value)
    if not value:
        return []
    if isinstance(value[0], list):
        first = value[0]
        if first and isinstance(first[0], (list, tuple)):
            return [_score_value(item) for item in first]
        return [_score_value(item) for item in first]
    return [_score_value(item) for item in value]


def _score_at(scores: list[float], index: int) -> float:
    if index >= len(scores):
        return 1.0
    return _score_value(scores[index])


def _score_value(value: Any) -> float:
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_python_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _extract_bbox(pred_instances: Any) -> list[float] | None:
    bboxes = getattr(pred_instances, "bboxes", None)
    if bboxes is None:
        return None
    if hasattr(bboxes, "detach"):
        bboxes = bboxes.detach().cpu().numpy()
    elif hasattr(bboxes, "cpu"):
        bboxes = bboxes.cpu().numpy()
    if hasattr(bboxes, "tolist"):
        bboxes = bboxes.tolist()
    if not bboxes:
        return None
    bbox = bboxes[0] if isinstance(bboxes[0], list) else bboxes
    return [float(value) for value in bbox[:4]]


def _prediction_area(prediction: Any) -> float:
    pred_instances = getattr(prediction, "pred_instances", None)
    if pred_instances is None:
        return 0.0
    bbox = _extract_bbox(pred_instances)
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _cfg_get(value: Any, path: list[str]) -> Any:
    current = value
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            source = current
            current = getattr(source, key, None)
            if current is None and hasattr(source, "get"):
                current = source.get(key)
    return current


def _dataset_meta_keypoints(dataset_meta: Any) -> list[str]:
    if not isinstance(dataset_meta, dict):
        return []
    names = dataset_meta.get("keypoint_id2name") or dataset_meta.get("keypoint_name2id") or dataset_meta.get("keypoint_names")
    if isinstance(names, dict):
        if all(isinstance(key, int) for key in names):
            return [str(names[index]) for index in sorted(names)]
        if all(str(key).isdigit() for key in names):
            return [str(names[key]) for key in sorted(names, key=lambda item: int(str(item)))]
        return [str(name) for name, _ in sorted(names.items(), key=lambda item: item[1])]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _first_int(*values: Any) -> int | None:
    for value in values:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_body_joint(name: str) -> bool:
    return name not in {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}

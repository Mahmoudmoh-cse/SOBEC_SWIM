from __future__ import annotations

from abc import ABC, abstractmethod
from math import acos, degrees
from pathlib import Path
from statistics import mean
from typing import Any

from app.core.config import get_settings
from app.models import Session, Swimmer, UploadedFile
from app.providers.coaching import CoachingTextGenerator, get_coaching_generator
from app.services.calculations import clamp_score

NO_POSE_MESSAGE = "No clear swimmer pose detected. Upload a side-view video where the full body is visible."
LOW_CONFIDENCE_MESSAGE = "Low confidence. Camera angle, water reflection, or body occlusion may affect analysis accuracy."
POOR_ANGLE_MESSAGE = "Video angle likely poor. Use a stable side-view or deck-level angle with the full swimmer visible."
VIDEO_TOO_SHORT_MESSAGE = "Video is very short. Upload at least 5 seconds of clear swimming footage for a more reliable report."
INVALID_VIDEO_MESSAGE = "Invalid video file. Upload a playable MP4 or MOV file recorded from a phone or camera."
PROCESSING_ERROR_MESSAGE = "Processing error. The video was saved, but AquaIQ could not complete pose analysis."


class TechniqueAnalyzer(ABC):
    @abstractmethod
    def analyze(self, swimmer: Swimmer, session: Session, uploaded_file: UploadedFile) -> dict[str, Any]:
        raise NotImplementedError


class MockTechniqueAnalyzer(TechniqueAnalyzer):
    def __init__(self, coaching: CoachingTextGenerator | None = None, provider_note: str | None = None) -> None:
        self.coaching = coaching or get_coaching_generator()
        self.provider_note = provider_note

    def analyze(self, swimmer: Swimmer, session: Session, uploaded_file: UploadedFile) -> dict[str, Any]:
        fatigue_penalty = max(0, session.rpe - 6) * 3
        distance_bonus = min(8, session.distance_m / 1000)
        recovery_bonus = max(0, session.mood_recovery - 5) * 1.5
        score = clamp_score(78 + distance_bonus + recovery_bonus - fatigue_penalty, 45, 96)

        faults = self._faults_for(swimmer.primary_stroke, session.rpe)
        top_fault = faults[0]["title"]
        time_gain = round((100 - score) / 60, 1)
        drills = self._drills_for(swimmer.primary_stroke, top_fault)
        trust = build_analysis_trust(
            frames_total=12,
            frames_analyzed=12,
            pose_detected_frames=12,
            average_visibility=0.82,
            demo=True,
        )

        return {
            "stroke": swimmer.primary_stroke,
            "overall_score": score,
            "dps_meters": round(1.55 + (score / 100) * 0.55, 2),
            "stroke_rate": clamp_score(44 - (score - 75) / 3, 28, 54),
            "faults": faults,
            "drill_prescriptions": drills,
            "keypoint_data": {
                "provider": "mock",
                "provider_note": self.provider_note,
                "source_file": uploaded_file.stored_path,
                "frames_sampled": trust["frames_analyzed"],
                "pose_frames_detected": trust["pose_detected_frames"],
                "confidence": trust["confidence_score"],
                "time_gain_possible": time_gain,
                "analysis_quality": "demo",
                "body_heatmap": _body_heatmap(faults),
                "metrics": {
                    "elbow_angle_mean": 151,
                    "hip_rotation_proxy": 37,
                    "kick_amplitude_proxy": 0.48,
                },
            },
            "processing_status": "complete",
            **trust,
            "analysis_overlay_video_url": None,
            "analysis_frame_urls": [],
            "analysis_events": [
                {
                    "timestamp_s": 0,
                    "type": "mock_analysis",
                    "label": "Mock analysis",
                    "message": "Deterministic demo report generated without MediaPipe video processing.",
                }
            ],
            "coaching_summary": self.coaching.technique_summary(swimmer.name, top_fault, score, time_gain),
        }

    def _faults_for(self, stroke: str, rpe: int) -> list[dict[str, Any]]:
        if stroke.lower() == "freestyle":
            return [
                {
                    "zone": "arms",
                    "severity": "critical" if rpe >= 7 else "needs_work",
                    "title": "Dropped elbow on catch",
                    "description": "Early vertical forearm lost - you are pushing water sideways, not back.",
                    "time_cost_seconds": 0.3,
                },
                {
                    "zone": "legs",
                    "severity": "critical" if rpe >= 8 else "needs_work",
                    "title": "Cross-body kick",
                    "description": "Left leg crosses centerline at peak kick. Creates drag and destabilizes rotation.",
                    "time_cost_seconds": 0.24,
                },
                {
                    "zone": "arms",
                    "severity": "needs_work",
                    "title": "Short front extension",
                    "description": "Hand entry too close to head. Missing 8-10cm of leverage on the pull phase.",
                    "time_cost_seconds": 0.18,
                },
                {
                    "zone": "legs",
                    "severity": "needs_work",
                    "title": "Kick depth too wide",
                    "description": "Kick amplitude sits above the target range. This slows stroke rate and wastes energy.",
                    "time_cost_seconds": 0.22,
                },
            ]

        stroke_faults = {
            "backstroke": ("Late hip rotation", "hips", "Start rotation before the recovering hand passes the shoulder."),
            "breaststroke": ("Wide kick recovery", "legs", "Recover the heels behind the hips before turning out."),
            "butterfly": ("Low second kick timing", "legs", "Connect the second kick to the hand exit."),
        }
        title, zone, description = stroke_faults.get(
            stroke.lower(),
            ("Stroke length drops under fatigue", "torso", "Protect distance per stroke late in the repeat."),
        )
        return [
            {
                "zone": zone,
                "severity": "critical" if rpe >= 8 else "needs_work",
                "title": title,
                "description": description,
                "time_cost_seconds": 0.35 if rpe < 8 else 0.55,
            },
            {
                "zone": "torso",
                "severity": "good",
                "title": "Body line stable",
                "description": "Alignment stays controlled enough to build from.",
                "time_cost_seconds": 0,
            },
        ]

    def _drills_for(self, stroke: str, top_fault: str) -> list[dict[str, Any]]:
        if stroke.lower() == "freestyle" or "Dropped elbow" in top_fault:
            return [
                {"name": "Fingertip drag", "focus": "Elbow position", "volume": "4x50m", "sets": 1},
                {"name": "Kickboard with fins", "focus": "Kick amplitude", "volume": "6x25m", "sets": 1},
                {"name": "Catch-up drill", "focus": "Front extension", "volume": "4x50m", "sets": 1},
            ]
        if "rotation" in top_fault:
            return [{"name": "6-kick switch", "focus": "Hip-led rotation", "volume": "8x25m", "sets": 1}]
        return [{"name": "Tempo contrast 25s", "focus": "Hold line under rhythm change", "volume": "8x25m", "sets": 1}]


def build_analysis_trust(
    *,
    frames_total: int,
    frames_analyzed: int,
    pose_detected_frames: int,
    average_visibility: float = 0,
    processing_error: str | None = None,
    demo: bool = False,
) -> dict[str, Any]:
    frames_total = max(0, int(frames_total or 0))
    frames_analyzed = max(0, int(frames_analyzed or 0))
    pose_detected_frames = max(0, min(int(pose_detected_frames or 0), frames_analyzed))
    pose_detection_rate = round(pose_detected_frames / frames_analyzed, 2) if frames_analyzed else 0
    visibility = max(0, min(1, float(average_visibility or 0)))
    confidence_score = round((pose_detection_rate * 0.65) + (visibility * 0.35), 2) if pose_detected_frames else 0

    if confidence_score >= 0.75:
        confidence_label = "high"
    elif confidence_score >= 0.45:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    analysis_status = "completed"
    analysis_warning = None
    analysis_error = None

    if processing_error:
        analysis_status = "failed"
        confidence_score = 0
        confidence_label = "low"
        analysis_error = processing_error
    elif pose_detected_frames == 0:
        analysis_status = "no_pose_detected"
        analysis_warning = NO_POSE_MESSAGE
    elif confidence_label == "low":
        analysis_status = "low_confidence"
        analysis_warning = POOR_ANGLE_MESSAGE if pose_detection_rate < 0.35 else LOW_CONFIDENCE_MESSAGE
    elif not demo and (frames_total < 90 or frames_analyzed < 5):
        analysis_warning = VIDEO_TOO_SHORT_MESSAGE

    return {
        "analysis_status": analysis_status,
        "frames_total": frames_total,
        "frames_analyzed": frames_analyzed,
        "pose_detected_frames": pose_detected_frames,
        "pose_detection_rate": pose_detection_rate,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "analysis_warning": analysis_warning,
        "analysis_error": analysis_error,
    }


class MediaPipeTechniqueAnalyzer(TechniqueAnalyzer):
    def __init__(self, coaching: CoachingTextGenerator | None = None) -> None:
        self.coaching = coaching or get_coaching_generator()
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"MediaPipe analyzer is unavailable: {exc}") from exc

        self.cv2 = cv2
        self.mp = mp
        self.np = np
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def analyze(self, swimmer: Swimmer, session: Session, uploaded_file: UploadedFile) -> dict[str, Any]:
        video_path = Path(uploaded_file.stored_path)
        cap = self.cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self._failed_report(swimmer, uploaded_file, INVALID_VIDEO_MESSAGE)

        fps = float(cap.get(self.cv2.CAP_PROP_FPS) or 30)
        total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_every = max(1, total_frames // 90) if total_frames else 5

        frame_index = 0
        frames_sampled = 0
        pose_frames_detected = 0
        confidence_values: list[float] = []
        elbow_angles: list[float] = []
        hip_rotation_proxy: list[float] = []
        kick_depth_proxy: list[float] = []
        sample_landmarks: list[dict[str, Any]] = []
        analysis_events: list[dict[str, Any]] = []
        snapshot_candidates: list[tuple[int, Any]] = []
        overlay_writer: Any = None
        overlay_path: Path | None = None
        overlay_error: str | None = None
        pose_seen = False
        low_confidence_seen = False
        high_confidence_seen = False

        try:
            pose_model = self.mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            try:
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_index % sample_every != 0:
                        frame_index += 1
                        continue

                    frames_sampled += 1
                    rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    result = pose_model.process(rgb)
                    annotated_frame = frame.copy()
                    frame_confidence: float | None = None
                    frame_elbow: float | None = None
                    frame_hip: float | None = None
                    if result.pose_landmarks:
                        pose_frames_detected += 1
                        landmarks = result.pose_landmarks.landmark
                        frame_confidence = _average_visibility(landmarks)
                        frame_elbows = _elbow_angles(landmarks)
                        frame_elbow = mean(frame_elbows)
                        frame_hip = _hip_rotation_proxy(landmarks)
                        confidence_values.append(frame_confidence)
                        elbow_angles.extend(frame_elbows)
                        hip_rotation_proxy.append(frame_hip)
                        kick_depth_proxy.append(_kick_depth_proxy(landmarks))
                        self.drawing_utils.draw_landmarks(
                            annotated_frame,
                            result.pose_landmarks,
                            self.mp.solutions.pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.drawing_styles.get_default_pose_landmarks_style(),
                        )
                        timestamp_s = round(frame_index / fps, 2) if fps else 0
                        if not pose_seen:
                            analysis_events.append(
                                {
                                    "timestamp_s": timestamp_s,
                                    "type": "pose_detected",
                                    "label": "Pose detected",
                                    "message": "MediaPipe found a swimmer pose and began drawing landmarks.",
                                }
                            )
                            pose_seen = True
                        if frame_confidence < 0.45 and not low_confidence_seen:
                            analysis_events.append(
                                {
                                    "timestamp_s": timestamp_s,
                                    "type": "low_confidence",
                                    "label": "Low confidence",
                                    "message": "Pose confidence dropped; camera angle, splash, or occlusion may affect this segment.",
                                }
                            )
                            low_confidence_seen = True
                        if frame_confidence >= 0.75 and not high_confidence_seen:
                            analysis_events.append(
                                {
                                    "timestamp_s": timestamp_s,
                                    "type": "high_confidence",
                                    "label": "High confidence tracking",
                                    "message": "Landmark visibility was strong enough for a reliable technique read.",
                                }
                            )
                            high_confidence_seen = True
                        if len(sample_landmarks) < 8:
                            sample_landmarks.append(_sample_landmarks(landmarks, frame_index, fps))
                    elif frames_sampled == 1:
                        analysis_events.append(
                            {
                                "timestamp_s": round(frame_index / fps, 2) if fps else 0,
                                "type": "no_pose_detected",
                                "label": "No pose detected",
                                "message": "No clear swimmer pose was detected on the first analyzed frame.",
                            }
                        )
                    status_text = "pose detected" if result.pose_landmarks else "not detected"
                    self._draw_overlay_text(
                        annotated_frame,
                        frame_index=frame_index,
                        timestamp_s=round(frame_index / fps, 2) if fps else 0,
                        status_text=status_text,
                        confidence=frame_confidence,
                        elbow_angle=frame_elbow,
                        hip_rotation=frame_hip,
                    )
                    if len(snapshot_candidates) < 5:
                        snapshot_candidates.append((frame_index, annotated_frame.copy()))
                    if overlay_error is None:
                        if overlay_writer is None:
                            overlay_path = self._analysis_asset_path("processed_videos", f"{uploaded_file.id}_analysis_overlay.mp4")
                            overlay_writer = self._create_video_writer(overlay_path, annotated_frame, fps, sample_every)
                            if overlay_writer is None:
                                overlay_error = "AI overlay video could not be generated. Showing skeleton snapshots if available."
                        if overlay_writer is not None:
                            try:
                                overlay_writer.write(annotated_frame)
                            except Exception:
                                overlay_error = "AI overlay video could not be generated. Showing skeleton snapshots if available."
                                overlay_writer.release()
                                overlay_writer = None
                    frame_index += 1
            finally:
                pose_model.close()
        except Exception:
            cap.release()
            if overlay_writer is not None:
                overlay_writer.release()
            return self._failed_report(swimmer, uploaded_file, PROCESSING_ERROR_MESSAGE, frames_total=total_frames, frames_analyzed=frames_sampled)
        finally:
            cap.release()
            if overlay_writer is not None:
                overlay_writer.release()

        overlay_url = str(overlay_path) if overlay_path and overlay_path.exists() and overlay_error is None else None
        snapshot_urls = self._write_snapshot_frames(uploaded_file.id, snapshot_candidates) if not overlay_url and snapshot_candidates else []

        if pose_frames_detected == 0:
            return self._empty_report(
                swimmer,
                session,
                uploaded_file,
                total_frames=total_frames,
                frames_analyzed=frames_sampled,
                overlay_url=overlay_url,
                snapshot_urls=snapshot_urls,
                analysis_events=analysis_events,
                overlay_error=overlay_error,
            )

        average_visibility = round(mean(confidence_values), 2)
        trust = build_analysis_trust(
            frames_total=total_frames or frame_index,
            frames_analyzed=frames_sampled,
            pose_detected_frames=pose_frames_detected,
            average_visibility=average_visibility,
        )
        elbow_mean = round(mean(elbow_angles), 1) if elbow_angles else None
        hip_mean = round(mean(hip_rotation_proxy), 1) if hip_rotation_proxy else None
        kick_mean = round(mean(kick_depth_proxy), 2) if kick_depth_proxy else None
        faults = _mediapipe_faults(swimmer.primary_stroke, elbow_mean, hip_mean, kick_mean, session.rpe)
        analysis_events.extend(_fault_events(faults, fps, sample_landmarks))
        penalty = sum(12 if item["severity"] == "critical" else 6 if item["severity"] == "needs_work" else 0 for item in faults)
        score = clamp_score(92 - penalty + (trust["confidence_score"] * 8), 35, 98)
        dps_meters = round(1.45 + score / 100 * 0.62, 2)
        stroke_rate = clamp_score(42 - (score - 75) / 4 + max(0, session.rpe - 7), 28, 58)
        time_gain = round(sum(float(item.get("time_cost_seconds", 0)) for item in faults), 1)
        top_fault = faults[0]["title"] if faults else "No major fault detected"

        if overlay_error and not trust["analysis_warning"]:
            trust["analysis_warning"] = overlay_error

        return {
            "stroke": swimmer.primary_stroke,
            "overall_score": score,
            "dps_meters": dps_meters,
            "stroke_rate": stroke_rate,
            "faults": faults,
            "drill_prescriptions": _drills_from_faults(faults, swimmer.primary_stroke),
            "keypoint_data": {
                "provider": "mediapipe",
                "source_file": uploaded_file.stored_path,
                "frames_read": frame_index,
                "frames_sampled": trust["frames_analyzed"],
                "pose_frames_detected": trust["pose_detected_frames"],
                "fps": round(fps, 2),
                "duration_seconds": round(frame_index / fps, 2) if fps else None,
                "confidence": trust["confidence_score"],
                "time_gain_possible": time_gain,
                "analysis_quality": trust["analysis_status"],
                "body_heatmap": _body_heatmap(faults),
                "metrics": {
                    "elbow_angle_mean": elbow_mean,
                    "hip_rotation_proxy": hip_mean,
                    "kick_amplitude_proxy": kick_mean,
                },
                "sample_landmarks": sample_landmarks,
            },
            "processing_status": "complete",
            **trust,
            "analysis_overlay_video_url": overlay_url,
            "analysis_frame_urls": snapshot_urls,
            "analysis_events": _dedupe_events(analysis_events),
            "coaching_summary": self.coaching.technique_summary(swimmer.name, top_fault, score, time_gain),
        }

    def _analysis_asset_path(self, folder: str, filename: str) -> Path:
        root = Path(get_settings().upload_dir)
        target_dir = root / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / filename

    def _create_video_writer(self, target_path: Path, sample_frame: Any, fps: float, sample_every: int) -> Any | None:
        height, width = sample_frame.shape[:2]
        output_fps = max(1, min(15, fps / max(sample_every, 1)))
        writer = self.cv2.VideoWriter(
            str(target_path),
            self.cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            return None
        return writer

    def _draw_overlay_text(
        self,
        frame: Any,
        *,
        frame_index: int,
        timestamp_s: float,
        status_text: str,
        confidence: float | None,
        elbow_angle: float | None,
        hip_rotation: float | None,
    ) -> None:
        lines = [
            f"Frame: {frame_index}  Time: {timestamp_s:.2f}s",
            f"Pose: {status_text}",
            f"Confidence: {confidence * 100:.0f}%" if confidence is not None else "Confidence: --",
            f"Elbow: {elbow_angle:.0f} deg" if elbow_angle is not None else "Elbow: --",
            f"Hip rotation: {hip_rotation:.0f}" if hip_rotation is not None else "Hip rotation: --",
            "Status: analyzing",
        ]
        x, y = 18, 30
        line_height = 24
        overlay = frame.copy()
        self.cv2.rectangle(overlay, (8, 8), (330, 8 + line_height * len(lines) + 10), (0, 0, 0), -1)
        self.cv2.addWeighted(overlay, 0.56, frame, 0.44, 0, frame)
        for index, line in enumerate(lines):
            self.cv2.putText(
                frame,
                line,
                (x, y + index * line_height),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (245, 245, 245),
                2,
                self.cv2.LINE_AA,
            )

    def _write_snapshot_frames(self, uploaded_file_id: str, candidates: list[tuple[int, Any]]) -> list[str]:
        target_dir = Path(get_settings().upload_dir) / "analysis_frames"
        target_dir.mkdir(parents=True, exist_ok=True)
        urls: list[str] = []
        for frame_index, frame in candidates[:5]:
            target_path = target_dir / f"{uploaded_file_id}_frame_{frame_index}.jpg"
            if self.cv2.imwrite(str(target_path), frame):
                urls.append(str(target_path))
        return urls

    def _empty_report(
        self,
        swimmer: Swimmer,
        session: Session,
        uploaded_file: UploadedFile,
        *,
        total_frames: int,
        frames_analyzed: int,
        overlay_url: str | None = None,
        snapshot_urls: list[str] | None = None,
        analysis_events: list[dict[str, Any]] | None = None,
        overlay_error: str | None = None,
    ) -> dict[str, Any]:
        fallback = MockTechniqueAnalyzer(self.coaching).analyze(swimmer, session, uploaded_file)
        trust = build_analysis_trust(frames_total=total_frames, frames_analyzed=frames_analyzed, pose_detected_frames=0)
        if overlay_error and not trust["analysis_warning"]:
            trust["analysis_warning"] = overlay_error
        fallback["keypoint_data"] = {
            **fallback["keypoint_data"],
            "provider": "mediapipe",
            "analysis_quality": "no_pose_detected",
            "pose_frames_detected": 0,
            "frames_sampled": frames_analyzed,
            "confidence": 0,
            "provider_note": NO_POSE_MESSAGE,
        }
        fallback["processing_status"] = "complete"
        fallback.update(trust)
        fallback["analysis_overlay_video_url"] = overlay_url
        fallback["analysis_frame_urls"] = snapshot_urls or []
        fallback["analysis_events"] = _dedupe_events(
            [
                *(analysis_events or []),
                {
                    "timestamp_s": 0,
                    "type": "no_pose_detected",
                    "label": "No pose detected",
                    "message": NO_POSE_MESSAGE,
                },
            ]
        )
        fallback["coaching_summary"] = f"{NO_POSE_MESSAGE} Showing deterministic fallback technique guidance for now."
        return fallback

    def _failed_report(
        self,
        swimmer: Swimmer,
        uploaded_file: UploadedFile,
        message: str,
        *,
        frames_total: int = 0,
        frames_analyzed: int = 0,
    ) -> dict[str, Any]:
        trust = build_analysis_trust(
            frames_total=frames_total,
            frames_analyzed=frames_analyzed,
            pose_detected_frames=0,
            processing_error=message,
        )
        return {
            "stroke": swimmer.primary_stroke,
            "overall_score": 0,
            "dps_meters": 0,
            "stroke_rate": 0,
            "faults": [],
            "drill_prescriptions": [],
            "keypoint_data": {
                "provider": "mediapipe",
                "source_file": uploaded_file.stored_path,
                "frames_sampled": frames_analyzed,
                "pose_frames_detected": 0,
                "confidence": 0,
                "analysis_quality": "failed",
                "provider_note": message,
            },
            "processing_status": "failed",
            **trust,
            "analysis_overlay_video_url": None,
            "analysis_frame_urls": [],
            "analysis_events": [
                {
                    "timestamp_s": 0,
                    "type": "processing_error",
                    "label": "Processing error",
                    "message": message,
                }
            ],
            "coaching_summary": message,
        }


def _average_visibility(landmarks: Any) -> float:
    selected = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    return float(mean(float(landmarks[index].visibility) for index in selected))


def _point(landmark: Any) -> tuple[float, float]:
    return float(landmark.x), float(landmark.y)


def _angle(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    denom = ((ba[0] ** 2 + ba[1] ** 2) ** 0.5) * ((bc[0] ** 2 + bc[1] ** 2) ** 0.5)
    if denom == 0:
        return 0
    cosine = max(-1, min(1, (ba[0] * bc[0] + ba[1] * bc[1]) / denom))
    return degrees(acos(cosine))


def _elbow_angles(landmarks: Any) -> list[float]:
    return [
        _angle(_point(landmarks[11]), _point(landmarks[13]), _point(landmarks[15])),
        _angle(_point(landmarks[12]), _point(landmarks[14]), _point(landmarks[16])),
    ]


def _hip_rotation_proxy(landmarks: Any) -> float:
    shoulder_width = abs(float(landmarks[11].x) - float(landmarks[12].x)) or 0.01
    hip_z_delta = abs(float(landmarks[23].z) - float(landmarks[24].z))
    return min(60, max(0, hip_z_delta / shoulder_width * 45))


def _kick_depth_proxy(landmarks: Any) -> float:
    left_depth = abs(float(landmarks[27].y) - float(landmarks[23].y))
    right_depth = abs(float(landmarks[28].y) - float(landmarks[24].y))
    return max(left_depth, right_depth)


def _sample_landmarks(landmarks: Any, frame_index: int, fps: float) -> dict[str, Any]:
    selected = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    return {
        "frame": frame_index,
        "timestamp_s": round(frame_index / fps, 2) if fps else 0,
        "landmarks": {
            str(index): {
                "x": round(float(landmarks[index].x), 4),
                "y": round(float(landmarks[index].y), 4),
                "z": round(float(landmarks[index].z), 4),
                "visibility": round(float(landmarks[index].visibility), 3),
            }
            for index in selected
        },
    }


def _mediapipe_faults(stroke: str, elbow_angle: float | None, hip_rotation: float | None, kick_depth: float | None, rpe: int) -> list[dict[str, Any]]:
    faults: list[dict[str, Any]] = []
    if elbow_angle is not None and elbow_angle > 148:
        faults.append(
            {
                "zone": "arms",
                "severity": "critical" if elbow_angle > 158 else "needs_work",
                "title": "Dropped elbow on catch",
                "description": f"Measured elbow angle averages {elbow_angle:.0f} degrees. Build a higher catch before pressure.",
                "time_cost_seconds": 0.32,
            }
        )
    if hip_rotation is not None and stroke.lower() in {"freestyle", "backstroke"} and hip_rotation < 35:
        faults.append(
            {
                "zone": "hips",
                "severity": "needs_work",
                "title": "Under-rotation",
                "description": f"Hip rotation proxy is {hip_rotation:.0f}. Target a stronger hip-led roll.",
                "time_cost_seconds": 0.22,
            }
        )
    if kick_depth is not None and kick_depth > 0.46:
        faults.append(
            {
                "zone": "legs",
                "severity": "needs_work" if rpe < 8 else "critical",
                "title": "Kick depth too wide",
                "description": f"Kick-depth proxy is {kick_depth:.2f}. Narrow the kick to reduce drag.",
                "time_cost_seconds": 0.24,
            }
        )
    if not faults:
        faults.append(
            {
                "zone": "torso",
                "severity": "good",
                "title": "Stable body line",
                "description": "No major pose-estimated fault crossed the Phase 1 thresholds.",
                "time_cost_seconds": 0,
            }
        )
    return faults


def _drills_from_faults(faults: list[dict[str, Any]], stroke: str) -> list[dict[str, Any]]:
    drills = []
    titles = " ".join(str(item.get("title", "")) for item in faults)
    if "Dropped elbow" in titles or stroke.lower() == "freestyle":
        drills.append({"name": "Fingertip drag", "focus": "Elbow position", "volume": "4x50m", "sets": 1})
    if "Kick" in titles:
        drills.append({"name": "Kickboard with fins", "focus": "Kick amplitude", "volume": "6x25m", "sets": 1})
    if "rotation" in titles.lower() or stroke.lower() in {"freestyle", "backstroke"}:
        drills.append({"name": "6-kick switch", "focus": "Hip rotation", "volume": "6x25m", "sets": 1})
    if not drills:
        drills.append({"name": "Catch-up drill", "focus": "Front extension", "volume": "4x50m", "sets": 1})
    return drills[:3]


def _body_heatmap(faults: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"zone": str(item.get("zone", "torso")), "severity": str(item.get("severity", "needs_work"))} for item in faults]


def _fault_events(faults: list[dict[str, Any]], fps: float, sample_landmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timestamp_s = 0
    if sample_landmarks:
        timestamp_s = float(sample_landmarks[min(1, len(sample_landmarks) - 1)].get("timestamp_s", 0))
    events: list[dict[str, Any]] = []
    for fault in faults:
        severity = str(fault.get("severity", "")).lower()
        if severity == "good":
            continue
        title = str(fault.get("title", "Technique warning"))
        event_type = title.lower().replace(" ", "_").replace("-", "_")
        events.append(
            {
                "timestamp_s": round(timestamp_s, 2),
                "type": event_type,
                "label": title,
                "message": str(fault.get("description", "MediaPipe detected a technique pattern worth coach review.")),
            }
        )
        timestamp_s += max(0.5, 12 / max(fps, 1))
    return events


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, float]] = set()
    deduped: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: float(item.get("timestamp_s", 0))):
        key = (str(event.get("type", "")), round(float(event.get("timestamp_s", 0)), 1))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "timestamp_s": round(float(event.get("timestamp_s", 0)), 2),
                "type": str(event.get("type", "analysis_event")),
                "label": str(event.get("label", "Analysis event")),
                "message": str(event.get("message", "")),
            }
        )
    return deduped[:12]


def failed_technique_report_payload(swimmer: Swimmer, uploaded_file: UploadedFile, message: str = PROCESSING_ERROR_MESSAGE) -> dict[str, Any]:
    trust = build_analysis_trust(frames_total=0, frames_analyzed=0, pose_detected_frames=0, processing_error=message)
    return {
        "stroke": swimmer.primary_stroke,
        "overall_score": 0,
        "dps_meters": 0,
        "stroke_rate": 0,
        "faults": [],
        "drill_prescriptions": [],
        "keypoint_data": {
            "provider": "mediapipe",
            "source_file": uploaded_file.stored_path,
            "frames_sampled": 0,
            "pose_frames_detected": 0,
            "confidence": 0,
            "analysis_quality": "failed",
            "provider_note": message,
        },
        "processing_status": "failed",
        **trust,
        "analysis_overlay_video_url": None,
        "analysis_frame_urls": [],
        "analysis_events": [
            {
                "timestamp_s": 0,
                "type": "processing_error",
                "label": "Processing error",
                "message": message,
            }
        ],
        "coaching_summary": message,
    }


def get_technique_analyzer() -> TechniqueAnalyzer:
    settings = get_settings()
    mode = settings.technique_analyzer.lower()
    if mode in {"auto", "mediapipe"}:
        try:
            return MediaPipeTechniqueAnalyzer()
        except RuntimeError as exc:
            if mode == "mediapipe":
                return MockTechniqueAnalyzer(provider_note=str(exc))
    return MockTechniqueAnalyzer()

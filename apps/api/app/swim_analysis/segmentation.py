from __future__ import annotations

from typing import Any

from app.swim_analysis.metrics import detect_stroke_cycles
from app.swim_analysis.pose.base import PoseFrameResult
from app.swim_analysis.quality import average_pose_confidence


def segment_swim_phases(
    side_frames: list[PoseFrameResult],
    velocity: dict[str, Any],
) -> list[dict[str, Any]]:
    duration = _duration(side_frames)
    if duration <= 0:
        return []
    pose_conf = _landmark_pose_confidence(side_frames)
    stroke_times = _merged_stroke_times(side_frames)
    if not stroke_times and pose_conf < 0.2:
        confidence = min(0.25, float(velocity.get("summary", {}).get("confidence", 0) or 0))
        start_end = min(1.5, duration * 0.1)
        finish_start = max(start_end, duration - min(1.5, duration * 0.1))
        return [
            _phase("start_push_off", 0.0, start_end, confidence, "Coarse timing only: no wrist landmarks were available for phase segmentation."),
            _phase("free_swim", start_end, finish_start, confidence, "Centroid-only fallback cannot separate underwater, breakout, turns, or stroke cycles reliably."),
            _phase("finish", finish_start, duration, confidence, "Final coarse segment of the uploaded clip."),
        ]
    first_cycle = stroke_times[0] if stroke_times else min(duration * 0.22, 3.0)
    breakout_center = min(max(first_cycle, 1.0), duration)
    start_end = min(1.5, duration * 0.1)
    underwater_end = min(duration, max(start_end, breakout_center - 0.25))
    finish_start = max(0.0, duration - min(1.5, duration * 0.1))

    phases = [
        _phase("start_push_off", 0.0, start_end, pose_conf * 0.65, "Early high-speed segment before regular stroke cycles."),
        _phase("underwater", start_end, underwater_end, pose_conf * 0.55, "Estimated from time before first repeatable wrist stroke peak."),
        _phase("breakout", max(0.0, breakout_center - 0.35), min(duration, breakout_center + 0.35), pose_conf * 0.68, "Centered on first detected stroke-cycle peak."),
    ]

    turns = _turn_candidates(velocity, duration)
    free_start = min(duration, max(underwater_end, breakout_center + 0.35))
    cursor = free_start
    for turn in turns:
        turn_start = max(cursor, float(turn["start_sec"]) - 0.5)
        turn_end = min(finish_start, float(turn["end_sec"]) + 0.5)
        if turn_start - cursor > 0.3:
            phases.append(_phase("free_swim", cursor, turn_start, pose_conf * 0.75, "Regular swimming between breakout and turn/finish."))
        phases.append(_phase("turn", turn_start, turn_end, min(float(turn.get("confidence", 0.4)), pose_conf), "Velocity valley suggests a turn or wall contact."))
        cursor = max(cursor, turn_end)

    if finish_start - cursor > 0.3:
        phases.append(_phase("free_swim", cursor, finish_start, pose_conf * 0.75, "Regular stroke cycles before the finish."))
    phases.append(_phase("finish", finish_start, duration, pose_conf * 0.62, "Final segment of the uploaded clip."))

    phases.extend(_stroke_cycle_segments(stroke_times, duration, pose_conf))
    phases.extend(_stroke_subphase_segments(side_frames, stroke_times, duration, pose_conf))
    phases.extend(_body_roll_and_kick_segments(side_frames, duration, pose_conf))
    return [phase for phase in phases if phase["end_sec"] > phase["start_sec"]]


def _phase(kind: str, start: float, end: float, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "type": kind,
        "start_sec": round(max(0.0, start), 3),
        "end_sec": round(max(0.0, end), 3),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "reason": reason,
    }


def _merged_stroke_times(frames: list[PoseFrameResult]) -> list[float]:
    times = sorted([*detect_stroke_cycles(frames, "left_wrist"), *detect_stroke_cycles(frames, "right_wrist")])
    output = []
    for value in times:
        if not output or value - output[-1] >= 0.25:
            output.append(value)
    return output


def _stroke_cycle_segments(stroke_times: list[float], duration: float, pose_conf: float) -> list[dict[str, Any]]:
    cycles = []
    for index, (start, end) in enumerate(zip(stroke_times, stroke_times[1:]), start=1):
        if end <= start:
            continue
        cycles.append(
            {
                "type": "stroke_cycle",
                "cycle_index": index,
                "start_sec": round(start, 3),
                "end_sec": round(min(end, duration), 3),
                "confidence": round(min(0.9, pose_conf * min(1.0, len(stroke_times) / 6)), 3),
                "reason": "Consecutive wrist-motion peaks define this stroke cycle.",
            }
        )
    return cycles[:80]


def _stroke_subphase_segments(
    frames: list[PoseFrameResult],
    stroke_times: list[float],
    duration: float,
    pose_conf: float,
) -> list[dict[str, Any]]:
    if pose_conf < 0.28 or len(stroke_times) < 2:
        return []
    output: list[dict[str, Any]] = []
    for cycle_index, (cycle_start, cycle_end) in enumerate(zip(stroke_times, stroke_times[1:]), start=1):
        cycle_end = min(cycle_end, duration)
        cycle_duration = cycle_end - cycle_start
        if cycle_duration < 0.35:
            continue
        arm, samples = _dominant_wrist_samples(frames, cycle_start, cycle_end)
        if len(samples) < 3:
            continue
        y_values = [sample[2] for sample in samples]
        amplitude = max(y_values) - min(y_values)
        support = min(1.0, len(samples) / 5.0)
        motion_confidence = max(0.0, min(1.0, pose_conf * support * (0.58 + min(0.32, amplitude / 140.0))))
        if motion_confidence < 0.22:
            continue
        boundaries = [
            cycle_start,
            cycle_start + cycle_duration * 0.22,
            cycle_start + cycle_duration * 0.48,
            cycle_start + cycle_duration * 0.72,
            cycle_end,
        ]
        descriptions = {
            "catch": "Early-cycle wrist trajectory starts the pressure phase; proxy only until denser validation data is available.",
            "pull": "Middle-cycle wrist displacement suggests the main propulsive pull window.",
            "push": "Late-cycle wrist motion indicates the push-through portion of the stroke.",
            "recovery": "Cycle reset/recovery window inferred from repeatable wrist motion.",
        }
        for phase_name, start, end in zip(["catch", "pull", "push", "recovery"], boundaries, boundaries[1:]):
            output.append(
                {
                    "type": phase_name,
                    "category": "stroke_subphase",
                    "cycle_index": cycle_index,
                    "arm": arm,
                    "start_sec": round(max(0.0, start), 3),
                    "end_sec": round(max(0.0, min(end, duration)), 3),
                    "confidence": round(min(0.82, motion_confidence), 3),
                    "supporting_frames": len(samples),
                    "amplitude_px": round(float(amplitude), 3),
                    "reason": descriptions[phase_name],
                }
            )
        if len(output) >= 24:
            break
    return output


def _body_roll_and_kick_segments(frames: list[PoseFrameResult], duration: float, pose_conf: float) -> list[dict[str, Any]]:
    if pose_conf < 0.28 or duration <= 0:
        return []
    phases: list[dict[str, Any]] = []
    shoulder_roll = _roll_proxy(frames, "shoulder")
    hip_roll = _roll_proxy(frames, "hip")
    roll_support = min(shoulder_roll["support"], hip_roll["support"])
    if roll_support >= max(4, len(frames) * 0.25):
        roll_confidence = min(0.78, pose_conf * min(1.0, roll_support / max(1, len(frames) * 0.55)))
        phases.append(
            {
                "type": "body_roll_timing",
                "category": "temporal_pattern",
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "confidence": round(roll_confidence, 3),
                "supporting_frames": int(roll_support),
                "reason": "Shoulder and hip centerline offsets were tracked over time as a lightweight body-roll proxy.",
                "evidence": {
                    "shoulder_offset_amplitude_px": shoulder_roll["amplitude"],
                    "hip_offset_amplitude_px": hip_roll["amplitude"],
                },
            }
        )
    kick_times = sorted([*detect_stroke_cycles(frames, "left_ankle"), *detect_stroke_cycles(frames, "right_ankle")])
    if len(kick_times) >= 3:
        intervals = [second - first for first, second in zip(kick_times, kick_times[1:]) if second > first]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0.0
        rhythm_confidence = min(0.78, pose_conf * min(1.0, len(kick_times) / 8.0))
        phases.append(
            {
                "type": "kick_rhythm",
                "category": "temporal_pattern",
                "start_sec": round(max(0.0, kick_times[0]), 3),
                "end_sec": round(min(duration, kick_times[-1]), 3),
                "confidence": round(rhythm_confidence, 3),
                "supporting_frames": len(kick_times),
                "reason": "Ankle-motion peaks provide a temporal kick rhythm proxy.",
                "evidence": {
                    "kick_peak_times_sec": [round(value, 3) for value in kick_times[:24]],
                    "average_interval_sec": round(avg_interval, 3) if avg_interval else None,
                },
            }
        )
    return phases


def _dominant_wrist_samples(frames: list[PoseFrameResult], start: float, end: float) -> tuple[str, list[tuple[float, float, float, float]]]:
    left = _joint_samples(frames, "left_wrist", start, end)
    right = _joint_samples(frames, "right_wrist", start, end)
    left_score = _sample_amplitude(left) * min(1.0, len(left) / 5.0)
    right_score = _sample_amplitude(right) * min(1.0, len(right) / 5.0)
    return ("left", left) if left_score >= right_score else ("right", right)


def _joint_samples(frames: list[PoseFrameResult], joint: str, start: float, end: float) -> list[tuple[float, float, float, float]]:
    samples = []
    for frame in frames:
        if frame.timestamp < start or frame.timestamp > end:
            continue
        point = frame.keypoint(joint, 0.18)
        if point is not None:
            samples.append((float(frame.timestamp), float(point.x), float(point.y), float(point.confidence)))
    return samples


def _sample_amplitude(samples: list[tuple[float, float, float, float]]) -> float:
    if len(samples) < 2:
        return 0.0
    ys = [sample[2] for sample in samples]
    xs = [sample[1] for sample in samples]
    return max(max(ys) - min(ys), (max(xs) - min(xs)) * 0.35)


def _roll_proxy(frames: list[PoseFrameResult], body_part: str) -> dict[str, Any]:
    left_name = f"left_{body_part}"
    right_name = f"right_{body_part}"
    offsets = []
    for frame in frames:
        left = frame.keypoint(left_name, 0.2)
        right = frame.keypoint(right_name, 0.2)
        if left is None or right is None:
            continue
        offsets.append(float(left.y) - float(right.y))
    amplitude = max(offsets) - min(offsets) if len(offsets) >= 2 else 0.0
    return {"support": len(offsets), "amplitude": round(float(amplitude), 3)}


def _turn_candidates(velocity: dict[str, Any], duration: float) -> list[dict[str, Any]]:
    if float(velocity.get("summary", {}).get("confidence", 0) or 0) < 0.45:
        return []
    if velocity.get("summary", {}).get("evidence_mode") == "centroid_only":
        return []
    spots = []
    for spot in velocity.get("dead_spots", []):
        center = (float(spot["start_sec"]) + float(spot["end_sec"])) / 2
        if duration * 0.18 < center < duration * 0.88:
            spots.append(spot)
    return spots[:4]


def _duration(frames: list[PoseFrameResult]) -> float:
    if len(frames) < 2:
        return 0.0
    return max(0.0, frames[-1].timestamp - frames[0].timestamp)


def _landmark_pose_confidence(frames: list[PoseFrameResult]) -> float:
    landmark_frames = [frame for frame in frames if frame.has_pose(min_confidence=0.2)]
    if not landmark_frames:
        return 0.0
    return average_pose_confidence(landmark_frames)

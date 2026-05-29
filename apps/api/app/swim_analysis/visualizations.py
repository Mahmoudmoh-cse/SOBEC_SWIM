from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from app.swim_analysis.phase1_schemas import ResearchFigureMetadata
from app.swim_analysis.pose.base import KEYPOINT_NAMES


TRAJECTORY_JOINTS = [
    "nose",
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

SWAP_PAIRS = [
    ("left_wrist", "right_wrist"),
    ("left_elbow", "right_elbow"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]


def generate_research_figures(output_dir: str | Path, trajectory_exports: dict[str, dict[str, Any]], velocity: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Generate SwimmerNET-inspired explainability figures from trajectory exports."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional local plotting stack
        return [], [f"Research visualizations were skipped because matplotlib is unavailable: {exc}"]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures: list[ResearchFigureMetadata] = []

    for view, export in trajectory_exports.items():
        figures.extend(_trajectory_figures(plt, output, view, export))
        figures.extend(_swap_figures(plt, output, view, export))
        figures.extend(_missing_percentage_figure(plt, output, view, export))
        figures.extend(_outlier_impact_figure(plt, output, view, export))
        figures.extend(_drift_figure(plt, output, view, export))
        figures.extend(_std_noise_figure(plt, output, view, export))
        figures.extend(_stroke_cycle_figure(plt, output, view, export))
        figures.extend(_reliability_figures(plt, output, view, export))
        figures.extend(_temporal_diagnostic_figures(plt, output, view, export))

    if velocity:
        figures.extend(_velocity_motion_figures(plt, output, velocity))

    return [figure.model_dump(mode="json") for figure in figures], []


def _trajectory_figures(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    figures = []
    raw = export.get("raw_keypoints", {})
    filtered = export.get("filtered_keypoints", {})
    for joint in TRAJECTORY_JOINTS:
        raw_series = raw.get(joint, [])
        filtered_series = filtered.get(joint, [])
        if not _has_points(raw_series) and not _has_points(filtered_series):
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        _plot_axis(ax, raw_series, "x", f"raw {joint} x", "#94a3b8", "--")
        _plot_axis(ax, raw_series, "y", f"raw {joint} y", "#cbd5e1", "--")
        _plot_axis(ax, filtered_series, "x", f"filtered {joint} x", "#1d4ed8", "-")
        _plot_axis(ax, filtered_series, "y", f"filtered {joint} y", "#0f766e", "-")
        ax.set_title(f"{view.title()} {joint.replace('_', ' ')} trajectory")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pixel coordinate")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        path = output / f"{view}_trajectory_{joint}.png"
        _save(fig, path)
        figures.append(
            _figure(
                figure_id=f"{view}_trajectory_{joint}",
                title=f"{view.title()} {joint.replace('_', ' ')} trajectory",
                type_="trajectories",
                path=path,
                description="Raw and filtered x/y landmark coordinates over time.",
            )
        )
    return figures


def _swap_figures(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    figures = []
    raw = export.get("raw_keypoints", {})
    corrected = export.get("corrected_keypoints", {}) or export.get("filtered_keypoints", {})
    for left, right in SWAP_PAIRS:
        if not (_has_points(raw.get(left, [])) or _has_points(raw.get(right, []))):
            continue
        fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
        _plot_axis(ax, raw.get(left, []), "x", f"raw {left}", "#94a3b8", "--")
        _plot_axis(ax, raw.get(right, []), "x", f"raw {right}", "#64748b", "--")
        _plot_axis(ax, corrected.get(left, []), "x", f"corrected {left}", "#1d4ed8", "-")
        _plot_axis(ax, corrected.get(right, []), "x", f"corrected {right}", "#dc2626", "-")
        ax.set_title(f"{view.title()} {left.split('_')[1]} left/right correction")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Horizontal position (px)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        path = output / f"{view}_swap_correction_{left.split('_')[1]}.png"
        _save(fig, path)
        figures.append(
            _figure(
                figure_id=f"{view}_swap_correction_{left.split('_')[1]}",
                title=f"{view.title()} {left.split('_')[1]} swap correction",
                type_="trajectories",
                path=path,
                description="Before/after temporal identity correction for left and right landmarks.",
            )
        )
    return figures


def _missing_percentage_figure(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    visibility = export.get("visibility_percentage", {})
    if not visibility:
        return []
    joints = [joint for joint in KEYPOINT_NAMES if joint in visibility]
    missing = [100.0 - float(visibility[joint]) for joint in joints]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    ax.bar([joint.replace("_", "\n") for joint in joints], missing, color="#1d4ed8")
    ax.set_title(f"{view.title()} missing target percentage")
    ax.set_ylabel("Missing / unrecognized (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=7)
    path = output / f"{view}_missing_target_percentage.png"
    _save(fig, path)
    return [
        _figure(
            figure_id=f"{view}_missing_target_percentage",
            title=f"{view.title()} missing target percentage",
            type_="tracking_quality",
            path=path,
            description="Percentage of frames where each joint was missing or too unreliable.",
        )
    ]


def _outlier_impact_figure(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    raw = export.get("raw_keypoints", {})
    filtered = export.get("filtered_keypoints", {})
    joints = [joint for joint in TRAJECTORY_JOINTS if _has_points(raw.get(joint, []))]
    if not joints:
        return []
    before = [_motion_noise_proxy(raw.get(joint, [])) for joint in joints]
    after = [_motion_noise_proxy(filtered.get(joint, [])) for joint in joints]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    positions = list(range(len(joints)))
    ax.bar([pos - 0.2 for pos in positions], before, width=0.4, color="#94a3b8", label="before outlier removal")
    ax.bar([pos + 0.2 for pos in positions], after, width=0.4, color="#0f766e", label="after outlier removal")
    ax.set_xticks(positions)
    ax.set_xticklabels([joint.replace("_", "\n") for joint in joints], fontsize=7)
    ax.set_title(f"{view.title()} outlier impact: trajectory noise proxy")
    ax.set_ylabel("Velocity/acceleration instability proxy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = output / f"{view}_outlier_impact_noise_proxy.png"
    _save(fig, path)
    return [
        _figure(
            figure_id=f"{view}_outlier_impact_noise_proxy",
            title=f"{view.title()} outlier impact",
            type_="error_noise",
            path=path,
            description="Trajectory noise proxy before and after outlier removal. This is not true localization error.",
        )
    ]


def _drift_figure(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    raw = export.get("raw_keypoints", {})
    filtered = export.get("filtered_keypoints", {})
    joints = [joint for joint in TRAJECTORY_JOINTS if _has_points(raw.get(joint, [])) and _has_points(filtered.get(joint, []))]
    if not joints:
        return []
    displacement = [_mean_displacement(raw.get(joint, []), filtered.get(joint, [])) for joint in joints]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    ax.bar([joint.replace("_", "\n") for joint in joints], displacement, color="#1d4ed8")
    ax.set_title(f"{view.title()} mean smoothing displacement proxy")
    ax.set_ylabel("Mean raw-to-filtered distance (px)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=7)
    path = output / f"{view}_mean_displacement_proxy.png"
    _save(fig, path)
    return [
        _figure(
            figure_id=f"{view}_mean_displacement_proxy",
            title=f"{view.title()} mean error/drift proxy",
            type_="error_noise",
            path=path,
            description="Average raw-to-filtered displacement. Labeled as a proxy because no ground-truth labels were provided.",
        )
    ]


def _std_noise_figure(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    filtered = export.get("filtered_keypoints", {})
    joints = [joint for joint in TRAJECTORY_JOINTS if _has_points(filtered.get(joint, []))]
    if not joints:
        return []
    std_x = [_axis_std(filtered.get(joint, []), "x") for joint in joints]
    std_y = [_axis_std(filtered.get(joint, []), "y") for joint in joints]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    positions = list(range(len(joints)))
    ax.bar([pos - 0.2 for pos in positions], std_x, width=0.4, color="#1d4ed8", label="horizontal noise proxy")
    ax.bar([pos + 0.2 for pos in positions], std_y, width=0.4, color="#0f766e", label="vertical noise proxy")
    ax.set_xticks(positions)
    ax.set_xticklabels([joint.replace("_", "\n") for joint in joints], fontsize=7)
    ax.set_title(f"{view.title()} standard deviation by body part")
    ax.set_ylabel("Standard deviation (px)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    path = output / f"{view}_std_noise_proxy.png"
    _save(fig, path)
    return [
        _figure(
            figure_id=f"{view}_std_noise_proxy",
            title=f"{view.title()} standard deviation",
            type_="error_noise",
            path=path,
            description="Horizontal and vertical motion-noise proxy per body part, not ground-truth error.",
        )
    ]


def _stroke_cycle_figure(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    filtered = export.get("filtered_keypoints", {})
    series_specs = [
        ("left_wrist", "#1d4ed8"),
        ("right_wrist", "#dc2626"),
        ("left_ankle", "#0f766e"),
        ("right_ankle", "#f59e0b"),
        ("left_hip", "#7c3aed"),
        ("right_hip", "#475569"),
    ]
    if not any(_has_points(filtered.get(joint, [])) for joint, _ in series_specs):
        return []
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
    for joint, color in series_specs:
        data = filtered.get(joint, [])
        if not _has_points(data):
            continue
        _plot_axis(ax, data, "y", joint.replace("_", " "), color, "-")
        peaks = _peak_points(data, "y")
        if peaks:
            ax.scatter([p[0] for p in peaks], [p[1] for p in peaks], s=16, color=color, alpha=0.7)
    ax.set_title(f"{view.title()} stroke-cycle movement curves")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vertical position (px)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path = output / f"{view}_stroke_cycle_curves.png"
    _save(fig, path)
    return [
        _figure(
            figure_id=f"{view}_stroke_cycle_curves",
            title=f"{view.title()} stroke-cycle curves",
            type_="stroke_cycles",
            path=path,
            description="Wrist, ankle, and hip vertical movement with peak markers for rhythm review.",
        )
    ]


def _reliability_figures(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    figures = []
    confidence = _mean_confidence_timeline(export)
    tracking = export.get("tracking_quality_timeline", [])
    frame_quality = export.get("frame_quality_timeline", [])

    if confidence:
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
        ax.plot([point["timestamp"] for point in confidence], [point["confidence"] for point in confidence], color="#1d4ed8")
        ax.set_ylim(0, 1)
        ax.set_title(f"{view.title()} joint confidence timeline")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean confidence")
        ax.grid(True, alpha=0.25)
        path = output / f"{view}_joint_confidence_timeline.png"
        _save(fig, path)
        figures.append(_figure(f"{view}_joint_confidence_timeline", f"{view.title()} joint confidence timeline", "tracking_quality", path, "Mean landmark confidence over time."))

    if tracking:
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
        ax.plot([point["timestamp"] for point in tracking], [point["quality"] for point in tracking], color="#0f766e")
        ax.set_ylim(0, 1)
        ax.set_title(f"{view.title()} tracking quality over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tracking quality")
        ax.grid(True, alpha=0.25)
        path = output / f"{view}_tracking_quality_timeline.png"
        _save(fig, path)
        figures.append(_figure(f"{view}_tracking_quality_timeline", f"{view.title()} tracking quality", "tracking_quality", path, "Frame-level tracking quality score over time."))

    if frame_quality:
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
        ax.plot([point["timestamp"] for point in frame_quality], [point["quality"] for point in frame_quality], color="#f59e0b")
        ax.set_ylim(0, 1)
        ax.set_title(f"{view.title()} frame quality over time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frame quality")
        ax.grid(True, alpha=0.25)
        path = output / f"{view}_frame_quality_timeline.png"
        _save(fig, path)
        figures.append(_figure(f"{view}_frame_quality_timeline", f"{view.title()} frame quality", "tracking_quality", path, "Frame quality score from blur, lighting, ROI, and tracking flags."))

    return figures


def _temporal_diagnostic_figures(plt: Any, output: Path, view: str, export: dict[str, Any]) -> list[ResearchFigureMetadata]:
    figures = []
    reliability = export.get("joint_reliability", {})
    if isinstance(reliability, dict) and reliability:
        joints = [joint for joint in KEYPOINT_NAMES if joint in reliability]
        stability = [float(reliability[joint].get("temporal_stability", 0) or 0) for joint in joints]
        smoothness = [float(reliability[joint].get("motion_smoothness", 0) or 0) for joint in joints]
        visibility = [float(reliability[joint].get("visibility_percent", 0) or 0) / 100.0 for joint in joints]
        fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=130)
        positions = list(range(len(joints)))
        ax.bar([pos - 0.25 for pos in positions], visibility, width=0.25, color="#1d4ed8", label="visibility")
        ax.bar(positions, stability, width=0.25, color="#0f766e", label="stability")
        ax.bar([pos + 0.25 for pos in positions], smoothness, width=0.25, color="#f59e0b", label="smoothness")
        ax.set_xticks(positions)
        ax.set_xticklabels([joint.replace("_", "\n") for joint in joints], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title(f"{view.title()} joint reliability heatmap inputs")
        ax.set_ylabel("Score")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        path = output / f"{view}_joint_reliability_heatmap.png"
        _save(fig, path)
        figures.append(
            _figure(
                f"{view}_joint_reliability_heatmap",
                f"{view.title()} joint reliability",
                "tracking_quality",
                path,
                "Visibility, temporal stability, and motion smoothness used to classify reliable, unstable, and hidden joints.",
            )
        )

    confidence_timeline = export.get("confidence_timeline", {})
    if isinstance(confidence_timeline, dict) and confidence_timeline:
        joints = [joint for joint in KEYPOINT_NAMES if joint in confidence_timeline]
        frame_count = max((len(confidence_timeline.get(joint, [])) for joint in joints), default=0)
        if joints and frame_count:
            matrix = []
            timestamps = []
            for joint in joints:
                row = []
                for point in confidence_timeline.get(joint, []):
                    row.append(float(point.get("confidence", 0) or 0))
                if len(row) < frame_count:
                    row.extend([0.0] * (frame_count - len(row)))
                matrix.append(row[:frame_count])
            first_row = confidence_timeline.get(joints[0], [])
            timestamps = [float(point.get("timestamp", 0) or 0) for point in first_row[:frame_count]]
            fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=130)
            image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_yticks(list(range(len(joints))))
            ax.set_yticklabels([joint.replace("_", " ") for joint in joints], fontsize=7)
            if timestamps:
                tick_positions = [0, len(timestamps) // 2, len(timestamps) - 1]
                ax.set_xticks(sorted(set(tick_positions)))
                ax.set_xticklabels([f"{timestamps[pos]:.1f}s" for pos in sorted(set(tick_positions))])
            ax.set_title(f"{view.title()} per-joint confidence heatmap")
            ax.set_xlabel("Time")
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Confidence")
            path = output / f"{view}_joint_confidence_heatmap.png"
            _save(fig, path)
            figures.append(
                _figure(
                    f"{view}_joint_confidence_heatmap",
                    f"{view.title()} confidence heatmap",
                    "tracking_quality",
                    path,
                    "Per-joint RTMPose confidence over time.",
                )
            )

    smoothing = export.get("smoothing_diagnostics", {})
    timeline = smoothing.get("frame_displacement_timeline", []) if isinstance(smoothing, dict) else []
    if timeline:
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
        ax.plot([float(point["timestamp"]) for point in timeline], [float(point.get("mean_displacement_px", 0) or 0) for point in timeline], color="#7c3aed")
        ax.set_title(f"{view.title()} smoothing before/after displacement")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean raw-to-smoothed distance (px)")
        ax.grid(True, alpha=0.25)
        path = output / f"{view}_smoothing_displacement_timeline.png"
        _save(fig, path)
        figures.append(
            _figure(
                f"{view}_smoothing_displacement_timeline",
                f"{view.title()} smoothing displacement",
                "error_noise",
                path,
                "Per-frame raw-to-smoothed displacement for temporal consistency review.",
            )
        )

    center = export.get("center_trajectory", [])
    if isinstance(center, list) and len(center) >= 2:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=130)
        xs = [float(point["x"]) for point in center if point.get("x") is not None]
        ys = [float(point["y"]) for point in center if point.get("y") is not None]
        if len(xs) >= 2 and len(ys) >= 2:
            ax.plot(xs, ys, color="#1d4ed8", linewidth=2)
            ax.scatter(xs[:1], ys[:1], color="#0f766e", s=36, label="start")
            ax.scatter(xs[-1:], ys[-1:], color="#dc2626", s=36, label="end")
            ax.invert_yaxis()
            ax.set_title(f"{view.title()} swimmer center trajectory")
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            path = output / f"{view}_center_trajectory.png"
            _save(fig, path)
            figures.append(_figure(f"{view}_center_trajectory", f"{view.title()} center trajectory", "trajectories", path, "Smoothed swimmer center path from cleaned pose bboxes."))

    roi_history = export.get("roi_history", [])
    if isinstance(roi_history, list) and roi_history:
        sources: dict[str, int] = {}
        for item in roi_history:
            source = str(item.get("source") or "unknown") if isinstance(item, dict) else "unknown"
            sources[source] = sources.get(source, 0) + 1
        fig, ax = plt.subplots(figsize=(8.5, 4.3), dpi=130)
        ax.bar(list(sources), list(sources.values()), color="#0f766e")
        ax.set_title(f"{view.title()} ROI source counts")
        ax.set_ylabel("Frames")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=20)
        path = output / f"{view}_roi_source_counts.png"
        _save(fig, path)
        figures.append(_figure(f"{view}_roi_source_counts", f"{view.title()} ROI source counts", "tracking_quality", path, "Selected ROI source distribution across sampled frames."))

        timeline = [item for item in roi_history if isinstance(item, dict) and item.get("timestamp") is not None]
        if timeline:
            times = [float(item.get("timestamp", 0) or 0) for item in timeline]
            widths = [float(item.get("width", 0) or 0) for item in timeline]
            heights = [float(item.get("height", 0) or 0) for item in timeline]
            fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
            ax.plot(times, widths, color="#1d4ed8", label="ROI width")
            ax.plot(times, heights, color="#0f766e", label="ROI height")
            ax.set_title(f"{view.title()} ROI size over time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pixels")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            path = output / f"{view}_roi_size_timeline.png"
            _save(fig, path)
            figures.append(_figure(f"{view}_roi_size_timeline", f"{view.title()} ROI size timeline", "tracking_quality", path, "Selected ROI crop width and height over time."))

            body_conf = [float(item.get("body_joint_confidence", item.get("confidence", 0)) or 0) for item in timeline]
            pose_conf = [float(item.get("confidence", 0) or 0) for item in timeline]
            fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=130)
            ax.plot(times, pose_conf, color="#64748b", label="pose confidence")
            ax.plot(times, body_conf, color="#dc2626", label="body-joint confidence")
            ax.set_ylim(0, 1)
            ax.set_title(f"{view.title()} ROI pose confidence over time")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Confidence")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            path = output / f"{view}_roi_pose_confidence_timeline.png"
            _save(fig, path)
            figures.append(_figure(f"{view}_roi_pose_confidence_timeline", f"{view.title()} ROI pose confidence", "tracking_quality", path, "Pose and body-joint confidence for selected ROI frames."))

            states = ["LOCKED", "SUSPECT", "LOST", "REACQUIRE", "REVIEW_ONLY"]
            state_values = [states.index(str(item.get("tracking_state"))) if str(item.get("tracking_state")) in states else -1 for item in timeline]
            if any(value >= 0 for value in state_values):
                fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=130)
                ax.step(times, state_values, where="post", color="#7c3aed")
                drift_times = [float(item.get("timestamp", 0) or 0) for item in timeline if item.get("drift_reasons")]
                if drift_times:
                    ax.scatter(drift_times, [state_values[index] for index, item in enumerate(timeline) if item.get("drift_reasons")], color="#dc2626", s=18, label="drift")
                ax.set_yticks(list(range(len(states))))
                ax.set_yticklabels(states, fontsize=8)
                ax.set_title(f"{view.title()} tracking state timeline")
                ax.set_xlabel("Time (s)")
                ax.grid(True, axis="x", alpha=0.25)
                if drift_times:
                    ax.legend(fontsize=8)
                path = output / f"{view}_tracking_state_timeline.png"
                _save(fig, path)
                figures.append(_figure(f"{view}_tracking_state_timeline", f"{view.title()} tracking state", "tracking_quality", path, "LOCKED/SUSPECT/LOST/REACQUIRE/REVIEW_ONLY state timeline with drift markers."))

    return figures


def _velocity_motion_figures(plt: Any, output: Path, velocity: dict[str, Any]) -> list[ResearchFigureMetadata]:
    series = [point for point in velocity.get("series", []) if isinstance(point, dict)]
    if len(series) < 2:
        return []
    figures: list[ResearchFigureMetadata] = []
    times = [float(point.get("timestamp", 0) or 0) for point in series]
    corrected = [float(point.get("velocity", 0) or 0) for point in series]
    raw = [float(point.get("raw_velocity", point.get("velocity", 0)) or 0) for point in series]
    accelerations = [float(point.get("acceleration", 0) or 0) for point in series]
    unit = str(velocity.get("summary", {}).get("velocity_unit") or "m/s")
    accel_unit = "m/s^2" if unit == "m/s" else "px/s^2"

    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=130)
    ax.plot(times, corrected, color="#0f766e", linewidth=2.0, label="corrected")
    ax.set_title("Smoothed velocity curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = output / "smoothed_velocity_curve.png"
    _save(fig, path)
    figures.append(_figure("smoothed_velocity_curve", "Smoothed velocity curve", "tracking_quality", path, "Physics-corrected swimmer velocity after median and rolling-average smoothing."))

    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=130)
    ax.plot(times, raw, color="#94a3b8", linewidth=1.4, linestyle="--", label="raw")
    ax.plot(times, corrected, color="#1d4ed8", linewidth=2.0, label="corrected")
    ax.set_title("Raw vs corrected velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    path = output / "raw_vs_corrected_velocity.png"
    _save(fig, path)
    figures.append(_figure("raw_vs_corrected_velocity", "Raw vs corrected velocity", "tracking_quality", path, "Comparison of raw centroid speed and corrected physics-aware speed."))

    fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=130)
    ax.plot(times, accelerations, color="#dc2626", linewidth=1.8)
    ax.axhline(0, color="#475569", linewidth=0.8)
    ax.set_title("Velocity acceleration")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(accel_unit)
    ax.grid(True, alpha=0.25)
    path = output / "velocity_acceleration_plot.png"
    _save(fig, path)
    figures.append(_figure("velocity_acceleration_plot", "Velocity acceleration", "tracking_quality", path, "Acceleration after velocity correction; spikes indicate residual tracking or camera artifacts."))

    anomaly_counts = velocity.get("summary", {}).get("motion_validation", {}).get("anomaly_counts", {})
    if isinstance(anomaly_counts, dict) and anomaly_counts:
        fig, ax = plt.subplots(figsize=(8.8, 4.5), dpi=130)
        labels = list(anomaly_counts.keys())
        values = [int(anomaly_counts[label]) for label in labels]
        ax.bar(labels, values, color="#7c2d12")
        ax.set_title("Motion anomaly diagnostics")
        ax.set_ylabel("Frames")
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=25)
        path = output / "motion_anomaly_diagnostics.png"
        _save(fig, path)
        figures.append(_figure("motion_anomaly_diagnostics", "Motion anomaly diagnostics", "tracking_quality", path, "Detected velocity, acceleration, plateau, ROI, and camera-motion anomalies."))

    return figures


def _plot_axis(ax: Any, data: list[dict[str, Any]], axis: str, label: str, color: str, style: str) -> None:
    samples = [(float(point["timestamp"]), point.get(axis)) for point in data if point.get(axis) is not None and point.get("visibility_state") not in {"missing", "skipped", "rejected_outlier"}]
    if not samples:
        return
    ax.plot([sample[0] for sample in samples], [float(sample[1]) for sample in samples], style, color=color, linewidth=1.8, label=label)


def _has_points(data: list[dict[str, Any]]) -> bool:
    return any(point.get("x") is not None and point.get("y") is not None and point.get("visibility_state") not in {"missing", "skipped", "rejected_outlier"} for point in data)


def _motion_noise_proxy(data: list[dict[str, Any]]) -> float:
    samples = _xy_samples(data)
    if len(samples) < 4:
        return 0.0
    velocities = []
    accelerations = []
    previous_velocity = 0.0
    for first, second in zip(samples, samples[1:]):
        dt = max(1e-3, second[0] - first[0])
        velocity = (((second[1] - first[1]) ** 2 + (second[2] - first[2]) ** 2) ** 0.5) / dt
        velocities.append(velocity)
        accelerations.append(abs(velocity - previous_velocity) / dt)
        previous_velocity = velocity
    velocity_std = pstdev(velocities) if len(velocities) > 1 else 0.0
    acceleration_std = pstdev(accelerations) if len(accelerations) > 1 else 0.0
    return round(velocity_std + acceleration_std * 0.015, 3)


def _mean_displacement(raw: list[dict[str, Any]], filtered: list[dict[str, Any]]) -> float:
    distances = []
    by_frame = {point["frame_index"]: point for point in filtered if point.get("x") is not None and point.get("y") is not None}
    for point in raw:
        other = by_frame.get(point.get("frame_index"))
        if other is None or point.get("x") is None or point.get("y") is None:
            continue
        distances.append(((float(point["x"]) - float(other["x"])) ** 2 + (float(point["y"]) - float(other["y"])) ** 2) ** 0.5)
    return round(mean(distances), 3) if distances else 0.0


def _axis_std(data: list[dict[str, Any]], axis: str) -> float:
    values = [float(point[axis]) for point in data if point.get(axis) is not None and point.get("visibility_state") not in {"missing", "skipped", "rejected_outlier"}]
    return round(pstdev(values), 3) if len(values) > 1 else 0.0


def _xy_samples(data: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    return [
        (float(point["timestamp"]), float(point["x"]), float(point["y"]))
        for point in data
        if point.get("x") is not None and point.get("y") is not None and point.get("visibility_state") not in {"missing", "skipped", "rejected_outlier"}
    ]


def _peak_points(data: list[dict[str, Any]], axis: str) -> list[tuple[float, float]]:
    samples = [(float(point["timestamp"]), float(point[axis])) for point in data if point.get(axis) is not None and point.get("visibility_state") not in {"missing", "skipped", "rejected_outlier"}]
    if len(samples) < 5:
        return []
    values = [sample[1] for sample in samples]
    span = max(values) - min(values)
    if span < 4:
        return []
    threshold = min(values) + span * 0.58
    peaks = []
    for index in range(1, len(samples) - 1):
        if values[index] >= values[index - 1] and values[index] > values[index + 1] and values[index] >= threshold:
            peaks.append(samples[index])
    return peaks[:32]


def _mean_confidence_timeline(export: dict[str, Any]) -> list[dict[str, float]]:
    timeline = export.get("confidence_timeline", {})
    if not timeline:
        return []
    frame_values: dict[int, dict[str, Any]] = {}
    for joint_values in timeline.values():
        for point in joint_values:
            frame_index = int(point["frame_index"])
            entry = frame_values.setdefault(frame_index, {"timestamp": float(point["timestamp"]), "values": []})
            entry["values"].append(float(point.get("confidence", 0) or 0))
    return [
        {"timestamp": entry["timestamp"], "confidence": round(mean(entry["values"]), 4) if entry["values"] else 0.0}
        for _, entry in sorted(frame_values.items())
    ]


def _figure(figure_id: str, title: str, type_: str, path: Path, description: str) -> ResearchFigureMetadata:
    return ResearchFigureMetadata(
        figure_id=figure_id,
        title=title,
        type=type_,
        file_path=str(path),
        description=description,
        metric_source="proxy",
    )


def _save(fig: Any, path: Path) -> None:
    try:
        fig.tight_layout()
    except MemoryError:
        pass
    fig.savefig(path, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.close(fig)
    except Exception:
        fig.clear()

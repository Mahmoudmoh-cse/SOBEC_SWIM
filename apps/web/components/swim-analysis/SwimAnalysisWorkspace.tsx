"use client";

import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock3,
  Download,
  Eye,
  FileVideo,
  Gauge,
  GaugeCircle,
  Image as ImageIcon,
  LocateFixed,
  RefreshCw,
  ShieldCheck,
  Split,
  Timer,
  UploadCloud,
  Waves,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AppShell } from "@/components/AppShell";
import { MetricCard } from "@/components/MetricCard";
import { StatusPill } from "@/components/StatusPill";
import { api } from "@/lib/api";
import type { SwimAnalysisMetric, SwimAnalysisReport, SwimAnalysisStatus, Swimmer } from "@/lib/types";
import { useAuthStore } from "@/store/auth";

const dashboardMetrics = [
  ["stroke_rate_spm", "Stroke Rate"],
  ["distance_per_stroke", "Distance/Stroke"],
  ["catch_angle", "Catch Angle"],
  ["body_alignment_score", "Body Alignment"],
  ["arm_symmetry_score", "Arm Symmetry"],
  ["head_stability_score", "Head Stability"],
  ["hip_stability_score", "Hip Stability"],
  ["kick_frequency", "Kick Frequency"],
  ["left_right_symmetry_estimate", "L/R Symmetry"],
  ["tracking_quality_score", "Tracking Quality"],
  ["joint_visibility_score", "Joint Visibility"]
] as const;

type ReviewEvent = {
  label: string;
  time: number;
  type: "phase" | "fault" | "finding" | "split";
  confidence: number;
};

type ReviewRow = {
  label: string;
  timeRange: string;
  strokes: string;
  tempo: string;
  dpc: string;
  split15: string;
  velocity: string;
  turnFinish: string;
  confidence: string;
};

type ResearchFigureItem = NonNullable<SwimAnalysisReport["research_figures"]>[number];

export default function SwimAnalysisPage() {
  const token = useAuthStore((state) => state.token);
  const [swimmers, setSwimmers] = useState<Swimmer[]>([]);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [status, setStatus] = useState<SwimAnalysisStatus | null>(null);
  const [report, setReport] = useState<SwimAnalysisReport | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!token) return;
    api.swimmers(token).then(setSwimmers).catch(() => setSwimmers([]));
  }, [token]);

  const refreshStatus = useCallback(async () => {
    if (!token || !analysisId) return;
    const nextStatus = await api.swimAnalysisStatus(token, analysisId);
    setStatus(nextStatus);
    if (nextStatus.status === "completed" || nextStatus.status === "failed") {
      const nextReport = await api.swimAnalysisReport(token, analysisId);
      setReport(nextReport);
    }
  }, [analysisId, token]);

  useEffect(() => {
    if (!analysisId || report || !token) return;
    const id = window.setInterval(() => {
      refreshStatus().catch((err) => setError(err instanceof Error ? err.message : "Unable to refresh analysis"));
    }, 2000);
    refreshStatus().catch((err) => setError(err instanceof Error ? err.message : "Unable to refresh analysis"));
    return () => window.clearInterval(id);
  }, [analysisId, refreshStatus, report, token]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    setUploading(true);
    setError(null);
    setReport(null);
    setStatus(null);
    try {
      const response = await api.uploadSwimAnalysis(token, form);
      setAnalysisId(response.analysis_id);
      setStatus({
        analysis_id: response.analysis_id,
        status: response.status,
        progress: 0,
        stroke_type: String(form.get("stroke_type") ?? "freestyle"),
        pose_backend: "rtmpose",
        swimmer_id: String(form.get("swimmer_id") || "") || null,
        video_metadata: {},
        processing_errors: [],
        error_message: null,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start swim analysis");
    } finally {
      setUploading(false);
    }
  }

  const qualityWarning = useMemo(() => {
    if (!report) return null;
    const singleVideoReport = isSingleVideoReport(report);
    const qualityViews = singleVideoReport
      ? [report.video_quality.video ?? report.video_quality.side ?? {}]
      : [report.video_quality.side ?? {}, report.video_quality.front ?? {}];
    const usable = Math.min(...qualityViews.map((view) => Number(view.usable_frame_ratio ?? 1)));
    return usable < 0.55 ? "Video quality was weak; low-confidence metrics should be treated as review prompts, not conclusions." : null;
  }, [report]);

  return (
    <AppShell>
      <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ink">Swim Motion Analysis</h1>
          <p className="mt-1 text-sm text-slate-500">Phase 1 analysis using one full-body smartphone video.</p>
        </div>
        {status ? (
          <button className="secondary-button" type="button" onClick={() => refreshStatus()} disabled={!analysisId}>
            <RefreshCw size={16} />
            Refresh
          </button>
        ) : null}
      </div>

      {error ? <div className="mb-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}

      <section className="panel p-5">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
              <FileVideo size={19} />
              Source Video
            </h2>
            <p className="mt-1 text-sm text-slate-500">Use a full-body, fixed-camera clip recorded during the swim.</p>
          </div>
          <StatusPill tone="violet">RTMPose primary / OpenCV fallback</StatusPill>
        </div>

        <form onSubmit={onSubmit} className="grid gap-4 lg:grid-cols-4">
          <label className="lg:col-span-2">
            <span className="label">Video</span>
            <input className="field mt-1" type="file" name="video" accept="video/*" required />
          </label>
          <label>
            <span className="label">Swimmer</span>
            <select className="field mt-1" name="swimmer_id" defaultValue="">
              <option value="">Unassigned</option>
              {swimmers.map((swimmer) => (
                <option key={swimmer.id} value={swimmer.id}>
                  {swimmer.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="label">Stroke</span>
            <select className="field mt-1" name="stroke_type" defaultValue="freestyle">
              <option value="freestyle">Freestyle</option>
              <option value="backstroke">Backstroke</option>
              <option value="breaststroke">Breaststroke</option>
              <option value="butterfly">Butterfly</option>
            </select>
          </label>
          <label>
            <span className="label">Quality Mode</span>
            <select className="field mt-1" name="quality_mode" defaultValue="balanced">
              <option value="high_accuracy">High Accuracy</option>
              <option value="balanced">Balanced</option>
            </select>
          </label>
          <label>
            <span className="label">Target FPS</span>
            <input className="field mt-1" type="number" name="target_fps" min={3} max={30} defaultValue={6} />
          </label>
          <label>
            <span className="label">Lane Length M</span>
            <input className="field mt-1" type="number" name="lane_length_m" min={5} max={100} step="0.5" placeholder="25 or 50" />
          </label>
          <div className="lg:col-span-3">
            <button className="primary-button" type="submit" disabled={uploading}>
              <UploadCloud size={16} />
              {uploading ? "Uploading" : "Start Analysis"}
            </button>
          </div>
        </form>
      </section>

      {status ? <StatusPanel status={status} /> : null}
      {qualityWarning ? (
        <div className="mt-6 flex items-start gap-3 rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <AlertTriangle size={18} />
          <span>{qualityWarning}</span>
        </div>
      ) : null}
      {report ? <ReportDashboard report={report} /> : null}
    </AppShell>
  );
}

function StatusPanel({ status }: { status: SwimAnalysisStatus }) {
  const tone = status.status === "failed" ? "coral" : status.status === "completed" ? "mint" : "water";
  return (
    <section className="panel mt-6 p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
          <Activity size={18} />
          Processing Status
        </h2>
        <StatusPill tone={tone}>{status.status}</StatusPill>
      </div>
      <div className="h-3 overflow-hidden rounded-md bg-slate-100">
        <div className="h-full bg-water transition-all" style={{ width: `${Math.max(0, Math.min(100, status.progress))}%` }} />
      </div>
      <div className="mt-3 grid gap-3 text-sm text-slate-600 md:grid-cols-3">
        <span>Progress: {status.progress}%</span>
        <span>Stroke: {status.stroke_type}</span>
        <span>Pose backend: {status.pose_backend}</span>
      </div>
      {status.error_message ? <p className="mt-3 text-sm text-red-700">{status.error_message}</p> : null}
    </section>
  );
}

function ReportDashboard({ report }: { report: SwimAnalysisReport }) {
  const videoUrl = (value: string | null | undefined) => (value ? api.videoUrl(value) : null);
  const sideVideo = videoUrl(report.artifacts.video_annotated_video_url ?? report.artifacts.side_annotated_video_url);
  const frontVideo = videoUrl(report.artifacts.front_annotated_video_url);
  const singleVideoReport = isSingleVideoReport(report);
  const reliability = report.summary.reliability_score ?? report.metrics.analysis_reliability_score?.value ?? report.summary.confidence_score;
  const overallValue =
    Number(report.summary.confidence_score ?? 0) < 0.35 || Number(report.summary.overall_score ?? 0) <= 0
      ? "Not enough confidence"
      : report.summary.overall_score;
  return (
    <div className="mt-6 space-y-6">
      <div className="grid gap-4 md:grid-cols-4">
        <MetricCard label="Overall Technique" value={overallValue} detail="Confidence-weighted landmark score" icon={Gauge} tone="water" />
        <MetricCard label="Reliability" value={typeof reliability === "number" ? formatPercent(reliability > 1 ? reliability / 100 : reliability) : reliability} detail="Tracking and visibility gate" icon={ShieldCheck} tone="mint" />
        <MetricCard label="Confidence" value={formatPercent(report.summary.confidence_score)} detail="Visibility and metric support" icon={ShieldCheck} tone="mint" />
        <MetricCard label="Data Quality" value={formatPercent(report.summary.data_quality_score)} detail="Blur, lighting, stability" icon={BarChart3} tone="violet" />
      </div>

      {report.warnings?.length ? <WarningsPanel warnings={report.warnings} /> : null}
      <PoseDiagnosticsPanel report={report} />

      <section className="grid gap-4 lg:grid-cols-5">
        {dashboardMetrics.map(([key, label]) => (
          <MetricTile key={key} label={label} metric={report.metrics[key]} />
        ))}
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
        <JointVisibilityChart report={report} singleVideo={singleVideoReport} />
        <TrackingQualitySummary report={report} singleVideo={singleVideoReport} />
      </section>

      <ResearchVisualizations report={report} singleVideo={singleVideoReport} />

      <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <VelocityChart report={report} />
        <PhaseTimeline report={report} />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <FaultList report={report} />
        <CoachingReport report={report} />
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <div className="panel p-5">
          <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
            <AlertTriangle size={18} />
            Findings
          </h2>
          <div className="mt-4 space-y-3">
            {report.findings.length ? (
              report.findings.map((finding, index) => (
                <div key={`${finding.type}-${index}`} className="rounded-md border border-slate-200 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-semibold text-ink">{finding.type.replace(/_/g, " ")}</p>
                    <StatusPill tone={finding.severity === "high" ? "coral" : "water"}>{finding.severity}</StatusPill>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-600">{finding.message}</p>
                  <p className="mt-1 text-xs font-semibold text-slate-500">Confidence {formatPercent(finding.confidence)}</p>
                </div>
              ))
            ) : (
              <p className="rounded-md border border-slate-200 p-3 text-sm text-slate-500">No medium or high-confidence technique warnings crossed the Phase 1 thresholds.</p>
            )}
          </div>
        </div>

        <div className="panel p-5">
          <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
            <Brain size={18} />
            Recommendations
          </h2>
          <div className="mt-4 space-y-3">
            {report.recommendations.map((item) => (
              <div key={`${item.priority}-${item.linked_metric}`} className="rounded-md border border-slate-200 p-3">
                <p className="text-xs font-black uppercase text-water">Priority {item.priority}</p>
                <p className="mt-2 text-sm leading-6 text-slate-700">{item.message}</p>
                <p className="mt-1 text-xs text-slate-500">{item.linked_metric.replace(/_/g, " ")}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="panel p-5">
        <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
          <Waves size={18} />
          {singleVideoReport ? "Annotated Video" : "Annotated Views"}
        </h2>
        {singleVideoReport ? (
          <div className="mt-4">
            <VideoPreview label="Video" src={sideVideo ?? frontVideo} />
          </div>
        ) : (
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <VideoPreview label="Side View" src={sideVideo} />
            <VideoPreview label="Front / Back View" src={frontVideo} />
          </div>
        )}
      </section>

      <button className="secondary-button" type="button" onClick={() => downloadReport(report)}>
        <Download size={16} />
        Download Coaching Report JSON
      </button>

      {report.processing_errors?.length ? (
        <section className="panel p-5">
          <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
            <CheckCircle2 size={18} />
            Processing Notes
          </h2>
          <ul className="mt-3 space-y-2 text-sm text-slate-600">
            {report.processing_errors.map((item, index) => (
              <li key={`${item}-${index}`}>{item}</li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function WarningsPanel({ warnings }: { warnings: string[] }) {
  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <AlertTriangle size={18} />
        Analysis Warnings
      </h2>
      <div className="mt-3 grid gap-2 md:grid-cols-2">
        {warnings.slice(0, 6).map((warning, index) => (
          <div key={`${warning}-${index}`} className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm leading-6 text-amber-900">
            {warning}
          </div>
        ))}
      </div>
    </section>
  );
}

function PoseDiagnosticsPanel({ report }: { report: SwimAnalysisReport }) {
  const diagnostics = isRecord(report.pose_diagnostics) ? report.pose_diagnostics : {};
  const roi = isRecord(diagnostics.roi) ? diagnostics.roi : {};
  const tracking = isRecord(diagnostics.tracking) ? diagnostics.tracking : {};
  const sourceCounts = isRecord(roi.source_counts) ? roi.source_counts : {};
  const sourceSummary = Object.entries(sourceCounts)
    .map(([source, count]) => `${source.replace(/_/g, " ")} ${count}`)
    .join(", ");
  const reliableJoints = stringArray(tracking.reliable_joints);
  const estimatedJoints = stringArray(tracking.estimated_joints);
  const unstableJoints = stringArray(tracking.unstable_joints);
  const hiddenJoints = stringArray(tracking.hidden_joints);
  const roiTimeline = Array.isArray(tracking.roi_source_timeline) ? tracking.roi_source_timeline.filter(isRecord).slice(0, 8) : [];
  const warnings = Array.isArray(diagnostics.warnings) ? diagnostics.warnings.map(String) : [];
  if (!Object.keys(diagnostics).length) return null;

  return (
    <section className="panel p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
          <LocateFixed size={18} />
          Pose Diagnostics
        </h2>
        <StatusPill tone={String(diagnostics.actual_backend ?? "") === "rtmpose" ? "mint" : "water"}>
          Actual {String(diagnostics.actual_backend ?? "unknown")}
        </StatusPill>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-4">
        <Stat label="Requested" value={diagnostics.requested_backend ?? "--"} />
        <Stat label="Actual" value={diagnostics.actual_backend ?? "--"} />
        <Stat label="Probable cause" value={diagnostics.probable_cause ?? "--"} />
        <Stat label="Sampled frames" value={diagnostics.frames_sampled ?? "--"} />
        <Stat label="Usable pose" value={diagnostics.usable_pose_frames ?? "--"} />
        <Stat label="Skipped" value={diagnostics.frames_skipped ?? "--"} />
        <Stat label="Avg confidence" value={formatDecimal(diagnostics.average_pose_confidence, 3)} />
        <Stat label="Visible gate" value={formatDecimal(diagnostics.temporal_visible_confidence, 2)} />
        <Stat label="Tracking quality" value={formatPercent(asNumber(tracking.average_tracking_quality) ?? 0)} />
        <Stat label="Temporal stability" value={formatPercent(asNumber(tracking.temporal_stability) ?? 0)} />
        <Stat label="Motion smoothness" value={formatPercent(asNumber(tracking.motion_smoothness) ?? 0)} />
        <Stat label="Smoothing" value={`${tracking.smoothing_status ?? "--"} ${formatDecimal(tracking.mean_smoothing_displacement_px, 1)}px`} />
        <Stat label="Phase confidence" value={formatPercent(asNumber(tracking.phase_detection_confidence) ?? 0)} />
        <Stat label="Raw joints/frame" value={formatDecimal(diagnostics.average_raw_joint_count, 1)} />
        <Stat label="Filtered joints/frame" value={formatDecimal(diagnostics.average_filtered_joint_count, 1)} />
        <Stat label="Removed by threshold" value={diagnostics.threshold_removed_joint_count ?? "--"} />
        <Stat label="Reliable joints" value={reliableJoints.length ? reliableJoints.map(cleanJointName).join(", ") : "--"} />
        <Stat label="Estimated joints" value={estimatedJoints.length ? estimatedJoints.map(cleanJointName).join(", ") : "--"} />
        <Stat label="Unstable joints" value={unstableJoints.length ? unstableJoints.map(cleanJointName).join(", ") : "--"} />
        <Stat label="Hidden joints" value={hiddenJoints.length ? hiddenJoints.map(cleanJointName).join(", ") : "--"} />
        <Stat label="ROI compare" value={truthyText(roi.compare_full_frame)} />
        <Stat label="Full-frame wins" value={roi.selected_full_frame_frames ?? "--"} />
        <Stat label="ROI wins" value={roi.selected_roi_frames ?? "--"} />
        <Stat label="Full fallback %" value={formatDecimal(roi.full_frame_fallback_percentage, 1)} />
        <Stat label="ROI fallback" value={roi.invalid_full_frame_fallback_frames ?? "--"} />
        <Stat label="Boundary crops" value={roi.boundary_clip_frames ?? "--"} />
        <Stat label="ROI score" value={formatDecimal(roi.average_selected_candidate_score, 3)} />
        <Stat label="ROI temporal jump" value={formatDecimal(roi.average_temporal_jump_ratio, 2)} />
        <Stat label="High ROI jumps" value={roi.high_temporal_jump_frames ?? "--"} />
        <Stat label="ROI sources" value={sourceSummary || "--"} />
      </div>
      {diagnostics.fallback_reason ? (
        <p className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          Fallback reason: {String(diagnostics.fallback_reason)}
        </p>
      ) : null}
      {roiTimeline.length ? (
        <div className="mt-4">
          <p className="label">ROI Source Timeline</p>
          <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {roiTimeline.map((entry, index) => (
              <div key={`${entry.view ?? "video"}-${entry.frame_index ?? index}`} className="rounded-md border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                <p className="font-bold text-ink">{formatDecimal(entry.timestamp, 2)}s / {String(entry.source ?? "unknown").replace(/_/g, " ")}</p>
                <p>{String(entry.selected ?? "roi").replace(/_/g, " ")} score {formatDecimal(entry.candidate_score, 2)}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {warnings.length ? (
        <div className="mt-4 grid gap-2 md:grid-cols-2">
          {warnings.slice(0, 4).map((warning, index) => (
            <div key={`${warning}-${index}`} className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm leading-6 text-amber-900">
              {warning}
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function isSingleVideoReport(report: SwimAnalysisReport) {
  if (report.video_metadata?.input_mode === "single_video") return true;
  if (report.artifacts.video_annotated_video_url) return true;
  if (report.video_quality.video) return true;
  if (report.trajectories?.video_visibility_percentage || report.trajectories?.video_tracking_quality) return true;
  const sideMetadataPath = metadataViewPath(report, "side");
  const frontMetadataPath = metadataViewPath(report, "front");
  return Boolean(sideMetadataPath && frontMetadataPath && sideMetadataPath === frontMetadataPath);
}

function metadataViewPath(report: SwimAnalysisReport, view: string) {
  const value = report.video_metadata?.[view];
  if (!isRecord(value)) return "";
  return String(value.path ?? "");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown) {
  return Array.isArray(value) ? value.map(String) : [];
}

function cleanJointName(value: string) {
  return value.replace(/_/g, " ");
}

function normalizeSingleVideoFigures(figures: ResearchFigureItem[]) {
  return figures
    .filter((figure) => !isFrontViewFigure(figure))
    .map((figure) => ({
      ...figure,
      figure_id: figure.figure_id.replace(/^side_/, "video_"),
      title: figure.title.replace(/^Side\b/, "Video")
    }));
}

function isFrontViewFigure(figure: ResearchFigureItem) {
  return figure.figure_id.startsWith("front_") || figure.title.startsWith("Front ");
}

function JointVisibilityChart({ report, singleVideo }: { report: SwimAnalysisReport; singleVideo: boolean }) {
  const video = report.trajectories?.video_visibility_percentage ?? report.trajectories?.side_visibility_percentage ?? {};
  const side = report.trajectories?.side_visibility_percentage ?? {};
  const front = report.trajectories?.front_visibility_percentage ?? {};
  const joints = singleVideo ? Object.keys(video) : Array.from(new Set([...Object.keys(side), ...Object.keys(front)]));
  const data = joints.map((joint) => ({
    joint: joint.replace(/_/g, " "),
    video: Number(video[joint] ?? 0),
    side: Number(side[joint] ?? 0),
    front: Number(front[joint] ?? 0)
  }));

  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <BarChart3 size={18} />
        Joint Visibility
      </h2>
      <div className="mt-4 h-80">
        {data.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.slice(0, 17)} margin={{ left: 0, right: 12, bottom: 56 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="joint" tickLine={false} axisLine={false} angle={-45} textAnchor="end" interval={0} height={72} tick={{ fontSize: 11 }} />
              <YAxis tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip />
              {singleVideo ? (
                <Bar dataKey="video" fill="#1d4ed8" name="Video" radius={[3, 3, 0, 0]} />
              ) : (
                <>
                  <Bar dataKey="side" fill="#1d4ed8" name="Side view" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="front" fill="#0f766e" name="Front/back view" radius={[3, 3, 0, 0]} />
                </>
              )}
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-md bg-slate-50 text-sm text-slate-500">Visibility data unavailable.</div>
        )}
      </div>
    </section>
  );
}

function TrackingQualitySummary({ report, singleVideo }: { report: SwimAnalysisReport; singleVideo: boolean }) {
  const video = qualitySeries(report.trajectories?.video_tracking_quality ?? report.trajectories?.side_tracking_quality);
  const side = qualitySeries(report.trajectories?.side_tracking_quality);
  const front = qualitySeries(report.trajectories?.front_tracking_quality);
  const combined = singleVideo ? video.map((point) => ({ time: point.time, video: point.quality })) : mergeQualitySeries(side, front);
  const videoAvg = averageQuality(video);
  const sideAvg = averageQuality(side);
  const frontAvg = averageQuality(front);

  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <LocateFixed size={18} />
        Tracking Quality
      </h2>
      <div className={singleVideo ? "mt-4 grid gap-3" : "mt-4 grid gap-3 sm:grid-cols-2"}>
        {singleVideo ? (
          <Stat label="Video avg" value={videoAvg === null ? "--" : formatPercent(videoAvg)} />
        ) : (
          <>
            <Stat label="Side avg" value={sideAvg === null ? "--" : formatPercent(sideAvg)} />
            <Stat label="Front avg" value={frontAvg === null ? "--" : formatPercent(frontAvg)} />
          </>
        )}
      </div>
      <div className="mt-4 h-56">
        {combined.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={combined}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} domain={[0, 1]} />
              <Tooltip />
              {singleVideo ? (
                <Line type="monotone" dataKey="video" stroke="#1d4ed8" dot={false} strokeWidth={2} />
              ) : (
                <>
                  <Line type="monotone" dataKey="side" stroke="#1d4ed8" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="front" stroke="#0f766e" dot={false} strokeWidth={2} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-md bg-slate-50 text-sm text-slate-500">Tracking quality timeline unavailable.</div>
        )}
      </div>
    </section>
  );
}

function ResearchVisualizations({ report, singleVideo }: { report: SwimAnalysisReport; singleVideo: boolean }) {
  const [tab, setTab] = useState("trajectories");
  const tabs = [
    ["trajectories", "Trajectories"],
    ["tracking_quality", "Tracking Quality"],
    ["error_noise", "Error/Noise Analysis"],
    ["stroke_cycles", "Stroke Cycles"]
  ] as const;
  const allFigures = singleVideo ? normalizeSingleVideoFigures(report.research_figures ?? []) : report.research_figures ?? [];
  const figures = allFigures.filter((figure) => figure.type === tab);

  if (!allFigures.length) {
    return (
      <section className="panel p-5">
        <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
          <ImageIcon size={18} />
          Research Visualizations
        </h2>
        <p className="mt-3 rounded-md border border-slate-200 p-3 text-sm text-slate-500">No research figures were generated for this run.</p>
      </section>
    );
  }

  return (
    <section className="panel p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
          <ImageIcon size={18} />
          Research Visualizations
        </h2>
        <div className="flex flex-wrap gap-2">
          {tabs.map(([value, label]) => (
            <button
              key={value}
              type="button"
              className={value === tab ? "primary-button px-3 py-2 text-xs" : "secondary-button px-3 py-2 text-xs"}
              onClick={() => setTab(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        {figures.length ? (
          figures.slice(0, 12).map((figure) => <ResearchFigureCard key={figure.figure_id} figure={figure} />)
        ) : (
          <p className="rounded-md border border-slate-200 p-3 text-sm text-slate-500">No figures in this category.</p>
        )}
      </div>
    </section>
  );
}

function ResearchFigureCard({ figure }: { figure: ResearchFigureItem }) {
  const src = api.videoUrl(figure.file_path);
  return (
    <div className="rounded-md border border-slate-200 p-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="font-bold text-ink">{figure.title}</p>
          <p className="mt-1 text-sm leading-6 text-slate-600">{figure.description}</p>
        </div>
        <StatusPill tone={figure.metric_source === "ground_truth" ? "mint" : "slate"}>{figure.metric_source}</StatusPill>
      </div>
      {src ? (
        <img className="mt-3 w-full rounded-md border border-slate-100 bg-white object-contain" src={src} alt={figure.title} />
      ) : (
        <div className="mt-3 flex aspect-video items-center justify-center rounded-md bg-slate-100 text-sm text-slate-500">Figure unavailable</div>
      )}
    </div>
  );
}

function CoachReviewPanel({
  report,
  sideVideo,
  frontVideo
}: {
  report: SwimAnalysisReport;
  sideVideo: string | null;
  frontVideo: string | null;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const events = reviewEvents(report);
  const rows = reviewRows(report);
  const primaryVideo = sideVideo ?? frontVideo;
  const velocitySummary = report.velocity?.summary ?? {};
  const evidenceMode = String(velocitySummary.evidence_mode ?? report.summary.pose_backend ?? "pose");
  const warnings = coachVisibleVelocityWarnings(velocitySummary.warnings);
  const duration = Number(report.summary.duration_sec || 0);
  const canSeek = Boolean(primaryVideo);

  const seekToEvent = useCallback(
    (time: number) => {
      const video = videoRef.current;
      if (!video || !Number.isFinite(time)) return;
      const maxTime = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : duration || time;
      video.currentTime = Math.max(0, Math.min(time, maxTime));
      video.focus();
    },
    [duration]
  );

  return (
    <section className="panel overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-slate-200 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
            <Eye size={18} />
            Coach Review
          </h2>
          <p className="mt-1 text-sm text-slate-500">
            Video-first review with phase markers, velocity context, and split-style metric rows.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusPill tone={evidenceMode === "centroid_only" ? "slate" : "mint"}>{evidenceMode.replace(/_/g, " ")}</StatusPill>
          <StatusPill tone={Number(velocitySummary.confidence ?? 0) >= 0.45 ? "mint" : "water"}>
            {formatConfidence(Number(velocitySummary.confidence ?? 0))} motion
          </StatusPill>
        </div>
      </div>

      <div className="grid gap-0 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.75fr)]">
        <div className="bg-slate-950">
          <div className="relative aspect-video min-h-[220px] overflow-hidden bg-slate-950 sm:min-h-[320px]">
            {primaryVideo ? (
              <video ref={videoRef} className="h-full w-full object-contain" src={primaryVideo} controls />
            ) : (
              <div className="flex h-full w-full items-center justify-center bg-slate-900 text-sm text-slate-300">
                Annotated review video unavailable
              </div>
            )}

            <div className="pointer-events-none absolute inset-x-0 top-0 hidden items-center justify-between bg-slate-950/70 px-4 py-3 text-white backdrop-blur sm:flex">
              <div>
                <p className="text-sm font-bold">AquaIQ Review</p>
                <p className="text-xs text-slate-300">
                  {report.summary.stroke_type} / {duration ? `${duration.toFixed(1)}s` : "duration unavailable"}
                </p>
              </div>
              <div className="flex items-center gap-3 text-xs font-semibold">
                <span>{formatCompactMetric(report.metrics.stroke_rate_spm)} SPM</span>
                <span>{formatCompactMetric(report.metrics.distance_per_stroke)} DPS</span>
                <span>{formatVelocityValue(velocitySummary.average_velocity, String(velocitySummary.velocity_unit ?? ""))}</span>
              </div>
            </div>
          </div>

          <DetectedEventsSection events={events} duration={duration} canSeek={canSeek} onSeek={seekToEvent} />

          {warnings.length ? (
            <div className="border-t border-amber-300/40 bg-amber-50 px-5 py-3 text-sm text-amber-900">
              {String(warnings[0])}
            </div>
          ) : null}
        </div>

        <div className="border-t border-slate-200 bg-white xl:border-l xl:border-t-0">
          <div className="grid grid-cols-2 gap-3 p-5">
            <ReviewStat label="Lap Time" value={formatClock(duration)} icon={Timer} />
            <ReviewStat label="Velocity" value={formatVelocity(report)} icon={GaugeCircle} />
            <ReviewStat label="Tempo" value={formatMetricValueForDisplay(report.metrics.stroke_rate_spm)} icon={Clock3} />
            <ReviewStat label="DPC" value={formatMetricValueForDisplay(report.metrics.distance_per_cycle)} icon={Split} />
          </div>
        </div>
      </div>

      <SplitMetricsTable rows={rows} />
    </section>
  );
}

function DetectedEventsSection({
  events,
  duration,
  canSeek,
  onSeek
}: {
  events: ReviewEvent[];
  duration: number;
  canSeek: boolean;
  onSeek: (time: number) => void;
}) {
  return (
    <div className="border-t border-slate-200 bg-white p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h3 className="text-sm font-black uppercase text-slate-500">Detected Events</h3>
        <span className="text-xs font-semibold text-slate-500">{duration ? formatClock(duration) : "Duration unavailable"}</span>
      </div>
      {events.length ? (
        <div className="grid gap-2 sm:grid-cols-2">
          {events.slice(0, 8).map((event, index) => (
            <button
              key={`${event.type}-${event.label}-${index}`}
              type="button"
              className="flex items-center justify-between gap-3 rounded-md border border-slate-200 px-3 py-2 text-left text-sm transition hover:border-water hover:bg-blue-50 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={!canSeek}
              onClick={() => onSeek(event.time)}
            >
              <span>
                <span className="block font-semibold capitalize text-ink">{event.label.replace(/_/g, " ")}</span>
                <span className="block text-xs text-slate-500">{formatClock(event.time)} / {event.type}</span>
              </span>
              <span className="text-xs font-bold text-slate-500">{formatPercent(event.confidence)}</span>
            </button>
          ))}
        </div>
      ) : (
        <p className="rounded-md border border-slate-200 px-3 py-2 text-sm text-slate-500">No reliable event markers available.</p>
      )}
    </div>
  );
}

function ReviewStat({ label, value, icon: Icon }: { label: string; value: unknown; icon: typeof Timer }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <div className="flex items-center gap-2 text-xs font-bold uppercase text-slate-500">
        <Icon size={15} />
        {label}
      </div>
      <p className="mt-2 break-words text-lg font-black leading-tight text-ink">{value === null || value === undefined || value === "" ? "Not enough confidence" : String(value)}</p>
    </div>
  );
}

function SplitMetricsTable({ rows }: { rows: ReviewRow[] }) {
  return (
    <div className="overflow-x-auto border-t border-slate-200">
      <table className="min-w-[980px] w-full text-left text-sm">
        <thead className="bg-slate-50 text-xs font-black uppercase text-slate-500">
          <tr>
            <th className="px-4 py-3">Phase</th>
            <th className="px-4 py-3">Time</th>
            <th className="px-4 py-3">Strokes</th>
            <th className="px-4 py-3">Tempo</th>
            <th className="px-4 py-3">DPC</th>
            <th className="px-4 py-3">15M Split</th>
            <th className="px-4 py-3">Velocity</th>
            <th className="px-4 py-3">Turn/Finish</th>
            <th className="px-4 py-3">Confidence</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200">
          {rows.map((row, index) => (
            <tr key={`${row.label}-${index}`}>
              <td className="px-4 py-3 font-bold text-ink">{row.label}</td>
              <td className="px-4 py-3 text-slate-700">{row.timeRange}</td>
              <td className="px-4 py-3 text-slate-700">{row.strokes}</td>
              <td className="px-4 py-3 text-slate-700">{row.tempo}</td>
              <td className="px-4 py-3 text-slate-700">{row.dpc}</td>
              <td className="px-4 py-3 text-slate-700">{row.split15}</td>
              <td className="px-4 py-3 text-slate-700">{row.velocity}</td>
              <td className="px-4 py-3 text-slate-700">{row.turnFinish}</td>
              <td className="px-4 py-3 text-slate-700">{row.confidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VelocityChart({ report }: { report: SwimAnalysisReport }) {
  const series = (report.velocity?.series ?? []).map((point) => ({
    time: Number(point.timestamp ?? 0),
    velocity: Number(point.velocity ?? 0),
    acceleration: Number(point.acceleration ?? 0)
  }));
  const summary = report.velocity?.summary ?? {};
  const warnings = coachVisibleVelocityWarnings(summary.warnings);
  return (
    <section className="panel p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
          <BarChart3 size={18} />
          Velocity Curve
        </h2>
        <StatusPill tone={Number(summary.confidence ?? 0) >= 0.45 ? "mint" : "slate"}>{String(summary.velocity_unit ?? "m/s")}</StatusPill>
      </div>
      <div className="h-72">
        {series.length ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tickLine={false} axisLine={false} />
              <YAxis tickLine={false} axisLine={false} />
              <Tooltip />
              <Line type="monotone" dataKey="velocity" stroke="#1a56a0" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-md bg-slate-50 text-sm text-slate-500">Velocity unavailable until swimmer tracking is reliable.</div>
        )}
      </div>
      <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <Stat label="Avg" value={summary.average_velocity} />
        <Stat label="Max" value={summary.max_velocity} />
        <Stat label="Dead spots" value={summary.dead_spot_count} />
      </dl>
      {warnings.length ? (
        <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-900">
          {warnings[0]}
        </div>
      ) : null}
    </section>
  );
}

function PhaseTimeline({ report }: { report: SwimAnalysisReport }) {
  const phases = (report.phase_segments ?? []).filter((phase) => phase.type !== "stroke_cycle").slice(0, 10);
  return (
    <section className="panel p-5">
      <h2 className="text-lg font-bold text-ink">Phase Timeline</h2>
      <div className="mt-4 space-y-3">
        {phases.length ? (
          phases.map((phase, index) => (
            <div key={`${phase.type}-${index}`} className="rounded-md border border-slate-200 p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold capitalize text-ink">{String(phase.type).replace(/_/g, " ")}</p>
                <span className="text-xs font-semibold text-slate-500">{formatConfidence(Number(phase.confidence ?? 0))}</span>
              </div>
              <p className="mt-1 text-sm text-slate-600">
                {Number(phase.start_sec ?? 0).toFixed(2)}s - {Number(phase.end_sec ?? 0).toFixed(2)}s
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-500">{String(phase.reason ?? "")}</p>
            </div>
          ))
        ) : (
          <p className="rounded-md border border-slate-200 p-3 text-sm text-slate-500">Phase segmentation unavailable.</p>
        )}
      </div>
    </section>
  );
}

function FaultList({ report }: { report: SwimAnalysisReport }) {
  const faults = report.faults ?? [];
  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <AlertTriangle size={18} />
        Technique Faults
      </h2>
      <div className="mt-4 space-y-3">
        {faults.length ? (
          faults.map((fault) => (
            <div key={fault.name} className="rounded-md border border-slate-200 p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold capitalize text-ink">{fault.name.replace(/_/g, " ")}</p>
                <StatusPill tone={fault.severity === "high" ? "coral" : "water"}>{fault.severity}</StatusPill>
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-600">{fault.coach_explanation}</p>
              <p className="mt-2 text-sm font-semibold text-water">{fault.recommended_drill}</p>
              <p className="mt-1 text-xs text-slate-500">Confidence {formatPercent(Number(fault.confidence ?? 0))}</p>
            </div>
          ))
        ) : (
          <p className="rounded-md border border-slate-200 p-3 text-sm text-slate-500">No high-confidence rule-based faults detected.</p>
        )}
      </div>
    </section>
  );
}

function CoachingReport({ report }: { report: SwimAnalysisReport }) {
  const coaching = report.coaching_report ?? {};
  const drills = Array.isArray(coaching.recommended_drills) ? coaching.recommended_drills : [];
  const plan = Array.isArray(coaching.weekly_correction_plan) ? coaching.weekly_correction_plan : [];
  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <Brain size={18} />
        Coaching Report
      </h2>
      <p className="mt-3 text-sm leading-6 text-slate-600">{String(coaching.technical_summary ?? "Report unavailable until analysis completes.")}</p>
      <div className="mt-4 space-y-3">
        {drills.slice(0, 3).map((item, index) => (
          <div key={index} className="rounded-md bg-slate-50 p-3 text-sm">
            <p className="font-semibold text-ink">{String(item.drill ?? "Drill")}</p>
            <p className="mt-1 text-slate-600">{String(item.why ?? "")}</p>
          </div>
        ))}
      </div>
      <div className="mt-4 grid gap-2 text-sm sm:grid-cols-2">
        {plan.slice(0, 4).map((item, index) => (
          <div key={index} className="rounded-md border border-slate-200 p-3">
            <p className="font-bold text-ink">Week {String(item.week ?? index + 1)}</p>
            <p className="mt-1 text-slate-600">{String(item.focus ?? "")}</p>
          </div>
        ))}
      </div>
      <p className="mt-4 text-xs text-slate-500">{String(coaching.safety_note ?? "Avoid medical claims.")}</p>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-md bg-slate-50 p-3">
      <dt className="text-xs font-semibold uppercase text-slate-500">{label}</dt>
      <dd className="mt-1 font-bold text-ink">{value === null || value === undefined ? "--" : String(value)}</dd>
    </div>
  );
}

function MetricTile({ label, metric }: { label: string; metric?: SwimAnalysisMetric }) {
  const reliability = metric?.reliability_category ? String(metric.reliability_category).replace(/_/g, " ") : formatConfidence(metric?.confidence);
  const support = typeof metric?.supporting_frame_count === "number" ? ` / ${metric.supporting_frame_count} frames` : "";
  return (
    <div className="panel p-4">
      <p className="label">{label}</p>
      <p className="mt-2 break-words text-2xl font-bold leading-tight text-ink">{formatMetricValueForDisplay(metric)}</p>
      <p className="mt-2 text-xs font-semibold text-slate-500">Reliability {reliability}{support}</p>
      <p className="mt-3 text-sm leading-6 text-slate-600">{metric?.reason ?? "Metric not available for this report."}</p>
    </div>
  );
}

function VideoPreview({ label, src }: { label: string; src: string | null }) {
  return (
    <div className="rounded-md border border-slate-200 p-3">
      <p className="mb-2 text-sm font-bold text-ink">{label}</p>
      {src ? <video className="aspect-video w-full rounded-md bg-slate-900" src={src} controls /> : <div className="flex aspect-video items-center justify-center rounded-md bg-slate-100 text-sm text-slate-500">Annotated video unavailable</div>}
    </div>
  );
}

function reviewEvents(report: SwimAnalysisReport): ReviewEvent[] {
  const splits: ReviewEvent[] = (report.velocity?.review_splits ?? []).map((split) => ({
    label: String(split.label ?? "split"),
    time: asNumber(split.end_sec) ?? 0,
    type: "split" as const,
    confidence: asNumber(split.confidence) ?? 0
  }));
  const phases: ReviewEvent[] = (report.phase_segments ?? [])
    .filter((phase) => String(phase.type ?? "") !== "stroke_cycle")
    .map((phase) => ({
      label: String(phase.type ?? "phase"),
      time: asNumber(phase.start_sec) ?? 0,
      type: "phase" as const,
      confidence: asNumber(phase.confidence) ?? 0
    }));
  const faults: ReviewEvent[] = (report.faults ?? []).map((fault) => ({
    label: fault.name,
    time: Array.isArray(fault.timestamp_range) ? asNumber(fault.timestamp_range[0]) ?? 0 : 0,
    type: "fault" as const,
    confidence: Number(fault.confidence ?? 0)
  }));
  const findings: ReviewEvent[] = (report.findings ?? []).map((finding) => ({
    label: finding.type,
    time: evidenceStartTime(finding.evidence) ?? 0,
    type: "finding" as const,
    confidence: Number(finding.confidence ?? 0)
  }));
  return [...phases, ...faults, ...findings, ...splits]
    .filter((event) => Number.isFinite(event.time))
    .sort((a, b) => a.time - b.time);
}

function reviewRows(report: SwimAnalysisReport): ReviewRow[] {
  const splits = report.velocity?.review_splits ?? [];
  if (splits.length) {
    return splits.slice(0, 8).map((split, index) => ({
      label: String(split.label ?? `S${index + 1}`),
      timeRange: `${formatClock(split.start_sec)} - ${formatClock(split.end_sec)}`,
      strokes: "--",
      tempo: "--",
      dpc: "--",
      split15: Number(split.distance_m ?? 0) === 15 ? `${Number(split.split_time_sec ?? 0).toFixed(2)}s` : "--",
      velocity: formatVelocityValue(split.average_velocity, String(split.velocity_unit ?? "m/s")),
      turnFinish: index === splits.length - 1 ? `${formatNumber(split.split_time_sec, 2)}s` : "--",
      confidence: formatConfidence(asNumber(split.confidence) ?? 0)
    }));
  }
  const phases = (report.phase_segments ?? []).filter((phase) => String(phase.type ?? "") !== "stroke_cycle");
  const rows = phases.length ? phases : [{ type: "clip", start_sec: 0, end_sec: report.summary.duration_sec, confidence: report.summary.confidence_score }];
  return rows.slice(0, 8).map((phase) => {
    const type = String(phase.type ?? "clip");
    const start = asNumber(phase.start_sec) ?? 0;
    const end = asNumber(phase.end_sec) ?? start;
    const phaseVelocity = phaseVelocitySummary(report, type);
    const velocity = asNumber(phaseVelocity?.average_velocity) ?? asNumber(report.velocity?.summary?.average_velocity);
    const velocityUnit = String(phaseVelocity?.velocity_unit ?? report.velocity?.summary?.velocity_unit ?? "");
    const velocityConfidence = asNumber(phaseVelocity?.confidence) ?? asNumber(report.velocity?.summary?.confidence) ?? 0;
    const canUseStrokeMetrics = type === "free_swim" || type === "clip";
    const showExactSpeedDerivedSplit = velocityConfidence >= 0.45 && velocityUnit === "m/s" && velocity && velocity > 0;

    return {
      label: type.replace(/_/g, " "),
      timeRange: `${formatClock(start)} - ${formatClock(end)}`,
      strokes: canUseStrokeMetrics ? String(formatMetricValue(report.metrics.stroke_count?.value)) : "--",
      tempo: canUseStrokeMetrics ? String(formatMetricValueForDisplay(report.metrics.stroke_rate_spm)) : "Not enough confidence",
      dpc: canUseStrokeMetrics ? String(formatMetricValueForDisplay(report.metrics.distance_per_cycle)) : "Not enough confidence",
      split15: showExactSpeedDerivedSplit ? `${(15 / velocity).toFixed(2)}s` : "--",
      velocity: velocity !== null && velocity !== undefined ? `${velocity.toFixed(2)} ${velocityUnit}` : "--",
      turnFinish: type.includes("turn") || type.includes("finish") ? `${Math.max(0, end - start).toFixed(2)}s` : "--",
      confidence: formatConfidence(asNumber(phase.confidence) ?? 0)
    };
  });
}

function phaseVelocitySummary(report: SwimAnalysisReport, phaseType: string) {
  return (report.velocity?.phase_summary ?? []).find((phase) => String(phase.phase ?? "") === phaseType);
}

function coachVisibleVelocityWarnings(value: unknown) {
  if (!Array.isArray(value)) return [];
  const hiddenWarnings = ["Manual lane length divided by clip duration implies an impossible swim speed"];
  return value.map(String).filter((warning) => !hiddenWarnings.some((hidden) => warning.includes(hidden)));
}

function qualitySeries(values: Array<Record<string, unknown>> | undefined) {
  return (values ?? [])
    .map((point) => ({
      time: asNumber(point.timestamp) ?? 0,
      quality: asNumber(point.quality) ?? 0
    }))
    .filter((point) => Number.isFinite(point.time));
}

function mergeQualitySeries(side: Array<{ time: number; quality: number }>, front: Array<{ time: number; quality: number }>) {
  const times = Array.from(new Set([...side.map((point) => point.time), ...front.map((point) => point.time)])).sort((a, b) => a - b);
  return times.map((time) => ({
    time,
    side: nearestQuality(side, time),
    front: nearestQuality(front, time)
  }));
}

function nearestQuality(series: Array<{ time: number; quality: number }>, time: number) {
  if (!series.length) return null;
  return series.reduce((best, point) => (Math.abs(point.time - time) < Math.abs(best.time - time) ? point : best), series[0]).quality;
}

function averageQuality(series: Array<{ time: number; quality: number }>) {
  if (!series.length) return null;
  return series.reduce((sum, point) => sum + point.quality, 0) / series.length;
}

function evidenceStartTime(evidence: Record<string, unknown>) {
  const range = evidence.time_range_sec;
  if (!Array.isArray(range)) return null;
  return asNumber(range[0]);
}

function formatVelocity(report: SwimAnalysisReport) {
  const value = asNumber(report.velocity?.summary?.average_velocity);
  if (value === null) return "--";
  return `${value.toFixed(2)} ${String(report.velocity?.summary?.velocity_unit ?? "")}`.trim();
}

function formatVelocityValue(value: unknown, unit: string) {
  const numeric = asNumber(value);
  if (numeric === null) return "--";
  return `${numeric.toFixed(2)} ${unit}`.trim();
}

function formatClock(seconds: unknown) {
  const value = asNumber(seconds);
  if (value === null) return "--";
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function asNumber(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return value;
}

function formatNumber(value: unknown, digits = 1) {
  const numeric = asNumber(value);
  return numeric === null ? "--" : numeric.toFixed(digits);
}

function formatDecimal(value: unknown, digits = 2) {
  const numeric = asNumber(value);
  return numeric === null ? "--" : numeric.toFixed(digits);
}

function truthyText(value: unknown) {
  if (typeof value === "boolean") return value ? "On" : "Off";
  if (value === null || value === undefined) return "--";
  return String(value);
}

function formatMetricValue(value: SwimAnalysisMetric["value"] | undefined) {
  if (value === null || value === undefined) return "--";
  return typeof value === "number" ? Number(value.toFixed(value < 10 && value !== Math.round(value) ? 2 : 1)) : value;
}

function formatMetricValueForDisplay(metric: SwimAnalysisMetric | undefined) {
  if (!hasEnoughMetricConfidence(metric)) return "Not enough confidence";
  return formatMetricValue(metric?.value);
}

function formatCompactMetric(metric: SwimAnalysisMetric | undefined) {
  if (!hasEnoughMetricConfidence(metric)) return "Low confidence";
  return formatMetricValue(metric?.value);
}

function hasEnoughMetricConfidence(metric: SwimAnalysisMetric | undefined) {
  if (!metric || metric.value === null || metric.value === undefined || metric.value === "") return false;
  if (metric.reliability_category === "insufficient_evidence") return false;
  if (metric.confidence === "low") return false;
  if (typeof metric.confidence === "number" && metric.confidence < 0.35) return false;
  return true;
}

function formatConfidence(value: SwimAnalysisMetric["confidence"] | undefined) {
  if (value === undefined || value === null) return "--";
  if (value === "low") return "low";
  return typeof value === "number" ? formatPercent(value) : value;
}

function formatPercent(value: number) {
  if (!Number.isFinite(value)) return "--";
  return `${Math.round(value * 100)}%`;
}

function downloadReport(report: SwimAnalysisReport) {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `aquaiq-${report.analysis_id}-coaching-report.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

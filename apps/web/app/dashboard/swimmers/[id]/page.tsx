"use client";

import {
  Activity,
  ArrowLeft,
  Brain,
  CalendarPlus,
  Dumbbell,
  FileVideo,
  Flag,
  RefreshCw,
  Save,
  ShieldCheck,
  Target,
  Trash2,
  Upload,
  Zap
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { Line, LineChart, PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AppShell } from "@/components/AppShell";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { StatusPill } from "@/components/StatusPill";
import { api } from "@/lib/api";
import type { AIOutput, MentalCheckin, RaceAnalysis, SwimSession, Swimmer, TechniqueReport, TrainingPlan } from "@/lib/types";
import { useAuthStore } from "@/store/auth";

type SwimmerWorkspace = {
  swimmer: Swimmer;
  sessions: SwimSession[];
  reports: TechniqueReport[];
  races: RaceAnalysis[];
  mental: MentalCheckin[];
  aiOutputs: AIOutput[];
  plan: TrainingPlan | null;
};

export default function SwimmerPage({ params }: { params: { id: string } }) {
  const router = useRouter();
  const token = useAuthStore((state) => state.token);
  const [workspace, setWorkspace] = useState<SwimmerWorkspace | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteSwimmerOpen, setDeleteSwimmerOpen] = useState(false);
  const [deletingSwimmer, setDeletingSwimmer] = useState(false);

  const loadWorkspace = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const [swimmer, sessions, reports, races, mental, aiOutputs] = await Promise.all([
        api.swimmer(token, params.id),
        api.sessions(token, params.id),
        api.techniqueReports(token, params.id),
        api.raceAnalyses(token, params.id),
        api.mentalCheckins(token, params.id),
        api.aiOutputs(token, params.id)
      ]);
      let plan: TrainingPlan | null = null;
      try {
        plan = await api.activePlan(token, params.id);
      } catch {
        plan = null;
      }
      setWorkspace({ swimmer, sessions, reports, races, mental, aiOutputs, plan });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load swimmer");
    } finally {
      setLoading(false);
    }
  }, [params.id, token]);

  useEffect(() => {
    loadWorkspace();
  }, [loadWorkspace]);

  async function deleteSwimmer() {
    if (!token || !workspace) return;
    setDeletingSwimmer(true);
    try {
      await api.deleteSwimmer(token, workspace.swimmer.id);
      setDeleteSwimmerOpen(false);
      router.replace("/dashboard");
    } finally {
      setDeletingSwimmer(false);
    }
  }

  const loadData = useMemo(
    () =>
      workspace?.sessions
        .slice()
        .reverse()
        .map((session) => ({
          date: new Date(session.session_date).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
          load: session.load_score,
          rpe: session.rpe
        })) ?? [],
    [workspace]
  );

  const mentalData = useMemo(() => {
    const latest = workspace?.mental[0];
    if (!latest) return [];
    return [
      { metric: "Focus", value: latest.mood_focus },
      { metric: "Confidence", value: latest.mood_confidence },
      { metric: "Energy", value: latest.mood_energy },
      { metric: "Calm", value: latest.mood_calm },
      { metric: "Recovery", value: latest.mood_recovery },
      { metric: "Motivation", value: latest.mood_motivation }
    ];
  }, [workspace]);

  return (
    <AppShell>
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <Link href="/dashboard" className="mb-3 inline-flex items-center gap-2 text-sm font-semibold text-water">
            <ArrowLeft size={16} />
            Dashboard
          </Link>
          <h1 className="text-2xl font-bold text-ink">{workspace?.swimmer.name ?? "Swimmer"}</h1>
          {workspace ? (
            <p className="mt-1 text-sm text-slate-500">
              {workspace.swimmer.primary_event} / {workspace.swimmer.level}
            </p>
          ) : null}
        </div>
        <div className="flex gap-2">
          <button className="secondary-button" type="button" onClick={loadWorkspace}>
            <RefreshCw size={16} />
            Refresh
          </button>
          {workspace ? (
            <button className="secondary-button border-red-200 text-red-700 hover:border-red-300 hover:text-red-800" type="button" onClick={() => setDeleteSwimmerOpen(true)}>
              <Trash2 size={16} />
              Delete
            </button>
          ) : null}
        </div>
      </div>

      {error ? <div className="mb-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}
      {loading ? <div className="panel p-6 text-sm text-slate-500">Loading swimmer workspace...</div> : null}

      {workspace ? (
        <div className="space-y-6">
          <section className="grid gap-4 lg:grid-cols-4">
            <ProfileSummary swimmer={workspace.swimmer} />
            <ActivePlanSummary plan={workspace.plan} />
            <LatestTechnique reports={workspace.reports} />
            <LatestRace races={workspace.races} />
          </section>

          <section className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
            <SwimmerCommandCenter workspace={workspace} />
            <SwimmerTimeline workspace={workspace} />
          </section>

          <SwimmerOrganizer workspace={workspace} onSaved={loadWorkspace} />

          <TechniqueAnalysisLab reports={workspace.reports} sessions={workspace.sessions} onSaved={loadWorkspace} />

          <section className="grid gap-6 xl:grid-cols-[1fr_0.8fr]">
            <div className="panel p-5">
              <h2 className="text-lg font-bold text-ink">Training Load</h2>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={loadData}>
                    <XAxis dataKey="date" tickLine={false} axisLine={false} />
                    <YAxis tickLine={false} axisLine={false} />
                    <Tooltip />
                    <Line type="monotone" dataKey="load" stroke="#1a56a0" strokeWidth={3} dot={{ r: 3 }} />
                    <Line type="monotone" dataKey="rpe" stroke="#d85a30" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel p-5">
              <h2 className="text-lg font-bold text-ink">Mental Shape</h2>
              <div className="mt-4 h-72">
                {mentalData.length ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={mentalData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="metric" />
                      <Radar dataKey="value" stroke="#0f6e56" fill="#0f6e56" fillOpacity={0.22} />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500">No check-ins yet.</div>
                )}
              </div>
            </div>
          </section>

          <section id="manual-tools" className="grid gap-6 xl:grid-cols-2">
            <ProfileEditor swimmer={workspace.swimmer} onSaved={loadWorkspace} />
            <PlanForm swimmer={workspace.swimmer} plan={workspace.plan} aiOutputs={workspace.aiOutputs} onSaved={loadWorkspace} />
            <RaceForm swimmer={workspace.swimmer} onSaved={loadWorkspace} />
            <MentalForm swimmerId={workspace.swimmer.id} onSaved={loadWorkspace} />
            <ReportsPanel reports={workspace.reports} sessions={workspace.sessions} />
          </section>

          <AICoachingPanel swimmerId={workspace.swimmer.id} outputs={workspace.aiOutputs} onSynced={loadWorkspace} />
          <DeleteConfirmDialog
            open={deleteSwimmerOpen}
            title="Delete swimmer?"
            itemName={workspace.swimmer.name}
            description="This will permanently remove the swimmer from AquaIQ and clear every linked record from the pilot dataset."
            details={["Profile and personal bests", "All logged sessions and uploaded video links", "Technique reports, training plans, races, and mental check-ins"]}
            confirmLabel="Delete swimmer"
            isDeleting={deletingSwimmer}
            onCancel={() => setDeleteSwimmerOpen(false)}
            onConfirm={deleteSwimmer}
          />
        </div>
      ) : null}
    </AppShell>
  );
}

const strokeTabs = ["freestyle", "backstroke", "breaststroke", "butterfly"];

function TechniqueAnalysisLab({ reports, sessions, onSaved }: { reports: TechniqueReport[]; sessions: SwimSession[]; onSaved: () => void }) {
  const latest = reports[0];
  const [activeStroke, setActiveStroke] = useState(latest?.stroke?.toLowerCase() ?? "freestyle");
  const [selectedHeatZone, setSelectedHeatZone] = useState<string | null>(null);
  const reportsByStroke = useMemo(() => {
    const grouped = new Map<string, TechniqueReport[]>();
    reports.forEach((report) => {
      const stroke = normalizeStroke(report.stroke);
      grouped.set(stroke, [...(grouped.get(stroke) ?? []), report]);
    });
    return grouped;
  }, [reports]);
  const activeStrokeReports = reportsByStroke.get(activeStroke) ?? [];
  const selectedStrokeHasReport = activeStrokeReports.length > 0;
  const selectedReport = activeStrokeReports[0] ?? latest ?? null;

  useEffect(() => {
    if (latest?.stroke) {
      setActiveStroke(normalizeStroke(latest.stroke));
    }
  }, [latest?.stroke]);

  useEffect(() => {
    setSelectedHeatZone(null);
  }, [selectedReport?.id]);

  if (!selectedReport) {
    return (
      <section id="video-analysis" className="rounded-md border border-stone-800 bg-[#171916] p-5 text-stone-100 shadow-panel">
        <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
          <div>
            <p className="text-xs font-black uppercase text-emerald-300">Video Analysis</p>
            <h2 className="mt-1 text-xl font-black">Technique Analysis Lab</h2>
            <p className="mt-2 max-w-2xl text-sm font-semibold leading-6 text-stone-400">
              Upload a swimmer video to generate stroke tabs, MediaPipe quality, clickable body zones, faults, and drill prescriptions.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {strokeTabs.map((stroke) => (
                <span key={stroke} className="rounded-md border border-stone-700 px-4 py-2 text-sm font-bold capitalize text-stone-400">
                  {stroke}
                  <span className="ml-2 text-[10px] uppercase text-stone-600">No report</span>
                </span>
              ))}
            </div>
          </div>
          <VideoUploadPanel sessions={sessions} onSaved={onSaved} variant="dark" />
        </div>
      </section>
    );
  }

  const report = selectedReport;
  const session = sessions.find((item) => item.id === report.session_id);
  const videoUrl = api.videoUrl(session?.video_url ?? null);
  const analysisFrameUrls = (report.analysis_frame_urls ?? []).map((path) => api.videoUrl(path)).filter((url): url is string => Boolean(url));
  const analysisEvents = report.analysis_events ?? [];
  const keypoints = report.keypoint_data ?? {};
  const provider = String(keypoints.provider ?? "mock");
  const providerNote = typeof keypoints.provider_note === "string" ? keypoints.provider_note : null;
  const isMediaPipe = provider === "mediapipe";
  const analysisStatus = report.analysis_status ?? String(keypoints.analysis_quality ?? "completed");
  const confidenceLabel = report.confidence_label ?? "low";
  const poseFrames = report.pose_detected_frames ?? numberFrom(keypoints.pose_frames_detected);
  const sampledFrames = report.frames_analyzed ?? numberFrom(keypoints.frames_sampled);
  const confidence = report.confidence_score ?? numberFrom(keypoints.confidence);
  const poseDetectionRate = report.pose_detection_rate ?? (sampledFrames ? poseFrames / sampledFrames : 0);
  const qualityMessage = report.analysis_error || report.analysis_warning || qualitySummary(analysisStatus, poseDetectionRate, confidenceLabel);
  const timeGain = numberFrom(keypoints.time_gain_possible, Math.max(0.1, (100 - report.overall_score) / 60));
  const metrics = isRecord(keypoints.metrics) ? keypoints.metrics : {};
  const faults = report.faults;
  const drills = report.drill_prescriptions;
  const topFault = faults[0] ?? null;
  const activeHeatZone = selectedHeatZone ?? zoneOf(topFault) ?? "torso";
  const activeHeatFault = selectedHeatZone ? faults.find((fault) => zoneOf(fault) === activeHeatZone) ?? null : topFault;
  const mediaPipeNote = mediaPipeStatusCopy(isMediaPipe, analysisStatus);

  return (
    <section id="video-analysis" className="rounded-md border border-stone-800 bg-[#171916] p-5 text-stone-100 shadow-panel">
      <div className="mb-4">
        <p className="text-xs font-black uppercase text-emerald-300">Video Analysis</p>
        <h2 className="mt-1 text-xl font-black">Technique Analysis Lab</h2>
        <p className="mt-2 max-w-3xl text-sm font-semibold leading-6 text-stone-400">
          Review MediaPipe quality, inspect body zones, and connect detected faults to prescribed drills.
        </p>
      </div>
      <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
        <div className="flex flex-wrap gap-2">
          {strokeTabs.map((stroke) => {
            const strokeReports = reportsByStroke.get(stroke) ?? [];
            const tabReport = strokeReports[0];
            const hasReport = strokeReports.length > 0;
            const tabProvider = tabReport ? reportProviderName(tabReport) : null;
            return (
              <button
                key={stroke}
                className={`rounded-md border px-4 py-2 text-left text-sm font-bold capitalize transition ${
                  activeStroke === stroke
                    ? "border-emerald-300/60 bg-emerald-500/10 text-white shadow-[0_0_0_1px_rgba(110,231,183,0.16)]"
                    : "border-stone-700 bg-stone-950/20 text-stone-300 hover:border-stone-500 hover:text-white"
                }`}
                type="button"
                onClick={() => setActiveStroke(stroke)}
              >
                <span className="flex items-center gap-2">
                  <span className={`h-2 w-2 rounded-full ${hasReport ? "bg-emerald-400" : "bg-stone-600"}`} />
                  {stroke}
                </span>
                <span className="mt-1 block text-[10px] font-black uppercase text-stone-500">
                  {hasReport ? `${tabProvider} / ${tabReport.analysis_status.replace(/_/g, " ")}` : "No report"}
                </span>
              </button>
            );
          })}
        </div>
        <div className="flex flex-wrap gap-2">
          <span
            className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-xs font-bold uppercase ${
              isMediaPipe ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200" : "border-amber-500/40 bg-amber-500/10 text-amber-200"
            }`}
          >
            <ShieldCheck size={15} />
            {isMediaPipe ? "MediaPipe active" : "Mock analysis"}
          </span>
          <span className="inline-flex items-center gap-2 rounded-md border border-stone-700 px-3 py-2 text-xs font-bold uppercase text-stone-300">
            <Activity size={15} />
            {analysisStatus.replace(/_/g, " ")}
          </span>
        </div>
      </div>
      <div className="mt-4">
        <VideoUploadPanel sessions={sessions} onSaved={onSaved} variant="dark" />
      </div>
      {!selectedStrokeHasReport ? (
        <p className="mt-4 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm font-semibold text-amber-100">
          No {strokeLabel(activeStroke)} report yet. Showing the latest {strokeLabel(report.stroke)} report as a reference until that stroke has its own upload.
        </p>
      ) : null}

      <div className="mt-5 grid gap-3 md:grid-cols-4">
        <AnalysisMetric value={report.overall_score.toString()} label="Technique score" />
        <AnalysisMetric value={`${report.dps_meters.toFixed(2)}m`} label="Distance per stroke" />
        <AnalysisMetric value={`${report.stroke_rate}/min`} label="Stroke rate" />
        <AnalysisMetric value={`+${timeGain.toFixed(1)}s`} label="Time gain possible" />
      </div>

      <VideoValidityPanel
        report={report}
        isMediaPipe={isMediaPipe}
        analysisStatus={analysisStatus}
        sampledFrames={sampledFrames}
        poseFrames={poseFrames}
        poseDetectionRate={poseDetectionRate}
        faults={faults}
        drills={drills}
      />

      <div className="mt-5 rounded-md border border-stone-700 bg-stone-950/30 p-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h3 className="text-sm font-bold text-stone-300">Analysis Quality</h3>
            <p className="mt-2 text-sm font-semibold leading-6 text-stone-400">
              {analysisStatus === "no_pose_detected" || analysisStatus === "failed"
                ? qualityMessage
                : `Pose detected in ${Math.round(poseDetectionRate * 100)}% of analyzed frames. Confidence: ${capitalize(confidenceLabel)}.`}
            </p>
            {(report.analysis_warning || report.analysis_error) && !["failed", "no_pose_detected"].includes(analysisStatus) ? (
              <p className="mt-1 text-sm text-amber-100">{qualityMessage}</p>
            ) : null}
          </div>
          <div className="grid gap-2 sm:grid-cols-4 lg:min-w-[520px]">
            <QualityBadge label="Status" value={analysisStatus.replace(/_/g, " ")} status={analysisStatus} />
            <QualityBadge label="Confidence" value={confidenceLabel} status={confidenceLabel} />
            <QualityBadge label="Pose rate" value={`${Math.round(poseDetectionRate * 100)}%`} status={analysisStatus} />
            <QualityBadge label="Frames" value={`${sampledFrames}/${report.frames_total || "?"}`} status="neutral" />
          </div>
        </div>
      </div>

      <AnalysisVideoPreview originalUrl={videoUrl} frameUrls={analysisFrameUrls} />
      <AnalysisTimeline events={analysisEvents} />

      <div className="mt-6 grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
        <div className="space-y-5">
          <div>
            <h3 className="text-sm font-bold text-stone-300">Body heat map</h3>
            <div className="mt-3 grid gap-4 md:grid-cols-[260px_1fr] xl:grid-cols-1 2xl:grid-cols-[260px_1fr]">
              <HeatMapFigure faults={faults} activeZone={activeHeatZone} onSelectZone={setSelectedHeatZone} />
              <HeatMapDetail fault={activeHeatFault} zone={activeHeatZone} onClear={() => setSelectedHeatZone(null)} />
            </div>
            <div className="mt-5 flex flex-wrap gap-4 text-xs font-semibold text-stone-400">
              <LegendDot tone="critical" label="Critical" />
              <LegendDot tone="needs_work" label="Needs work" />
              <LegendDot tone="good" label="Good" />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-bold text-stone-300">AI-detected faults</h3>
          <div className="mt-4 divide-y divide-stone-700">
            {faults.map((fault, index) => (
              <FaultRow
                key={`${fault.title}-${index}`}
                fault={fault}
                drill={linkedDrillForFault(fault, drills, index)}
                active={zoneOf(fault) === activeHeatZone}
                onSelect={() => setSelectedHeatZone(zoneOf(fault))}
              />
            ))}
          </div>

          <div className="mt-4 grid gap-3 rounded-md border border-stone-700 bg-stone-950/30 p-3 text-xs text-stone-400 md:grid-cols-4">
            <ProviderStat label="Provider" value={isMediaPipe ? "MediaPipe" : "Mock"} />
            <ProviderStat label="Pose frames" value={poseFrames ? `${poseFrames}/${sampledFrames || "?"}` : "-"} />
            <ProviderStat label="Confidence" value={confidence ? `${Math.round(confidence * 100)}%` : "-"} />
            <ProviderStat label="Elbow angle" value={numberFrom(metrics.elbow_angle_mean) ? `${numberFrom(metrics.elbow_angle_mean).toFixed(0)}deg` : "-"} />
          </div>
          {providerNote ? <p className="mt-3 rounded-md bg-amber-500/10 px-3 py-2 text-sm text-amber-100">{providerNote}</p> : null}
        </div>
      </div>

      <div className="mt-7">
        <h3 className="text-sm font-bold text-stone-300">Prescribed drills</h3>
        <div className="mt-3 space-y-2">
          {drills.map((drill, index) => (
            <div key={`${drill.name}-${index}`} className="flex items-center justify-between gap-4 rounded-md border border-stone-700 px-4 py-3">
              <div className="min-w-0">
                <p className="font-bold text-stone-100">{textFrom(drill.name, "Drill")}</p>
                <p className="text-sm font-semibold text-stone-400">Focus: {textFrom(drill.focus, "Technique")}</p>
                <p className="mt-1 text-xs font-semibold text-stone-500">For: {textFrom(linkedFaultForDrill(drill, faults, index)?.title, "Technique maintenance")}</p>
              </div>
              <div className="flex items-center gap-3">
                <span className="rounded border border-stone-700 px-3 py-1 text-sm font-bold text-stone-300">{textFrom(drill.volume, "4x50m")}</span>
                <Dumbbell size={18} className="hidden text-stone-500 sm:block" />
              </div>
            </div>
          ))}
        </div>
      </div>

      <p className="mt-4 flex items-start gap-2 text-xs leading-5 text-stone-500">
        <Zap className="mt-0.5 shrink-0" size={14} />
        {mediaPipeNote}
      </p>
    </section>
  );
}

function AnalysisMetric({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-md bg-stone-800 px-4 py-4 text-center">
      <p className="text-2xl font-black text-white">{value}</p>
      <p className="mt-1 text-sm font-bold text-stone-400">{label}</p>
    </div>
  );
}

function QualityBadge({ label, value, status }: { label: string; value: string; status: string }) {
  return (
    <div className={`rounded-md border px-3 py-2 ${qualityBadgeClasses(status)}`}>
      <p className="text-[10px] font-black uppercase tracking-normal opacity-80">{label}</p>
      <p className="mt-1 text-sm font-black capitalize">{value}</p>
    </div>
  );
}

function VideoValidityPanel({
  report,
  isMediaPipe,
  analysisStatus,
  sampledFrames,
  poseFrames,
  poseDetectionRate,
  faults,
  drills
}: {
  report: TechniqueReport;
  isMediaPipe: boolean;
  analysisStatus: string;
  sampledFrames: number;
  poseFrames: number;
  poseDetectionRate: number;
  faults: TechniqueReport["faults"];
  drills: TechniqueReport["drill_prescriptions"];
}) {
  const validity = videoValidityState(isMediaPipe, analysisStatus, poseDetectionRate);
  const criticalCount = faults.filter((fault) => severityOf(fault) === "critical").length;
  const needsWorkCount = faults.filter((fault) => severityOf(fault) === "needs_work").length;
  const usefulFaults = criticalCount + needsWorkCount;
  const providerText = isMediaPipe ? "Real MediaPipe pose extraction" : "Mock demo analysis";

  return (
    <div className="mt-5 rounded-md border border-stone-700 bg-stone-950/30 p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h3 className="text-sm font-bold text-stone-200">Video Validity</h3>
          <p className="mt-2 max-w-2xl text-sm font-semibold leading-6 text-stone-400">{validity.message}</p>
        </div>
        <span className={`inline-flex w-fit rounded-md border px-3 py-2 text-xs font-black uppercase ${qualityBadgeClasses(validity.status)}`}>
          {validity.label}
        </span>
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-5">
        <ValidityStat label="Provider" value={providerText} />
        <ValidityStat label="Frames read" value={`${sampledFrames}/${report.frames_total || "?"}`} />
        <ValidityStat label="Pose frames" value={`${poseFrames}/${sampledFrames || "?"}`} />
        <ValidityStat label="Pose rate" value={`${Math.round(poseDetectionRate * 100)}%`} />
        <ValidityStat label="Faults / drills" value={`${usefulFaults} / ${drills.length}`} />
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-3">
        <VideoCheck label="Validity" value={validity.label} status={validity.status} />
        <VideoCheck label="Fault clarity" value={usefulFaults ? `${criticalCount} critical, ${needsWorkCount} needs work` : "No major threshold crossed"} status={usefulFaults ? "needs_work" : "good"} />
        <VideoCheck label="Drill links" value={drills.length ? "Each fault has a linked drill" : "No drill prescribed yet"} status={drills.length ? "good" : "needs_work"} />
      </div>
    </div>
  );
}

function ValidityStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-stone-800 bg-stone-950/50 px-3 py-3">
      <p className="text-[10px] font-black uppercase text-stone-500">{label}</p>
      <p className="mt-1 text-sm font-black text-stone-200">{value}</p>
    </div>
  );
}

function VideoCheck({ label, value, status }: { label: string; value: string; status: string }) {
  return (
    <div className={`rounded-md border px-3 py-3 ${qualityBadgeClasses(status)}`}>
      <p className="text-[10px] font-black uppercase opacity-70">{label}</p>
      <p className="mt-1 text-sm font-black">{value}</p>
    </div>
  );
}

function FaultRow({
  fault,
  drill,
  active,
  onSelect
}: {
  fault: Record<string, string | number>;
  drill: Record<string, string | number> | null;
  active: boolean;
  onSelect: () => void;
}) {
  const severity = severityOf(fault);
  const cost = numberFrom(fault.time_cost_seconds, 0);
  return (
    <button
      className={`grid w-full grid-cols-[14px_1fr] gap-3 rounded-md px-2 py-4 text-left transition first:pt-0 ${
        active ? "bg-stone-900/60" : "hover:bg-stone-900/40"
      }`}
      type="button"
      onClick={onSelect}
    >
      <span className={`mt-1.5 h-2.5 w-2.5 rounded-full ${severityClasses(severity).dot}`} />
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <p className="font-bold text-stone-100">{textFrom(fault.title, "Technique flag")}</p>
          <span className={`rounded border px-2 py-0.5 text-[10px] font-black uppercase ${qualityBadgeClasses(severity)}`}>{severity.replace(/_/g, " ")}</span>
        </div>
        <p className="mt-1 text-sm font-semibold leading-6 text-stone-400">
          {textFrom(fault.description, "Review the uploaded stroke video for this movement pattern.")}
          {cost > 0 ? ` Costs ~${cost.toFixed(1)}s per 50m.` : ""}
        </p>
        <div className="mt-3 grid gap-2 rounded-md border border-stone-800 bg-stone-950/40 p-3 text-xs sm:grid-cols-[1fr_auto]">
          <div>
            <p className="font-black uppercase text-stone-500">Linked drill</p>
            <p className="mt-1 font-bold text-stone-200">
              {drill ? `${textFrom(drill.name, "Drill")} / ${textFrom(drill.focus, "Technique")}` : "No drill linked"}
            </p>
          </div>
          <span className="self-center rounded border border-stone-700 px-3 py-1 font-black text-stone-300">{drill ? textFrom(drill.volume, "4x50m") : "-"}</span>
        </div>
      </div>
    </button>
  );
}

function AnalysisVideoPreview({ originalUrl, frameUrls }: { originalUrl: string | null; frameUrls: string[] }) {
  return (
    <div className="mt-5 rounded-md border border-stone-700 bg-stone-950/30 p-4">
      <div>
        <h3 className="text-sm font-bold text-stone-200">Uploaded Video Preview</h3>
        <p className="mt-1 text-sm font-semibold text-stone-500">
          Use this to verify the clip attached to the report. Skeleton snapshots appear below when available.
        </p>
      </div>
      <div className="mt-4">
        <VideoPane title="Original Video" url={originalUrl} emptyText="Original video preview is unavailable." />
      </div>
      {frameUrls.length ? (
        <div className="mt-4">
          <p className="text-xs font-black uppercase text-stone-500">Skeleton snapshots</p>
          <div className="mt-2 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {frameUrls.map((url, index) => (
              <Image
                key={url}
                className="aspect-video w-full rounded-md border border-stone-800 bg-black object-contain"
                src={url}
                alt={`AI skeleton snapshot ${index + 1}`}
                width={320}
                height={180}
                unoptimized
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function VideoPane({ title, url, emptyText }: { title: string; url: string | null; emptyText: string }) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-2 text-sm font-bold text-stone-300">
        <FileVideo size={16} />
        {title}
      </div>
      {url ? (
        <video className="aspect-video w-full rounded-md border border-stone-800 bg-black object-contain" controls preload="metadata" playsInline src={url} />
      ) : (
        <div className="flex aspect-video items-center justify-center rounded-md border border-dashed border-stone-700 bg-stone-950/50 px-4 text-center text-sm font-semibold text-stone-500">
          {emptyText}
        </div>
      )}
    </div>
  );
}

function AnalysisTimeline({ events }: { events: TechniqueReport["analysis_events"] }) {
  return (
    <div className="mt-5 rounded-md border border-stone-700 bg-stone-950/30 p-4">
      <h3 className="text-sm font-bold text-stone-200">Analysis Timeline</h3>
      <div className="mt-3 space-y-3">
        {events.length ? (
          events.map((event, index) => (
            <div key={`${event.type}-${event.timestamp_s}-${index}`} className="grid grid-cols-[64px_12px_1fr] gap-3">
              <p className="pt-0.5 text-xs font-black text-stone-500">{event.timestamp_s.toFixed(1)}s</p>
              <span className="mt-1.5 h-2.5 w-2.5 rounded-full bg-emerald-400" />
              <div>
                <p className="text-sm font-black text-stone-100">{event.label}</p>
                <p className="mt-0.5 text-sm font-semibold leading-5 text-stone-500">{event.message}</p>
              </div>
            </div>
          ))
        ) : (
          <p className="text-sm font-semibold text-stone-500">No timeline events were recorded for this report.</p>
        )}
      </div>
    </div>
  );
}

function qualityBadgeClasses(status: string) {
  if (["completed", "high", "good"].includes(status)) {
    return "border-emerald-500/40 bg-emerald-500/10 text-emerald-100";
  }
  if (["low_confidence", "medium", "low", "needs_work"].includes(status)) {
    return "border-amber-500/40 bg-amber-500/10 text-amber-100";
  }
  if (["failed", "no_pose_detected", "critical"].includes(status)) {
    return "border-rose-500/40 bg-rose-500/10 text-rose-100";
  }
  return "border-stone-700 bg-stone-900 text-stone-200";
}

function qualitySummary(status: string, poseDetectionRate: number, confidenceLabel: string) {
  if (status === "no_pose_detected") {
    return "No clear swimmer pose detected. Upload a side-view video where the full body is visible.";
  }
  if (status === "low_confidence") {
    return "Low confidence. Camera angle, water reflection, or body occlusion may affect analysis accuracy.";
  }
  if (status === "failed") {
    return "Processing error. The video was saved, but AquaIQ could not complete pose analysis.";
  }
  return `Pose detected in ${Math.round(poseDetectionRate * 100)}% of analyzed frames. Confidence: ${capitalize(confidenceLabel)}.`;
}

function videoValidityState(isMediaPipe: boolean, status: string, poseDetectionRate: number) {
  if (!isMediaPipe) {
    return {
      label: "Mock only",
      status: "needs_work",
      message: "This is deterministic mock analysis. It is useful for UI testing, but not proof that MediaPipe detected this swimmer."
    };
  }
  if (status === "failed") {
    return {
      label: "Processing failed",
      status: "failed",
      message: "The file was uploaded, but pose processing failed. Try a smaller MP4 with the full swimmer visible."
    };
  }
  if (status === "no_pose_detected") {
    return {
      label: "Not usable",
      status: "no_pose_detected",
      message: "No swimmer pose was detected. Re-upload a side-view or above-water clip with the full body in frame."
    };
  }
  if (status === "low_confidence" || poseDetectionRate < 0.45) {
    return {
      label: "Use with caution",
      status: "low_confidence",
      message: "MediaPipe found some pose data, but the clip is low confidence. Treat faults as review prompts, not final truth."
    };
  }
  if (poseDetectionRate < 0.75) {
    return {
      label: "Usable with review",
      status: "medium",
      message: "The video is usable, but the coach should review the original clip alongside the heat map before acting on small details."
    };
  }
  return {
    label: "Usable video",
    status: "completed",
    message: "The video has strong enough pose coverage for Phase 1 technique review. Use faults and drills as the session focus."
  };
}

function capitalize(value: string) {
  return value ? value.charAt(0).toUpperCase() + value.slice(1) : value;
}

function HeatMapFigure({
  faults,
  activeZone,
  onSelectZone
}: {
  faults: TechniqueReport["faults"];
  activeZone: string;
  onSelectZone: (zone: string) => void;
}) {
  return (
    <div className="relative mx-auto h-80 w-64 rounded-md border border-stone-800 bg-stone-950/20">
      {heatMapSegments.map((segment) => {
        const severity = severityForZone(faults, segment.zone);
        const active = activeZone === segment.zone;
        return (
          <button
            key={segment.key}
            aria-label={`Select ${segment.label}`}
            aria-pressed={active}
            className={`absolute transition hover:scale-[1.03] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-stone-300 ${segment.className} ${heatZoneClasses(
              severity,
              active
            )}`}
            type="button"
            onClick={() => onSelectZone(segment.zone)}
            title={segment.label}
          />
        );
      })}
      <div className="absolute right-3 top-3 rounded border border-stone-700 bg-stone-950/80 px-2 py-1 text-[10px] font-black uppercase text-stone-400">
        Click zones
      </div>
    </div>
  );
}

function HeatMapDetail({
  fault,
  zone,
  onClear
}: {
  fault: Record<string, string | number> | null;
  zone: string;
  onClear: () => void;
}) {
  const severity = fault ? severityOf(fault) : "good";
  const timeCost = fault ? numberFrom(fault.time_cost_seconds, 0) : 0;
  const hasIssue = Boolean(fault && severity !== "good");
  return (
    <div className={`rounded-md border p-4 ${zonePanelClasses(severity)}`}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-black uppercase opacity-70">Zone inspector</p>
          <h4 className="mt-1 text-lg font-black text-stone-100">{zoneLabel(zone)}</h4>
        </div>
        <span className={`rounded border px-2 py-1 text-[10px] font-black uppercase ${qualityBadgeClasses(severity)}`}>{severity.replace(/_/g, " ")}</span>
      </div>
      <p className="mt-4 text-base font-black text-stone-100">{fault ? textFrom(fault.title, "No issue flagged") : "No issue flagged in this zone"}</p>
      <p className="mt-2 text-sm font-semibold leading-6 text-stone-300/85">
        {fault ? textFrom(fault.description, "This zone is currently within Phase 1 thresholds.") : "MediaPipe did not flag a fault here. Keep this area as a quick check while reviewing the video."}
      </p>
      <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border border-white/10 bg-black/20 px-3 py-2">
          <p className="font-black uppercase opacity-60">Time cost</p>
          <p className="mt-1 font-black text-stone-200">{timeCost > 0 ? `~${timeCost.toFixed(1)}s / 50m` : "None flagged"}</p>
        </div>
        <div className="rounded border border-white/10 bg-black/20 px-3 py-2">
          <p className="font-black uppercase opacity-60">Next action</p>
          <p className="mt-1 font-black text-stone-200">{hasIssue ? "Review matching drill" : "Maintain pattern"}</p>
        </div>
      </div>
      <button className="mt-4 rounded border border-white/10 px-3 py-2 text-xs font-bold text-stone-300 transition hover:border-white/30 hover:text-white" type="button" onClick={onClear}>
        Reset to top fault
      </button>
    </div>
  );
}

const heatMapSegments = [
  { key: "head", zone: "head", label: "Head and breathing", className: "left-[102px] top-4 h-14 w-14 rounded-full" },
  { key: "torso", zone: "torso", label: "Torso and body line", className: "left-[106px] top-[84px] h-20 w-12 rounded-lg" },
  { key: "left-arm", zone: "arms", label: "Arms and catch", className: "left-[76px] top-[84px] h-[74px] w-7 rounded-lg" },
  { key: "right-arm", zone: "arms", label: "Arms and catch", className: "left-[160px] top-[84px] h-[74px] w-7 rounded-lg" },
  { key: "hips", zone: "hips", label: "Hips and rotation", className: "left-[110px] top-[170px] h-[62px] w-10 rounded-lg" },
  { key: "left-leg", zone: "legs", label: "Legs and kick", className: "left-[96px] top-[242px] h-[64px] w-8 rounded-lg" },
  { key: "right-leg", zone: "legs", label: "Legs and kick", className: "left-[138px] top-[242px] h-[64px] w-8 rounded-lg" }
];

function normalizeStroke(value: string) {
  const normalized = String(value || "freestyle").toLowerCase();
  return strokeTabs.includes(normalized) ? normalized : "freestyle";
}

function strokeLabel(value: string) {
  const normalized = String(value || "freestyle").toLowerCase();
  if (normalized === "im") return "IM";
  return normalized
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map(capitalize)
    .join(" ");
}

function zoneOf(fault?: Record<string, string | number> | null) {
  const raw = String(fault?.zone ?? fault?.title ?? "torso").toLowerCase();
  if (["arms", "arm", "catch", "shoulder", "elbow", "wrist", "hand", "extension"].some((token) => raw.includes(token))) return "arms";
  if (["legs", "leg", "kick", "knee", "ankle", "feet", "foot"].some((token) => raw.includes(token))) return "legs";
  if (["hips", "hip", "rotation", "roll"].some((token) => raw.includes(token))) return "hips";
  if (["head", "breath", "breathing", "turn"].some((token) => raw.includes(token))) return "head";
  return "torso";
}

function zoneLabel(zone: string) {
  const labels: Record<string, string> = {
    arms: "Arms / Catch",
    legs: "Legs / Kick",
    hips: "Hips / Rotation",
    head: "Head / Breathing",
    torso: "Torso / Body Line"
  };
  return labels[zone] ?? "Body Line";
}

function severityForZone(faults: TechniqueReport["faults"], zone: string) {
  const zoneFaults = faults.filter((fault) => zoneOf(fault) === zone);
  if (!zoneFaults.length) return "good";
  if (zoneFaults.some((fault) => severityOf(fault) === "critical")) return "critical";
  if (zoneFaults.some((fault) => severityOf(fault) === "needs_work")) return "needs_work";
  return "good";
}

function heatZoneClasses(severity: string, active: boolean) {
  const color =
    severity === "critical"
      ? "border border-rose-300/80 bg-rose-300/85 shadow-[0_0_22px_rgba(251,113,133,0.22)]"
      : severity === "needs_work"
        ? "border border-amber-200/80 bg-amber-100/90 shadow-[0_0_22px_rgba(251,191,36,0.18)]"
      : "border border-lime-400/70 bg-lime-400/55 shadow-[0_0_18px_rgba(132,204,22,0.16)]";
  return `${color} ${active ? "ring-4 ring-white/30" : "opacity-80 hover:opacity-100"}`;
}

function zonePanelClasses(severity: string) {
  if (severity === "critical") {
    return "border-rose-400/35 bg-rose-500/10";
  }
  if (severity === "needs_work") {
    return "border-amber-300/35 bg-amber-500/10";
  }
  return "border-emerald-400/25 bg-emerald-500/10";
}

function mediaPipeStatusCopy(isMediaPipe: boolean, status: string) {
  if (!isMediaPipe) {
    return "This report is deterministic demo data. Run the API with TECHNIQUE_ANALYZER=mediapipe to validate real pose extraction.";
  }
  if (status === "completed") {
    return "MediaPipe completed pose extraction on the uploaded video. Use the heat map and fault list together to review what the model saw.";
  }
  if (status === "low_confidence") {
    return "MediaPipe found a swimmer pose, but confidence is low. Re-upload a clearer side-view or above-water clip if the faults look suspicious.";
  }
  if (status === "no_pose_detected") {
    return "MediaPipe did not detect a clear swimmer pose. Upload footage with the full swimmer visible and less water reflection.";
  }
  if (status === "failed") {
    return "MediaPipe could not complete this analysis. Try a smaller MP4 recorded from a phone or camera.";
  }
  return "MediaPipe is processing the uploaded file. Clear side-view or above-water footage gives the best validation.";
}

function LegendDot({ tone, label }: { tone: "critical" | "needs_work" | "good"; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`h-2.5 w-2.5 rounded-full border ${severityClasses(tone).legend}`} />
      {label}
    </span>
  );
}

function ProviderStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-bold uppercase text-stone-500">{label}</p>
      <p className="mt-1 font-bold text-stone-200">{value}</p>
    </div>
  );
}

function severityOf(fault: Record<string, string | number>) {
  const severity = String(fault.severity ?? "").toLowerCase();
  if (["critical", "high"].includes(severity)) return "critical";
  if (["good", "low"].includes(severity)) return severity === "good" ? "good" : "needs_work";
  return "needs_work";
}

function severityClasses(severity: string) {
  if (severity === "critical") {
    return { dot: "bg-rose-400", legend: "border-rose-400 bg-rose-950" };
  }
  if (severity === "good") {
    return { dot: "bg-lime-500", legend: "border-lime-500 bg-lime-950" };
  }
  return { dot: "bg-amber-400", legend: "border-amber-400 bg-amber-950" };
}

function linkedDrillForFault(fault: Record<string, string | number>, drills: TechniqueReport["drill_prescriptions"], index: number) {
  if (!drills.length) return null;
  const zone = zoneOf(fault);
  const title = textFrom(fault.title, "").toLowerCase();
  const description = textFrom(fault.description, "").toLowerCase();
  return (
    drills.find((drill) => {
      const focus = textFrom(drill.focus, "").toLowerCase();
      const name = textFrom(drill.name, "").toLowerCase();
      return focus.includes(zone) || title.includes(focus) || description.includes(focus) || title.includes(name);
    }) ??
    drills[index % drills.length] ??
    null
  );
}

function linkedFaultForDrill(drill: Record<string, string | number>, faults: TechniqueReport["faults"], index: number) {
  if (!faults.length) return null;
  const focus = textFrom(drill.focus, "").toLowerCase();
  const name = textFrom(drill.name, "").toLowerCase();
  return (
    faults.find((fault) => {
      const title = textFrom(fault.title, "").toLowerCase();
      const description = textFrom(fault.description, "").toLowerCase();
      const zone = zoneOf(fault);
      return focus.includes(zone) || title.includes(focus) || description.includes(focus) || title.includes(name);
    }) ??
    faults[index % faults.length] ??
    null
  );
}

function textFrom(value: unknown, fallback: string) {
  return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
}

function numberFrom(value: unknown, fallback = 0) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function reportProviderName(report: TechniqueReport) {
  return String(report.keypoint_data.provider ?? "mock") === "mediapipe" ? "MediaPipe" : "Mock";
}

function providerLabel(provider: string) {
  if (provider === "anthropic") return "Claude";
  if (provider === "local") return "Local AI";
  return capitalize(provider);
}

function formatOutputType(type: string) {
  return type
    .split("_")
    .filter(Boolean)
    .map(capitalize)
    .join(" ");
}

function latestAIOutput(outputs: AIOutput[], outputType: string) {
  return outputs.find((output) => output.output_type === outputType) ?? null;
}

function planMessage(plan: TrainingPlan | null) {
  const log = plan?.adaptation_log ?? [];
  const aiEntry = [...log].reverse().find((entry) => entry.event === "ai_rationale");
  if (typeof aiEntry?.message === "string") return aiEntry.message;
  const created = [...log].reverse().find((entry) => entry.event === "created");
  return typeof created?.message === "string" ? created.message : "";
}

function formatDateLabel(dateString: string) {
  return new Date(`${dateString}T12:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric"
  });
}

function suggestNextVisit(session: SwimSession | null) {
  if (!session) return "After first log";
  const date = new Date(`${session.session_date}T12:00:00`);
  const daysToAdd = session.rpe >= 8 || session.mood_recovery <= 5 ? 2 : 1;
  date.setDate(date.getDate() + daysToAdd);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function buildCoachDecision(workspace: SwimmerWorkspace): {
  label: string;
  title: string;
  detail: string;
  action: string;
  href: string;
  tone: "water" | "mint" | "coral" | "violet" | "slate";
  reason: Array<{ label: string; value: string }>;
  recommendation: {
    nextSession: string;
    why: string;
    risk: string;
    coachAction: string;
  };
} {
  const session = workspace.sessions[0] ?? null;
  const report = workspace.reports[0] ?? null;
  const mental = workspace.mental[0] ?? null;
  const plan = workspace.plan;
  const currentWeek = plan?.weekly_plans.find((week) => week.week_number === plan.current_week) ?? plan?.weekly_plans[0];
  const nextSet = currentWeek?.sessions?.[0];
  const topFault = report?.faults?.[0];
  const topDrill = report?.drill_prescriptions?.[0];

  if (!session) {
    return {
      label: "Needs first data",
      title: "Log the first real practice",
      detail: "AquaIQ needs one session to calculate load, readiness, and a useful next action for this swimmer.",
      action: "Add session",
      href: "#session-record",
      tone: "water",
      reason: [
        { label: "Training", value: "No sessions yet" },
        { label: "Video", value: "Add after first swim" },
        { label: "Plan", value: plan ? `${plan.current_phase} ready` : "Generate after profile" }
      ],
      recommendation: {
        nextSession: "Next: log one normal practice before changing the plan.",
        why: "Reason: AquaIQ has no load, RPE, or recovery baseline for this swimmer yet.",
        risk: "Risk: recommendations will be generic until the first real session is saved.",
        coachAction: "Coach action: save date, session type, distance, RPE, sleep, mood scores, and notes."
      }
    };
  }

  if (session.rpe >= 8 || session.mood_recovery <= 5) {
    return {
      label: "Watch load",
      title: "Make the next visit recovery-led",
      detail: `Latest RPE is ${session.rpe} and recovery is ${session.mood_recovery}/10. Keep the next swim technical or easy aerobic before another hard set.`,
      action: "Review sessions",
      href: "#session-record",
      tone: "coral",
      reason: [
        { label: "Last came", value: formatDateLabel(session.session_date) },
        { label: "Load", value: `${session.load_score}` },
        { label: "Next visit", value: suggestNextVisit(session) }
      ],
      recommendation: {
        nextSession: "Next: recovery-led technique + easy aerobic, reduced volume.",
        why: `Reason: last RPE was ${session.rpe} and recovery was ${session.mood_recovery}/10.`,
        risk: "Risk: stacking another hard set may hide fatigue or push the swimmer into poor technique.",
        coachAction: "Coach action: keep intensity low, watch stroke quality, and reassess recovery after the session."
      }
    };
  }

  if (!mental) {
    return {
      label: "Need check-in",
      title: "Capture mental readiness",
      detail: "The swimmer has training data, but no mental check-in. Add scores so confidence, focus, calm, and recovery influence the plan.",
      action: "Add check-in",
      href: "#manual-tools",
      tone: "water",
      reason: [
        { label: "Last came", value: formatDateLabel(session.session_date) },
        { label: "Recovery", value: `${session.mood_recovery}/10 from session` },
        { label: "Mental", value: "Missing latest check-in" }
      ],
      recommendation: {
        nextSession: "Next: keep the planned set, but add a mental check-in first.",
        why: "Reason: training load exists, but focus/confidence/calm are missing from the decision.",
        risk: "Risk: the swimmer may look physically ready while mentally underprepared.",
        coachAction: "Coach action: capture six mood scores, then use the lowest score as the pre-set coaching cue."
      }
    };
  }

  if (!report) {
    return {
      label: "Need video",
      title: "Upload technique evidence",
      detail: "The swimmer has session and readiness data. Add a clear video so the next recommendation can connect training load to stroke quality.",
      action: "Upload video",
      href: "#video-analysis",
      tone: "violet",
      reason: [
        { label: "Readiness", value: `${mental.composite_score}/10` },
        { label: "Training", value: `${session.session_type.replace(/_/g, " ")} / ${session.distance_m}m` },
        { label: "Video", value: "No report yet" }
      ],
      recommendation: {
        nextSession: "Next: run the planned session and record one clear technique clip.",
        why: "Reason: AquaIQ can see load and readiness, but cannot verify stroke quality yet.",
        risk: "Risk: the plan may improve fitness while missing the technical limiter.",
        coachAction: "Coach action: upload a side-view or above-water video with the full swimmer visible."
      }
    };
  }

  if (!plan) {
    return {
      label: "Need plan",
      title: "Generate the training plan",
      detail: "Technique and readiness data exist. Generate an active plan so each next session has a target and phase rationale.",
      action: "Generate plan",
      href: "#manual-tools",
      tone: "violet",
      reason: [
        { label: "Technique", value: `${report.overall_score}/100` },
        { label: "Readiness", value: `${mental.composite_score}/10` },
        { label: "Plan", value: "Missing" }
      ],
      recommendation: {
        nextSession: "Next: generate the active training plan before prescribing the next set.",
        why: "Reason: the swimmer has session, mental, and technique evidence, but no periodized target.",
        risk: "Risk: sessions may become disconnected from race date and target time.",
        coachAction: "Coach action: enter race date, race event, and target time, then generate the plan."
      }
    };
  }

  const confidenceRisk =
    report.analysis_status === "low_confidence" || report.confidence_score < 0.45
      ? "Risk: video confidence is low, so confirm the fault manually before changing the set."
      : topFault && numberFrom(topFault.time_cost_seconds, 0) > 0
        ? `Risk: ${textFrom(topFault.title, "the top fault")} may cost about ${numberFrom(topFault.time_cost_seconds).toFixed(1)}s per 50m if ignored.`
        : "Risk: low immediate risk; keep monitoring technique under fatigue.";
  const drillAction = topDrill
    ? `Coach action: start with ${textFrom(topDrill.volume, "4x50m")} ${textFrom(topDrill.name, "prescribed drill")} before the main set.`
    : "Coach action: run the planned set and check whether technique holds in the final reps.";

  return {
    label: "Ready",
    title: nextSet ? `${strokeLabel(nextSet.type)} next session` : "Follow active plan",
    detail: nextSet
      ? `${nextSet.main_set}. ${nextSet.target}${topDrill ? ` Add ${textFrom(topDrill.name, "the top drill")} for ${textFrom(topDrill.focus, "technique")}.` : ""}`
      : planMessage(plan) || "Use the current phase and latest report to guide the next session.",
    action: "Review plan",
    href: "#manual-tools",
    tone: "mint",
    reason: [
      { label: "Phase", value: `${plan.current_phase} week ${plan.current_week}` },
      { label: "Technique", value: `${report.overall_score}/100${topFault ? ` / ${textFrom(topFault.title, "top focus")}` : ""}` },
      { label: "Readiness", value: `${mental.composite_score}/10` }
    ],
    recommendation: {
      nextSession: nextSet ? `Next: ${nextSet.main_set}.` : `Next: follow ${plan.current_phase} week ${plan.current_week}.`,
      why: `Reason: current phase is ${plan.current_phase}, latest readiness is ${mental.composite_score}/10, and technique score is ${report.overall_score}/100.`,
      risk: confidenceRisk,
      coachAction: drillAction
    }
  };
}

function buildTimeline(workspace: SwimmerWorkspace) {
  const items: Array<{
    id: string;
    date: string;
    sortDate: string;
    type: string;
    title: string;
    detail: string;
    tone: "water" | "mint" | "coral" | "violet" | "slate";
  }> = [
    ...workspace.sessions.map((session) => ({
      id: `session-${session.id}`,
      date: session.session_date,
      sortDate: `${session.session_date}T12:00:00`,
      type: "Session",
      title: `${capitalize(session.session_type.replace(/_/g, " "))} / ${session.distance_m}m`,
      detail: `RPE ${session.rpe}, load ${session.load_score}, recovery ${session.mood_recovery}/10.`,
      tone: "water" as const
    })),
    ...workspace.reports.map((report) => ({
      id: `report-${report.id}`,
      date: report.created_at.slice(0, 10),
      sortDate: report.created_at,
      type: "Video",
      title: `${reportProviderName(report)} ${report.overall_score}/100`,
      detail: `${capitalize(report.confidence_label)} confidence, ${Math.round(report.pose_detection_rate * 100)}% pose rate.`,
      tone: "violet" as const
    })),
    ...workspace.races.map((race) => ({
      id: `race-${race.id}`,
      date: race.race_date,
      sortDate: `${race.race_date}T12:00:00`,
      type: "Race",
      title: `${race.event} / ${race.official_time_seconds}s`,
      detail: `${race.strategy_type} split, strategy score ${race.strategy_score}/100.`,
      tone: "coral" as const
    })),
    ...workspace.mental.map((checkin) => ({
      id: `mental-${checkin.id}`,
      date: checkin.checkin_date.slice(0, 10),
      sortDate: checkin.checkin_date,
      type: "Mind",
      title: `${checkin.checkin_type.replace(/_/g, " ")} / ${checkin.composite_score}/10`,
      detail: `Focus ${checkin.mood_focus}, confidence ${checkin.mood_confidence}, recovery ${checkin.mood_recovery}.`,
      tone: "mint" as const
    }))
  ];

  if (workspace.plan) {
    items.push({
      id: `plan-${workspace.plan.id}`,
      date: workspace.plan.updated_at.slice(0, 10),
      sortDate: workspace.plan.updated_at,
      type: "Plan",
      title: `${capitalize(workspace.plan.current_phase)} plan / week ${workspace.plan.current_week}`,
      detail: planMessage(workspace.plan) || `${workspace.plan.weeks_total}-week plan for ${workspace.plan.race_event}.`,
      tone: "slate" as const
    });
  }

  return items.sort((a, b) => new Date(b.sortDate).getTime() - new Date(a.sortDate).getTime()).slice(0, 8);
}

function SwimmerCommandCenter({ workspace }: { workspace: SwimmerWorkspace }) {
  const decision = buildCoachDecision(workspace);
  const latestSession = workspace.sessions[0] ?? null;
  const latestReport = workspace.reports[0] ?? null;
  const latestMental = workspace.mental[0] ?? null;

  return (
    <section className="panel p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-black uppercase tracking-wide text-water">Coach Decision Center</p>
          <h2 className="mt-1 text-xl font-bold text-ink">{decision.title}</h2>
        </div>
        <StatusPill tone={decision.tone}>{decision.label}</StatusPill>
      </div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{decision.detail}</p>

      <div className="mt-4 rounded-md border border-blue-100 bg-blue-50 p-4">
        <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-water">Recommendation Engine</p>
            <h3 className="mt-1 text-lg font-bold text-ink">{decision.recommendation.nextSession}</h3>
          </div>
          <StatusPill tone={decision.tone}>{decision.label}</StatusPill>
        </div>
        <div className="mt-4 grid gap-3 md:grid-cols-3">
          <RecommendationFact label="Why" value={decision.recommendation.why} />
          <RecommendationFact label="Risk" value={decision.recommendation.risk} />
          <RecommendationFact label="Exact coach action" value={decision.recommendation.coachAction} />
        </div>
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {decision.reason.map((item) => (
          <div key={item.label} className="rounded-md border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-black uppercase text-slate-400">{item.label}</p>
            <p className="mt-1 text-sm font-bold text-ink">{item.value}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 grid gap-3 rounded-md border border-slate-200 p-3 text-sm md:grid-cols-3">
        <MiniEvidence label="Last came" value={latestSession ? formatDateLabel(latestSession.session_date) : "No session"} />
        <MiniEvidence label="Latest report" value={latestReport ? `${latestReport.overall_score}/100 ${reportProviderName(latestReport)}` : "No video report"} />
        <MiniEvidence label="Readiness" value={latestMental ? `${latestMental.composite_score}/10 mental` : latestSession ? `${latestSession.mood_recovery}/10 recovery` : "No check-in"} />
      </div>

      <a className="primary-button mt-4" href={decision.href}>
        <Target size={16} />
        {decision.action}
      </a>
    </section>
  );
}

function RecommendationFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-blue-100 bg-white p-3">
      <p className="text-xs font-black uppercase text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-bold leading-5 text-ink">{value}</p>
    </div>
  );
}

function MiniEvidence({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs font-black uppercase text-slate-400">{label}</p>
      <p className="mt-1 font-semibold text-ink">{value}</p>
    </div>
  );
}

function SwimmerTimeline({ workspace }: { workspace: SwimmerWorkspace }) {
  const items = buildTimeline(workspace);
  return (
    <section className="panel p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-black uppercase tracking-wide text-water">Swimmer Timeline</p>
          <h2 className="mt-1 text-lg font-bold text-ink">What changed recently</h2>
        </div>
        <StatusPill tone={items.length ? "mint" : "slate"}>{items.length} events</StatusPill>
      </div>

      <div className="mt-4 space-y-3">
        {items.length ? (
          items.map((item) => <TimelineRow key={item.id} item={item} />)
        ) : (
          <div className="rounded-md border border-dashed border-slate-300 p-4 text-sm text-slate-500">
            Log a session, upload video, create a plan, or add a race result to build this swimmer timeline.
          </div>
        )}
      </div>
    </section>
  );
}

function TimelineRow({ item }: { item: ReturnType<typeof buildTimeline>[number] }) {
  const dotClasses = {
    water: "bg-water",
    mint: "bg-mint",
    coral: "bg-coral",
    violet: "bg-violet",
    slate: "bg-slate-400"
  };
  return (
    <div className="grid grid-cols-[88px_1fr] gap-3 rounded-md border border-slate-200 p-3">
      <div>
        <p className="text-xs font-black uppercase text-slate-400">{formatDateLabel(item.date)}</p>
        <div className="mt-2 flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${dotClasses[item.tone]}`} />
          <span className="text-xs font-bold text-slate-500">{item.type}</span>
        </div>
      </div>
      <div>
        <p className="font-bold text-ink">{item.title}</p>
        <p className="mt-1 text-sm leading-5 text-slate-600">{item.detail}</p>
      </div>
    </div>
  );
}

function ProfileSummary({ swimmer }: { swimmer: Swimmer }) {
  const pb = Object.entries(swimmer.personal_bests)[0];
  return (
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <StatusPill tone="water">Profile</StatusPill>
        <Target size={18} className="text-water" />
      </div>
      <h2 className="text-lg font-bold text-ink">{swimmer.primary_stroke}</h2>
      <p className="mt-1 text-sm text-slate-500">{swimmer.primary_event}</p>
      <p className="mt-4 text-2xl font-bold text-ink">{pb ? `${pb[1]}s` : "No PB"}</p>
      <p className="text-sm text-slate-500">{pb?.[0] ?? "Personal best"}</p>
    </section>
  );
}

function ActivePlanSummary({ plan }: { plan: TrainingPlan | null }) {
  return (
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <StatusPill tone="mint">Plan</StatusPill>
        <CalendarPlus size={18} className="text-mint" />
      </div>
      {plan ? (
        <>
          <h2 className="text-lg font-bold capitalize text-ink">{plan.current_phase}</h2>
          <p className="mt-1 text-sm text-slate-500">
            Week {plan.current_week} of {plan.weeks_total}
          </p>
          <p className="mt-4 text-2xl font-bold text-ink">{plan.target_time_seconds}s</p>
          <p className="text-sm text-slate-500">{plan.race_event}</p>
        </>
      ) : (
        <p className="text-sm text-slate-500">No active plan.</p>
      )}
    </section>
  );
}

function LatestTechnique({ reports }: { reports: TechniqueReport[] }) {
  const latest = reports[0];
  return (
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <StatusPill tone="violet">Technique</StatusPill>
        <FileVideo size={18} className="text-violet" />
      </div>
      {latest ? (
        <>
          <h2 className="text-lg font-bold text-ink">{latest.overall_score}/100</h2>
          <p className="mt-1 text-sm text-slate-500">{reportProviderName(latest)} analysis</p>
          <p className="mt-4 text-sm text-slate-600">{latest.coaching_summary}</p>
        </>
      ) : (
        <p className="text-sm text-slate-500">No technique report.</p>
      )}
    </section>
  );
}

function LatestRace({ races }: { races: RaceAnalysis[] }) {
  const latest = races[0];
  return (
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <StatusPill tone="coral">Race</StatusPill>
        <Flag size={18} className="text-coral" />
      </div>
      {latest ? (
        <>
          <h2 className="text-lg font-bold text-ink">{latest.official_time_seconds}s</h2>
          <p className="mt-1 text-sm text-slate-500 capitalize">{latest.strategy_type} split</p>
          <p className="mt-4 text-sm text-slate-600">{latest.ai_insights[0]}</p>
        </>
      ) : (
        <p className="text-sm text-slate-500">No race analysis.</p>
      )}
    </section>
  );
}

function SwimmerOrganizer({ workspace, onSaved }: { workspace: SwimmerWorkspace; onSaved: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<SwimSession | null>(null);
  const sessionRows = workspace.sessions.map((session) => {
    const report = workspace.reports.find((item) => item.session_id === session.id) ?? null;
    const mental = workspace.mental.find((item) => item.checkin_date.slice(0, 10) === session.session_date) ?? null;
    return { session, report, mental };
  });

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const created = await api.createSession(token, {
        swimmer_id: workspace.swimmer.id,
        session_date: form.get("session_date"),
        session_type: form.get("session_type"),
        distance_m: Number(form.get("distance_m")),
        duration_min: Number(form.get("duration_min")),
        rpe: Number(form.get("rpe")),
        mood_focus: Number(form.get("mood_focus")),
        mood_confidence: Number(form.get("mood_confidence")),
        mood_energy: Number(form.get("mood_energy")),
        mood_calm: Number(form.get("mood_calm")),
        mood_recovery: Number(form.get("mood_recovery")),
        mood_motivation: Number(form.get("mood_motivation")),
        sleep_hours: Number(form.get("sleep_hours")),
        notes: form.get("notes")
      });
      formElement.reset();
      setMessage(`${created.session_date} ${created.session_type.replace(/_/g, " ")} saved. Load ${created.load_score}.`);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save session");
    } finally {
      setSaving(false);
    }
  }

  async function deleteSession() {
    if (!token || !pendingDelete) return;
    const session = pendingDelete;
    setDeletingId(session.id);
    try {
      await api.deleteSession(token, session.id);
      setPendingDelete(null);
      setMessage(`${session.session_date} ${session.session_type.replace(/_/g, " ")} session deleted.`);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to delete session");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section id="session-record" className="panel p-5">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="text-xs font-black uppercase tracking-wide text-water">Swimmer Organizer</p>
          <h2 className="mt-1 text-lg font-bold text-ink">Attendance and session record</h2>
        </div>
        <StatusPill tone={sessionRows.length ? "mint" : "slate"}>{sessionRows.length} records</StatusPill>
      </div>
      <p className="mt-1 text-sm text-slate-500">Every row is one swimmer visit: when they came, what they did, readiness, sleep, video status, and notes.</p>

      <details className="mt-4 rounded-md border border-slate-200 bg-slate-50 p-4">
        <summary className="cursor-pointer text-sm font-black text-ink">
          <span className="inline-flex items-center gap-2">
            <CalendarPlus size={16} className="text-water" />
            Add session to this swimmer
          </span>
        </summary>
        <form onSubmit={onSubmit} className="mt-4 grid gap-3 md:grid-cols-4 xl:grid-cols-6">
          <label>
            <span className="label">Date</span>
            <input className="field mt-1" type="date" name="session_date" defaultValue={new Date().toISOString().slice(0, 10)} required />
          </label>
          <label>
            <span className="label">Type</span>
            <select className="field mt-1" name="session_type" defaultValue="technique">
              <option value="base">Base</option>
              <option value="threshold">Threshold</option>
              <option value="vo2">VO2</option>
              <option value="technique">Technique</option>
              <option value="race_pace">Race Pace</option>
              <option value="recovery">Recovery</option>
            </select>
          </label>
          <NumberField name="distance_m" label="Distance" value={3200} />
          <NumberField name="duration_min" label="Minutes" value={75} />
          <NumberField name="rpe" label="RPE" value={7} min={1} max={10} />
          <NumberField name="sleep_hours" label="Sleep" value={7.5} step="0.25" />
          {["focus", "confidence", "energy", "calm", "recovery", "motivation"].map((field) => (
            <NumberField key={field} name={`mood_${field}`} label={field} value={7} min={1} max={10} />
          ))}
          <label className="md:col-span-4 xl:col-span-5">
            <span className="label">Notes</span>
            <textarea className="field mt-1 min-h-10" name="notes" />
          </label>
          <div className="flex items-end">
            <button className="primary-button w-full" type="submit" disabled={saving}>
              <CalendarPlus size={16} />
              {saving ? "Saving" : "Save Session"}
            </button>
          </div>
        </form>
      </details>

      {message ? <p className="mt-3 rounded-md bg-emerald-50 px-3 py-2 text-sm font-semibold text-emerald-800">{message}</p> : null}
      {error ? <p className="mt-3 rounded-md bg-red-50 px-3 py-2 text-sm font-semibold text-red-700">{error}</p> : null}

      <div className="mt-4 overflow-x-auto rounded-md border border-slate-200">
        <table className="min-w-[1060px] w-full text-left text-sm">
          <thead className="bg-slate-50 text-xs font-black uppercase text-slate-500">
            <tr>
              <th className="px-3 py-3">Came</th>
              <th className="px-3 py-3">Exercise</th>
              <th className="px-3 py-3">Load</th>
              <th className="px-3 py-3">Readiness</th>
              <th className="px-3 py-3">Sleep</th>
              <th className="px-3 py-3">Video / Report</th>
              <th className="px-3 py-3">Next Visit</th>
              <th className="px-3 py-3">Notes</th>
              <th className="px-3 py-3">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {sessionRows.length ? (
              sessionRows.map(({ session, report, mental }) => (
                <tr key={session.id} className="align-top">
                  <td className="px-3 py-3 font-semibold text-ink">{formatDateLabel(session.session_date)}</td>
                  <td className="px-3 py-3">
                    <p className="font-bold capitalize text-ink">{session.session_type.replace(/_/g, " ")}</p>
                    <p className="text-xs text-slate-500">{session.distance_m}m / {session.duration_min}min</p>
                  </td>
                  <td className="px-3 py-3">
                    <p className="font-bold text-ink">{session.load_score}</p>
                    <p className="text-xs text-slate-500">RPE {session.rpe}</p>
                  </td>
                  <td className="px-3 py-3">
                    <p className="font-bold text-ink">{mental ? `${mental.composite_score}/10` : `${session.mood_recovery}/10`}</p>
                    <p className="text-xs text-slate-500">{mental ? "mental" : "recovery"}</p>
                  </td>
                  <td className="px-3 py-3 font-semibold text-ink">{session.sleep_hours ? `${session.sleep_hours}h` : "-"}</td>
                  <td className="px-3 py-3">
                    <div className="flex flex-wrap gap-1.5">
                      <StatusPill tone={session.video_url ? "mint" : "slate"}>{session.video_url ? "Video" : "No video"}</StatusPill>
                      <StatusPill tone={report ? "violet" : "slate"}>{report ? `${report.overall_score}/100` : "No report"}</StatusPill>
                    </div>
                    {report ? <p className="mt-1 text-xs text-slate-500">{reportProviderName(report)} / {report.confidence_label}</p> : null}
                  </td>
                  <td className="px-3 py-3 font-semibold text-ink">{suggestNextVisit(session)}</td>
                  <td className="px-3 py-3 text-xs leading-5 text-slate-600">{session.notes?.trim() || "-"}</td>
                  <td className="px-3 py-3">
                    <button
                      className="icon-button text-red-600 hover:border-red-300 hover:text-red-700"
                      type="button"
                      title="Delete session"
                      aria-label={`Delete ${session.session_date} session`}
                      disabled={deletingId === session.id}
                      onClick={() => setPendingDelete(session)}
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-3 py-6 text-center text-sm text-slate-500" colSpan={9}>
                  No sessions yet. Save a session to start the swimmer record.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <DeleteConfirmDialog
        open={Boolean(pendingDelete)}
        title="Delete session?"
        itemName={pendingDelete ? `${pendingDelete.session_date} - ${pendingDelete.session_type}` : "Session"}
        description="This removes the selected training entry from the swimmer timeline and updates the organizer table from the remaining records."
        details={
          pendingDelete
            ? [`${pendingDelete.distance_m}m session entry`, `RPE ${pendingDelete.rpe} and load ${pendingDelete.load_score}`, "Any linked video/report records for this session"]
            : []
        }
        confirmLabel="Delete session"
        isDeleting={Boolean(deletingId)}
        onCancel={() => setPendingDelete(null)}
        onConfirm={deleteSession}
      />
    </section>
  );
}

function AICoachingPanel({ swimmerId, outputs, onSynced }: { swimmerId: string; outputs: AIOutput[]; onSynced: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [syncing, setSyncing] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const latest = outputs[0];
  const provider = latest ? latest.provider : "local";
  const outputCounts = outputs.reduce<Record<string, number>>((acc, output) => {
    acc[output.output_type] = (acc[output.output_type] ?? 0) + 1;
    return acc;
  }, {});

  async function syncOutputs() {
    if (!token) return;
    setSyncing(true);
    setMessage(null);
    try {
      const created = await api.backfillAIOutputs(token, swimmerId);
      setMessage(created.length ? `Created ${created.length} AI audit outputs.` : "AI outputs are already synced.");
      onSynced();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to sync AI outputs.");
    } finally {
      setSyncing(false);
    }
  }

  return (
    <section className="panel p-0">
      <details className="group">
        <summary className="flex cursor-pointer list-none flex-col gap-3 p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-black uppercase tracking-wide text-slate-400">Developer audit</p>
            <h2 className="mt-1 flex items-center gap-2 text-base font-bold text-ink">
              <Zap size={16} className="text-slate-400" />
              AI output records
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              {outputs.length ? `${outputs.length} backend records / ${providerLabel(provider)} provider` : "No AI audit records yet."}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {["technique_summary", "training_rationale", "race_debrief", "mental_routine"].map((type) => (
              <span key={type} className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs font-black text-slate-600">
                {formatOutputType(type)}: {outputCounts[type] ?? 0}
              </span>
            ))}
          </div>
        </summary>

        <div className="border-t border-slate-200 p-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <p className="max-w-2xl text-sm text-slate-500">
              This is a developer audit trail for debugging provider calls. The coaching workspace above already shows the useful swimmer-facing outputs.
            </p>
            <button className="secondary-button" type="button" onClick={syncOutputs} disabled={syncing}>
              <RefreshCw size={16} />
              {syncing ? "Syncing" : "Sync existing data"}
            </button>
          </div>

          {message ? <div className="mt-4 rounded-md bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-800">{message}</div> : null}

          <div className="mt-4 space-y-3">
            {outputs.length ? (
              outputs.slice(0, 4).map((output) => (
                <div key={output.id} className="rounded-md border border-slate-200 p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <StatusPill tone={output.status === "completed" ? "mint" : "coral"}>{output.status}</StatusPill>
                    <span className="text-xs font-bold uppercase text-slate-400">{formatOutputType(output.output_type)}</span>
                    <span className="text-xs text-slate-400">{providerLabel(output.provider)} / {output.prompt_version}</span>
                  </div>
                  <p className="mt-2 text-sm text-slate-700">{output.parsed_json.summary || "No summary returned."}</p>
                  {output.error ? <p className="mt-2 text-xs font-semibold text-amber-700">{output.error}</p> : null}
                </div>
              ))
            ) : (
              <div className="rounded-md border border-dashed border-slate-300 p-4 text-sm text-slate-500">
                Generate a new report, plan, race debrief, or mental check-in to create AI output records, or sync the existing swimmer data now.
              </div>
            )}
          </div>
        </div>
      </details>
    </section>
  );
}

function ProfileEditor({ swimmer, onSaved }: { swimmer: Swimmer; onSaved: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pb = Object.entries(swimmer.personal_bests)[0];

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const primaryEvent = String(form.get("primary_event"));
    const pbEvent = String(form.get("pb_event") || primaryEvent);
    const pbSeconds = Number(form.get("pb_seconds"));
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      await api.updateSwimmer(token, swimmer.id, {
        name: form.get("name"),
        date_of_birth: form.get("date_of_birth") || null,
        primary_stroke: form.get("primary_stroke"),
        primary_event: primaryEvent,
        level: form.get("level"),
        personal_bests: Number.isFinite(pbSeconds) && pbSeconds > 0 ? { [pbEvent]: pbSeconds } : {}
      });
      setMessage("Profile updated.");
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update swimmer");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <Save size={18} />
        Fix Profile
      </h2>
      <form onSubmit={onSubmit} className="mt-4 grid gap-4 md:grid-cols-2">
        <label>
          <span className="label">Name</span>
          <input className="field mt-1" name="name" defaultValue={swimmer.name} required />
        </label>
        <label>
          <span className="label">Birth Date</span>
          <input className="field mt-1" name="date_of_birth" type="date" defaultValue={swimmer.date_of_birth ?? ""} />
        </label>
        <label>
          <span className="label">Stroke</span>
          <select className="field mt-1" name="primary_stroke" defaultValue={swimmer.primary_stroke}>
            <option value="freestyle">Freestyle</option>
            <option value="backstroke">Backstroke</option>
            <option value="breaststroke">Breaststroke</option>
            <option value="butterfly">Butterfly</option>
            <option value="IM">IM</option>
          </select>
        </label>
        <label>
          <span className="label">Level</span>
          <select className="field mt-1" name="level" defaultValue={swimmer.level}>
            <option value="junior">Junior</option>
            <option value="age_group">Age Group</option>
            <option value="elite">Elite</option>
            <option value="masters">Masters</option>
          </select>
        </label>
        <label>
          <span className="label">Primary Event</span>
          <input className="field mt-1" name="primary_event" defaultValue={swimmer.primary_event} required />
        </label>
        <label>
          <span className="label">PB Event</span>
          <input className="field mt-1" name="pb_event" defaultValue={pb?.[0] ?? swimmer.primary_event} />
        </label>
        <label>
          <span className="label">PB Seconds</span>
          <input className="field mt-1" name="pb_seconds" type="number" step="0.01" defaultValue={pb?.[1] ?? ""} />
        </label>
        <div className="flex items-end">
          <button className="primary-button w-full" type="submit" disabled={saving}>
            <Save size={16} />
            {saving ? "Saving" : "Save Fixes"}
          </button>
        </div>
        {message ? <p className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-800 md:col-span-2">{message}</p> : null}
        {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 md:col-span-2">{error}</p> : null}
      </form>
    </section>
  );
}

function NumberField({
  name,
  label,
  value,
  min,
  max,
  step
}: {
  name: string;
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: string;
}) {
  return (
    <label>
      <span className="label capitalize">{label}</span>
      <input className="field mt-1" type="number" name={name} defaultValue={value} min={min} max={max} step={step ?? "1"} />
    </label>
  );
}

function VideoUploadPanel({ sessions, onSaved, variant = "light" }: { sessions: SwimSession[]; onSaved: () => void; variant?: "light" | "dark" }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const isDark = variant === "dark";

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    const sessionId = String(form.get("session_id"));
    const file = form.get("file");
    if (!(file instanceof File) || !file.size) {
      setMessage("Choose a video file.");
      return;
    }
    setSaving(true);
    setMessage(null);
    try {
      const result = await api.uploadVideo(token, sessionId, file);
      setMessage(`Technique report ${result.report.overall_score}/100 created.`);
      onSaved();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to upload video");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className={isDark ? "rounded-md border border-stone-700 bg-stone-950/35 p-4 text-stone-100" : "panel p-5"}>
      <h2 className={`flex items-center gap-2 text-lg font-bold ${isDark ? "text-stone-100" : "text-ink"}`}>
        <Upload size={18} />
        Upload analysis video
      </h2>
      <form onSubmit={onSubmit} className="mt-4 space-y-4">
        <label className="block">
          <span className={isDark ? "text-xs font-black uppercase text-stone-500" : "label"}>Session</span>
          <select className={isDark ? "mt-1 w-full rounded-md border border-stone-700 bg-stone-900 px-3 py-2 text-sm text-stone-100" : "field mt-1"} name="session_id" required>
            {sessions.map((session) => (
              <option key={session.id} value={session.id}>
                {session.session_date} / {session.session_type} / load {session.load_score}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <span className={isDark ? "text-xs font-black uppercase text-stone-500" : "label"}>Video</span>
          <input
            className={
              isDark
                ? "mt-1 w-full rounded-md border border-dashed border-stone-700 bg-stone-900 px-3 py-2 text-sm text-stone-300 file:mr-3 file:rounded file:border-0 file:bg-emerald-500/15 file:px-3 file:py-1.5 file:text-sm file:font-bold file:text-emerald-100"
                : "field mt-1"
            }
            type="file"
            name="file"
            accept="video/*"
          />
        </label>
        {message ? (
          <p className={isDark ? "rounded-md border border-stone-700 bg-stone-900 px-3 py-2 text-sm text-stone-300" : "rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-700"}>
            {message}
          </p>
        ) : null}
        <button className="primary-button" type="submit" disabled={saving || sessions.length === 0}>
          <FileVideo size={16} />
          {saving ? "Analyzing" : "Upload"}
        </button>
      </form>
    </section>
  );
}

function PlanForm({ swimmer, plan, aiOutputs, onSaved }: { swimmer: Swimmer; plan: TrainingPlan | null; aiOutputs: AIOutput[]; onSaved: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);
  const [generatedPlan, setGeneratedPlan] = useState<TrainingPlan | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pb = Object.entries(swimmer.personal_bests)[0]?.[1] ?? 60;
  const shownPlan = generatedPlan ?? plan;
  const trainingOutput = latestAIOutput(aiOutputs, "training_rationale");
  const existingPlanMessage = planMessage(shownPlan);
  const shownMessage = message || existingPlanMessage || trainingOutput?.parsed_json.summary || null;

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    setSaving(true);
    setMessage(null);
    setError(null);
    try {
      const created = await api.generatePlan(token, {
        swimmer_id: swimmer.id,
        race_date: form.get("race_date"),
        race_event: form.get("race_event"),
        target_time_seconds: Number(form.get("target_time_seconds"))
      });
      setGeneratedPlan(created);
      setMessage(
        `Generated ${created.weeks_total}-week plan: ${created.current_phase} phase, week ${created.current_week}, target ${created.target_time_seconds}s for ${created.race_event}.`
      );
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to generate plan");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel p-5 xl:col-span-2">
      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <div>
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-black uppercase tracking-wide text-water">Plan Generator</p>
              <h2 className="mt-1 flex items-center gap-2 text-xl font-bold text-ink">
                <Target size={20} />
                Training Plan
              </h2>
              <p className="mt-2 text-sm text-slate-500">Set the race target, then AquaIQ creates the phase plan and stores the AI rationale.</p>
            </div>
            {shownPlan ? <StatusPill tone="mint">Active plan</StatusPill> : <StatusPill>No plan</StatusPill>}
          </div>
          <form onSubmit={onSubmit} className="mt-5 grid gap-4 md:grid-cols-3">
            <label>
              <span className="label">Race Date</span>
              <input
                className="field mt-1"
                name="race_date"
                type="date"
                defaultValue={plan?.race_date ?? new Date(Date.now() + 56 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)}
              />
            </label>
            <label>
              <span className="label">Target Time</span>
              <input className="field mt-1" name="target_time_seconds" type="number" step="0.01" defaultValue={plan?.target_time_seconds ?? Math.max(1, pb - 0.5)} />
            </label>
            <label>
              <span className="label">Race Event</span>
              <input className="field mt-1" name="race_event" defaultValue={plan?.race_event ?? swimmer.primary_event} />
            </label>
            {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 md:col-span-3">{error}</p> : null}
            <button className="primary-button md:col-span-3" type="submit" disabled={saving}>
              <Target size={16} />
              {saving ? "Generating plan and AI message" : "Generate Plan"}
            </button>
          </form>
        </div>

        <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <PlanStat label="Phase" value={shownPlan ? capitalize(shownPlan.current_phase) : "-"} />
            <PlanStat label="Length" value={shownPlan ? `${shownPlan.weeks_total} weeks` : "-"} />
            <PlanStat label="Target" value={shownPlan ? `${shownPlan.target_time_seconds}s` : "-"} />
          </div>
          <div className="mt-4 rounded-md border border-blue-100 bg-white p-4">
            <p className="text-xs font-black uppercase text-water">Generated message</p>
            <p className="mt-2 text-sm leading-6 text-slate-700">
              {shownMessage || "Generate a plan to see the backend-created message here."}
            </p>
          </div>
          {shownPlan ? (
            <div className="mt-4">
              <p className="text-xs font-black uppercase text-slate-500">Phase split</p>
              <div className="mt-2 grid grid-cols-4 gap-2">
                {Object.entries(shownPlan.phase_config).map(([phase, weeks]) => (
                  <div key={phase} className="rounded bg-white px-3 py-2">
                    <p className="text-xs font-bold capitalize text-slate-500">{phase}</p>
                    <p className="mt-1 text-lg font-black text-ink">{weeks}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function PlanStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-white px-3 py-3">
      <p className="text-xs font-black uppercase text-slate-400">{label}</p>
      <p className="mt-1 text-lg font-black text-ink">{value}</p>
    </div>
  );
}

function RaceForm({ swimmer, onSaved }: { swimmer: Swimmer; onSaved: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    const splits = String(form.get("splits_actual"))
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value) && value > 0);
    setSaving(true);
    try {
      await api.createRaceAnalysis(token, {
        swimmer_id: swimmer.id,
        race_date: form.get("race_date"),
        event: form.get("event"),
        official_time_seconds: Number(form.get("official_time_seconds")),
        splits_actual: splits,
        reaction_time_ms: Number(form.get("reaction_time_ms")),
        turn_times: []
      });
      onSaved();
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <Flag size={18} />
        Race Analysis
      </h2>
      <form onSubmit={onSubmit} className="mt-4 grid gap-4 md:grid-cols-2">
        <label>
          <span className="label">Date</span>
          <input className="field mt-1" name="race_date" type="date" defaultValue={new Date().toISOString().slice(0, 10)} />
        </label>
        <label>
          <span className="label">Official Time</span>
          <input className="field mt-1" name="official_time_seconds" type="number" step="0.01" defaultValue="61.50" />
        </label>
        <label>
          <span className="label">Reaction ms</span>
          <input className="field mt-1" name="reaction_time_ms" type="number" defaultValue="720" />
        </label>
        <label>
          <span className="label">Event</span>
          <input className="field mt-1" name="event" defaultValue={swimmer.primary_event} />
        </label>
        <label className="md:col-span-2">
          <span className="label">Splits</span>
          <input className="field mt-1" name="splits_actual" defaultValue="15.1, 15.5, 15.7, 15.2" />
        </label>
        <button className="primary-button md:col-span-2" type="submit" disabled={saving}>
          <Flag size={16} />
          {saving ? "Analyzing" : "Save Race"}
        </button>
      </form>
    </section>
  );
}

function MentalForm({ swimmerId, onSaved }: { swimmerId: string; onSaved: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    setSaving(true);
    try {
      await api.createMentalCheckin(token, {
        swimmer_id: swimmerId,
        checkin_type: form.get("checkin_type"),
        mood_focus: Number(form.get("mood_focus")),
        mood_confidence: Number(form.get("mood_confidence")),
        mood_energy: Number(form.get("mood_energy")),
        mood_calm: Number(form.get("mood_calm")),
        mood_recovery: Number(form.get("mood_recovery")),
        mood_motivation: Number(form.get("mood_motivation")),
        notes: form.get("notes")
      });
      onSaved();
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel p-5">
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
        <Brain size={18} />
        Mental Check-in
      </h2>
      <form onSubmit={onSubmit} className="mt-4 grid gap-4 md:grid-cols-2">
        <label className="md:col-span-2">
          <span className="label">Type</span>
          <select className="field mt-1" name="checkin_type" defaultValue="pre_session">
            <option value="pre_session">Pre Session</option>
            <option value="pre_race">Pre Race</option>
            <option value="post_race">Post Race</option>
          </select>
        </label>
        {["focus", "confidence", "energy", "calm", "recovery", "motivation"].map((field) => (
          <NumberField key={field} name={`mood_${field}`} label={field} value={7} min={1} max={10} />
        ))}
        <label className="md:col-span-2">
          <span className="label">Notes</span>
          <textarea className="field mt-1 min-h-20" name="notes" />
        </label>
        <button className="primary-button md:col-span-2" type="submit" disabled={saving}>
          <Brain size={16} />
          {saving ? "Saving" : "Save Check-in"}
        </button>
      </form>
    </section>
  );
}

function ReportsPanel({ reports, sessions }: { reports: TechniqueReport[]; sessions: SwimSession[] }) {
  return (
    <section className="panel p-5">
      <h2 className="text-lg font-bold text-ink">Reports</h2>
      <div className="mt-4 space-y-3">
        {reports.length ? (
          reports.slice(0, 5).map((report) => {
            const session = sessions.find((item) => item.id === report.session_id);
            return (
              <div key={report.id} className="rounded-md border border-slate-200 p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <p className="font-semibold text-ink">
                    {report.overall_score}/100 / {session?.session_date ?? "Session"}
                  </p>
                  <StatusPill tone="violet">{report.processing_status}</StatusPill>
                </div>
                <p className="text-sm text-slate-600">{report.coaching_summary}</p>
              </div>
            );
          })
        ) : (
          <p className="text-sm text-slate-500">No reports yet.</p>
        )}
      </div>
    </section>
  );
}

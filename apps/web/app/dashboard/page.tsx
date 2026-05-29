"use client";

import {
  AlertTriangle,
  CalendarClock,
  CheckCircle2,
  ClipboardList,
  FileVideo,
  Plus,
  RefreshCw,
  Timer,
  Trash2,
  Upload,
  Users
} from "lucide-react";
import Link from "next/link";
import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AppShell } from "@/components/AppShell";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";
import { MetricCard } from "@/components/MetricCard";
import { StatusPill } from "@/components/StatusPill";
import { api } from "@/lib/api";
import type { DashboardOverview, PilotBoard, PilotBoardCard, PilotSessionEntryResponse } from "@/lib/types";
import { useAuthStore } from "@/store/auth";

export default function DashboardPage() {
  const token = useAuthStore((state) => state.token);
  const [overview, setOverview] = useState<DashboardOverview | null>(null);
  const [pilotBoard, setPilotBoard] = useState<PilotBoard | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  const loadDashboard = useCallback(async () => {
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const [overviewData, pilotData] = await Promise.all([api.dashboard(token), api.pilotBoard(token)]);
      setOverview(overviewData);
      setPilotBoard(pilotData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load dashboard");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  const chartData = useMemo(
    () =>
      overview?.recent_sessions
        .slice()
        .reverse()
        .map((session) => ({
          date: new Date(session.session_date).toLocaleDateString(undefined, { month: "short", day: "numeric" }),
          load: session.load_score
        })) ?? [],
    [overview]
  );

  return (
    <AppShell>
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ink">Pilot Board</h1>
          <p className="mt-1 text-sm text-slate-500">Fast daily entry and readiness tracking for your first swimmers.</p>
        </div>
        <div className="flex gap-2">
          <button className="secondary-button" type="button" onClick={loadDashboard}>
            <RefreshCw size={16} />
            Refresh
          </button>
          <button className="primary-button" type="button" onClick={() => setShowCreate((value) => !value)}>
            <Plus size={16} />
            Swimmer
          </button>
        </div>
      </div>

      {showCreate ? (
        <CreateSwimmerPanel
          onCreated={() => {
            setShowCreate(false);
            loadDashboard();
          }}
        />
      ) : null}

      {error ? <div className="mb-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}
      {loading ? <div className="panel p-6 text-sm text-slate-500">Loading pilot board...</div> : null}

      {overview && pilotBoard ? (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard label="Swimmers" value={overview.swimmers_count} detail="Active pilot profiles" icon={Users} />
            <MetricCard
              label="Sessions"
              value={overview.sessions_count}
              detail="Logged training records"
              icon={CalendarClock}
              tone="mint"
            />
            <MetricCard
              label="Need Video"
              value={pilotBoard.totals.needs_video}
              detail="Missing technique evidence"
              icon={FileVideo}
              tone="violet"
            />
            <MetricCard
              label="Watch"
              value={pilotBoard.totals.at_risk}
              detail="High RPE or low recovery"
              icon={AlertTriangle}
              tone="coral"
            />
          </div>

          <TodayCoachBrief board={pilotBoard} />

          <FastEntryPanel board={pilotBoard} onSaved={loadDashboard} />

          <section className="grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
            <div className="panel p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-lg font-bold text-ink">Swimmer Status</h2>
                <span className="text-sm text-slate-500">
                  Updated {new Date(pilotBoard.generated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </span>
              </div>

              {pilotBoard.swimmers.length ? (
                <div className="grid gap-4 lg:grid-cols-2">
                  {pilotBoard.swimmers.map((card) => (
                    <PilotSwimmerCard key={card.swimmer.id} card={card} onDeleted={loadDashboard} />
                  ))}
                </div>
              ) : (
                <EmptyPilotState onCreate={() => setShowCreate(true)} />
              )}
            </div>

            <div className="space-y-6">
              <section className="panel p-5">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="text-lg font-bold text-ink">Recent Load</h2>
                  <StatusPill tone="mint">Live</StatusPill>
                </div>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="date" tickLine={false} axisLine={false} />
                      <YAxis tickLine={false} axisLine={false} />
                      <Tooltip />
                      <Bar dataKey="load" fill="#1a56a0" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>

              <section className="panel p-5">
                <h2 className="text-lg font-bold text-ink">Pilot Gaps</h2>
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                  <GapTile label="Need session" value={pilotBoard.totals.needs_session} />
                  <GapTile label="Need check-in" value={pilotBoard.totals.needs_checkin} />
                  <GapTile label="Need video" value={pilotBoard.totals.needs_video} />
                  <GapTile label="Watch" value={pilotBoard.totals.at_risk} />
                </div>
              </section>
            </div>
          </section>
        </div>
      ) : null}
    </AppShell>
  );
}

function FastEntryPanel({ board, onSaved }: { board: PilotBoard; onSaved: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [selectedSwimmerId, setSelectedSwimmerId] = useState("");
  const [sessionType, setSessionType] = useState("technique");
  const [saving, setSaving] = useState(false);
  const [result, setResult] = useState<PilotSessionEntryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const savedType = window.localStorage.getItem("aquaiq_last_session_type");
    if (savedType) {
      setSessionType(savedType);
    }
  }, []);

  useEffect(() => {
    if (!selectedSwimmerId && board.swimmers[0]) {
      setSelectedSwimmerId(board.swimmers[0].swimmer.id);
    }
  }, [board.swimmers, selectedSwimmerId]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    const nextSessionType = String(form.get("session_type"));
    setSaving(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.createPilotSessionEntry(token, {
        swimmer_id: form.get("swimmer_id"),
        session_date: form.get("session_date"),
        session_type: nextSessionType,
        distance_m: Number(form.get("distance_m")),
        duration_min: Number(form.get("duration_min")),
        rpe: Number(form.get("rpe")),
        sleep_hours: form.get("sleep_hours") ? Number(form.get("sleep_hours")) : null,
        mood_focus: Number(form.get("mood_focus")),
        mood_confidence: Number(form.get("mood_confidence")),
        mood_energy: Number(form.get("mood_energy")),
        mood_calm: Number(form.get("mood_calm")),
        mood_recovery: Number(form.get("mood_recovery")),
        mood_motivation: Number(form.get("mood_motivation")),
        notes: form.get("notes"),
        create_mental_checkin: form.get("create_mental_checkin") === "on",
        checkin_type: "post_session",
        checkin_notes: form.get("notes")
      });
      window.localStorage.setItem("aquaiq_last_session_type", nextSessionType);
      setSessionType(nextSessionType);
      setResult(response);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to save pilot entry");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-bold text-ink">
            <ClipboardList size={18} />
            Fast Session Entry
          </h2>
          <p className="mt-1 text-sm text-slate-500">One form for today&apos;s session and optional mental check-in.</p>
        </div>
        <StatusPill tone="water">Pilot</StatusPill>
      </div>

      <form onSubmit={onSubmit} className="grid gap-4 md:grid-cols-4">
        <label className="md:col-span-2">
          <span className="label">Swimmer</span>
          <select
            className="field mt-1"
            name="swimmer_id"
            value={selectedSwimmerId}
            onChange={(event) => setSelectedSwimmerId(event.target.value)}
            required
          >
            {board.swimmers.map((card) => (
              <option key={card.swimmer.id} value={card.swimmer.id}>
                {card.swimmer.name} - {card.swimmer.primary_event}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span className="label">Date</span>
          <input className="field mt-1" type="date" name="session_date" defaultValue={new Date().toISOString().slice(0, 10)} />
        </label>
        <label>
          <span className="label">Type</span>
          <select
            className="field mt-1"
            name="session_type"
            value={sessionType}
            onChange={(event) => setSessionType(event.target.value)}
          >
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
        <label className="md:col-span-2">
          <span className="label">Notes</span>
          <textarea className="field mt-1 min-h-20" name="notes" />
        </label>
        <label className="flex items-center gap-2 pt-6 text-sm font-semibold text-slate-700 md:col-span-2">
          <input type="checkbox" name="create_mental_checkin" defaultChecked className="h-4 w-4" />
          Save these mood scores as a mental check-in
        </label>

        {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 md:col-span-4">{error}</p> : null}
        {result ? (
          <div className="rounded-md bg-emerald-50 px-3 py-2 text-sm text-emerald-800 md:col-span-4">
            <div className="flex items-center gap-2 font-semibold">
              <CheckCircle2 size={16} />
              {result.entry_summary}
            </div>
            {result.duplicate_warning ? <p className="mt-1 text-amber-800">{result.duplicate_warning}</p> : null}
          </div>
        ) : null}

        <div className="md:col-span-4">
          <button className="primary-button" type="submit" disabled={saving || board.swimmers.length === 0}>
            <Timer size={16} />
            {saving ? "Saving" : "Save Session"}
          </button>
        </div>
      </form>
    </section>
  );
}

function TodayCoachBrief({ board }: { board: PilotBoard }) {
  const priorities = board.swimmers
    .map((card) => ({ card, priority: dailyPriority(card) }))
    .sort((a, b) => b.priority.score - a.priority.score)
    .slice(0, 5);
  const top = priorities[0];
  const readyCount = board.swimmers.filter((card) => card.status === "ready").length;

  return (
    <section className="panel p-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-black uppercase tracking-wide text-water">Today Board</p>
          <h2 className="mt-1 text-xl font-bold text-ink">Coach priorities</h2>
          <p className="mt-1 max-w-2xl text-sm text-slate-500">
            Start here before practice: who needs attention, what data is missing, and the next action for each swimmer.
          </p>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
          <BriefStat label="Ready" value={readyCount} tone="mint" />
          <BriefStat label="Watch" value={board.totals.at_risk} tone="coral" />
          <BriefStat label="Need video" value={board.totals.needs_video} tone="violet" />
          <BriefStat label="Need check-in" value={board.totals.needs_checkin} tone="water" />
        </div>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-md border border-slate-200 bg-slate-50 p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-black uppercase text-slate-400">Top next action</p>
              <h3 className="mt-2 text-lg font-bold text-ink">{top ? top.card.swimmer.name : "Add pilot swimmers"}</h3>
            </div>
            <StatusPill tone={top?.priority.tone ?? "slate"}>{top?.priority.label ?? "No data"}</StatusPill>
          </div>
          <p className="mt-3 text-sm leading-6 text-slate-600">
            {top ? top.priority.reason : "Create 3-5 swimmer profiles, then log one real session after practice."}
          </p>
          {top ? (
            <Link className="primary-button mt-4" href={`/dashboard/swimmers/${top.card.swimmer.id}`}>
              <ClipboardList size={16} />
              {top.priority.action}
            </Link>
          ) : null}
        </div>

        <div className="overflow-x-auto rounded-md border border-slate-200">
          <table className="min-w-[720px] w-full text-left text-sm">
            <thead className="bg-slate-50 text-xs font-black uppercase text-slate-500">
              <tr>
                <th className="px-3 py-3">Swimmer</th>
                <th className="px-3 py-3">Last work</th>
                <th className="px-3 py-3">Readiness</th>
                <th className="px-3 py-3">Priority</th>
                <th className="px-3 py-3">Open</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {priorities.length ? (
                priorities.map(({ card, priority }) => (
                  <tr key={card.swimmer.id}>
                    <td className="px-3 py-3">
                      <p className="font-bold text-ink">{card.swimmer.name}</p>
                      <p className="text-xs text-slate-500">{card.swimmer.primary_event}</p>
                    </td>
                    <td className="px-3 py-3 text-slate-600">
                      {card.latest_session
                        ? `${card.latest_session.session_type.replace(/_/g, " ")} / ${card.latest_session.distance_m}m`
                        : "No session yet"}
                    </td>
                    <td className="px-3 py-3 font-semibold text-ink">
                      {card.latest_checkin ? `${card.latest_checkin.composite_score}/10` : card.latest_session ? `${card.latest_session.mood_recovery}/10` : "-"}
                    </td>
                    <td className="px-3 py-3">
                      <StatusPill tone={priority.tone}>{priority.label}</StatusPill>
                      <p className="mt-1 text-xs text-slate-500">{priority.action}</p>
                    </td>
                    <td className="px-3 py-3">
                      <Link className="secondary-button px-3 py-1.5" href={`/dashboard/swimmers/${card.swimmer.id}`}>
                        Open
                      </Link>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="px-3 py-6 text-center text-slate-500" colSpan={5}>
                    No swimmers yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function BriefStat({ label, value, tone }: { label: string; value: number; tone: "water" | "mint" | "coral" | "violet" | "slate" }) {
  const toneClasses = {
    water: "border-blue-100 bg-blue-50 text-water",
    mint: "border-emerald-100 bg-emerald-50 text-mint",
    coral: "border-orange-100 bg-orange-50 text-coral",
    violet: "border-violet-100 bg-violet-50 text-violet",
    slate: "border-slate-200 bg-slate-50 text-slate-700"
  };
  return (
    <div className={`rounded-md border px-3 py-2 ${toneClasses[tone]}`}>
      <p className="text-[10px] font-black uppercase">{label}</p>
      <p className="mt-1 text-xl font-black">{value}</p>
    </div>
  );
}

function dailyPriority(card: PilotBoardCard): {
  label: string;
  action: string;
  reason: string;
  tone: "water" | "mint" | "coral" | "violet" | "slate";
  score: number;
} {
  const session = card.latest_session;
  if (!session) {
    return {
      label: "Need session",
      action: "Log first session",
      reason: `${card.swimmer.name} has no training record yet. Start with one normal practice so AquaIQ can calculate load and readiness.`,
      tone: "water",
      score: 100
    };
  }

  if (card.status === "watch" || session.rpe >= 8 || session.mood_recovery <= 5) {
    return {
      label: "Watch",
      action: "Review load",
      reason: `${card.swimmer.name} logged RPE ${session.rpe} and recovery ${session.mood_recovery}/10. Keep the next practice controlled before another hard set.`,
      tone: "coral",
      score: 90
    };
  }

  if (!card.latest_checkin) {
    return {
      label: "Need check-in",
      action: "Add mental scores",
      reason: `${card.swimmer.name} has training data but no latest mental check-in. Capture focus, confidence, energy, calm, recovery, and motivation.`,
      tone: "water",
      score: 80
    };
  }

  if (!card.latest_report) {
    return {
      label: "Need video",
      action: "Upload analysis video",
      reason: `${card.swimmer.name} has recent work logged. Add a technique video so the next recommendation has stroke evidence, not only training load.`,
      tone: "violet",
      score: 70
    };
  }

  return {
    label: "Ready",
    action: "Open plan",
    reason: `${card.swimmer.name} has session, mental, and video data. Use the swimmer page to follow the current plan and review the latest report.`,
    tone: "mint",
    score: 50
  };
}

function PilotSwimmerCard({ card, onDeleted }: { card: PilotBoardCard; onDeleted: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [deleting, setDeleting] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const latestSession = card.latest_session;
  const latestReport = card.latest_report;
  const latestCheckin = card.latest_checkin;
  const tone = card.status === "watch" ? "coral" : card.status === "ready" ? "mint" : "water";

  async function deleteSwimmer() {
    if (!token) return;
    setDeleting(true);
    try {
      await api.deleteSwimmer(token, card.swimmer.id);
      setConfirmOpen(false);
      onDeleted();
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div className="rounded-md border border-slate-200 p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <Link href={`/dashboard/swimmers/${card.swimmer.id}`} className="font-bold text-ink hover:text-water">
            {card.swimmer.name}
          </Link>
          <p className="mt-1 text-sm text-slate-500">{card.swimmer.primary_event}</p>
        </div>
        <div className="flex items-center gap-2">
          <StatusPill tone={tone}>{card.status}</StatusPill>
          <button
            className="icon-button h-8 w-8 text-red-600 hover:border-red-300 hover:text-red-700"
            type="button"
            title="Delete swimmer"
            aria-label={`Delete ${card.swimmer.name}`}
            disabled={deleting}
            onClick={() => setConfirmOpen(true)}
          >
            <Trash2 size={15} />
          </button>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div>
          <dt className="text-slate-500">RPE</dt>
          <dd className="font-semibold text-ink">{latestSession ? latestSession.rpe : "-"}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Recovery</dt>
          <dd className="font-semibold text-ink">{latestSession ? latestSession.mood_recovery : "-"}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Load</dt>
          <dd className="font-semibold text-ink">{latestSession ? latestSession.load_score : "-"}</dd>
        </div>
      </dl>

      <div className="mt-4 space-y-2 text-sm text-slate-600">
        <div className="grid gap-2 rounded-md bg-slate-50 p-3 sm:grid-cols-3">
          <div>
            <p className="text-xs font-bold uppercase text-slate-400">Last came</p>
            <p className="mt-1 font-semibold text-ink">{latestSession ? formatDateLabel(latestSession.session_date) : "-"}</p>
          </div>
          <div>
            <p className="text-xs font-bold uppercase text-slate-400">Latest exercise</p>
            <p className="mt-1 font-semibold capitalize text-ink">{latestSession ? latestSession.session_type.replace(/_/g, " ") : "-"}</p>
          </div>
          <div>
            <p className="text-xs font-bold uppercase text-slate-400">Next visit</p>
            <p className="mt-1 font-semibold text-ink">{latestSession ? suggestNextVisit(latestSession) : "After first log"}</p>
          </div>
        </div>
        <p>{latestSession ? `${latestSession.distance_m}m / ${latestSession.duration_min}min / load ${latestSession.load_score}` : "No session logged yet."}</p>
        <div className="flex flex-wrap gap-2">
          <StatusPill tone={latestSession?.video_url ? "mint" : "slate"}>
            {latestSession?.video_url ? "Video attached" : "Video needed"}
          </StatusPill>
          <StatusPill tone={latestReport ? "violet" : "slate"}>
            {latestReport ? `${reportProviderName(latestReport)} ${latestReport.overall_score}/100` : "Analysis pending"}
          </StatusPill>
          <StatusPill tone={latestCheckin ? "mint" : "slate"}>
            {latestCheckin ? `Mind ${latestCheckin.composite_score}` : "Check-in needed"}
          </StatusPill>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <Link className="secondary-button px-3 py-1.5" href={`/dashboard/swimmers/${card.swimmer.id}`}>
          <Upload size={15} />
          {card.next_action}
        </Link>
        <Link className="secondary-button px-3 py-1.5" href={`/dashboard/swimmers/${card.swimmer.id}`}>
          <FileVideo size={15} />
          Video/Race
        </Link>
      </div>
      <DeleteConfirmDialog
        open={confirmOpen}
        title="Delete swimmer?"
        itemName={card.swimmer.name}
        description="This will permanently remove this swimmer from the pilot board and erase all data connected to their profile."
        details={["Swimmer profile and personal bests", "Sessions and video links", "Technique reports, plans, races, and mental check-ins"]}
        confirmLabel="Delete swimmer"
        isDeleting={deleting}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={deleteSwimmer}
      />
    </div>
  );
}

function reportProviderName(report: PilotBoardCard["latest_report"]) {
  return String(report?.keypoint_data.provider ?? "mock") === "mediapipe" ? "MediaPipe" : "Mock";
}

function formatDateLabel(dateString: string) {
  return new Date(`${dateString}T12:00:00`).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric"
  });
}

function suggestNextVisit(session: PilotBoardCard["latest_session"]) {
  if (!session) return "After first log";
  const date = new Date(`${session.session_date}T12:00:00`);
  const daysToAdd = session.rpe >= 8 || session.mood_recovery <= 5 ? 2 : 1;
  date.setDate(date.getDate() + daysToAdd);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function EmptyPilotState({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="rounded-md border border-dashed border-slate-300 p-8 text-center">
      <Users className="mx-auto text-water" size={28} />
      <h3 className="mt-3 text-lg font-bold text-ink">Add your first pilot swimmer</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-slate-500">
        Start with 3-5 real swimmers, then log one session and one check-in after each practice.
      </p>
      <button className="primary-button mt-4" type="button" onClick={onCreate}>
        <Plus size={16} />
        Add Swimmer
      </button>
    </div>
  );
}

function GapTile({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-slate-200 bg-slate-50 p-3">
      <p className="text-xs font-semibold uppercase text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-bold text-ink">{value}</p>
    </div>
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

function CreateSwimmerPanel({ onCreated }: { onCreated: () => void }) {
  const token = useAuthStore((state) => state.token);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!token) return;
    const form = new FormData(event.currentTarget);
    setSaving(true);
    setError(null);
    try {
      await api.createSwimmer(token, {
        name: form.get("name"),
        date_of_birth: form.get("date_of_birth") || null,
        primary_stroke: form.get("primary_stroke"),
        primary_event: form.get("primary_event"),
        level: form.get("level"),
        personal_bests: {
          [String(form.get("primary_event"))]: Number(form.get("personal_best_seconds"))
        }
      });
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create swimmer");
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel mb-6 p-5">
      <h2 className="text-lg font-bold text-ink">New Swimmer</h2>
      <form onSubmit={onSubmit} className="mt-4 grid gap-4 md:grid-cols-3">
        <label>
          <span className="label">Name</span>
          <input className="field mt-1" name="name" required />
        </label>
        <label>
          <span className="label">Birth Date</span>
          <input className="field mt-1" name="date_of_birth" type="date" />
        </label>
        <label>
          <span className="label">Level</span>
          <select className="field mt-1" name="level" defaultValue="age_group">
            <option value="junior">Junior</option>
            <option value="age_group">Age Group</option>
            <option value="elite">Elite</option>
            <option value="masters">Masters</option>
          </select>
        </label>
        <label>
          <span className="label">Stroke</span>
          <select className="field mt-1" name="primary_stroke" defaultValue="freestyle">
            <option value="freestyle">Freestyle</option>
            <option value="backstroke">Backstroke</option>
            <option value="breaststroke">Breaststroke</option>
            <option value="butterfly">Butterfly</option>
            <option value="IM">IM</option>
          </select>
        </label>
        <label>
          <span className="label">Primary Event</span>
          <input className="field mt-1" name="primary_event" defaultValue="100m freestyle" required />
        </label>
        <label>
          <span className="label">PB Seconds</span>
          <input className="field mt-1" name="personal_best_seconds" type="number" step="0.01" defaultValue="60.00" />
        </label>
        {error ? <p className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 md:col-span-3">{error}</p> : null}
        <div className="md:col-span-3">
          <button className="primary-button" type="submit" disabled={saving}>
            <Plus size={16} />
            {saving ? "Creating" : "Create"}
          </button>
        </div>
      </form>
    </section>
  );
}

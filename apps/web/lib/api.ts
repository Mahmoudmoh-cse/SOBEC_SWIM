import type {
  AIOutput,
  DashboardOverview,
  MentalCheckin,
  PilotBoard,
  PilotSessionEntryResponse,
  RaceAnalysis,
  SwimSession,
  SwimAnalysisMetric,
  SwimAnalysisReport,
  SwimAnalysisStatus,
  Swimmer,
  TechniqueReport,
  Token,
  TrainingPlan,
  User
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";
const API_ORIGIN = API_BASE.replace(/\/api\/v1\/?$/, "");

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type RequestOptions = {
  token?: string | null;
  body?: unknown;
  method?: string;
};

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: HeadersInit = {};
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method: options.method ?? "GET",
    headers,
    body: options.body instanceof FormData ? options.body : options.body ? JSON.stringify(options.body) : undefined,
    cache: "no-store"
  });

  if (!response.ok) {
    let detail = "Request failed";
    try {
      const payload = await response.json();
      detail = payload.detail ?? detail;
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail, response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

export const api = {
  videoUrl: (storedPath: string | null) => {
    if (!storedPath) return null;
    const normalized = storedPath.replace(/\\/g, "/");
    const uploadIndex = normalized.indexOf("uploads/");
    const relativePath = uploadIndex >= 0 ? normalized.slice(uploadIndex) : normalized.replace(/^\/+/, "");
    return `${API_ORIGIN}/${relativePath}`;
  },
  login: (email: string, password: string) =>
    request<Token>("/auth/login", {
      method: "POST",
      body: { email, password }
    }),
  me: (token: string) => request<User>("/auth/me", { token }),
  dashboard: (token: string) => request<DashboardOverview>("/dashboard/overview", { token }),
  pilotBoard: (token: string) => request<PilotBoard>("/pilot/board", { token }),
  createPilotSessionEntry: (token: string, body: Record<string, unknown>) =>
    request<PilotSessionEntryResponse>("/pilot/session-entry", { token, method: "POST", body }),
  swimmers: (token: string) => request<Swimmer[]>("/swimmers", { token }),
  swimmer: (token: string, swimmerId: string) => request<Swimmer>(`/swimmers/${swimmerId}`, { token }),
  createSwimmer: (token: string, body: Record<string, unknown>) =>
    request<Swimmer>("/swimmers", { token, method: "POST", body }),
  updateSwimmer: (token: string, swimmerId: string, body: Record<string, unknown>) =>
    request<Swimmer>(`/swimmers/${swimmerId}`, { token, method: "PATCH", body }),
  deleteSwimmer: (token: string, swimmerId: string) =>
    request<void>(`/swimmers/${swimmerId}`, { token, method: "DELETE" }),
  sessions: (token: string, swimmerId?: string) =>
    request<SwimSession[]>(`/sessions${swimmerId ? `?swimmer_id=${swimmerId}` : ""}`, { token }),
  createSession: (token: string, body: Record<string, unknown>) =>
    request<SwimSession>("/sessions", { token, method: "POST", body }),
  deleteSession: (token: string, sessionId: string) =>
    request<void>(`/sessions/${sessionId}`, { token, method: "DELETE" }),
  uploadVideo: async (token: string, sessionId: string, file: File) => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/video`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": file.type || "application/octet-stream",
        "X-Filename": encodeURIComponent(file.name || "video.mp4")
      },
      body: file,
      cache: "no-store"
    });

    if (!response.ok) {
      let detail = "Video upload failed";
      try {
        const payload = await response.json();
        detail = payload.detail ?? detail;
      } catch {
        detail = response.statusText;
      }
      throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
    }

    return (await response.json()) as { report: TechniqueReport };
  },
  techniqueReports: (token: string, swimmerId: string) =>
    request<TechniqueReport[]>(`/technique-reports?swimmer_id=${swimmerId}`, { token }),
  activePlan: (token: string, swimmerId: string) =>
    request<TrainingPlan>(`/training-plans/active/${swimmerId}`, { token }),
  generatePlan: (token: string, body: Record<string, unknown>) =>
    request<TrainingPlan>("/training-plans/generate", { token, method: "POST", body }),
  raceAnalyses: (token: string, swimmerId: string) =>
    request<RaceAnalysis[]>(`/race-analyses?swimmer_id=${swimmerId}`, { token }),
  createRaceAnalysis: (token: string, body: Record<string, unknown>) =>
    request<RaceAnalysis>("/race-analyses", { token, method: "POST", body }),
  mentalCheckins: (token: string, swimmerId: string) =>
    request<MentalCheckin[]>(`/mental-checkins?swimmer_id=${swimmerId}`, { token }),
  createMentalCheckin: (token: string, body: Record<string, unknown>) =>
    request<MentalCheckin>("/mental-checkins", { token, method: "POST", body }),
  aiOutputs: (token: string, swimmerId: string) =>
    request<AIOutput[]>(`/ai-outputs?swimmer_id=${swimmerId}`, { token }),
  backfillAIOutputs: (token: string, swimmerId: string) =>
    request<AIOutput[]>(`/ai-outputs/backfill/${swimmerId}`, { token, method: "POST" }),
  uploadSwimAnalysis: async (token: string, formData: FormData) => {
    const response = await fetch(`${API_ORIGIN}/api/swim-analysis/upload`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`
      },
      body: formData,
      cache: "no-store"
    });
    if (!response.ok) {
      let detail = "Swim analysis upload failed";
      try {
        const payload = await response.json();
        detail = payload.detail ?? detail;
      } catch {
        detail = response.statusText;
      }
      throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
    }
    return (await response.json()) as { analysis_id: string; status: string };
  },
  swimAnalysisStatus: async (token: string, analysisId: string) => {
    const response = await fetch(`${API_ORIGIN}/api/swim-analysis/${analysisId}`, {
      headers: {
        Authorization: `Bearer ${token}`
      },
      cache: "no-store"
    });
    if (!response.ok) {
      let detail = "Unable to load swim analysis status";
      try {
        const payload = await response.json();
        detail = payload.detail ?? detail;
      } catch {
        detail = response.statusText;
      }
      throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
    }
    return (await response.json()) as SwimAnalysisStatus;
  },
  swimAnalysisReport: async (token: string, analysisId: string) => {
    const response = await fetch(`${API_ORIGIN}/api/swim-analysis/${analysisId}/report`, {
      headers: {
        Authorization: `Bearer ${token}`
      },
      cache: "no-store"
    });
    if (!response.ok) {
      let detail = "Swim analysis report is not ready";
      try {
        const payload = await response.json();
        detail = payload.detail ?? detail;
      } catch {
        detail = response.statusText;
      }
      throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
    }
    return (await response.json()) as SwimAnalysisReport;
  },
  swimAnalysisMetrics: async (token: string, analysisId: string) => {
    const response = await fetch(`${API_ORIGIN}/api/swim-analysis/${analysisId}/metrics`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store"
    });
    if (!response.ok) throw new Error("Unable to load swim analysis metrics");
    return (await response.json()) as { analysis_id: string; status: string; metrics: Record<string, SwimAnalysisMetric> };
  },
  swimAnalysisVelocity: async (token: string, analysisId: string) => {
    const response = await fetch(`${API_ORIGIN}/api/swim-analysis/${analysisId}/velocity`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store"
    });
    if (!response.ok) throw new Error("Unable to load swim analysis velocity");
    return (await response.json()) as Pick<SwimAnalysisReport, "velocity" | "phase_segments" | "calibration"> & { analysis_id: string; status: string };
  },
  swimAnalysisAnnotatedVideo: async (token: string, analysisId: string) => {
    const response = await fetch(`${API_ORIGIN}/api/swim-analysis/${analysisId}/annotated-video`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store"
    });
    if (!response.ok) throw new Error("Unable to load annotated video URLs");
    return (await response.json()) as { analysis_id: string; status: string; videos: Record<string, string | null>; confidence_warning: string | null };
  }
};

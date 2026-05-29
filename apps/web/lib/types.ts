export type User = {
  id: string;
  email: string;
  full_name: string;
  role: string;
  created_at: string;
};

export type Token = {
  access_token: string;
  token_type: string;
  user: User;
};

export type Swimmer = {
  id: string;
  coach_id: string;
  name: string;
  date_of_birth: string | null;
  primary_stroke: string;
  primary_event: string;
  level: string;
  personal_bests: Record<string, number>;
  technique_profile: Record<string, unknown>;
  mental_profile: Record<string, unknown>;
  created_at: string;
};

export type SwimSession = {
  id: string;
  swimmer_id: string;
  session_date: string;
  session_type: string;
  distance_m: number;
  duration_min: number;
  rpe: number;
  load_score: number;
  mood_focus: number;
  mood_confidence: number;
  mood_energy: number;
  mood_calm: number;
  mood_recovery: number;
  mood_motivation: number;
  sleep_hours: number | null;
  hrv_morning: number | null;
  notes: string | null;
  video_url: string | null;
  created_at: string;
};

export type TechniqueReport = {
  id: string;
  session_id: string;
  swimmer_id: string;
  stroke: string;
  overall_score: number;
  dps_meters: number;
  stroke_rate: number;
  faults: Array<Record<string, string | number>>;
  drill_prescriptions: Array<Record<string, string | number>>;
  keypoint_data: Record<string, unknown>;
  processing_status: string;
  analysis_status: "uploaded" | "processing" | "completed" | "failed" | "no_pose_detected" | "low_confidence" | string;
  frames_total: number;
  frames_analyzed: number;
  pose_detected_frames: number;
  pose_detection_rate: number;
  confidence_score: number;
  confidence_label: "high" | "medium" | "low" | string;
  analysis_warning: string | null;
  analysis_error: string | null;
  analysis_overlay_video_url: string | null;
  analysis_frame_urls: string[];
  analysis_events: Array<{
    timestamp_s: number;
    type: string;
    label: string;
    message: string;
  }>;
  coaching_summary: string;
  created_at: string;
};

export type TrainingPlan = {
  id: string;
  swimmer_id: string;
  race_date: string;
  race_event: string;
  target_time_seconds: number;
  weeks_total: number;
  current_phase: string;
  current_week: number;
  phase_config: Record<string, number>;
  weekly_plans: Array<{
    week_number: number;
    phase: string;
    target_load: number;
    focus: string;
    sessions: Array<{ type: string; main_set: string; target: string }>;
  }>;
  adaptation_log: Array<Record<string, string>>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
};

export type RaceAnalysis = {
  id: string;
  swimmer_id: string;
  race_date: string;
  event: string;
  official_time_seconds: number;
  splits_actual: number[];
  splits_predicted: number[];
  strategy_type: string;
  strategy_score: number;
  reaction_time_ms: number | null;
  turn_times: Array<Record<string, string | number>>;
  phase_analysis: Record<string, string | number | null>;
  time_vs_pb_seconds: number | null;
  ai_insights: string[];
  created_at: string;
};

export type MentalCheckin = {
  id: string;
  swimmer_id: string;
  checkin_type: string;
  checkin_date: string;
  mood_focus: number;
  mood_confidence: number;
  mood_energy: number;
  mood_calm: number;
  mood_recovery: number;
  mood_motivation: number;
  composite_score: number;
  routine_generated: Record<string, string>;
  race_result_id: string | null;
  performance_delta: number | null;
  notes: string | null;
};

export type AIOutput = {
  id: string;
  swimmer_id: string;
  session_id: string | null;
  technique_report_id: string | null;
  training_plan_id: string | null;
  race_analysis_id: string | null;
  mental_checkin_id: string | null;
  output_type: string;
  provider: string;
  model: string;
  prompt_version: string;
  parsed_json: {
    summary?: string;
    key_findings?: string[];
    recommendations?: string[];
    risk_flags?: string[];
    next_steps?: string[];
  };
  status: string;
  error: string | null;
  latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  created_at: string;
};

export type DashboardOverview = {
  swimmers_count: number;
  sessions_count: number;
  average_load: number;
  high_risk_count: number;
  swimmers: Swimmer[];
  recent_sessions: SwimSession[];
  active_plans: TrainingPlan[];
  alerts: Array<{
    type: string;
    swimmer_id: string;
    swimmer_name: string;
    message: string;
  }>;
};

export type PilotBoardCard = {
  swimmer: Swimmer;
  latest_session: SwimSession | null;
  latest_report: TechniqueReport | null;
  latest_checkin: MentalCheckin | null;
  latest_race: RaceAnalysis | null;
  missing_flags: string[];
  next_action: string;
  status: "ready" | "needs_data" | "watch" | "incomplete" | string;
};

export type PilotBoard = {
  generated_at: string;
  swimmers: PilotBoardCard[];
  totals: {
    swimmers: number;
    needs_session: number;
    needs_video: number;
    needs_checkin: number;
    at_risk: number;
  };
};

export type PilotSessionEntryResponse = {
  session: SwimSession;
  mental_checkin: MentalCheckin | null;
  duplicate_warning: string | null;
  entry_summary: string;
};

export type SwimAnalysisMetric = {
  value: string | number | null;
  confidence: number | "low" | string;
  evidence: Record<string, unknown>;
  reason: string;
  unit?: string | null;
  interpretation?: string;
  coaching_meaning?: string;
  supporting_frame_count?: number;
  reliability_category?: "reliable" | "estimated" | "insufficient_evidence" | string;
};

export type ResearchFigure = {
  figure_id: string;
  title: string;
  type: "trajectories" | "tracking_quality" | "error_noise" | "stroke_cycles" | string;
  file_path: string;
  description: string;
  metric_source: "ground_truth" | "proxy" | string;
};

export type SwimAnalysisReport = {
  analysis_id: string;
  status: "completed" | "failed" | string;
  summary: {
    stroke_type: string;
    duration_sec: number;
    overall_score: number;
    confidence_score: number;
    reliability_score?: number;
    data_quality_score: number;
    pose_backend?: string;
  };
  video_quality: Record<string, Record<string, unknown>>;
  video_metadata?: Record<string, unknown>;
  metrics: Record<string, SwimAnalysisMetric>;
  calibration?: Record<string, unknown>;
  velocity?: {
    summary?: Record<string, unknown>;
    series?: Array<Record<string, unknown>>;
    dead_spots?: Array<Record<string, unknown>>;
    phase_summary?: Array<Record<string, unknown>>;
    review_splits?: Array<Record<string, unknown>>;
    breathing_velocity_loss?: Array<Record<string, unknown>>;
    chart_ready?: boolean;
  };
  phase_segments?: Array<Record<string, unknown>>;
  faults?: Array<{
    name: string;
    severity: string;
    evidence_metrics: Record<string, unknown>;
    timestamp_range: Array<number | null>;
    confidence: number;
    recommended_drill: string;
    coach_explanation: string;
  }>;
  findings: Array<{
    type: string;
    severity: string;
    message: string;
    confidence: number;
    evidence: Record<string, unknown>;
  }>;
  recommendations: Array<{
    priority: number;
    message: string;
    linked_metric: string;
  }>;
  coaching_report?: Record<string, unknown>;
  artifacts: Record<string, string | null>;
  warnings?: string[];
  pose_diagnostics?: Record<string, unknown>;
  trajectories?: {
    trajectory_data_url?: string | null;
    video_visibility_percentage?: Record<string, number>;
    side_visibility_percentage?: Record<string, number>;
    front_visibility_percentage?: Record<string, number>;
    video_tracking_quality?: Array<Record<string, unknown>>;
    side_tracking_quality?: Array<Record<string, unknown>>;
    front_tracking_quality?: Array<Record<string, unknown>>;
    video_joint_reliability?: Record<string, Record<string, unknown>>;
    side_joint_reliability?: Record<string, Record<string, unknown>>;
    front_joint_reliability?: Record<string, Record<string, unknown>>;
    video_roi_history?: Array<Record<string, unknown>>;
    side_roi_history?: Array<Record<string, unknown>>;
    front_roi_history?: Array<Record<string, unknown>>;
  };
  research_figures?: ResearchFigure[];
  debug_info?: Record<string, unknown>;
  processing_errors?: string[];
};

export type SwimAnalysisStatus = {
  analysis_id?: string;
  id?: string;
  status: "queued" | "processing" | "completed" | "failed" | string;
  progress: number;
  stroke_type: string;
  pose_backend: string;
  swimmer_id: string | null;
  video_metadata: Record<string, unknown>;
  processing_errors: string[];
  error_message: string | null;
  created_at: string;
  updated_at: string;
};

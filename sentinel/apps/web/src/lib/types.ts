export type Severity = "low" | "medium" | "high" | "critical";
export type EventStatus = "new" | "acknowledged" | "resolved" | "dismissed";
export type CameraSourceType = "rtsp" | "onvif" | "file" | "webcam" | "synthetic";
export type ZoneType =
  | "restricted"
  | "hazardous"
  | "loading"
  | "entrance"
  | "emergency"
  | "parking"
  | "waiting"
  | "custom";
export type EvidenceType = "snapshot" | "clip";
export type CameraHealthStatus =
  | "online"
  | "degraded"
  | "retrying"
  | "offline"
  | "disabled"
  | "unknown";

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface WhoAmI {
  user: string;
  roles: string[];
  permissions: string[];
  org_id: string | null;
  auth_mode: string;
  warnings: string[];
}

export interface RetentionPolicy {
  pre_seconds: number;
  post_seconds: number;
  days_by_severity: Record<string, number>;
}

export interface Camera {
  id: string;
  camera_id: string;
  name: string;
  location: string | null;
  site_id: string | null;
  source_type: CameraSourceType;
  enabled: boolean;
  detection_enabled: boolean;
  recording_enabled: boolean;
  detection_fps: number;
  width: number | null;
  height: number | null;
  timezone: string;
  model_profile: string;
  rule_profile: string;
  retention: RetentionPolicy;
  metadata: Record<string, unknown>;
  stream_url_set: boolean;
  deleted_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface CameraCreate {
  camera_id: string;
  name: string;
  stream_url: string;
  location?: string | null;
  source_type?: CameraSourceType;
  enabled?: boolean;
  detection_enabled?: boolean;
  recording_enabled?: boolean;
  detection_fps?: number;
  timezone?: string;
  model_profile?: string;
  rule_profile?: string;
}

export interface CameraUpdate {
  name?: string | null;
  location?: string | null;
  enabled?: boolean | null;
  detection_enabled?: boolean | null;
  recording_enabled?: boolean | null;
  detection_fps?: number | null;
  model_profile?: string | null;
  rule_profile?: string | null;
  stream_url?: string | null;
}

export interface CameraHealthSummary {
  camera_id: string;
  name: string;
  location: string | null;
  site_id: string | null;
  enabled: boolean;
  status: CameraHealthStatus;
  stale: boolean;
  state: string | null;
  health: string | null;
  ai_status: string | null;
  fps: number | null;
  frame_drops: number | null;
  reconnect_count: number | null;
  last_frame_at: string | null;
  error: string | null;
  health_ts: string | null;
}

export interface CameraHealthSnapshot {
  camera_id: string;
  state: string;
  health: string;
  fps: number;
  frame_drops: number;
  latency_ms: number | null;
  reconnect_count: number;
  frames_processed: number;
  last_frame_at: string | null;
  error: string | null;
  ai_status: string;
  ts: string;
  details: Record<string, unknown>;
}

export interface ConditionEvidence {
  name: string;
  operator: string;
  actual: string | number | boolean | null;
  threshold: string | number | boolean | null;
  satisfied: boolean;
}

export interface Event {
  event_id: string;
  camera_id: string;
  timestamp: string;
  event_type: string;
  severity: Severity;
  status: EventStatus;
  confidence: number | null;
  summary: string;
  track_ids: number[];
  zone_id: string | null;
  zone_name: string | null;
  rule_id: string | null;
  rule_name: string | null;
  conditions: ConditionEvidence[];
  model_versions: Record<string, string>;
  evidence_ids: string[];
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  acknowledged_by: string | null;
  acknowledged_at: string | null;
  resolved_at: string | null;
}

export interface EvidenceItem {
  evidence_id: string;
  event_id: string;
  camera_id: string;
  type: EvidenceType;
  uri: string;
  sha256: string | null;
  size_bytes: number;
  content_type: string | null;
  width: number | null;
  height: number | null;
  duration_ms: number | null;
  captured_at: string;
  expires_at: string | null;
  storage_backend: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface AlertOut {
  id: string;
  event_id: string;
  channel: string;
  status: string;
  target: string | null;
  attempts: number;
  error: string | null;
  sent_at: string | null;
  created_at: string;
}

export interface ScheduleWindow {
  days: string[];
  start: string;
  end: string;
}

export interface Schedule {
  windows: ScheduleWindow[];
  timezone: string;
}

export interface Zone {
  id: string;
  camera_id: string;
  name: string;
  zone_type: ZoneType;
  polygon: number[][];
  anchor: string;
  enabled: boolean;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
}

export interface ZoneCreate {
  name: string;
  zone_type: ZoneType;
  polygon: number[][];
  anchor?: string;
  enabled?: boolean;
}

export interface RuleOut {
  id: string;
  rule_id: string;
  name: string;
  rule_type: string;
  event_type: string;
  severity: Severity;
  enabled: boolean;
  zone_ids: string[];
  line: number[][] | null;
  params: Record<string, unknown>;
  schedule: Schedule;
  cooldown_seconds: number;
  confirm_seconds: number;
  min_confidence: number;
  camera_id: string | null;
  site_id: string | null;
  version: string;
  created_at: string;
  updated_at: string;
}

export interface RuleCreate {
  rule_id: string;
  name: string;
  rule_type: string;
  event_type: string;
  severity?: Severity;
  enabled?: boolean;
  zone_ids?: string[];
  line?: number[][] | null;
  params?: Record<string, unknown>;
  schedule?: Schedule;
  cooldown_seconds?: number;
  confirm_seconds?: number;
  min_confidence?: number;
  camera_id?: string | null;
  site_id?: string | null;
}

export interface StreamEnvelope {
  topic: string;
  payload: Record<string, unknown>;
}

export type StreamTopic =
  | "events.created"
  | "events.updated"
  | "camera.health"
  | "alerts.updated";

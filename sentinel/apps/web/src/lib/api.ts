import type {
  AlertOut,
  Camera,
  CameraCreate,
  CameraHealthSnapshot,
  CameraHealthSummary,
  CameraUpdate,
  Event,
  EvidenceItem,
  Page,
  RuleCreate,
  RuleOut,
  StreamEnvelope,
  StreamTopic,
  WhoAmI,
  Zone,
  ZoneCreate,
} from "./types";

export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "/api/v1";

const KEY_STORAGE = "sentinel.apiKey";

let apiKey: string | null = (typeof localStorage !== "undefined"
  ? localStorage.getItem(KEY_STORAGE)
  : null) ?? null;

export function getApiKey(): string | null {
  return apiKey;
}

export function setApiKey(key: string | null): void {
  apiKey = key;
  if (typeof localStorage === "undefined") return;
  if (key) localStorage.setItem(KEY_STORAGE, key);
  else localStorage.removeItem(KEY_STORAGE);
}

export class ApiError extends Error {
  code: string;
  status: number;
  requestId?: string;
  details?: unknown;

  constructor(
    code: string,
    message: string,
    status: number,
    requestId?: string,
    details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.requestId = requestId;
    this.details = details;
  }

  get isUnauthorized(): boolean {
    return this.status === 401;
  }

  get isForbidden(): boolean {
    return this.status === 403;
  }

  get isNotFound(): boolean {
    return this.status === 404;
  }

  get isConflict(): boolean {
    return this.status === 409;
  }
}

type Query = Record<
  string,
  string | number | boolean | null | undefined | (string | number)[]
>;

function buildQuery(query?: Query): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const v of value) params.append(key, String(v));
    } else {
      params.set(key, String(value));
    }
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

export interface RequestOptions {
  method?: string;
  body?: unknown;
  query?: Query;
  signal?: AbortSignal;
  raw?: boolean;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;
  if (options.body !== undefined) headers["Content-Type"] = "application/json";

  const response = await fetch(`${API_BASE}${path}${buildQuery(options.query)}`, {
    method: options.method ?? "GET",
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
    signal: options.signal,
  });

  if (!response.ok) {
    let code = "unknown_error";
    let message = `request failed with status ${response.status}`;
    let requestId: string | undefined = response.headers.get("X-Request-ID") ?? undefined;
    let details: unknown;
    try {
      const payload = await response.json();
      const err = payload?.error;
      if (err && typeof err === "object") {
        code = err.code ?? code;
        message = err.message ?? message;
        requestId = err.request_id ?? requestId;
        details = err.details;
      }
    } catch {
      // non-JSON error body
    }
    throw new ApiError(code, message, response.status, requestId, details);
  }

  if (response.status === 204) return undefined as T;
  if (options.raw) return response as unknown as T;
  return (await response.json()) as T;
}

// -- auth -----------------------------------------------------------------

export function fetchMe(signal?: AbortSignal): Promise<WhoAmI> {
  return request<WhoAmI>("/auth/me", { signal });
}

// -- health ---------------------------------------------------------------

export interface SystemHealth {
  status: "healthy" | "degraded" | "error";
  checks: Record<string, { status: string; detail?: string; dialect?: string; backend?: string }>;
  app: { name: string; version: string; env: string; uptime_seconds: number };
  ts: string;
}

export async function fetchSystemHealth(signal?: AbortSignal): Promise<SystemHealth> {
  // degraded/error responses are 503 but still carry the full body
  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;
  const response = await fetch(`${API_BASE}/system/health`, { headers, signal });
  if (!response.ok && response.status !== 503) {
    throw new ApiError(
      `health_error_${response.status}`,
      `health check failed with status ${response.status}`,
      response.status,
    );
  }
  return (await response.json()) as SystemHealth;
}

// -- cameras --------------------------------------------------------------

export interface CameraListParams {
  limit?: number;
  offset?: number;
  enabled?: boolean;
  include_deleted?: boolean;
}

export function fetchCameras(
  params: CameraListParams = {},
  signal?: AbortSignal,
): Promise<Page<Camera>> {
  return request<Page<Camera>>("/cameras", { query: { ...params }, signal });
}

export function fetchCamera(cameraId: string, signal?: AbortSignal): Promise<Camera> {
  return request<Camera>(`/cameras/${encodeURIComponent(cameraId)}`, { signal });
}

export function createCamera(body: CameraCreate): Promise<Camera> {
  return request<Camera>("/cameras", { method: "POST", body });
}

export function updateCamera(
  cameraId: string,
  body: CameraUpdate,
): Promise<Camera> {
  return request<Camera>(`/cameras/${encodeURIComponent(cameraId)}`, {
    method: "PATCH",
    body,
  });
}

export function deleteCamera(cameraId: string): Promise<void> {
  return request<void>(`/cameras/${encodeURIComponent(cameraId)}`, {
    method: "DELETE",
  });
}

export function fetchHealthSummary(signal?: AbortSignal): Promise<CameraHealthSummary[]> {
  return request<CameraHealthSummary[]>("/cameras/health/summary", { signal });
}

export function fetchCameraHealth(
  cameraId: string,
  limit = 50,
  signal?: AbortSignal,
): Promise<CameraHealthSnapshot[]> {
  return request<CameraHealthSnapshot[]>(
    `/cameras/${encodeURIComponent(cameraId)}/health`,
    { query: { limit }, signal },
  );
}

export interface PreviewFrame {
  blob: Blob;
  capturedAt: string | null;
}

export async function fetchPreviewFrame(
  cameraId: string,
  signal?: AbortSignal,
): Promise<PreviewFrame> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;
  const response = await fetch(
    `${API_BASE}/cameras/${encodeURIComponent(cameraId)}/preview.jpg`,
    { headers, signal },
  );
  if (!response.ok) {
    let code = "preview_unavailable";
    let message = "preview unavailable";
    try {
      const payload = await response.json();
      code = payload?.error?.code ?? code;
      message = payload?.error?.message ?? message;
    } catch {
      // keep defaults
    }
    throw new ApiError(code, message, response.status);
  }
  return {
    blob: await response.blob(),
    capturedAt: response.headers.get("X-Frame-Timestamp"),
  };
}

// -- events ---------------------------------------------------------------

export interface EventListParams {
  camera_id?: string;
  status?: string;
  severity?: string;
  event_type?: string;
  rule_id?: string;
  since?: string;
  until?: string;
  limit?: number;
  offset?: number;
}

export function fetchEvents(
  params: EventListParams = {},
  signal?: AbortSignal,
): Promise<Page<Event>> {
  return request<Page<Event>>("/events", { query: { ...params }, signal });
}

export function fetchEvent(eventId: string, signal?: AbortSignal): Promise<Event> {
  return request<Event>(`/events/${encodeURIComponent(eventId)}`, { signal });
}

export function updateEventStatus(
  eventId: string,
  status: string,
  note?: string,
): Promise<Event> {
  return request<Event>(`/events/${encodeURIComponent(eventId)}/status`, {
    method: "POST",
    body: { status, note: note ?? null },
  });
}

// -- evidence -------------------------------------------------------------

export interface EvidenceListParams {
  event_id?: string;
  camera_id?: string;
  limit?: number;
  offset?: number;
}

export function fetchEvidence(
  params: EvidenceListParams = {},
  signal?: AbortSignal,
): Promise<Page<EvidenceItem>> {
  return request<Page<EvidenceItem>>("/evidence", { query: { ...params }, signal });
}

export async function downloadEvidence(
  evidenceId: string,
): Promise<{ blob: Blob; filename: string }> {
  const headers: Record<string, string> = {};
  if (apiKey) headers["X-API-Key"] = apiKey;
  const response = await fetch(
    `${API_BASE}/evidence/${encodeURIComponent(evidenceId)}/download`,
    { headers },
  );
  if (!response.ok) {
    let code = "unknown_error";
    let message = `download failed with status ${response.status}`;
    try {
      const payload = await response.json();
      code = payload?.error?.code ?? code;
      message = payload?.error?.message ?? message;
    } catch {
      // keep defaults
    }
    throw new ApiError(code, message, response.status);
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/.exec(disposition);
  return {
    blob: await response.blob(),
    filename: match ? match[1] : evidenceId,
  };
}

// -- alerts ---------------------------------------------------------------

export interface AlertListParams {
  event_id?: string;
  channel?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export function fetchAlerts(
  params: AlertListParams = {},
  signal?: AbortSignal,
): Promise<Page<AlertOut>> {
  return request<Page<AlertOut>>("/alerts", { query: { ...params }, signal });
}

// -- zones ----------------------------------------------------------------

export interface ZoneListParams {
  camera_id?: string;
  enabled?: boolean;
  limit?: number;
  offset?: number;
}

export function fetchZones(
  params: ZoneListParams = {},
  signal?: AbortSignal,
): Promise<Page<Zone>> {
  return request<Page<Zone>>("/zones", { query: { ...params }, signal });
}

export function createZone(cameraId: string, body: ZoneCreate): Promise<Zone> {
  return request<Zone>(`/cameras/${encodeURIComponent(cameraId)}/zones`, {
    method: "POST",
    body,
  });
}

export function updateZone(zoneId: string, body: Partial<ZoneCreate>): Promise<Zone> {
  return request<Zone>(`/zones/${encodeURIComponent(zoneId)}`, {
    method: "PATCH",
    body,
  });
}

export function deleteZone(zoneId: string): Promise<void> {
  return request<void>(`/zones/${encodeURIComponent(zoneId)}`, { method: "DELETE" });
}

// -- rules ----------------------------------------------------------------

export interface RuleListParams {
  camera_id?: string;
  enabled?: boolean;
  rule_type?: string;
  limit?: number;
  offset?: number;
}

export function fetchRules(
  params: RuleListParams = {},
  signal?: AbortSignal,
): Promise<Page<RuleOut>> {
  return request<Page<RuleOut>>("/rules", { query: { ...params }, signal });
}

export function createRule(body: RuleCreate): Promise<RuleOut> {
  return request<RuleOut>("/rules", { method: "POST", body });
}

export function updateRule(
  ruleId: string,
  body: Partial<RuleCreate>,
): Promise<RuleOut> {
  return request<RuleOut>(`/rules/${encodeURIComponent(ruleId)}`, {
    method: "PATCH",
    body,
  });
}

export function deleteRule(ruleId: string): Promise<void> {
  return request<void>(`/rules/${encodeURIComponent(ruleId)}`, { method: "DELETE" });
}

// -- SSE (fetch-based so X-API-Key works) ---------------------------------

export interface StreamHandlers {
  onMessage: (envelope: StreamEnvelope) => void;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: unknown) => void;
}

export type StreamTopicFilter = StreamTopic[];

/**
 * Opens the event stream with fetch (EventSource cannot send headers).
 * Returns a close function. The caller owns fallback timers.
 */
export function openEventStream(
  handlers: StreamHandlers,
  topics?: StreamTopicFilter,
): () => void {
  const controller = new AbortController();
  const headers: Record<string, string> = { Accept: "text/event-stream" };
  if (apiKey) headers["X-API-Key"] = apiKey;

  const query = topics && topics.length ? `?topics=${topics.join(",")}` : "";

  (async () => {
    try {
      const response = await fetch(`${API_BASE}/stream${query}`, {
        headers,
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new ApiError(
          `stream_error_${response.status}`,
          `stream failed with status ${response.status}`,
          response.status,
        );
      }
      handlers.onOpen?.();
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const dataLine = block
            .split("\n")
            .find((line) => line.startsWith("data: "));
          if (dataLine) {
            try {
              handlers.onMessage(JSON.parse(dataLine.slice(6)) as StreamEnvelope);
            } catch {
              // ignore malformed frames
            }
          }
          boundary = buffer.indexOf("\n\n");
        }
      }
      handlers.onClose?.();
    } catch (error) {
      if (controller.signal.aborted) {
        handlers.onClose?.();
      } else {
        handlers.onError?.(error);
      }
    }
  })();

  return () => controller.abort();
}

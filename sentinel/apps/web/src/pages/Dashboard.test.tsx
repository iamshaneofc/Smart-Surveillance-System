import { screen } from "@testing-library/react";
import Dashboard from "./Dashboard";
import {
  ADMIN,
  errorResponse,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

const HEALTH = {
  status: "healthy",
  checks: {
    database: { status: "healthy", dialect: "sqlite" },
    event_bus: { status: "healthy", backend: "memory" },
  },
  app: { name: "SENTINEL", version: "0.1.0", env: "test", uptime_seconds: 42 },
  ts: "2026-10-03T12:00:00Z",
};

const SUMMARY = [
  {
    camera_id: "cam1",
    name: "Lobby",
    location: "Entrance",
    site_id: null,
    enabled: true,
    status: "online",
    stale: false,
    state: "running",
    health: "healthy",
    ai_status: "enabled",
    fps: 5.0,
    frame_drops: 0,
    reconnect_count: 0,
    last_frame_at: "2026-10-03T12:00:00Z",
    error: null,
    health_ts: "2026-10-03T12:00:00Z",
  },
  {
    camera_id: "cam2",
    name: "Yard",
    location: null,
    site_id: null,
    enabled: true,
    status: "offline",
    stale: true,
    state: "stopped",
    health: "error",
    ai_status: "disabled",
    fps: 0.0,
    frame_drops: 12,
    reconnect_count: 3,
    last_frame_at: null,
    error: "connection refused",
    health_ts: "2026-10-03T11:00:00Z",
  },
];

const EVENT = {
  event_id: "evt_9",
  camera_id: "cam1",
  timestamp: "2026-10-03T11:59:00Z",
  event_type: "zone_enter",
  severity: "medium",
  status: "new",
  confidence: 0.8,
  summary: "Person crossed into dock zone",
  track_ids: [1],
  zone_id: "z1",
  zone_name: "Dock",
  rule_id: "r1",
  rule_name: "Dock entry",
  conditions: [],
  model_versions: {},
  evidence_ids: [],
  metadata: {},
  created_at: "2026-10-03T11:59:00Z",
  updated_at: "2026-10-03T11:59:00Z",
  acknowledged_by: null,
  acknowledged_at: null,
  resolved_at: null,
};

afterEach(() => {
  vi.restoreAllMocks();
});

function setup(routes: Parameters<typeof mockFetch>[0]) {
  return mockFetch([["/auth/me", jsonResponse(ADMIN)], ...routes]);
}

test("shows system summary, camera health and recent events", async () => {
  setup([
    ["/system/health", jsonResponse(HEALTH)],
    ["/cameras/health/summary", jsonResponse(SUMMARY)],
    ["/events", jsonResponse({ items: [EVENT], total: 1, limit: 20, offset: 0 })],
  ]);

  renderPage(<Dashboard />);

  expect(
    await screen.findByRole("heading", { name: "Command Center" }),
  ).toBeInTheDocument();

  // System stat card
  expect(screen.getByText("healthy")).toBeInTheDocument();
  expect(screen.getByText(/v0\.1\.0 · test · up 42s/)).toBeInTheDocument();

  // Camera stat + health rows
  expect(screen.getByText("1 / 2")).toBeInTheDocument();
  expect(screen.getByText("1 need attention")).toBeInTheDocument();
  expect(await screen.findByTestId("status-cam1")).toHaveTextContent("online");
  expect(screen.getByTestId("status-cam2")).toHaveTextContent("offline · stale");

  // Recent events
  expect(await screen.findByText("Person crossed into dock zone")).toBeInTheDocument();
});

test("surfaces system health errors without hiding the rest", async () => {
  setup([
    ["/system/health", errorResponse("health_error", "health endpoint down", 500)],
    ["/cameras/health/summary", jsonResponse([])],
    ["/events", jsonResponse({ items: [], total: 0, limit: 20, offset: 0 })],
  ]);

  renderPage(<Dashboard />);

  const panel = await screen.findByTestId("error-panel");
  expect(panel).toHaveTextContent("health check failed with status 500");
  expect(
    await screen.findByText("No cameras yet"),
  ).toBeInTheDocument();
});

test("renders degraded 503 health bodies honestly", async () => {
  setup([
    [
      "/system/health",
      jsonResponse(
        {
          status: "degraded",
          checks: {
            database: { status: "error", detail: "connection refused" },
          },
          app: {
            name: "SENTINEL",
            version: "0.1.0",
            env: "test",
            uptime_seconds: 9,
          },
          ts: "2026-10-03T12:00:00Z",
        },
        503,
      ),
    ],
    ["/cameras/health/summary", jsonResponse([])],
    ["/events", jsonResponse({ items: [], total: 0, limit: 20, offset: 0 })],
  ]);

  renderPage(<Dashboard />);

  expect(await screen.findByText("degraded")).toBeInTheDocument();
  expect(screen.queryByTestId("error-panel")).not.toBeInTheDocument();
});

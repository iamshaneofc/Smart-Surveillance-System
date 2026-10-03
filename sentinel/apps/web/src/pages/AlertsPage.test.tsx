import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import AlertsPage from "./AlertsPage";
import {
  OPERATOR,
  errorResponse,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

const ALERT = {
  id: "alr_1",
  event_id: "evt_7",
  channel: "in_app",
  status: "sent",
  target: null,
  attempts: 1,
  error: null,
  sent_at: "2026-10-03T10:00:05Z",
  created_at: "2026-10-03T10:00:01Z",
};

const EVENT_TITLE = {
  event_id: "evt_7",
  camera_id: "cam1",
  timestamp: "2026-10-03T10:00:00Z",
  event_type: "zone_enter",
  severity: "high",
  status: "new",
  confidence: 0.9,
  summary: "Forklift entered dock lane",
  track_ids: [],
  zone_id: null,
  zone_name: null,
  rule_id: null,
  rule_name: null,
  conditions: [],
  model_versions: {},
  evidence_ids: [],
  metadata: {},
  created_at: "2026-10-03T10:00:00Z",
  updated_at: "2026-10-03T10:00:00Z",
  acknowledged_by: null,
  acknowledged_at: null,
  resolved_at: null,
};

afterEach(() => {
  vi.restoreAllMocks();
});

test("renders deliveries with the resolved event title", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    [
      "/alerts",
      jsonResponse({ items: [ALERT], total: 1, limit: 25, offset: 0 }),
    ],
    [
      "/events",
      jsonResponse({ items: [EVENT_TITLE], total: 1, limit: 100, offset: 0 }),
    ],
  ]);

  renderPage(<AlertsPage />);

  expect(await screen.findByTestId("alerts-table")).toBeInTheDocument();
  const row = screen.getByTestId("alert-alr_1");
  expect(row).toHaveTextContent("Forklift entered dock lane");
  expect(row).toHaveTextContent("in_app");
  expect(row).toHaveTextContent("sent");
  expect(row).toHaveTextContent("1");
});

test("permission errors render the error panel", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    [
      "/alerts",
      errorResponse("forbidden", "missing permission: alerts:read", 403),
    ],
    ["/events", jsonResponse({ items: [], total: 0, limit: 100, offset: 0 })],
  ]);

  renderPage(<AlertsPage />);

  const panel = await screen.findByTestId("error-panel");
  expect(panel).toHaveTextContent("missing permission: alerts:read");
  expect(screen.queryByTestId("alerts-table")).not.toBeInTheDocument();
});

test("status filter is applied through the URL", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    [
      "/alerts",
      jsonResponse({ items: [ALERT], total: 1, limit: 25, offset: 0 }),
    ],
    ["/events", jsonResponse({ items: [EVENT_TITLE], total: 1, limit: 100, offset: 0 })],
  ]);

  renderPage(<AlertsPage />);
  await screen.findByTestId("alerts-table");

  await userEvent.selectOptions(screen.getByTestId("alert-status"), "failed");
  await userEvent.click(screen.getByRole("button", { name: "Apply" }));

  await waitFor(() => {
    const calls = fetchSpy.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.includes("/alerts?") && url.includes("status=failed"));
    expect(calls.length).toBeGreaterThan(0);
  });
});

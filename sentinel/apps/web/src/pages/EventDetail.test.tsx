import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EventDetail from "./EventDetail";
import {
  OPERATOR,
  VIEWER,
  errorResponse,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

const CONDITION = {
  name: "zone",
  operator: "in",
  actual: "Restricted",
  threshold: "z1",
  satisfied: true,
};

const UNMET = {
  name: "dwell_seconds",
  operator: ">=",
  actual: 3,
  threshold: 10,
  satisfied: false,
};

function makeEvent(overrides: Record<string, unknown> = {}) {
  return {
    event_id: "evt_42",
    camera_id: "cam1",
    timestamp: "2026-10-03T10:00:00Z",
    event_type: "restricted_zone_intrusion",
    severity: "critical",
    status: "new",
    confidence: 0.91,
    summary: "Intrusion detected at north gate",
    track_ids: [9],
    zone_id: "z1",
    zone_name: "Restricted",
    rule_id: "restricted-zone-entry",
    rule_name: "Restricted zone entry",
    conditions: [CONDITION, UNMET],
    model_versions: { detector: "opencv-hog-people:4.14.0-dev1" },
    evidence_ids: ["evid1"],
    metadata: {},
    created_at: "2026-10-03T10:00:00Z",
    updated_at: "2026-10-03T10:00:00Z",
    acknowledged_by: null,
    acknowledged_at: null,
    resolved_at: null,
    ...overrides,
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

test("explains the event with stored ConditionEvidence", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/events/evt_42", jsonResponse(makeEvent())],
    ["/evidence", jsonResponse({ items: [], total: 0, limit: 50, offset: 0 })],
  ]);

  renderPage(<EventDetail />, { route: "/events/evt_42", path: "/events/:eventId" });

  expect(await screen.findByTestId("conditions-table")).toBeInTheDocument();
  const satisfied = screen.getByTestId("condition-0");
  expect(satisfied).toHaveTextContent("zone");
  expect(satisfied).toHaveTextContent("Restricted");
  expect(satisfied).toHaveTextContent("satisfied");

  const unmet = screen.getByTestId("condition-1");
  expect(unmet).toHaveTextContent("dwell_seconds");
  expect(unmet).toHaveTextContent("not met");

  expect(
    screen.getByText(/confirmed this event when 1 of 2 stored conditions/),
  ).toBeInTheDocument();
  expect(screen.getByText(/opencv-hog-people/)).toBeInTheDocument();
});

test("viewer does not see status actions", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/events/evt_42", jsonResponse(makeEvent())],
    ["/evidence", jsonResponse({ items: [], total: 0, limit: 50, offset: 0 })],
  ]);

  renderPage(<EventDetail />, { route: "/events/evt_42", path: "/events/:eventId" });
  await screen.findByTestId("conditions-table");

  expect(screen.queryByTestId("action-acknowledged")).not.toBeInTheDocument();
  expect(screen.queryByTestId("action-dismissed")).not.toBeInTheDocument();
  expect(
    screen.getByText(/require the events:ack \/ events:dismiss/),
  ).toBeInTheDocument();
});

test("operator acknowledge posts the transition", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    ["/events/evt_42/status", jsonResponse(makeEvent({ status: "acknowledged" }))],
    ["/events/evt_42", jsonResponse(makeEvent())],
    ["/evidence", jsonResponse({ items: [], total: 0, limit: 50, offset: 0 })],
  ]);

  renderPage(<EventDetail />, { route: "/events/evt_42", path: "/events/:eventId" });
  await screen.findByTestId("conditions-table");

  await userEvent.click(screen.getByTestId("action-acknowledged"));

  await waitFor(() => {
    const posts = fetchSpy.mock.calls.filter(
      (call) =>
        String(call[0]).includes("/events/evt_42/status") &&
        (call[1] as RequestInit)?.method === "POST",
    );
    expect(posts.length).toBeGreaterThan(0);
  });
  expect(await screen.findByText("Event acknowledged.")).toBeInTheDocument();
});

test("invalid transition surfaces backend 409", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    [
      "/events/evt_42/status",
      errorResponse(
        "invalid_transition",
        "cannot transition from resolved to acknowledged",
        409,
      ),
    ],
    [
      "/events/evt_42",
      jsonResponse(makeEvent({ status: "resolved" })),
    ],
    ["/evidence", jsonResponse({ items: [], total: 0, limit: 50, offset: 0 })],
  ]);

  renderPage(<EventDetail />, { route: "/events/evt_42", path: "/events/:eventId" });
  await screen.findByTestId("conditions-table");

  await userEvent.click(screen.getByTestId("action-acknowledged"));

  expect(await screen.findByTestId("error-panel")).toHaveTextContent(
    "cannot transition from resolved to acknowledged",
  );
});


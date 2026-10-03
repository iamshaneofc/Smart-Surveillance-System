import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import EventFeed from "./EventFeed";
import {
  EMPTY_PAGE,
  VIEWER,
  errorResponse,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

const EVENT = {
  event_id: "evt_1",
  camera_id: "cam1",
  timestamp: "2026-10-03T10:00:00Z",
  event_type: "restricted_zone_intrusion",
  severity: "high",
  status: "new",
  confidence: 0.91,
  summary: "Person entered restricted zone",
  track_ids: [7],
  zone_id: "z1",
  zone_name: "Restricted",
  rule_id: "restricted-zone-entry",
  rule_name: "Restricted zone entry",
  conditions: [],
  model_versions: { detector: "opencv-hog-people:4.14.0-dev1" },
  evidence_ids: ["ev1"],
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

test("renders event rows with severity and status", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/events", jsonResponse({ items: [EVENT], total: 1, limit: 20, offset: 0 })],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<EventFeed />);

  expect(await screen.findByTestId("event-row-evt_1")).toBeInTheDocument();
  expect(screen.getByText("Person entered restricted zone")).toBeInTheDocument();
  expect(screen.getByText("camera: cam1")).toBeInTheDocument();
  expect(screen.getByText(/1–1 of 1/)).toBeInTheDocument();
});

test("shows forbidden error panel when API rejects with 403", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/events", errorResponse("forbidden", "missing permission: events:read", 403)],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<EventFeed />);

  const panel = await screen.findByTestId("error-panel");
  expect(panel).toHaveTextContent("Permission required");
  expect(panel).toHaveTextContent("missing permission: events:read");
  expect(panel).toHaveTextContent("req_test");
});

test("shows empty state when no events match", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/events", jsonResponse(EMPTY_PAGE)],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<EventFeed />);

  expect(
    await screen.findByText("No events yet"),
  ).toBeInTheDocument();
});

test("applying status filter refetches with the filter", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/events", jsonResponse(EMPTY_PAGE)],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<EventFeed />);
  await screen.findByTestId("apply-filters");

  await userEvent.selectOptions(screen.getByTestId("filter-status"), "dismissed");
  await userEvent.click(screen.getByTestId("apply-filters"));

  await waitFor(() => {
    const eventCalls = fetchSpy.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.includes("/events?") && url.includes("status=dismissed"));
    expect(eventCalls.length).toBeGreaterThan(0);
  });
});

test("pagination advances offset", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    [
      "/events",
      jsonResponse({ items: [EVENT], total: 45, limit: 20, offset: 0 }),
    ],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<EventFeed />);
  await screen.findByTestId("event-row-evt_1");

  await userEvent.click(screen.getByTestId("page-next"));

  await waitFor(() => {
    const paged = fetchSpy.mock.calls
      .map((call) => String(call[0]))
      .filter((url) => url.includes("/events?") && url.includes("offset=20"));
    expect(paged.length).toBeGreaterThan(0);
  });
  expect(screen.getByTestId("page-prev")).not.toBeDisabled();
});

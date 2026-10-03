import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import RulesPage from "./RulesPage";
import {
  ADMIN,
  EMPTY_PAGE,
  OPERATOR,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

const RULE = {
  id: "db-rule-1",
  rule_id: "restricted-zone-entry",
  name: "Restricted zone entry",
  rule_type: "restricted_zone_intrusion",
  event_type: "restricted_zone_intrusion",
  camera_id: "cam1",
  severity: "high",
  enabled: true,
  version: 1,
  zone_ids: ["z1"],
  line: null,
  params: { dwell_seconds: 0 },
  cooldown_seconds: 60,
  confirm_seconds: 2,
  min_confidence: 0.4,
  schedule: { timezone: "UTC", days: [], windows: [] },
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01T00:00:00Z",
};

afterEach(() => {
  vi.restoreAllMocks();
});

test("renders rules with type, severity and camera scope", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    [
      "/rules",
      jsonResponse({ items: [RULE], total: 1, limit: 200, offset: 0 }),
    ],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<RulesPage />);

  expect(await screen.findByTestId("rule-restricted-zone-entry")).toBeInTheDocument();
  expect(screen.getByText("restricted_zone_intrusion")).toBeInTheDocument();
  expect(screen.queryByTestId("add-rule")).not.toBeInTheDocument();
});

test("operator without rules:manage cannot create or delete", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(OPERATOR)],
    ["/rules", jsonResponse(EMPTY_PAGE)],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<RulesPage />);

  await screen.findByText("No rules match");
  expect(screen.queryByTestId("add-rule")).not.toBeInTheDocument();
});

test("admin creates a rule through the form", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    [
      "/rules",
      (_url: string, init?: RequestInit) =>
        init?.method === "POST"
          ? jsonResponse({ ...RULE, rule_id: "dock-dwell" }, 201)
          : jsonResponse(EMPTY_PAGE),
    ],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<RulesPage />);
  await screen.findByTestId("add-rule");

  await userEvent.click(screen.getByTestId("add-rule"));
  expect(screen.getByTestId("rule-form")).toBeInTheDocument();

  await userEvent.type(screen.getByTestId("rule-form-id"), "dock-dwell");
  await userEvent.type(screen.getByTestId("rule-form-name"), "Dock dwell");
  await userEvent.click(screen.getByTestId("rule-form-submit"));

  await waitFor(() => {
    const post = fetchSpy.mock.calls.find(
      (call) =>
        String(call[0]).includes("/rules") &&
        (call[1] as RequestInit)?.method === "POST",
    );
    expect(post).toBeDefined();
    const body = JSON.parse(String((post as any[])[1].body)) as {
      rule_id: string;
      rule_type: string;
      params: Record<string, unknown>;
    };
    expect(body.rule_id).toBe("dock-dwell");
    expect(body.rule_type).toBe("restricted_zone_intrusion");
  });
  expect(await screen.findByText("Rule created.")).toBeInTheDocument();
});

test("rejects invalid params JSON before hitting the API", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/rules", jsonResponse(EMPTY_PAGE)],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<RulesPage />);
  await screen.findByTestId("add-rule");
  await userEvent.click(screen.getByTestId("add-rule"));

  await userEvent.type(screen.getByTestId("rule-form-id"), "bad-json");
  await userEvent.type(screen.getByTestId("rule-form-name"), "Bad");
  const paramsField = screen.getByTestId("rule-form-params");
  fireEvent.change(paramsField, { target: { value: "{not json" } });
  await userEvent.click(screen.getByTestId("rule-form-submit"));

  expect(await screen.findByTestId("error-panel")).toHaveTextContent(
    "params must be valid JSON",
  );
  const posts = fetchSpy.mock.calls.filter(
    (call) => (call[1] as RequestInit)?.method === "POST",
  );
  expect(posts).toHaveLength(0);
});

test("rejects line points outside the normalized range", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/rules", jsonResponse(EMPTY_PAGE)],
    ["/cameras", jsonResponse(EMPTY_PAGE)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<RulesPage />);
  await screen.findByTestId("add-rule");
  await userEvent.click(screen.getByTestId("add-rule"));

  await userEvent.type(screen.getByTestId("rule-form-id"), "line-rule");
  await userEvent.type(screen.getByTestId("rule-form-name"), "Line");
  const lineField = screen.getByLabelText(/Line points/);
  fireEvent.change(lineField, { target: { value: "100,200 400,250" } });
  await userEvent.click(screen.getByTestId("rule-form-submit"));

  expect(await screen.findByTestId("error-panel")).toHaveTextContent(
    "normalized 0–1",
  );
  const posts = fetchSpy.mock.calls.filter(
    (call) => (call[1] as RequestInit)?.method === "POST",
  );
  expect(posts).toHaveLength(0);
});


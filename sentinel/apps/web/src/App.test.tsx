import { render, screen, waitFor } from "@testing-library/react";
import App from "./App";

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "Content-Type": "application/json" }),
    json: async () => body,
  } as unknown as Response;
}

const WHO_AM_I = {
  user: "dev-admin",
  roles: ["admin"],
  permissions: ["events:read", "cameras:read", "health:read"],
  org_id: null,
  auth_mode: "disabled",
  warnings: ["authentication disabled"],
};

const HEALTH = {
  status: "healthy",
  checks: {
    database: { status: "healthy", dialect: "sqlite" },
    event_bus: { status: "healthy", backend: "memory" },
  },
  app: { name: "SENTINEL", version: "0.1.0", env: "test", uptime_seconds: 12 },
  ts: "2026-10-03T12:00:00Z",
};

const EMPTY_PAGE = { items: [], total: 0, limit: 20, offset: 0 };

function routeFetch(url: string): Response {
  if (url.includes("/auth/me")) return jsonResponse(WHO_AM_I);
  if (url.includes("/system/health")) return jsonResponse(HEALTH);
  if (url.includes("/cameras/health/summary")) return jsonResponse([]);
  if (url.includes("/stream")) {
    return { ok: true, status: 200, body: null } as unknown as Response;
  }
  if (url.includes("/cameras") || url.includes("/events") || url.includes("/alerts") || url.includes("/rules") || url.includes("/zones")) {
    return jsonResponse(EMPTY_PAGE);
  }
  return jsonResponse({});
}

test("opens the console shell once authenticated", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL) =>
    routeFetch(String(input)),
  ) as unknown as typeof fetch;

  render(<App />);
  expect(screen.getByRole("status")).toBeInTheDocument();

  await waitFor(() =>
    expect(screen.getAllByText("SENTINEL").length).toBeGreaterThan(0),
  );
  expect(
    await screen.findByRole("heading", { name: "Command Center" }),
  ).toBeInTheDocument();
  expect(screen.getByTestId("live-badge")).toBeInTheDocument();

  globalThis.fetch = originalFetch;
});

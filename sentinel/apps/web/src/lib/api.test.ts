import { ApiError, fetchEvents, fetchSystemHealth, request, setApiKey } from "./api";

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "Content-Type": "application/json" }),
    json: async () => body,
  } as unknown as Response;
}

afterEach(() => {
  setApiKey(null);
  vi.restoreAllMocks();
});

test("request sends API key header and parses error envelopes", async () => {
  setApiKey("viewer-key");
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
    jsonResponse(
      {
        error: {
          code: "forbidden",
          message: "missing permission: events:ack",
          request_id: "req_abc",
        },
      },
      403,
    ),
  );

  await expect(request("/events")).rejects.toMatchObject({
    code: "forbidden",
    status: 403,
    requestId: "req_abc",
  });

  const [, init] = fetchSpy.mock.calls[0];
  expect((init as RequestInit).headers).toMatchObject({
    "X-API-Key": "viewer-key",
  });
});

test("ApiError flags unauthorized / forbidden / conflict", () => {
  expect(new ApiError("x", "y", 401).isUnauthorized).toBe(true);
  expect(new ApiError("x", "y", 403).isForbidden).toBe(true);
  expect(new ApiError("x", "y", 404).isNotFound).toBe(true);
  expect(new ApiError("x", "y", 409).isConflict).toBe(true);
  expect(new ApiError("x", "y", 500).isUnauthorized).toBe(false);
});

test("fetchEvents serializes filter params", async () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
    jsonResponse({ items: [], total: 0, limit: 20, offset: 0 }),
  );

  await fetchEvents({
    camera_id: "cam1",
    status: "new",
    severity: "high",
    limit: 20,
    offset: 40,
  });

  const url = String(fetchSpy.mock.calls[0][0]);
  expect(url).toContain("/events?");
  expect(url).toContain("camera_id=cam1");
  expect(url).toContain("status=new");
  expect(url).toContain("severity=high");
  expect(url).toContain("limit=20");
  expect(url).toContain("offset=40");
});

test("fetchEvents omits empty filters", async () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
    jsonResponse({ items: [], total: 0, limit: 20, offset: 0 }),
  );

  await fetchEvents({ camera_id: "", status: undefined });

  const url = String(fetchSpy.mock.calls[0][0]);
  expect(url).toContain("/events");
  expect(url).not.toContain("camera_id");
  expect(url).not.toContain("status");
});

test("fetchSystemHealth accepts 503 degraded body", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue(
    jsonResponse(
      {
        status: "degraded",
        checks: { database: { status: "error", detail: "down" } },
        app: { name: "SENTINEL", version: "0.1.0", env: "test", uptime_seconds: 1 },
        ts: "2026-10-03T00:00:00Z",
      },
      503,
    ),
  );

  const health = await fetchSystemHealth();
  expect(health.status).toBe("degraded");
  expect(health.checks.database.status).toBe("error");
});

test("request returns undefined on 204", async () => {
  vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok: true,
    status: 204,
    headers: new Headers(),
  } as unknown as Response);

  await expect(request("/cameras/cam1")).resolves.toBeUndefined();
});

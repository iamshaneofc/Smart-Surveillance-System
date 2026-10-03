import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { ADMIN, jsonResponse } from "./test/utils";

afterEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
});

test("401 shows the login gate; entering an API key opens the console", async () => {
  const fetchSpy = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/auth/me")) {
        const headers = (init?.headers ?? {}) as Record<string, string>;
        if (headers["X-API-Key"] === "good-key") {
          return jsonResponse(ADMIN);
        }
        return jsonResponse(
          { error: { code: "unauthorized", message: "invalid api key", request_id: "r1" } },
          401,
        );
      }
      if (url.includes("/system/health")) {
        return jsonResponse({
          status: "healthy",
          checks: {},
          app: { name: "SENTINEL", version: "0.1.0", env: "test", uptime_seconds: 1 },
          ts: "2026-10-03T00:00:00Z",
        });
      }
      if (url.includes("/cameras/health/summary")) {
        return jsonResponse([]);
      }
      if (url.includes("/stream")) {
        return { ok: true, status: 200, body: null } as unknown as Response;
      }
      return jsonResponse({ items: [], total: 0, limit: 20, offset: 0 });
    },
  );
  globalThis.fetch = fetchSpy as unknown as typeof fetch;

  render(<App />);

  expect(await screen.findByTestId("api-key-input")).toBeInTheDocument();
  expect(screen.getByText(/authentication disabled/i)).toBeInTheDocument();

  await userEvent.type(screen.getByTestId("api-key-input"), "good-key");
  await userEvent.click(screen.getByTestId("api-key-submit"));

  await waitFor(() => {
    expect(
      screen.getByRole("heading", { name: "Command Center" }),
    ).toBeInTheDocument();
  });
  expect(localStorage.getItem("sentinel.apiKey")).toBe("good-key");
});

test("bad API key shows the backend error", async () => {
  localStorage.setItem("sentinel.apiKey", "wrong-key");
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/auth/me")) {
      const headers = (init?.headers ?? {}) as Record<string, string>;
      if (headers["X-API-Key"] === "wrong-key") {
        return jsonResponse(
          {
            error: {
              code: "unauthorized",
              message: "invalid api key",
              request_id: "r2",
            },
          },
          401,
        );
      }
      return jsonResponse(
        { error: { code: "unauthorized", message: "no key", request_id: "r3" } },
        401,
      );
    }
    return jsonResponse({});
  }) as unknown as typeof fetch;

  render(<App />);

  expect(await screen.findByTestId("api-key-input")).toBeInTheDocument();
  await userEvent.type(screen.getByTestId("api-key-input"), "wrong-key");
  await userEvent.click(screen.getByTestId("api-key-submit"));

  expect(await screen.findByTestId("error-panel")).toHaveTextContent(
    "invalid api key",
  );
});

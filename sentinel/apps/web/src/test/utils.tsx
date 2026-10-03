import { render, type RenderResult } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { AuthProvider } from "../lib/auth";
import { LiveProvider } from "../lib/live";

export function renderPage(
  ui: React.ReactElement,
  { route = "/", path = route }: { route?: string; path?: string } = {},
): RenderResult {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <AuthProvider>
        <LiveProvider enabled={false}>
          <Routes>
            <Route path={path} element={ui} />
          </Routes>
        </LiveProvider>
      </AuthProvider>
    </MemoryRouter>,
  );
}

export function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "Content-Type": "application/json" }),
    json: async () => body,
  } as unknown as Response;
}

export function errorResponse(
  code: string,
  message: string,
  status: number,
): Response {
  return jsonResponse({ error: { code, message, request_id: "req_test" } }, status);
}

export const VIEWER = {
  user: "viewer1",
  roles: ["viewer"],
  permissions: ["events:read", "cameras:read", "evidence:read", "health:read"],
  org_id: null,
  auth_mode: "api_key",
  warnings: [],
};

export const OPERATOR = {
  user: "operator1",
  roles: ["operator"],
  permissions: [
    "events:read",
    "events:ack",
    "events:dismiss",
    "cameras:read",
    "evidence:read",
    "health:read",
    "zones:read",
    "rules:read",
    "alerts:read",
  ],
  org_id: null,
  auth_mode: "api_key",
  warnings: [],
};

export const ADMIN = {
  user: "admin1",
  roles: ["admin"],
  permissions: [
    "events:read",
    "events:ack",
    "events:dismiss",
    "cameras:read",
    "cameras:manage",
    "evidence:read",
    "evidence:export",
    "health:read",
    "zones:read",
    "zones:manage",
    "rules:read",
    "rules:manage",
    "alerts:read",
  ],
  org_id: null,
  auth_mode: "api_key",
  warnings: [],
};

export const EMPTY_PAGE = { items: [], total: 0, limit: 20, offset: 0 };

export function mockFetch(
  routes: Array<[substring: string, response: Response | ((url: string, init?: RequestInit) => Response)]>,
): ReturnType<typeof vi.fn> {
  const spy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    for (const [substring, response] of routes) {
      if (url.includes(substring)) {
        return typeof response === "function" ? response(url, init) : response;
      }
    }
    return jsonResponse({});
  });
  globalThis.fetch = spy as unknown as typeof fetch;
  return spy;
}

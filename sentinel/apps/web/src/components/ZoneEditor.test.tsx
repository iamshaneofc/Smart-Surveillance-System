import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ZoneEditor } from "./ZoneEditor";
import {
  ADMIN,
  EMPTY_PAGE,
  VIEWER,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

function mockCanvasRect() {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    width: 640,
    height: 360,
    left: 0,
    top: 0,
    right: 640,
    bottom: 360,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
}

function clickCanvas(x: number, y: number) {
  fireEvent.click(screen.getByTestId("zone-canvas"), { clientX: x, clientY: y });
}

afterEach(() => {
  vi.restoreAllMocks();
});

test("admin draws a polygon and saves it with >= 3 points", async () => {
  mockCanvasRect();
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    [
      "/zones",
      (_url: string, init?: RequestInit) =>
        init?.method === "POST"
          ? jsonResponse({ id: "z1", name: "Perimeter" }, 201)
          : jsonResponse(EMPTY_PAGE),
    ],
  ]);

  renderPage(<ZoneEditor cameraId="cam1" frameWidth={640} frameHeight={360} />);
  await screen.findByTestId("zone-editor");

  clickCanvas(10, 10);
  clickCanvas(100, 10);
  clickCanvas(100, 100);
  expect(screen.getByText("3 points selected")).toBeInTheDocument();

  await userEvent.type(screen.getByTestId("zone-name"), "Perimeter");
  await userEvent.click(screen.getByTestId("zone-save"));

  await waitFor(() => {
    const post = fetchSpy.mock.calls.find(
      (call) =>
        String(call[0]).includes("/zones") &&
        (call[1] as RequestInit)?.method === "POST",
    );
    expect(post).toBeDefined();
    const body = JSON.parse(String((post as any[])[1].body)) as {
      polygon: number[][];
      zone_type: string;
      name: string;
    };
    expect(body.name).toBe("Perimeter");
    expect(body.polygon).toHaveLength(3);
    // Drawn in frame pixels, stored normalized [0,1].
    expect(body.polygon[0]).toEqual([0.015625, 0.027778]);
    for (const [x, y] of body.polygon) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
    }
  });
  expect(await screen.findByText("Zone created.")).toBeInTheDocument();
});

test("client blocks saving a zone with fewer than 3 points", async () => {
  mockCanvasRect();
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<ZoneEditor cameraId="cam1" frameWidth={640} frameHeight={360} />);
  await screen.findByTestId("zone-editor");

  clickCanvas(10, 10);
  await userEvent.type(screen.getByTestId("zone-name"), "Too small");
  await userEvent.click(screen.getByTestId("zone-save"));

  expect(await screen.findByTestId("error-panel")).toHaveTextContent(
    "at least 3 points",
  );
  const posts = fetchSpy.mock.calls.filter(
    (call) => (call[1] as RequestInit)?.method === "POST",
  );
  expect(posts).toHaveLength(0);
});

test("viewer cannot draw or save zones", async () => {
  mockCanvasRect();
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<ZoneEditor cameraId="cam1" />);
  await screen.findByTestId("zone-editor");

  clickCanvas(10, 10);
  expect(screen.getByText("0 points selected")).toBeInTheDocument();
  expect(screen.queryByTestId("zone-save")).not.toBeInTheDocument();
  expect(
    screen.getByText(/zones:manage/),
  ).toBeInTheDocument();
});

test("canvas is labelled approximate when frame size is unknown", async () => {
  mockCanvasRect();
  mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/zones", jsonResponse(EMPTY_PAGE)],
  ]);

  renderPage(<ZoneEditor cameraId="cam1" />);
  await screen.findByTestId("zone-editor");

  expect(
    screen.getByText(/1280×720 px · approximate \(frame size unknown\)/),
  ).toBeInTheDocument();
});

test("create sends anchor and enabled from the form controls", async () => {
  mockCanvasRect();
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    [
      "/zones",
      (_url: string, init?: RequestInit) =>
        init?.method === "POST"
          ? jsonResponse({ id: "z1", name: "Loading" }, 201)
          : jsonResponse(EMPTY_PAGE),
    ],
  ]);

  renderPage(<ZoneEditor cameraId="cam1" frameWidth={640} frameHeight={360} />);
  await screen.findByTestId("zone-editor");

  clickCanvas(10, 10);
  clickCanvas(100, 10);
  clickCanvas(100, 100);
  await userEvent.type(screen.getByTestId("zone-name"), "Loading");
  await userEvent.selectOptions(screen.getByTestId("zone-anchor"), "top_center");
  fireEvent.click(screen.getByTestId("zone-enabled"));
  await userEvent.click(screen.getByTestId("zone-save"));

  await waitFor(() => {
    const post = fetchSpy.mock.calls.find(
      (call) =>
        String(call[0]).includes("/zones") &&
        (call[1] as RequestInit)?.method === "POST",
    );
    expect(post).toBeDefined();
    const body = JSON.parse(String((post as any[])[1].body)) as {
      anchor: string;
      enabled: boolean;
    };
    expect(body.anchor).toBe("top_center");
    expect(body.enabled).toBe(false);
  });
  expect(await screen.findByText("Zone created.")).toBeInTheDocument();
});

test("admin can disable and re-enable a zone from the list", async () => {
  mockCanvasRect();
  const zone = {
    id: "z1",
    camera_id: "cam1",
    name: "Gate",
    zone_type: "restricted",
    polygon: [
      [0, 0],
      [0.5, 0],
      [0.5, 0.5],
    ],
    anchor: "center",
    enabled: true,
    metadata: {},
    created_at: null,
    updated_at: null,
  };
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    [
      "/zones",
      (_url: string, init?: RequestInit) =>
        init?.method === "PATCH"
          ? jsonResponse({ ...zone, enabled: false })
          : jsonResponse({ items: [zone], total: 1, limit: 100, offset: 0 }),
    ],
  ]);

  renderPage(<ZoneEditor cameraId="cam1" frameWidth={640} frameHeight={360} />);
  await screen.findByTestId("zone-Gate");

  await userEvent.click(screen.getByRole("button", { name: "Disable" }));

  await waitFor(() => {
    const patch = fetchSpy.mock.calls.find(
      (call) => (call[1] as RequestInit)?.method === "PATCH",
    );
    expect(patch).toBeDefined();
    const [, init] = patch as unknown as [string, RequestInit];
    const body = JSON.parse(String(init.body));
    expect(body).toEqual({ enabled: false });
  });
  expect(await screen.findByText("Gate disabled.")).toBeInTheDocument();
});


import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CameraWall from "./CameraWall";
import {
  ADMIN,
  VIEWER,
  errorResponse,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

const CAMERA = {
  id: "db-row-1",
  camera_id: "cam1",
  name: "Lobby",
  stream_url: "rtsp://example/stream",
  source_type: "rtsp",
  location: "Entrance",
  enabled: true,
  detection_fps: 5,
  inference_mode: "edge",
  created_at: "2026-10-01T00:00:00Z",
  updated_at: "2026-10-01T00:00:00Z",
  deleted_at: null,
};

const SUMMARY_ONLINE = [
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
];

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

test("viewer sees tiles but no management controls", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/cameras/health/summary", jsonResponse(SUMMARY_ONLINE)],
    ["preview.jpg", errorResponse("preview_unavailable", "no pipeline", 409)],
    ["/cameras", jsonResponse({ items: [CAMERA], total: 1, limit: 500, offset: 0 })],
  ]);

  renderPage(<CameraWall />);

  expect(await screen.findByTestId("camera-tile-cam1")).toBeInTheDocument();
  expect(screen.getByTestId("status-cam1")).toHaveTextContent("online");
  expect(screen.queryByTestId("add-camera")).not.toBeInTheDocument();
  expect(screen.queryByText("Edit")).not.toBeInTheDocument();
  expect(
    await screen.findByTestId("preview-unavailable-cam1"),
  ).toBeInTheDocument();
});

test("admin can open the add-camera form and create a camera", async () => {
  const fetchSpy = mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/cameras/health/summary", jsonResponse(SUMMARY_ONLINE)],
    ["preview.jpg", errorResponse("preview_unavailable", "no pipeline", 409)],
    [
      "/cameras",
      (_url: string, init?: RequestInit) =>
        init?.method === "POST"
          ? jsonResponse({ ...CAMERA, camera_id: "cam2" }, 201)
          : jsonResponse({ items: [CAMERA], total: 1, limit: 500, offset: 0 }),
    ],
  ]);

  renderPage(<CameraWall />);

  await userEvent.click(await screen.findByTestId("add-camera"));
  expect(screen.getByTestId("camera-form")).toBeInTheDocument();

  await userEvent.type(screen.getByTestId("camera-form-id"), "cam2");
  await userEvent.type(screen.getByTestId("camera-form-name"), "Dock");
  await userEvent.type(
    screen.getByTestId("camera-form-url"),
    "rtsp://example/dock",
  );
  await userEvent.click(screen.getByTestId("camera-form-submit"));

  await waitFor(() => {
    const post = fetchSpy.mock.calls.find(
      (call) =>
        String(call[0]).includes("/cameras") &&
        (call[1] as RequestInit)?.method === "POST",
    );
    expect(post).toBeDefined();
    expect(String((post as unknown[])[0])).toContain("/cameras");
  });
  expect(await screen.findByText("Camera created.")).toBeInTheDocument();
});

test("status filter narrows the visible cameras", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/cameras/health/summary", jsonResponse(SUMMARY_ONLINE)],
    ["preview.jpg", errorResponse("preview_unavailable", "no pipeline", 409)],
    ["/cameras", jsonResponse({ items: [CAMERA], total: 1, limit: 500, offset: 0 })],
  ]);

  renderPage(<CameraWall />);
  await screen.findByTestId("camera-tile-cam1");

  await userEvent.click(screen.getByTestId("filter-offline"));

  expect(screen.queryByTestId("camera-tile-cam1")).not.toBeInTheDocument();
  expect(
    await screen.findByText("No cameras with status 'offline'"),
  ).toBeInTheDocument();
});


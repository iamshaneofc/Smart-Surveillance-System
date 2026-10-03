import { screen } from "@testing-library/react";
import { EvidencePanel } from "./EvidencePanel";
import { fetchEvidence } from "../lib/api";
import { useQuery } from "../lib/useQuery";
import type { EvidenceItem, Page } from "../lib/types";
import {
  ADMIN,
  VIEWER,
  jsonResponse,
  mockFetch,
  renderPage,
} from "../test/utils";

function makeItem(overrides: Partial<EvidenceItem> = {}): EvidenceItem {
  return {
    evidence_id: "evid_1",
    event_id: "evt_1",
    camera_id: "cam1",
    type: "clip",
    uri: "file:///data/evidence/evid_1.mp4",
    sha256: "abc123def4567890abc123def4567890abc123def4567890abc123def4567890",
    size_bytes: 2048,
    content_type: "video/mp4",
    width: null,
    height: null,
    duration_ms: 3500,
    captured_at: "2026-10-03T10:00:00Z",
    expires_at: null,
    storage_backend: "filesystem",
    metadata: {},
    created_at: "2026-10-03T10:00:01Z",
    ...overrides,
  };
}

function Harness({ items }: { items: EvidenceItem[] }) {
  const evidence = useQuery(
    (signal) => fetchEvidence({ event_id: "evt_1" }, signal),
    [],
  );
  // Inject fixtures without a second network round-trip shape mismatch.
  const patched = { ...evidence, data: { items, total: items.length, limit: 50, offset: 0 } as Page<EvidenceItem> };
  return <EvidencePanel eventId="evt_1" evidence={patched} canExport={false} />;
}

function ExportHarness({ items }: { items: EvidenceItem[] }) {
  const evidence = useQuery(
    (signal) => fetchEvidence({ event_id: "evt_1" }, signal),
    [],
  );
  const patched = { ...evidence, data: { items, total: items.length, limit: 50, offset: 0 } as Page<EvidenceItem> };
  return <EvidencePanel eventId="evt_1" evidence={patched} canExport />;
}

afterEach(() => {
  vi.restoreAllMocks();
});

test("viewer sees metadata but no download controls", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(VIEWER)],
    ["/evidence", jsonResponse({ items: [makeItem()], total: 1, limit: 50, offset: 0 })],
  ]);

  renderPage(<Harness items={[makeItem()]} />);

  expect(await screen.findByTestId("evidence-evid_1")).toBeInTheDocument();
  expect(
    screen.getByText(/Previewing and downloading files requires the/),
  ).toBeInTheDocument();
  expect(screen.queryByTestId("download-evid_1")).not.toBeInTheDocument();
  expect(screen.getByText(/sha256 abc123def456789/)).toBeInTheDocument();
  expect(screen.getByText(/3\.5s/)).toBeInTheDocument();
});

test("export permission reveals download buttons", async () => {
  mockFetch([
    ["/auth/me", jsonResponse(ADMIN)],
    ["/evidence", jsonResponse({ items: [makeItem()], total: 1, limit: 50, offset: 0 })],
    ["/evidence/evid_1/download", errorResponse404()],
  ]);

  renderPage(<ExportHarness items={[makeItem()]} />);

  expect(await screen.findByTestId("download-evid_1")).toBeInTheDocument();
  expect(
    screen.queryByText(/requires the evidence:export/),
  ).not.toBeInTheDocument();
});

function errorResponse404(): Response {
  return jsonResponse(
    { error: { code: "not_found", message: "evidence expired", request_id: "r9" } },
    404,
  );
}

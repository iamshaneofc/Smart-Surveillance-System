import { useMemo, useRef, useState, type MouseEvent } from "react";
import {
  ApiError,
  createZone,
  deleteZone,
  fetchZones,
  updateZone,
} from "../lib/api";
import { useAuth } from "../lib/auth";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";
import type { Zone, ZoneType } from "../lib/types";
import { ErrorPanel, Spinner, EmptyState } from "./States";

const ZONE_TYPES: ZoneType[] = [
  "restricted",
  "hazardous",
  "loading",
  "entrance",
  "emergency",
  "parking",
  "waiting",
  "custom",
];

const ZONE_ANCHORS = ["center", "top_center", "bottom_center"];

const FALLBACK_WIDTH = 1280;
const FALLBACK_HEIGHT = 720;

/**
 * Zone polygon editor.
 *
 * Coordinates are frame pixels. When the camera preview is available the
 * coordinate space matches the actual frame; otherwise a fixed working canvas
 * is used and labelled as approximate. Polygons are validated client-side
 * (>= 3 points) and the server still enforces simplicity (422 invalid_polygon).
 */
export function ZoneEditor({
  cameraId,
  frameWidth,
  frameHeight,
}: {
  cameraId: string;
  frameWidth?: number | null;
  frameHeight?: number | null;
}) {
  const { can } = useAuth();
  const { tick } = useLive();
  const zones = useQuery(
    (signal) => fetchZones({ camera_id: cameraId, limit: 100 }, signal),
    [cameraId, tick],
  );
  const canManage = can("zones:manage");

  const [points, setPoints] = useState<number[][]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [zoneType, setZoneType] = useState<ZoneType>("restricted");
  const [anchor, setAnchor] = useState("center");
  const [enabled, setEnabled] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const width = frameWidth && frameWidth > 0 ? frameWidth : FALLBACK_WIDTH;
  const height = frameHeight && frameHeight > 0 ? frameHeight : FALLBACK_HEIGHT;
  const approximate = !frameWidth || !frameHeight;

  const previewPath = useMemo(
    () => (points.length > 1 ? points.map((p) => p.join(",")).join(" ") : null),
    [points],
  );

  function handleCanvasClick(event: MouseEvent<SVGSVGElement>) {
    if (!canManage) return;
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = Math.round(((event.clientX - rect.left) / rect.width) * width);
    const y = Math.round(((event.clientY - rect.top) / rect.height) * height);
    setPoints((current) => [...current, [x, y]]);
    setNotice(null);
  }

  function reset() {
    setPoints([]);
    setEditingId(null);
    setName("");
    setZoneType("restricted");
    setAnchor("center");
    setEnabled(true);
    setError(null);
    setNotice(null);
  }

  function startEdit(zone: Zone) {
    setEditingId(zone.id);
    // Stored polygons are normalized [0,1]; the canvas works in frame pixels.
    setPoints(zone.polygon.map(([x, y]) => [Math.round(x * width), Math.round(y * height)]));
    setName(zone.name);
    setZoneType(zone.zone_type);
    setAnchor(zone.anchor);
    setEnabled(zone.enabled);
    setError(null);
    setNotice(null);
  }

  function toNormalized(points: number[][]): number[][] {
    return points.map(([x, y]) => [
      Math.min(1, Math.max(0, Number((x / width).toFixed(6)))),
      Math.min(1, Math.max(0, Number((y / height).toFixed(6)))),
    ]);
  }

  async function save() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      if (points.length < 3) {
        throw new ApiError(
          "too_few_points",
          "A zone needs at least 3 points — click the canvas to add them.",
          422,
        );
      }
      const polygon = toNormalized(points);
      const wasEdit = Boolean(editingId);
      if (editingId) {
        await updateZone(editingId, { polygon, anchor, enabled });
      } else {
        if (!name.trim()) {
          throw new ApiError("name_required", "Enter a zone name.", 422);
        }
        await createZone(cameraId, {
          name: name.trim(),
          zone_type: zoneType,
          polygon,
          anchor,
          enabled,
        });
      }
      reset();
      setNotice(wasEdit ? "Zone polygon updated." : "Zone created.");
      zones.reload();
    } catch (err) {
      setError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  async function remove(zone: Zone) {
    setBusy(true);
    setError(null);
    try {
      await deleteZone(zone.id);
      if (editingId === zone.id) reset();
      zones.reload();
    } catch (err) {
      setError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  async function toggleEnabled(zone: Zone) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await updateZone(zone.id, { enabled: !zone.enabled });
      setNotice(`${zone.name} ${zone.enabled ? "disabled" : "enabled"}.`);
      zones.reload();
    } catch (err) {
      setError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2" data-testid="zone-editor">
      <div>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-300">
            Zones on this camera
          </h3>
          <span className="text-xs text-slate-500">{zones.data?.total ?? 0} total</span>
        </div>
        {zones.loading ? <Spinner label="Loading zones" /> : null}
        {zones.error ? <ErrorPanel error={zones.error} onRetry={zones.reload} /> : null}
        {zones.data && zones.data.items.length === 0 ? (
          <EmptyState
            title="No zones yet"
            hint={
              canManage
                ? "Click the canvas to draw the first polygon."
                : "Zones appear here once an operator draws them."
            }
          />
        ) : null}
        <ul className="space-y-2">
          {(zones.data?.items ?? []).map((zone) => (
            <li
              key={zone.id}
              className="rounded-lg border border-slate-800 bg-slate-900/60 p-2.5"
              data-testid={`zone-${zone.name}`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <span className="text-sm font-medium text-slate-200">{zone.name}</span>
                  <span className="ml-2 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] uppercase text-slate-400">
                    {zone.zone_type}
                  </span>
                  {!zone.enabled ? (
                    <span className="ml-2 text-[10px] uppercase text-slate-500">
                      disabled
                    </span>
                  ) : null}
                </div>
                <div className="flex shrink-0 gap-1">
                  <span className="text-xs text-slate-500">
                    {zone.polygon.length} pts
                  </span>
                  {canManage ? (
                    <>
                      <button
                        type="button"
                        onClick={() => startEdit(zone)}
                        className="rounded px-2 py-0.5 text-xs text-sky-400 hover:bg-slate-800"
                      >
                        Edit
                      </button>
                      <button
                        type="button"
                        onClick={() => void toggleEnabled(zone)}
                        disabled={busy}
                        className="rounded px-2 py-0.5 text-xs text-amber-400 hover:bg-slate-800 disabled:opacity-50"
                      >
                        {zone.enabled ? "Disable" : "Enable"}
                      </button>
                      <button
                        type="button"
                        onClick={() => void remove(zone)}
                        className="rounded px-2 py-0.5 text-xs text-rose-400 hover:bg-slate-800"
                      >
                        Delete
                      </button>
                    </>
                  ) : null}
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-300">
          {editingId ? "Edit polygon" : "Draw a zone"}
        </h3>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="none"
          className={`h-64 w-full rounded-lg border border-slate-700 bg-slate-950 ${
            canManage ? "cursor-crosshair" : "cursor-not-allowed"
          }`}
          onClick={handleCanvasClick}
          data-testid="zone-canvas"
          role="application"
          aria-label="Zone drawing canvas"
        >
          <defs>
            <pattern id="zone-grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" strokeWidth="1" />
            </pattern>
          </defs>
          <rect width={width} height={height} fill="url(#zone-grid)" />
          {(zones.data?.items ?? []).map((zone) => (
            <polygon
              key={zone.id}
              points={zone.polygon.map((p) => p.join(",")).join(" ")}
              fill={
                editingId === zone.id
                  ? "rgba(56,189,248,0.25)"
                  : "rgba(148,163,184,0.15)"
              }
              stroke={editingId === zone.id ? "#38bdf8" : "#64748b"}
              strokeWidth={2}
            />
          ))}
          {previewPath ? (
            <polygon
              points={previewPath}
              fill="rgba(52,211,153,0.2)"
              stroke="#34d399"
              strokeWidth={2}
              strokeDasharray="6 4"
            />
          ) : null}
          {points.map((p, index) => (
            <circle key={index} cx={p[0]} cy={p[1]} r={4} fill="#34d399" />
          ))}
        </svg>
        <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
          <span>
            Canvas {width}×{height} px
            {approximate ? " · approximate (frame size unknown)" : ""}
          </span>
          <span>{points.length} points selected</span>
        </div>

        {canManage ? (
          <div className="mt-3 space-y-2 rounded-lg border border-slate-800 bg-slate-900/60 p-3">
            <div className="grid gap-2 sm:grid-cols-2">
              <label className="block text-xs text-slate-400">
                Zone name
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={Boolean(editingId)}
                  placeholder={editingId ? "(kept)" : "Perimeter"}
                  className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100 disabled:text-slate-500"
                  data-testid="zone-name"
                />
              </label>
              <label className="block text-xs text-slate-400">
                Type
                <select
                  value={zoneType}
                  onChange={(e) => setZoneType(e.target.value as ZoneType)}
                  disabled={Boolean(editingId)}
                  className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                  data-testid="zone-type"
                >
                  {ZONE_TYPES.map((type) => (
                    <option key={type} value={type}>
                      {type}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-xs text-slate-400">
                Anchor
                <select
                  value={anchor}
                  onChange={(e) => setAnchor(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                  data-testid="zone-anchor"
                >
                  {ZONE_ANCHORS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => setEnabled(e.target.checked)}
                  className="size-4 accent-emerald-500"
                  data-testid="zone-enabled"
                />
                Enabled
              </label>
            </div>
            {error ? <ErrorPanel error={error} /> : null}
            {notice ? (
              <p className="text-sm text-emerald-400" role="status">
                {notice}
              </p>
            ) : null}
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => void save()}
                disabled={busy}
                className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white disabled:opacity-50"
                data-testid="zone-save"
              >
                {busy
                  ? "Saving…"
                  : editingId
                    ? "Update polygon"
                    : "Create zone"}
              </button>
              <button
                type="button"
                onClick={() => setPoints((current) => current.slice(0, -1))}
                disabled={points.length === 0}
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900 disabled:opacity-40"
              >
                Undo point
              </button>
              <button
                type="button"
                onClick={reset}
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900"
              >
                Clear
              </button>
            </div>
            <p className="text-xs text-slate-600">
              Polygon must have ≥ 3 points and no self-intersections. You draw
              in frame pixels; points are stored normalized 0–1 (the server
              rejects invalid polygons with 422 invalid_polygon).
            </p>
          </div>
        ) : (
          <p className="mt-3 text-xs text-slate-500">
            You need the <span className="font-mono">zones:manage</span> permission to
            draw or edit zones.
          </p>
        )}
      </div>
    </div>
  );
}

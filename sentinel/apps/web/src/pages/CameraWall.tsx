import { useMemo, useState, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router";
import { CameraPreview } from "../components/CameraPreview";
import { EmptyState, ErrorPanel, PageHeader, Spinner } from "../components/States";
import {
  ApiError,
  createCamera,
  deleteCamera,
  fetchCameras,
  fetchHealthSummary,
  updateCamera,
} from "../lib/api";
import { useAuth } from "../lib/auth";
import { healthStatusClass, relativeTime } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";
import type { Camera, CameraHealthSummary, CameraUpdate } from "../lib/types";

const STATUS_FILTERS = [
  "all",
  "online",
  "degraded",
  "retrying",
  "offline",
  "disabled",
  "unknown",
] as const;

type StatusFilter = (typeof STATUS_FILTERS)[number];

export default function CameraWall() {
  const { can } = useAuth();
  const { tick } = useLive();
  const [searchParams] = useSearchParams();
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [showForm, setShowForm] = useState(searchParams.get("manage") === "1");
  const [editing, setEditing] = useState<Camera | null>(null);

  const summary = useQuery((signal) => fetchHealthSummary(signal), [tick]);
  const cameras = useQuery(
    (signal) => fetchCameras({ limit: 500 }, signal),
    [tick],
  );

  const byId = useMemo(() => {
    const map = new Map<string, CameraHealthSummary>();
    for (const item of summary.data ?? []) map.set(item.camera_id, item);
    return map;
  }, [summary.data]);

  const visible = useMemo(() => {
    const liveOnly = (cameras.data?.items ?? []).filter((c) => !c.deleted_at);
    if (statusFilter === "all") return liveOnly;
    return liveOnly.filter(
      (c) => byId.get(c.camera_id)?.status === statusFilter,
    );
  }, [cameras.data, byId, statusFilter]);

  const canManage = can("cameras:manage");

  return (
    <div>
      <PageHeader
        title="Camera Wall"
        subtitle="All configured cameras with development previews and health status."
        actions={
          canManage ? (
            <button
              type="button"
              onClick={() => {
                setEditing(null);
                setShowForm((open) => !open);
              }}
              className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white"
              data-testid="add-camera"
            >
              {showForm ? "Close form" : "Add camera"}
            </button>
          ) : null
        }
      />

      {canManage && showForm ? (
        <div className="mb-5">
          <CameraForm
            camera={editing}
            onDone={() => {
              setShowForm(false);
              setEditing(null);
              summary.reload();
              cameras.reload();
            }}
          />
        </div>
      ) : null}

      <div className="mb-4 flex flex-wrap gap-2" role="group" aria-label="Filter by status">
        {STATUS_FILTERS.map((status) => (
          <button
            key={status}
            type="button"
            onClick={() => setStatusFilter(status)}
            className={`rounded-full px-3 py-1 text-xs font-medium capitalize ring-1 transition-colors ${
              statusFilter === status
                ? "bg-slate-100 text-slate-900 ring-slate-100"
                : "bg-slate-900 text-slate-400 ring-slate-700 hover:text-slate-200"
            }`}
            data-testid={`filter-${status}`}
          >
            {status}
          </button>
        ))}
      </div>

      {summary.error ? (
        <div className="mb-4">
          <ErrorPanel error={summary.error} onRetry={summary.reload} />
        </div>
      ) : null}
      {cameras.error ? (
        <div className="mb-4">
          <ErrorPanel error={cameras.error} onRetry={cameras.reload} />
        </div>
      ) : null}

      {summary.loading || cameras.loading ? (
        <Spinner label="Loading cameras" />
      ) : visible.length === 0 ? (
        <EmptyState
          title={statusFilter === "all" ? "No cameras configured" : `No cameras with status '${statusFilter}'`}
          hint={
            statusFilter === "all" && canManage
              ? "Use 'Add camera' to register your first source."
              : "Adjust the filter or add a camera."
          }
        />
      ) : (
        <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3" data-testid="camera-grid">
          {visible.map((camera) => {
            const health = byId.get(camera.camera_id);
            return (
              <li
                key={camera.id}
                className="overflow-hidden rounded-xl border border-slate-800 bg-slate-900/60"
                data-testid={`camera-tile-${camera.camera_id}`}
              >
                <Link
                  to={`/cameras/${encodeURIComponent(camera.camera_id)}`}
                  className="block"
                >
                  <CameraPreview cameraId={camera.camera_id} className="h-40 w-full" />
                </Link>
                <div className="p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <Link
                        to={`/cameras/${encodeURIComponent(camera.camera_id)}`}
                        className="truncate font-medium text-slate-200 hover:text-white"
                      >
                        {camera.name}
                      </Link>
                      <div className="truncate text-xs text-slate-500">
                        {camera.location ?? camera.camera_id} · {camera.source_type}
                      </div>
                    </div>
                    {health ? (
                      <span
                        className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${healthStatusClass(health.status)}`}
                        data-testid={`status-${camera.camera_id}`}
                      >
                        {health.status}
                      </span>
                    ) : (
                      <span className="shrink-0 rounded-full bg-slate-900 px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ring-slate-700 text-slate-400">
                        unknown
                      </span>
                    )}
                  </div>
                  <div className="mt-2 flex items-center justify-between text-[11px] text-slate-500">
                    <span>
                      {health?.fps != null ? `${health.fps.toFixed(1)} fps` : "no fps data"}
                      {health?.error ? ` · ${health.error}` : ""}
                    </span>
                    <span>
                      {health?.health_ts ? `health ${relativeTime(health.health_ts)}` : ""}
                      {health?.stale ? " (stale)" : ""}
                    </span>
                  </div>
                  {canManage ? (
                    <div className="mt-2 flex gap-2 border-t border-slate-800 pt-2">
                      <button
                        type="button"
                        onClick={() => {
                          setEditing(camera);
                          setShowForm(true);
                        }}
                        className="rounded-lg px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                      >
                        Edit
                      </button>
                      <DeleteCameraButton
                        cameraId={camera.camera_id}
                        onDeleted={() => {
                          summary.reload();
                          cameras.reload();
                        }}
                      />
                    </div>
                  ) : null}
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function DeleteCameraButton({
  cameraId,
  onDeleted,
}: {
  cameraId: string;
  onDeleted: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await deleteCamera(cameraId);
      onDeleted();
    } catch (err) {
      setError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
      setConfirming(false);
    }
  }

  if (error) {
    return (
      <span className="text-xs text-rose-400" role="alert">
        {error.message}
      </span>
    );
  }
  if (!confirming) {
    return (
      <button
        type="button"
        onClick={() => setConfirming(true)}
        className="rounded-lg px-2 py-1 text-xs text-rose-400 hover:bg-slate-800"
      >
        Delete
      </button>
    );
  }
  return (
    <span className="flex items-center gap-1 text-xs">
      <button
        type="button"
        onClick={() => void remove()}
        disabled={busy}
        className="rounded-lg bg-rose-950 px-2 py-1 text-rose-300 ring-1 ring-rose-800 disabled:opacity-50"
      >
        {busy ? "Deleting…" : "Confirm"}
      </button>
      <button
        type="button"
        onClick={() => setConfirming(false)}
        className="rounded-lg px-2 py-1 text-slate-400 hover:bg-slate-800"
      >
        Cancel
      </button>
    </span>
  );
}

export function CameraForm({
  camera,
  onDone,
}: {
  camera: Camera | null;
  onDone: () => void;
}) {
  const isEdit = camera !== null;
  const [cameraId, setCameraId] = useState(camera?.camera_id ?? "");
  const [name, setName] = useState(camera?.name ?? "");
  const [streamUrl, setStreamUrl] = useState("");
  const [sourceType, setSourceType] = useState(camera?.source_type ?? "rtsp");
  const [location, setLocation] = useState(camera?.location ?? "");
  const [detectionFps, setDetectionFps] = useState(
    String(camera?.detection_fps ?? 5),
  );
  const [enabled, setEnabled] = useState(camera?.enabled ?? true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [done, setDone] = useState<string | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      if (isEdit && camera) {
        const patch: CameraUpdate = {
          name,
          location: location || null,
          enabled,
          detection_fps: Number(detectionFps) || 1,
        };
        if (streamUrl.trim()) patch.stream_url = streamUrl.trim();
        await updateCamera(camera.camera_id, patch);
        setDone("Camera updated.");
      } else {
        await createCamera({
          camera_id: cameraId.trim(),
          name: name.trim(),
          stream_url: streamUrl.trim(),
          source_type: sourceType as Camera["source_type"],
          location: location || null,
          detection_fps: Number(detectionFps) || 1,
          enabled,
        });
        setDone("Camera created.");
      }
      setTimeout(onDone, 400);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err
          : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      onSubmit={submit}
      className="rounded-xl border border-slate-800 bg-slate-900/60 p-4"
      data-testid="camera-form"
    >
      <h2 className="text-sm font-semibold text-slate-200">
        {isEdit ? `Edit camera · ${camera.camera_id}` : "Add camera"}
      </h2>
      <div className="mt-3 grid gap-3 sm:grid-cols-2">
        <label className="block text-xs text-slate-400">
          Camera ID {isEdit ? "" : "(required)"}
          <input
            value={isEdit ? camera.camera_id : cameraId}
            onChange={(e) => setCameraId(e.target.value)}
            disabled={isEdit}
            required={!isEdit}
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100 disabled:text-slate-500"
            data-testid="camera-form-id"
          />
        </label>
        <label className="block text-xs text-slate-400">
          Name (required)
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
            data-testid="camera-form-name"
          />
        </label>
        <label className="block text-xs text-slate-400 sm:col-span-2">
          Stream URL {isEdit ? "(leave blank to keep current)" : "(required)"}
          <input
            value={streamUrl}
            onChange={(e) => setStreamUrl(e.target.value)}
            required={!isEdit}
            placeholder={isEdit ? "rtsp://…" : "rtsp://user:pass@host/stream"}
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
            data-testid="camera-form-url"
          />
        </label>
        <label className="block text-xs text-slate-400">
          Source type
          <select
            value={sourceType}
            onChange={(e) => setSourceType(e.target.value as Camera["source_type"])}
            disabled={isEdit}
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
          >
            <option value="rtsp">rtsp</option>
            <option value="file">file</option>
            <option value="webcam">webcam</option>
            <option value="synthetic">synthetic</option>
            <option value="onvif">onvif</option>
          </select>
        </label>
        <label className="block text-xs text-slate-400">
          Location
          <input
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
          />
        </label>
        <label className="block text-xs text-slate-400">
          Detection FPS
          <input
            type="number"
            min={1}
            step={1}
            value={detectionFps}
            onChange={(e) => setDetectionFps(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
          />
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => setEnabled(e.target.checked)}
            className="size-4 accent-emerald-500"
          />
          Enabled
        </label>
      </div>
      {error ? (
        <div className="mt-3">
          <ErrorPanel error={error} />
        </div>
      ) : null}
      {done ? (
        <p className="mt-3 text-sm text-emerald-400" role="status">
          {done}
        </p>
      ) : null}
      <div className="mt-3 flex gap-2">
        <button
          type="submit"
          disabled={busy}
          className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white disabled:opacity-50"
          data-testid="camera-form-submit"
        >
          {busy ? "Saving…" : isEdit ? "Save changes" : "Create camera"}
        </button>
        <button
          type="button"
          onClick={onDone}
          className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900"
        >
          Cancel
        </button>
      </div>
    </form>
  );
}

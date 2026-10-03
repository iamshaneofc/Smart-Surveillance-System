import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";
import { CameraPreview } from "../components/CameraPreview";
import { ZoneEditor } from "../components/ZoneEditor";
import { ErrorPanel, PageHeader, Spinner, EmptyState } from "../components/States";
import {
  fetchCamera,
  fetchCameraHealth,
  fetchEvents,
  fetchHealthSummary,
} from "../lib/api";
import { useAuth } from "../lib/auth";
import { eventStatusClass, formatTimestamp, healthStatusClass, relativeTime, severityClass } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";
import { CameraForm } from "./CameraWall";

type Tab = "overview" | "zones";

export default function CameraDetail() {
  const { cameraId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const { can } = useAuth();
  const { tick } = useLive();
  const [tab, setTab] = useState<Tab>(
    searchParams.get("tab") === "zones" ? "zones" : "overview",
  );
  const [editing, setEditing] = useState(false);

  const camera = useQuery(
    (signal) => fetchCamera(cameraId, signal),
    [cameraId, tick],
  );
  const summary = useQuery((signal) => fetchHealthSummary(signal), [tick]);
  const health = useQuery(
    (signal) => fetchCameraHealth(cameraId, 10, signal),
    [cameraId, tick],
  );
  const events = useQuery(
    (signal) => fetchEvents({ camera_id: cameraId, limit: 5 }, signal),
    [cameraId, tick],
  );

  const healthRow = (summary.data ?? []).find((s) => s.camera_id === cameraId);

  function selectTab(next: Tab) {
    setTab(next);
    const params = new URLSearchParams(searchParams);
    if (next === "overview") params.delete("tab");
    else params.set("tab", next);
    setSearchParams(params, { replace: true });
  }

  if (camera.loading) return <Spinner label="Loading camera" />;
  if (camera.error) {
    return (
      <div>
        <PageHeader title={`Camera · ${cameraId}`} />
        <ErrorPanel error={camera.error} onRetry={camera.reload} />
        <Link to="/cameras" className="mt-3 inline-block text-sm text-sky-400">
          ← Camera wall
        </Link>
      </div>
    );
  }
  const cam = camera.data!;
  const canManage = can("cameras:manage");

  return (
    <div>
      <PageHeader
        title={cam.name}
        subtitle={`${cam.camera_id}${cam.location ? ` · ${cam.location}` : ""}`}
        actions={
          <div className="flex items-center gap-2">
            {healthRow ? (
              <span
                className={`rounded-full px-2.5 py-1 text-xs font-semibold uppercase ring-1 ${healthStatusClass(healthRow.status)}`}
                data-testid="camera-status"
              >
                {healthRow.status}
                {healthRow.stale ? " · stale" : ""}
              </span>
            ) : null}
            {canManage ? (
              <button
                type="button"
                onClick={() => setEditing((open) => !open)}
                className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900"
              >
                {editing ? "Close editor" : "Edit camera"}
              </button>
            ) : null}
            <Link to="/cameras" className="text-sm text-sky-400 hover:text-sky-300">
              ← Wall
            </Link>
          </div>
        }
      />

      {canManage && editing ? (
        <div className="mb-5">
          <CameraForm
            camera={cam}
            onDone={() => {
              setEditing(false);
              camera.reload();
              summary.reload();
            }}
          />
        </div>
      ) : null}

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <CameraPreview cameraId={cam.camera_id} className="h-72 w-full rounded-xl" />
          <p className="mt-1.5 text-xs text-slate-500">
            Frame preview from the development pipeline buffer — this is a
            buffered still refresh, not a live video stream.
          </p>

          <div className="mt-4 flex gap-2" role="tablist" aria-label="Camera sections">
            {(["overview", "zones"] as const).map((key) => (
              <button
                key={key}
                type="button"
                role="tab"
                aria-selected={tab === key}
                onClick={() => selectTab(key)}
                className={`rounded-lg px-3 py-1.5 text-sm capitalize ${
                  tab === key
                    ? "bg-slate-100 text-slate-900"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200"
                }`}
                data-testid={`tab-${key}`}
              >
                {key}
              </button>
            ))}
          </div>

          {tab === "overview" ? (
            <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
                <Detail label="Source type" value={cam.source_type} />
                <Detail label="Enabled" value={cam.enabled ? "yes" : "no"} />
                <Detail label="Detection" value={cam.detection_enabled ? "on" : "off"} />
                <Detail label="Recording" value={cam.recording_enabled ? "on" : "off"} />
                <Detail label="Detection FPS" value={String(cam.detection_fps)} />
                <Detail label="Timezone" value={cam.timezone} />
                <Detail label="Model profile" value={cam.model_profile} />
                <Detail label="Rule profile" value={cam.rule_profile} />
                <Detail
                  label="Stream URL"
                  value={cam.stream_url_set ? "configured (write-only)" : "not set"}
                />
                <Detail
                  label="Frame size"
                  value={cam.width && cam.height ? `${cam.width}×${cam.height}` : "unknown"}
                />
                <Detail label="Retention" value={`${cam.retention.days_by_severity.high ?? 0}d high`} />
                <Detail label="Created" value={formatTimestamp(cam.created_at)} />
              </dl>
              {cam.deleted_at ? (
                <p className="mt-3 rounded-lg bg-rose-950/60 p-2 text-xs text-rose-300">
                  This camera is soft-deleted; history is preserved but the camera is
                  excluded from active listings.
                </p>
              ) : null}
            </div>
          ) : (
            <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/60 p-4">
              <ZoneEditor
                cameraId={cam.camera_id}
                frameWidth={cam.width}
                frameHeight={cam.height}
              />
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Latest health
            </h2>
            {health.loading ? <Spinner label="Loading health" /> : null}
            {health.error ? <ErrorPanel error={health.error} onRetry={health.reload} /> : null}
            {health.data && health.data.length === 0 ? (
              <p className="text-sm text-slate-500">No health rows recorded yet.</p>
            ) : null}
            {health.data && health.data.length > 0 ? (
              <ul className="mt-2 space-y-1.5 text-xs">
                {health.data.slice(0, 5).map((snap) => (
                  <li
                    key={snap.ts + String(snap.frames_processed)}
                    className="flex items-center justify-between gap-2"
                  >
                    <span className="text-slate-400">
                      {snap.state} / {snap.health} / ai:{snap.ai_status}
                    </span>
                    <span className="whitespace-nowrap text-slate-600">
                      {relativeTime(snap.ts)}
                    </span>
                  </li>
                ))}
              </ul>
            ) : null}
            <Link
              to="/health"
              className="mt-2 inline-block text-xs text-sky-400 hover:text-sky-300"
            >
              Full health page →
            </Link>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Recent events
            </h2>
            {events.loading ? <Spinner label="Loading events" /> : null}
            {events.error ? <ErrorPanel error={events.error} onRetry={events.reload} /> : null}
            {events.data && events.data.items.length === 0 ? (
              <EmptyState title="No events on this camera" />
            ) : null}
            <ul className="mt-2 space-y-2">
              {(events.data?.items ?? []).map((event) => (
                <li key={event.event_id}>
                  <Link
                    to={`/events/${encodeURIComponent(event.event_id)}`}
                    className="block rounded-lg border border-slate-800 p-2 hover:border-slate-700"
                  >
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase ring-1 ${severityClass(event.severity)}`}
                      >
                        {event.severity}
                      </span>
                      <span
                        className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase ring-1 ${eventStatusClass(event.status)}`}
                      >
                        {event.status}
                      </span>
                      <span className="ml-auto text-[10px] text-slate-500">
                        {relativeTime(event.timestamp)}
                      </span>
                    </div>
                    <div className="mt-1 truncate text-xs text-slate-300">
                      {event.summary || event.event_type}
                    </div>
                  </Link>
                </li>
              ))}
            </ul>
          </section>
        </aside>
      </div>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-slate-800/70 pb-1.5">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right text-slate-300">{value}</dd>
    </div>
  );
}

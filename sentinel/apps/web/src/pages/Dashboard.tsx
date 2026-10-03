import { Link } from "react-router";
import { CameraPreview } from "../components/CameraPreview";
import { ErrorPanel, PageHeader, Spinner, EmptyState } from "../components/States";
import { fetchEvents, fetchHealthSummary, fetchSystemHealth } from "../lib/api";
import { useLive } from "../lib/live";
import { eventStatusClass, healthStatusClass, relativeTime, severityClass } from "../lib/format";
import { useQuery } from "../lib/useQuery";

export default function Dashboard() {
  const { tick } = useLive();

  const health = useQuery((signal) => fetchSystemHealth(signal), [tick]);
  const summary = useQuery((signal) => fetchHealthSummary(signal), [tick]);
  const events = useQuery(
    (signal) => fetchEvents({ limit: 8 }, signal),
    [tick],
  );
  const openEvents = useQuery(
    (signal) => fetchEvents({ status: "new", limit: 1 }, signal),
    [tick],
  );
  const ackEvents = useQuery(
    (signal) => fetchEvents({ status: "acknowledged", limit: 1 }, signal),
    [tick],
  );

  const statuses = summary.data ?? [];
  const online = statuses.filter((s) => s.status === "online").length;
  const attention = statuses.filter((s) =>
    ["degraded", "offline", "retrying", "unknown"].includes(s.status),
  ).length;

  return (
    <div>
      <PageHeader
        title="Command Center"
        subtitle="System status, camera health and the latest detections."
      />

      {health.error ? (
        <div className="mb-4">
          <ErrorPanel error={health.error} onRetry={health.reload} />
        </div>
      ) : null}

      <section className="grid grid-cols-2 gap-3 md:grid-cols-4" aria-label="System summary">
        <StatCard
          label="System"
          value={health.data?.status ?? (health.loading ? "…" : "unknown")}
          tone={
            health.data?.status === "healthy"
              ? "good"
              : health.data?.status === "degraded"
                ? "warn"
                : "bad"
          }
          detail={
            health.data
              ? `v${health.data.app?.version ?? "?"} · ${health.data.app?.env ?? "?"} · up ${Math.round(health.data.app?.uptime_seconds ?? 0)}s`
              : undefined
          }
        />
        <StatCard
          label="Cameras online"
          value={summary.loading ? "…" : `${online} / ${statuses.length}`}
          tone="good"
          detail={`${attention} need attention`}
        />
        <StatCard
          label="Open events"
          value={
            openEvents.loading
              ? "…"
              : String(openEvents.data?.total ?? 0)
          }
          tone="info"
          detail="status: new"
        />
        <StatCard
          label="Acknowledged"
          value={ackEvents.loading ? "…" : String(ackEvents.data?.total ?? 0)}
          tone="info"
          detail="awaiting resolution"
        />
      </section>

      <div className="mt-6 grid gap-5 lg:grid-cols-5">
        <section className="lg:col-span-3" aria-label="Camera health">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Camera health
            </h2>
            <Link to="/health" className="text-xs text-sky-400 hover:text-sky-300">
              View all →
            </Link>
          </div>
          {summary.loading ? <Spinner label="Loading cameras" /> : null}
          {summary.error ? (
            <ErrorPanel error={summary.error} onRetry={summary.reload} />
          ) : null}
          {summary.data && summary.data.length === 0 ? (
            <EmptyState
              title="No cameras yet"
              hint="Add a camera to start the pipeline."
            />
          ) : null}
          <ul className="space-y-2">
            {(summary.data ?? []).slice(0, 5).map((cam) => (
              <li key={cam.camera_id}>
                <Link
                  to={`/cameras/${encodeURIComponent(cam.camera_id)}`}
                  className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900/60 p-3 transition-colors hover:border-slate-700"
                >
                  <CameraPreview
                    cameraId={cam.camera_id}
                    className="h-14 w-24 shrink-0 rounded"
                    refreshMs={8_000}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium text-slate-200">
                      {cam.name}
                    </div>
                    <div className="truncate text-xs text-slate-500">
                      {cam.location ?? cam.camera_id}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-1">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${healthStatusClass(cam.status)}`}
                      data-testid={`status-${cam.camera_id}`}
                    >
                      {cam.status}
                      {cam.stale && cam.status !== "disabled" && cam.status !== "unknown"
                        ? " · stale"
                        : ""}
                    </span>
                    <span className="text-[10px] text-slate-600">
                      {cam.health_ts ? `health ${relativeTime(cam.health_ts)}` : "no health data"}
                    </span>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        </section>

        <section className="lg:col-span-2" aria-label="Recent events">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Recent events
            </h2>
            <Link to="/events" className="text-xs text-sky-400 hover:text-sky-300">
              Event feed →
            </Link>
          </div>
          {events.loading ? <Spinner label="Loading events" /> : null}
          {events.error ? <ErrorPanel error={events.error} onRetry={events.reload} /> : null}
          {events.data && events.data.items.length === 0 ? (
            <EmptyState title="No events yet" hint="Detections will appear here." />
          ) : null}
          <ul className="space-y-2">
            {(events.data?.items ?? []).map((event) => (
              <li key={event.event_id}>
                <Link
                  to={`/events/${encodeURIComponent(event.event_id)}`}
                  className="block rounded-xl border border-slate-800 bg-slate-900/60 p-3 transition-colors hover:border-slate-700"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${severityClass(event.severity)}`}
                    >
                      {event.severity}
                    </span>
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${eventStatusClass(event.status)}`}
                    >
                      {event.status}
                    </span>
                    <span className="ml-auto text-[10px] text-slate-500">
                      {relativeTime(event.timestamp)}
                    </span>
                  </div>
                  <div className="mt-1.5 truncate text-sm text-slate-200">
                    {event.summary || event.event_type}
                  </div>
                  <div className="text-xs text-slate-500">{event.camera_id}</div>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail?: string;
  tone: "good" | "warn" | "bad" | "info";
}) {
  const toneClass =
    tone === "good"
      ? "text-emerald-300"
      : tone === "warn"
        ? "text-amber-300"
        : tone === "bad"
          ? "text-rose-300"
          : "text-slate-100";
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`mt-1 text-xl font-semibold ${toneClass}`}>{value}</div>
      {detail ? <div className="mt-0.5 text-xs text-slate-500">{detail}</div> : null}
    </div>
  );
}

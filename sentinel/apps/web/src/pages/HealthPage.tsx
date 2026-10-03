import { useState } from "react";
import { Link } from "react-router";
import { ErrorPanel, PageHeader, Spinner, EmptyState } from "../components/States";
import { fetchCameraHealth, fetchHealthSummary, fetchSystemHealth } from "../lib/api";
import { formatTimestamp, healthStatusClass, relativeTime } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";
import type { CameraHealthSummary } from "../lib/types";

export default function HealthPage() {
  const { tick } = useLive();
  const system = useQuery((signal) => fetchSystemHealth(signal), [tick]);
  const summary = useQuery((signal) => fetchHealthSummary(signal), [tick]);
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div>
      <PageHeader
        title="Camera & System Health"
        subtitle="Latest health per camera (server-derived status) and core service checks."
      />

      {system.error ? (
        <div className="mb-4">
          <ErrorPanel error={system.error} onRetry={system.reload} />
        </div>
      ) : null}

      <section className="mb-6 rounded-xl border border-slate-800 bg-slate-900/60 p-4" aria-label="System checks">
        <div className="flex flex-wrap items-center gap-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            System
          </h2>
          {system.data ? (
            <span
              className={`rounded-full px-2 py-0.5 text-xs font-semibold uppercase ring-1 ${
                system.data.status === "healthy"
                  ? "bg-emerald-950 text-emerald-300 ring-emerald-800"
                  : system.data.status === "degraded"
                    ? "bg-amber-950 text-amber-300 ring-amber-800"
                    : "bg-rose-950 text-rose-300 ring-rose-800"
              }`}
              data-testid="system-status"
            >
              {system.data.status}
            </span>
          ) : (
            <span className="text-sm text-slate-500">loading…</span>
          )}
          {system.data ? (
            <span className="text-xs text-slate-500">
              {system.data.app.name} v{system.data.app.version} · {system.data.app.env} ·
              uptime {Math.round(system.data.app.uptime_seconds)}s
            </span>
          ) : null}
        </div>
        {system.data ? (
          <ul className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(system.data.checks).map(([name, check]) => (
              <li
                key={name}
                className="flex items-center justify-between rounded-lg bg-slate-950 px-3 py-2 text-sm"
              >
                <span className="text-slate-300">{name}</span>
                <span className="flex items-center gap-2">
                  {check.backend || check.dialect ? (
                    <span className="text-xs text-slate-600">
                      {check.backend ?? check.dialect}
                    </span>
                  ) : null}
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${
                      check.status === "healthy"
                        ? "bg-emerald-950 text-emerald-300 ring-emerald-800"
                        : "bg-rose-950 text-rose-300 ring-rose-800"
                    }`}
                  >
                    {check.status}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        ) : null}
      </section>

      <section aria-label="Camera health">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-slate-400">
          Cameras
        </h2>
        {summary.loading ? <Spinner label="Loading health" /> : null}
        {summary.error ? (
          <ErrorPanel error={summary.error} onRetry={summary.reload} />
        ) : null}
        {summary.data && summary.data.length === 0 ? (
          <EmptyState title="No cameras configured" />
        ) : null}

        <ul className="space-y-2">
          {(summary.data ?? []).map((row) => (
            <li
              key={row.camera_id}
              className="rounded-xl border border-slate-800 bg-slate-900/60"
              data-testid={`health-${row.camera_id}`}
            >
              <button
                type="button"
                className="flex w-full flex-wrap items-center gap-3 p-3 text-left"
                onClick={() =>
                  setExpanded((current) =>
                    current === row.camera_id ? null : row.camera_id,
                  )
                }
                aria-expanded={expanded === row.camera_id}
              >
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${healthStatusClass(row.status)}`}
                >
                  {row.status}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium text-slate-200">
                    {row.name}
                  </span>
                  <span className="block truncate text-xs text-slate-500">
                    {row.camera_id}
                    {row.location ? ` · ${row.location}` : ""}
                    {row.error ? ` · ${row.error}` : ""}
                  </span>
                </span>
                <span className="text-right text-xs text-slate-500">
                  {row.fps != null ? `${row.fps.toFixed(1)} fps` : "—"}
                  <br />
                  {row.health_ts ? (
                    <span className={row.stale ? "text-amber-400" : ""}>
                      {relativeTime(row.health_ts)}
                      {row.stale ? " · stale" : ""}
                    </span>
                  ) : (
                    "no data"
                  )}
                </span>
              </button>
              {expanded === row.camera_id ? (
                <HealthDetail cameraId={row.camera_id} row={row} />
              ) : null}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function HealthDetail({
  cameraId,
  row,
}: {
  cameraId: string;
  row: CameraHealthSummary;
}) {
  const { tick } = useLive();
  const snapshots = useQuery(
    (signal) => fetchCameraHealth(cameraId, 20, signal),
    [cameraId, tick],
  );

  return (
    <div className="border-t border-slate-800 p-3" data-testid={`health-detail-${cameraId}`}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <dl className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-400">
          <div>
            <dt className="inline text-slate-500">state:</dt>{" "}
            <dd className="inline text-slate-300">{row.state ?? "—"}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">health:</dt>{" "}
            <dd className="inline text-slate-300">{row.health ?? "—"}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">detector/ai:</dt>{" "}
            <dd className="inline text-slate-300">{row.ai_status ?? "—"}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">reconnects:</dt>{" "}
            <dd className="inline text-slate-300">{row.reconnect_count ?? "—"}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">frame drops:</dt>{" "}
            <dd className="inline text-slate-300">{row.frame_drops ?? "—"}</dd>
          </div>
          <div>
            <dt className="inline text-slate-500">last frame:</dt>{" "}
            <dd className="inline text-slate-300">
              {row.last_frame_at ? formatTimestamp(row.last_frame_at) : "—"}
            </dd>
          </div>
        </dl>
        <Link
          to={`/cameras/${encodeURIComponent(cameraId)}`}
          className="text-xs text-sky-400 hover:text-sky-300"
        >
          Open camera →
        </Link>
      </div>

      {snapshots.loading ? <Spinner label="Loading history" /> : null}
      {snapshots.error ? (
        <ErrorPanel error={snapshots.error} onRetry={snapshots.reload} />
      ) : null}
      {snapshots.data && snapshots.data.length === 0 ? (
        <p className="text-xs text-slate-500">No health rows recorded yet.</p>
      ) : null}
      {snapshots.data && snapshots.data.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-500">
              <tr>
                <th className="py-1 pr-3 font-medium">Time</th>
                <th className="py-1 pr-3 font-medium">State</th>
                <th className="py-1 pr-3 font-medium">Health</th>
                <th className="py-1 pr-3 font-medium">AI</th>
                <th className="py-1 pr-3 font-medium">FPS</th>
                <th className="py-1 pr-3 font-medium">Reconnects</th>
                <th className="py-1 font-medium">Error</th>
              </tr>
            </thead>
            <tbody className="text-slate-300">
              {snapshots.data.map((snap) => (
                <tr key={snap.ts + String(snap.frames_processed)} className="border-t border-slate-800/70">
                  <td className="py-1 pr-3 whitespace-nowrap">{formatTimestamp(snap.ts)}</td>
                  <td className="py-1 pr-3">{snap.state}</td>
                  <td className="py-1 pr-3">{snap.health}</td>
                  <td className="py-1 pr-3">{snap.ai_status}</td>
                  <td className="py-1 pr-3">{snap.fps.toFixed(1)}</td>
                  <td className="py-1 pr-3">{snap.reconnect_count}</td>
                  <td className="py-1 text-rose-300">{snap.error ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

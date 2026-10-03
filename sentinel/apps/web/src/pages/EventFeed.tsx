import { useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router";
import { EmptyState, ErrorPanel, PageHeader, Spinner } from "../components/States";
import { fetchCameras, fetchEvents } from "../lib/api";
import { eventStatusClass, formatTimestamp, relativeTime, severityClass } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";

const PAGE_SIZE = 20;

const STATUSES = ["", "new", "acknowledged", "resolved", "dismissed"];
const SEVERITIES = ["", "low", "medium", "high", "critical"];

export default function EventFeed() {
  const { tick } = useLive();
  const [searchParams, setSearchParams] = useSearchParams();

  const cameraId = searchParams.get("camera_id") ?? "";
  const status = searchParams.get("status") ?? "";
  const severity = searchParams.get("severity") ?? "";
  const eventType = searchParams.get("event_type") ?? "";
  const ruleId = searchParams.get("rule_id") ?? "";
  const since = searchParams.get("since") ?? "";
  const until = searchParams.get("until") ?? "";
  const offset = Number(searchParams.get("offset") ?? "0") || 0;

  const [draft, setDraft] = useState({
    camera_id: cameraId,
    status,
    severity,
    event_type: eventType,
    rule_id: ruleId,
    since,
    until,
  });

  const filters = useMemo(
    () => ({
      camera_id: cameraId || undefined,
      status: status || undefined,
      severity: severity || undefined,
      event_type: eventType || undefined,
      rule_id: ruleId || undefined,
      since: since ? new Date(since).toISOString() : undefined,
      until: until ? new Date(until).toISOString() : undefined,
      limit: PAGE_SIZE,
      offset,
    }),
    [cameraId, status, severity, eventType, ruleId, since, until, offset],
  );

  const events = useQuery((signal) => fetchEvents(filters, signal), [
    JSON.stringify(filters),
    tick,
  ]);
  const cameras = useQuery(
    (signal) => fetchCameras({ limit: 500 }, signal),
    [],
  );

  function applyFilters() {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(draft)) {
      if (value) params.set(key, value);
    }
    setSearchParams(params);
  }

  function resetFilters() {
    setDraft({
      camera_id: "",
      status: "",
      severity: "",
      event_type: "",
      rule_id: "",
      since: "",
      until: "",
    });
    setSearchParams({});
  }

  function setOffset(next: number) {
    const params = new URLSearchParams(searchParams);
    if (next <= 0) params.delete("offset");
    else params.set("offset", String(next));
    setSearchParams(params);
  }

  const total = events.data?.total ?? 0;
  const hasFilters = Boolean(
    cameraId || status || severity || eventType || ruleId || since || until,
  );

  return (
    <div>
      <PageHeader
        title="Event Feed"
        subtitle="Detections with filters, deterministic ordering (newest first) and pagination."
      />

      <section
        className="mb-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4"
        aria-label="Event filters"
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <label className="block text-xs text-slate-400">
            Camera
            <select
              value={draft.camera_id}
              onChange={(e) => setDraft((d) => ({ ...d, camera_id: e.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              data-testid="filter-camera"
            >
              <option value="">All cameras</option>
              {(cameras.data?.items ?? [])
                .filter((c) => !c.deleted_at)
                .map((c) => (
                  <option key={c.camera_id} value={c.camera_id}>
                    {c.name}
                  </option>
                ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Status
            <select
              value={draft.status}
              onChange={(e) => setDraft((d) => ({ ...d, status: e.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              data-testid="filter-status"
            >
              {STATUSES.map((value) => (
                <option key={value} value={value}>
                  {value || "Any"}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Severity
            <select
              value={draft.severity}
              onChange={(e) => setDraft((d) => ({ ...d, severity: e.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              data-testid="filter-severity"
            >
              {SEVERITIES.map((value) => (
                <option key={value} value={value}>
                  {value || "Any"}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Event type
            <input
              value={draft.event_type}
              onChange={(e) => setDraft((d) => ({ ...d, event_type: e.target.value }))}
              placeholder="restricted_zone_intrusion"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              data-testid="filter-event-type"
            />
          </label>
          <label className="block text-xs text-slate-400">
            Rule ID
            <input
              value={draft.rule_id}
              onChange={(e) => setDraft((d) => ({ ...d, rule_id: e.target.value }))}
              placeholder="restricted-zone-entry"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
            />
          </label>
          <label className="block text-xs text-slate-400">
            From
            <input
              type="datetime-local"
              value={draft.since}
              onChange={(e) => setDraft((d) => ({ ...d, since: e.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
            />
          </label>
          <label className="block text-xs text-slate-400">
            To
            <input
              type="datetime-local"
              value={draft.until}
              onChange={(e) => setDraft((d) => ({ ...d, until: e.target.value }))}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
            />
          </label>
          <div className="flex items-end gap-2">
            <button
              type="button"
              onClick={applyFilters}
              className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white"
              data-testid="apply-filters"
            >
              Apply
            </button>
            <button
              type="button"
              onClick={resetFilters}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900"
            >
              Reset
            </button>
          </div>
        </div>
      </section>

      {events.error ? (
        <div className="mb-4">
          <ErrorPanel error={events.error} onRetry={events.reload} />
        </div>
      ) : null}
      {events.loading ? <Spinner label="Loading events" /> : null}

      {!events.loading && !events.error && events.data?.items.length === 0 ? (
        <EmptyState
          title={hasFilters ? "No events match these filters" : "No events yet"}
          hint={
            hasFilters
              ? "Try widening the time range or clearing filters."
              : "Detections confirmed by rules appear here."
          }
        />
      ) : null}

      {events.data && events.data.items.length > 0 ? (
        <>
          <ul className="space-y-2" data-testid="event-list">
            {events.data.items.map((event) => (
              <li key={event.event_id}>
                <Link
                  to={`/events/${encodeURIComponent(event.event_id)}`}
                  className="block rounded-xl border border-slate-800 bg-slate-900/60 p-3 transition-colors hover:border-slate-700"
                  data-testid={`event-row-${event.event_id}`}
                >
                  <div className="flex flex-wrap items-center gap-2">
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
                    <span className="text-xs font-medium text-slate-300">
                      {event.event_type}
                    </span>
                    <span className="ml-auto text-xs text-slate-500">
                      {formatTimestamp(event.timestamp)}{" "}
                      <span className="text-slate-600">({relativeTime(event.timestamp)})</span>
                    </span>
                  </div>
                  <p className="mt-1.5 text-sm text-slate-200">
                    {event.summary || event.event_type}
                  </p>
                  <div className="mt-1 flex flex-wrap gap-x-4 text-xs text-slate-500">
                    <span>camera: {event.camera_id}</span>
                    {event.zone_name ? <span>zone: {event.zone_name}</span> : null}
                    {event.rule_name ? <span>rule: {event.rule_name}</span> : null}
                    {event.evidence_ids.length > 0 ? (
                      <span>evidence: {event.evidence_ids.length}</span>
                    ) : null}
                  </div>
                </Link>
              </li>
            ))}
          </ul>

          <nav
            className="mt-4 flex items-center justify-between text-sm"
            aria-label="Pagination"
          >
            <button
              type="button"
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-slate-300 hover:bg-slate-900 disabled:opacity-40"
              data-testid="page-prev"
            >
              ← Previous
            </button>
            <span className="text-xs text-slate-500" data-testid="page-info">
              {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
            </span>
            <button
              type="button"
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-slate-300 hover:bg-slate-900 disabled:opacity-40"
              data-testid="page-next"
            >
              Next →
            </button>
          </nav>
        </>
      ) : null}
    </div>
  );
}

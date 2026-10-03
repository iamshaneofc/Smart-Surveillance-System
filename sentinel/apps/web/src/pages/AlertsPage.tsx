import { useState } from "react";
import { Link, useSearchParams } from "react-router";
import { EmptyState, ErrorPanel, PageHeader, Spinner } from "../components/States";
import { fetchAlerts, fetchEvents } from "../lib/api";
import { formatTimestamp, relativeTime } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";

const PAGE_SIZE = 25;
const CHANNELS = ["", "in_app", "webhook"];
const STATUSES = ["", "sent", "skipped", "failed"];

export default function AlertsPage() {
  const { tick } = useLive();
  const [searchParams, setSearchParams] = useSearchParams();
  const channel = searchParams.get("channel") ?? "";
  const status = searchParams.get("status") ?? "";
  const eventId = searchParams.get("event_id") ?? "";
  const offset = Number(searchParams.get("offset") ?? "0") || 0;

  const [draftChannel, setDraftChannel] = useState(channel);
  const [draftStatus, setDraftStatus] = useState(status);
  const [draftEventId, setDraftEventId] = useState(eventId);

  const alerts = useQuery(
    (signal) =>
      fetchAlerts(
        {
          channel: channel || undefined,
          status: status || undefined,
          event_id: eventId || undefined,
          limit: PAGE_SIZE,
          offset,
        },
        signal,
      ),
    [channel, status, eventId, offset, tick],
  );

  const recentEvents = useQuery(
    (signal) => fetchEvents({ limit: 100 }, signal),
    [tick],
  );

  const eventTitles = new Map(
    (recentEvents.data?.items ?? []).map((e) => [
      e.event_id,
      e.summary || e.event_type,
    ]),
  );

  function apply() {
    const params = new URLSearchParams();
    if (draftChannel) params.set("channel", draftChannel);
    if (draftStatus) params.set("status", draftStatus);
    if (draftEventId) params.set("event_id", draftEventId);
    setSearchParams(params);
  }

  function reset() {
    setDraftChannel("");
    setDraftStatus("");
    setDraftEventId("");
    setSearchParams({});
  }

  function setOffset(next: number) {
    const params = new URLSearchParams(searchParams);
    if (next <= 0) params.delete("offset");
    else params.set("offset", String(next));
    setSearchParams(params);
  }

  const total = alerts.data?.total ?? 0;

  return (
    <div>
      <PageHeader
        title="Alert Center"
        subtitle="Delivery attempts per event across channels (in-app, webhook)."
      />

      <section
        className="mb-4 rounded-xl border border-slate-800 bg-slate-900/60 p-4"
        aria-label="Alert filters"
      >
        <div className="grid gap-3 sm:grid-cols-4">
          <label className="block text-xs text-slate-400">
            Channel
            <select
              value={draftChannel}
              onChange={(e) => setDraftChannel(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              data-testid="alert-channel"
            >
              {CHANNELS.map((value) => (
                <option key={value} value={value}>
                  {value || "Any"}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Status
            <select
              value={draftStatus}
              onChange={(e) => setDraftStatus(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
              data-testid="alert-status"
            >
              {STATUSES.map((value) => (
                <option key={value} value={value}>
                  {value || "Any"}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-slate-400">
            Event ID
            <input
              value={draftEventId}
              onChange={(e) => setDraftEventId(e.target.value)}
              placeholder="evt_…"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
            />
          </label>
          <div className="flex items-end gap-2">
            <button
              type="button"
              onClick={apply}
              className="rounded-lg bg-slate-100 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-white"
            >
              Apply
            </button>
            <button
              type="button"
              onClick={reset}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-900"
            >
              Reset
            </button>
          </div>
        </div>
      </section>

      {alerts.error ? (
        <div className="mb-4">
          <ErrorPanel error={alerts.error} onRetry={alerts.reload} />
        </div>
      ) : null}
      {alerts.loading ? <Spinner label="Loading alerts" /> : null}

      {!alerts.loading && !alerts.error && alerts.data?.items.length === 0 ? (
        <EmptyState
          title="No alert deliveries yet"
          hint="Alerts are recorded when rules confirm events and a channel dispatch runs."
        />
      ) : null}

      {alerts.data && alerts.data.items.length > 0 ? (
        <>
          <div className="overflow-x-auto rounded-xl border border-slate-800">
            <table className="w-full text-left text-sm" data-testid="alerts-table">
              <thead className="bg-slate-900 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2 font-medium">Created</th>
                  <th className="px-3 py-2 font-medium">Event</th>
                  <th className="px-3 py-2 font-medium">Channel</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Attempts</th>
                  <th className="px-3 py-2 font-medium">Error</th>
                </tr>
              </thead>
              <tbody>
                {alerts.data.items.map((alert) => (
                  <tr
                    key={alert.id}
                    className="border-t border-slate-800/70 text-slate-300"
                    data-testid={`alert-${alert.id}`}
                  >
                    <td className="px-3 py-2 whitespace-nowrap">
                      {formatTimestamp(alert.created_at)}
                      <div className="text-[10px] text-slate-600">
                        {relativeTime(alert.created_at)}
                      </div>
                    </td>
                    <td className="px-3 py-2">
                      <Link
                        to={`/events/${encodeURIComponent(alert.event_id)}`}
                        className="block max-w-52 truncate text-sky-400 hover:text-sky-300"
                        title={eventTitles.get(alert.event_id) ?? alert.event_id}
                      >
                        {eventTitles.get(alert.event_id) ?? alert.event_id}
                      </Link>
                      <span className="text-[10px] text-slate-600">{alert.event_id}</span>
                    </td>
                    <td className="px-3 py-2">
                      <span className="rounded bg-slate-800 px-1.5 py-0.5 text-xs">
                        {alert.channel}
                      </span>
                      {alert.target ? (
                        <div className="max-w-40 truncate text-[10px] text-slate-600">
                          {alert.target}
                        </div>
                      ) : null}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${
                          alert.status === "sent"
                            ? "bg-emerald-950 text-emerald-300 ring-emerald-800"
                            : alert.status === "failed"
                              ? "bg-rose-950 text-rose-300 ring-rose-800"
                              : "bg-slate-900 text-slate-400 ring-slate-700"
                        }`}
                      >
                        {alert.status}
                      </span>
                      {alert.sent_at ? (
                        <div className="text-[10px] text-slate-600">
                          {relativeTime(alert.sent_at)}
                        </div>
                      ) : null}
                    </td>
                    <td className="px-3 py-2">{alert.attempts}</td>
                    <td className="max-w-48 truncate px-3 py-2 text-rose-300/80">
                      {alert.error ?? ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <nav
            className="mt-4 flex items-center justify-between text-sm"
            aria-label="Pagination"
          >
            <button
              type="button"
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-slate-300 hover:bg-slate-900 disabled:opacity-40"
            >
              ← Previous
            </button>
            <span className="text-xs text-slate-500">
              {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
            </span>
            <button
              type="button"
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total}
              className="rounded-lg border border-slate-700 px-3 py-1.5 text-slate-300 hover:bg-slate-900 disabled:opacity-40"
            >
              Next →
            </button>
          </nav>
        </>
      ) : null}
    </div>
  );
}

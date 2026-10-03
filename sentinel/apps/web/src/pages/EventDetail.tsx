import { useState } from "react";
import { Link, useParams } from "react-router";
import {
  ApiError,
  fetchEvent,
  fetchEvidence,
  updateEventStatus,
} from "../lib/api";
import { useAuth } from "../lib/auth";
import { formatTimestamp, relativeTime } from "../lib/format";
import { useLive } from "../lib/live";
import { useQuery } from "../lib/useQuery";
import { ErrorPanel, PageHeader, Spinner } from "../components/States";
import { EvidencePanel } from "../components/EvidencePanel";

const STATUS_ACTIONS = [
  { status: "acknowledged", label: "Acknowledge", permission: "events:ack" },
  { status: "resolved", label: "Resolve", permission: "events:ack" },
  { status: "dismissed", label: "Dismiss", permission: "events:dismiss" },
] as const;

export default function EventDetail() {
  const { eventId = "" } = useParams();
  const { can } = useAuth();
  const { tick } = useLive();

  const event = useQuery((signal) => fetchEvent(eventId, signal), [eventId, tick]);
  const evidence = useQuery(
    (signal) => fetchEvidence({ event_id: eventId, limit: 50 }, signal),
    [eventId],
  );

  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<ApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  async function act(status: string) {
    setBusy(true);
    setActionError(null);
    setNotice(null);
    try {
      await updateEventStatus(eventId, status, note || undefined);
      setNotice(`Event ${status}.`);
      setNote("");
      event.reload();
    } catch (err) {
      setActionError(
        err instanceof ApiError ? err : new ApiError("network_error", String(err), 0),
      );
    } finally {
      setBusy(false);
    }
  }

  if (event.loading) return <Spinner label="Loading event" />;
  if (event.error) {
    return (
      <div>
        <PageHeader title={`Event ${eventId}`} />
        <ErrorPanel error={event.error} onRetry={event.reload} />
        <Link to="/events" className="mt-3 inline-block text-sm text-sky-400">
          ← Event feed
        </Link>
      </div>
    );
  }

  const item = event.data!;
  const satisfiedCount = item.conditions.filter((c) => c.satisfied).length;

  return (
    <div>
      <PageHeader
        title={item.summary || item.event_type}
        subtitle={`Event ${item.event_id} · ${formatTimestamp(item.timestamp)} (${relativeTime(item.timestamp)})`}
        actions={
          <Link to="/events" className="text-sm text-sky-400 hover:text-sky-300">
            ← Event feed
          </Link>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-slate-900 px-2.5 py-1 text-xs font-semibold uppercase ring-1 ring-slate-700 text-slate-300">
          {item.severity}
        </span>
        <span
          className="rounded-full bg-slate-900 px-2.5 py-1 text-xs font-semibold uppercase ring-1 ring-slate-700 text-slate-200"
          data-testid="event-status"
        >
          {item.status}
        </span>
        <span className="text-sm text-slate-300">{item.event_type}</span>
        {item.confidence != null ? (
          <span className="text-xs text-slate-500">
            detector confidence {item.confidence.toFixed(2)}
          </span>
        ) : null}
      </div>

      {actionError ? (
        <div className="mb-4">
          <ErrorPanel error={actionError} />
        </div>
      ) : null}
      {notice ? (
        <p className="mb-4 text-sm text-emerald-400" role="status">
          {notice}
        </p>
      ) : null}

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Why this event fired
            </h2>
            <p className="mt-1 text-sm text-slate-300">
              Rule <span className="font-medium">{item.rule_name ?? "unknown"}</span>{" "}
              confirmed this event when {satisfiedCount} of {item.conditions.length}{" "}
              stored conditions were satisfied at detection time.
            </p>
            {item.conditions.length === 0 ? (
              <p className="mt-2 text-sm text-slate-500">
                No condition evidence was stored for this event.
              </p>
            ) : (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-sm" data-testid="conditions-table">
                  <thead className="text-xs uppercase text-slate-500">
                    <tr>
                      <th className="py-1.5 pr-3 font-medium">Condition</th>
                      <th className="py-1.5 pr-3 font-medium">Operator</th>
                      <th className="py-1.5 pr-3 font-medium">Actual</th>
                      <th className="py-1.5 pr-3 font-medium">Threshold</th>
                      <th className="py-1.5 font-medium">Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {item.conditions.map((cond, index) => (
                      <tr
                        key={`${cond.name}-${index}`}
                        className="border-t border-slate-800"
                        data-testid={`condition-${index}`}
                      >
                        <td className="py-1.5 pr-3 text-slate-200">{cond.name}</td>
                        <td className="py-1.5 pr-3 font-mono text-xs text-slate-400">
                          {cond.operator}
                        </td>
                        <td className="py-1.5 pr-3 text-slate-200">
                          {String(cond.actual)}
                        </td>
                        <td className="py-1.5 pr-3 text-slate-400">
                          {String(cond.threshold)}
                        </td>
                        <td className="py-1.5">
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase ring-1 ${
                              cond.satisfied
                                ? "bg-emerald-950 text-emerald-300 ring-emerald-800"
                                : "bg-slate-900 text-slate-400 ring-slate-700"
                            }`}
                          >
                            {cond.satisfied ? "satisfied" : "not met"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {Object.keys(item.model_versions).length > 0 ? (
              <dl className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-500">
                {Object.entries(item.model_versions).map(([key, value]) => (
                  <div key={key}>
                    <dt className="inline text-slate-600">{key}:</dt>{" "}
                    <dd className="inline font-mono text-slate-400">{value}</dd>
                  </div>
                ))}
                {item.rule_id ? (
                  <div>
                    <dt className="inline text-slate-600">rule:</dt>{" "}
                    <dd className="inline font-mono text-slate-400">
                      {item.rule_id}
                    </dd>
                  </div>
                ) : null}
              </dl>
            ) : null}
          </section>

          <EvidencePanel
            eventId={eventId}
            evidence={evidence}
            canExport={can("evidence:export")}
          />
        </div>

        <aside className="space-y-4">
          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Details
            </h2>
            <dl className="mt-2 space-y-1.5 text-sm">
              <Row label="Camera">
                <Link
                  to={`/cameras/${encodeURIComponent(item.camera_id)}`}
                  className="text-sky-400 hover:text-sky-300"
                >
                  {item.camera_id}
                </Link>
              </Row>
              {item.zone_name ? <Row label="Zone">{item.zone_name}</Row> : null}
              {item.rule_name ? <Row label="Rule">{item.rule_name}</Row> : null}
              <Row label="Tracks">{item.track_ids.join(", ") || "-"}</Row>
              <Row label="Created">{formatTimestamp(item.created_at)}</Row>
              {item.acknowledged_by ? (
                <Row label="Acknowledged by">
                  {item.acknowledged_by} · {relativeTime(item.acknowledged_at)}
                </Row>
              ) : null}
              {item.resolved_at ? (
                <Row label="Resolved">{formatTimestamp(item.resolved_at)}</Row>
              ) : null}
            </dl>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Actions
            </h2>
            <label className="mt-2 block text-xs text-slate-400">
              Note (optional, stored with the audit trail)
              <textarea
                value={note}
                onChange={(e) => setNote(e.target.value)}
                rows={2}
                className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-sm text-slate-100"
                data-testid="status-note"
              />
            </label>
            <div className="mt-2 flex flex-wrap gap-2">
              {STATUS_ACTIONS.map((action) =>
                can(action.permission) ? (
                  <button
                    key={action.status}
                    type="button"
                    onClick={() => void act(action.status)}
                    disabled={busy || item.status === action.status}
                    className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-900 disabled:opacity-40"
                    data-testid={`action-${action.status}`}
                  >
                    {action.label}
                  </button>
                ) : null,
              )}
            </div>
            {!can("events:ack") && !can("events:dismiss") ? (
              <p className="mt-2 text-xs text-slate-500">
                Status transitions require the events:ack / events:dismiss
                permissions. The API rejects unauthorized attempts with 403.
              </p>
            ) : null}
          </section>
        </aside>
      </div>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-right text-slate-300">{children}</dd>
    </div>
  );
}

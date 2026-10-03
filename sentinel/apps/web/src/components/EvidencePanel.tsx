import { useEffect, useState } from "react";
import { ApiError, downloadEvidence } from "../lib/api";
import type { EvidenceItem, Page } from "../lib/types";
import { formatBytes, formatTimestamp } from "../lib/format";
import { EmptyState, ErrorPanel, Spinner } from "./States";
import type { useQuery } from "../lib/useQuery";

type EvidenceQuery = ReturnType<typeof useQuery<Page<EvidenceItem>>>;

const INLINE_SNAPSHOT_LIMIT = 4;

function useSnapshotUrl(item: EvidenceItem, enabled: boolean) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!enabled || item.type !== "snapshot") return;
    let active = true;
    let objectUrl: string | null = null;
    (async () => {
      try {
        const { blob } = await downloadEvidence(item.evidence_id);
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      } catch {
        if (active) setFailed(true);
      }
    })();
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [item.evidence_id, item.type, enabled]);

  return { url, failed };
}

/**
 * Evidence viewer (phase F3-H): metadata for every item, inline image preview
 * and downloads only when the principal holds `evidence:export` (the server
 * enforces the same rule with 403).
 */
export function EvidencePanel({
  eventId,
  evidence,
  canExport,
}: {
  eventId: string;
  evidence: EvidenceQuery;
  canExport: boolean;
}) {
  const items = evidence.data?.items ?? [];

  if (evidence.loading) return <Spinner label="Loading evidence" />;
  if (evidence.error) {
    return (
      <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Evidence
        </h2>
        <div className="mt-2">
          <ErrorPanel error={evidence.error} onRetry={evidence.reload} />
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Evidence
        </h2>
        <span className="text-xs text-slate-500">{evidence.data?.total ?? 0} items</span>
      </div>

      {items.length === 0 ? (
        <div className="mt-2">
          <EmptyState
            title="No evidence stored for this event"
            hint={`Event ${eventId} has no snapshots or clips.`}
          />
        </div>
      ) : (
        <>
          {!canExport ? (
            <p className="mt-2 rounded-lg bg-slate-950 p-2 text-xs text-slate-500">
              You can inspect evidence metadata. Previewing and downloading files
              requires the <span className="font-mono">evidence:export</span>{" "}
              permission (the API returns 403 otherwise).
            </p>
          ) : null}
          <ul className="mt-3 grid gap-3 sm:grid-cols-2" data-testid="evidence-list">
            {items.map((item, index) => (
              <EvidenceCard
                key={item.evidence_id}
                item={item}
                canExport={canExport}
                inline={canExport && index < INLINE_SNAPSHOT_LIMIT}
              />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function EvidenceCard({
  item,
  canExport,
  inline,
}: {
  item: EvidenceItem;
  canExport: boolean;
  inline: boolean;
}) {
  const { url, failed } = useSnapshotUrl(item, inline);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function download() {
    setDownloading(true);
    setError(null);
    try {
      const { blob, filename } = await downloadEvidence(item.evidence_id);
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.message : "Download failed.",
      );
    } finally {
      setDownloading(false);
    }
  }

  return (
    <li
      className="overflow-hidden rounded-lg border border-slate-800 bg-slate-950"
      data-testid={`evidence-${item.evidence_id}`}
    >
      {item.type === "snapshot" ? (
        <div className="flex h-36 items-center justify-center bg-black">
          {url ? (
            <img
              src={url}
              alt={`Snapshot ${item.evidence_id}`}
              className="max-h-36 w-full object-contain"
            />
          ) : (
            <span className="text-xs text-slate-600">
              {failed
                ? "Preview unavailable"
                : inline
                  ? "Loading preview…"
                  : "Preview requires evidence:export"}
            </span>
          )}
        </div>
      ) : (
        <div className="flex h-36 flex-col items-center justify-center gap-1 bg-slate-900">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
            Clip {item.duration_ms ? `${(item.duration_ms / 1000).toFixed(1)}s` : ""}
          </span>
          <span className="text-xs text-slate-600">video evidence</span>
        </div>
      )}
      <div className="space-y-1 p-2.5 text-xs">
        <div className="flex items-center justify-between">
          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] font-semibold uppercase text-slate-300">
            {item.type}
          </span>
          <span className="text-slate-500">{formatBytes(item.size_bytes)}</span>
        </div>
        <div className="text-slate-400">{formatTimestamp(item.captured_at)}</div>
        <div className="truncate font-mono text-[10px] text-slate-600" title={item.uri}>
          {item.uri}
        </div>
        {item.sha256 ? (
          <div className="truncate font-mono text-[10px] text-slate-600" title={item.sha256}>
            sha256 {item.sha256.slice(0, 16)}…
          </div>
        ) : null}
        {error ? (
          <div className="text-rose-400" role="alert">
            {error}
          </div>
        ) : null}
        {canExport ? (
          <button
            type="button"
            onClick={() => void download()}
            disabled={downloading}
            className="mt-1 w-full rounded-lg border border-slate-700 py-1 text-slate-200 hover:bg-slate-900 disabled:opacity-50"
            data-testid={`download-${item.evidence_id}`}
          >
            {downloading ? "Preparing…" : "Download"}
          </button>
        ) : null}
      </div>
    </li>
  );
}

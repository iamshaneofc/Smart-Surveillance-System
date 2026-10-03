export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function relativeTime(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "-";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  const diffMs = now - date.getTime();
  const abs = Math.abs(diffMs);
  const suffix = diffMs >= 0 ? "ago" : "from now";
  if (abs < 5_000) return "just now";
  if (abs < 60_000) return `${Math.round(abs / 1_000)}s ${suffix}`;
  if (abs < 3_600_000) return `${Math.round(abs / 60_000)}m ${suffix}`;
  if (abs < 86_400_000) return `${Math.round(abs / 3_600_000)}h ${suffix}`;
  return `${Math.round(abs / 86_400_000)}d ${suffix}`;
}

export function severityClass(severity: string): string {
  switch (severity) {
    case "critical":
      return "bg-rose-950 text-rose-300 ring-rose-800";
    case "high":
      return "bg-orange-950 text-orange-300 ring-orange-800";
    case "medium":
      return "bg-amber-950 text-amber-300 ring-amber-800";
    default:
      return "bg-slate-900 text-slate-300 ring-slate-700";
  }
}

export function eventStatusClass(status: string): string {
  switch (status) {
    case "new":
      return "bg-sky-950 text-sky-300 ring-sky-800";
    case "acknowledged":
      return "bg-amber-950 text-amber-300 ring-amber-800";
    case "resolved":
      return "bg-emerald-950 text-emerald-300 ring-emerald-800";
    case "dismissed":
      return "bg-slate-900 text-slate-400 ring-slate-700";
    default:
      return "bg-slate-900 text-slate-300 ring-slate-700";
  }
}

export function healthStatusClass(status: string): string {
  switch (status) {
    case "online":
      return "bg-emerald-950 text-emerald-300 ring-emerald-800";
    case "degraded":
      return "bg-amber-950 text-amber-300 ring-amber-800";
    case "retrying":
      return "bg-sky-950 text-sky-300 ring-sky-800";
    case "offline":
      return "bg-rose-950 text-rose-300 ring-rose-800";
    case "disabled":
      return "bg-slate-900 text-slate-400 ring-slate-700";
    default:
      return "bg-slate-900 text-slate-400 ring-slate-700";
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

import type { ReactNode } from "react";
import { ApiError } from "../lib/api";

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div
      className="flex items-center gap-3 text-slate-400 text-sm"
      role="status"
      aria-live="polite"
    >
      <span className="size-4 animate-spin rounded-full border-2 border-slate-600 border-t-slate-200" />
      <span>{label}…</span>
    </div>
  );
}

export function EmptyState({
  title,
  hint,
  icon,
}: {
  title: string;
  hint?: string;
  icon?: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-8 text-center">
      {icon ? <div className="mb-2 text-3xl opacity-60">{icon}</div> : null}
      <p className="text-slate-300 font-medium">{title}</p>
      {hint ? <p className="mt-1 text-sm text-slate-500">{hint}</p> : null}
    </div>
  );
}

export function ErrorPanel({
  error,
  onRetry,
}: {
  error: ApiError | Error | null;
  onRetry?: () => void;
}) {
  if (!error) return null;
  const apiError = error instanceof ApiError ? error : null;
  const forbidden = apiError?.isForbidden ?? false;
  const title = forbidden
    ? "Permission required"
    : apiError?.isNotFound
      ? "Not found"
      : apiError?.status === 429
        ? "Rate limited"
        : apiError?.status === 0
          ? "Backend unreachable"
          : "Something went wrong";
  const detail = error.message || "Unexpected error.";

  return (
    <div
      className="rounded-xl border border-rose-900/60 bg-rose-950/40 p-4"
      role="alert"
      data-testid="error-panel"
    >
      <p className="font-medium text-rose-200">{title}</p>
      <p className="mt-1 text-sm text-rose-300/80 break-words">{detail}</p>
      {forbidden ? (
        <p className="mt-1 text-xs text-rose-300/60">
          Your account does not have this permission. The API enforces access —
          hiding UI controls is not authorization.
        </p>
      ) : null}
      {apiError?.requestId ? (
        <p className="mt-1 font-mono text-xs text-rose-300/50">
          request_id: {apiError.requestId}
        </p>
      ) : null}
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-lg bg-slate-800 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
        >
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function UnauthorizedNotice({ message }: { message?: string }) {
  return (
    <div
      className="rounded-xl border border-amber-800/60 bg-amber-950/40 p-4 text-sm text-amber-200"
      role="alert"
    >
      {message ?? "You are not signed in."}
    </div>
  );
}

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold text-slate-100">{title}</h1>
        {subtitle ? (
          <p className="mt-0.5 text-sm text-slate-400">{subtitle}</p>
        ) : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

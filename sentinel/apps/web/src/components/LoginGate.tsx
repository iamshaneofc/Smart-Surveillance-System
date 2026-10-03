import { useState, type FormEvent } from "react";
import { ApiError } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ErrorPanel } from "./States";

export function LoginGate() {
  const { login } = useAuth();
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!key.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await login(key.trim());
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
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
      <div className="w-full max-w-sm rounded-2xl border border-slate-800 bg-slate-900/70 p-6">
        <h1 className="text-lg font-semibold text-slate-100">SENTINEL</h1>
        <p className="mt-1 text-sm text-slate-400">
          Enter your API key to open the operations console.
        </p>
        <form onSubmit={submit} className="mt-4 space-y-3">
          <label className="block">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              API key
            </span>
            <input
              type="password"
              value={key}
              onChange={(e) => setKey(e.target.value)}
              autoFocus
              autoComplete="off"
              className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 placeholder-slate-600 focus:border-slate-500 focus:outline-none"
              placeholder="sentinel-dev-key"
              data-testid="api-key-input"
            />
          </label>
          <button
            type="submit"
            disabled={busy || !key.trim()}
            className="w-full rounded-lg bg-slate-100 px-3 py-2 text-sm font-medium text-slate-900 hover:bg-white disabled:opacity-50"
            data-testid="api-key-submit"
          >
            {busy ? "Checking…" : "Open console"}
          </button>
        </form>
        {error ? (
          <div className="mt-3">
            <ErrorPanel error={error} />
          </div>
        ) : null}
        <p className="mt-4 text-xs text-slate-500">
          If the backend runs with authentication disabled
          (<span className="font-mono">SENTINEL_AUTH_MODE=disabled</span>), no
          key is required and this screen will not appear.
        </p>
      </div>
    </div>
  );
}

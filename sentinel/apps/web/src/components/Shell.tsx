import { useState, type ReactNode } from "react";
import { NavLink } from "react-router";
import { useAuth } from "../lib/auth";
import { useLive } from "../lib/live";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/cameras", label: "Cameras" },
  { to: "/events", label: "Events" },
  { to: "/alerts", label: "Alerts" },
  { to: "/rules", label: "Rules" },
  { to: "/health", label: "Health" },
];

export function LiveBadge() {
  const { mode } = useLive();
  if (mode === "live") {
    return (
      <span
        data-testid="live-badge"
        className="inline-flex items-center gap-1.5 rounded-full bg-emerald-950 px-2.5 py-1 text-xs font-medium text-emerald-300 ring-1 ring-emerald-800"
      >
        <span className="size-1.5 animate-pulse rounded-full bg-emerald-400" />
        LIVE
      </span>
    );
  }
  if (mode === "polling") {
    return (
      <span
        data-testid="live-badge"
        className="inline-flex items-center gap-1.5 rounded-full bg-amber-950 px-2.5 py-1 text-xs font-medium text-amber-300 ring-1 ring-amber-800"
        title="Live stream unavailable - refreshing every 10 seconds"
      >
        <span className="size-1.5 rounded-full bg-amber-400" />
        POLLING · 10s
      </span>
    );
  }
  return (
    <span
      data-testid="live-badge"
      className="inline-flex items-center gap-1.5 rounded-full bg-rose-950 px-2.5 py-1 text-xs font-medium text-rose-300 ring-1 ring-rose-800"
      title="Backend unreachable - data may be stale"
    >
      <span className="size-1.5 rounded-full bg-rose-400" />
      OFFLINE
    </span>
  );
}

function navLinkClass(isActive: boolean): string {
  return [
    "block rounded-lg px-3 py-2 text-sm transition-colors",
    isActive
      ? "bg-slate-800 text-slate-50 font-medium"
      : "text-slate-400 hover:bg-slate-900 hover:text-slate-200",
  ].join(" ");
}

export function Shell({ children }: { children: ReactNode }) {
  const { me, logout, can } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
        <div className="flex items-center gap-3 px-4 py-3">
          <button
            type="button"
            className="rounded-lg p-2 text-slate-400 hover:bg-slate-900 md:hidden"
            aria-label="Toggle navigation"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((open) => !open)}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <div className="flex items-center gap-2">
            <span className="font-semibold tracking-wide text-slate-100">SENTINEL</span>
            <span className="hidden rounded bg-slate-900 px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-slate-500 sm:inline">
              {me?.auth_mode === "api_key" ? "api key" : (me?.auth_mode ?? "auth")}
            </span>
          </div>
          <div className="ml-auto flex items-center gap-3">
            <LiveBadge />
            <div className="flex items-center gap-2">
              <div className="hidden text-right sm:block">
                <div className="text-sm text-slate-200">{me?.user ?? "-"}</div>
                <div className="text-xs text-slate-500">
                  {me?.roles.join(", ") ?? ""}
                </div>
              </div>
              {me?.auth_mode === "api_key" ? (
                <button
                  type="button"
                  onClick={logout}
                  className="rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs text-slate-300 hover:bg-slate-900"
                >
                  Sign out
                </button>
              ) : null}
            </div>
          </div>
        </div>
        {me?.warnings?.length ? (
          <div className="border-t border-amber-900/50 bg-amber-950/40 px-4 py-1.5 text-xs text-amber-300">
            {me.warnings.join(" · ")}
          </div>
        ) : null}
      </header>

      <div className="flex">
        <nav
          className={[
            "fixed inset-y-0 left-0 z-10 w-56 transform border-r border-slate-800 bg-slate-950 pt-16 transition-transform md:sticky md:top-[57px] md:h-[calc(100vh-57px)] md:translate-x-0 md:pt-0",
            menuOpen ? "translate-x-0" : "-translate-x-full",
          ].join(" ")}
          aria-label="Primary"
        >
          <ul className="space-y-1 p-3">
            {NAV_ITEMS.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.end}
                  className={({ isActive }) => navLinkClass(isActive)}
                  onClick={() => setMenuOpen(false)}
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
            {can("cameras:manage") ? (
              <>
                <li className="pt-3 text-xs uppercase tracking-wider text-slate-600">
                  Administration
                </li>
                <li>
                  <NavLink
                    to="/cameras?manage=1"
                    className={({ isActive }) => navLinkClass(isActive)}
                    onClick={() => setMenuOpen(false)}
                  >
                    Manage cameras
                  </NavLink>
                </li>
              </>
            ) : null}
          </ul>
        </nav>
        {menuOpen ? (
          <button
            type="button"
            aria-label="Close navigation"
            className="fixed inset-0 z-5 bg-black/60 md:hidden"
            onClick={() => setMenuOpen(false)}
          />
        ) : null}

        <main className="min-w-0 flex-1 px-4 py-5 md:px-6">
          <div className="mx-auto max-w-6xl">{children}</div>
        </main>
      </div>
    </div>
  );
}

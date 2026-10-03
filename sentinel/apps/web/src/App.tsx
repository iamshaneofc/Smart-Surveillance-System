import { BrowserRouter, Navigate, Route, Routes } from "react-router";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { LoginGate } from "./components/LoginGate";
import { Shell } from "./components/Shell";
import { ErrorPanel, Spinner } from "./components/States";
import { AuthProvider, useAuth } from "./lib/auth";
import { LiveProvider } from "./lib/live";
import AlertsPage from "./pages/AlertsPage";
import CameraDetail from "./pages/CameraDetail";
import CameraWall from "./pages/CameraWall";
import Dashboard from "./pages/Dashboard";
import EventDetail from "./pages/EventDetail";
import EventFeed from "./pages/EventFeed";
import HealthPage from "./pages/HealthPage";
import RulesPage from "./pages/RulesPage";

function Root() {
  const { state, error, reload } = useAuth();

  if (state === "loading") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950">
        <Spinner label="Connecting to SENTINEL" />
      </div>
    );
  }
  if (state === "error") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
        <div className="w-full max-w-md">
          <ErrorPanel
            error={
              error
                ? new Error(error)
                : new Error("backend unreachable")
            }
            onRetry={reload}
          />
        </div>
      </div>
    );
  }
  if (state === "unauthenticated") return <LoginGate />;

  return (
    <LiveProvider>
      <Shell>
        <ErrorBoundary>
          <Routes>
            <Route index element={<Dashboard />} />
            <Route path="cameras" element={<CameraWall />} />
            <Route path="cameras/:cameraId" element={<CameraDetail />} />
            <Route path="events" element={<EventFeed />} />
            <Route path="events/:eventId" element={<EventDetail />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="rules" element={<RulesPage />} />
            <Route path="health" element={<HealthPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </Shell>
    </LiveProvider>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Root />
      </BrowserRouter>
    </AuthProvider>
  );
}

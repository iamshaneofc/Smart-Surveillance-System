import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, fetchPreviewFrame } from "../lib/api";
import { formatTimestamp, relativeTime } from "../lib/format";

type PreviewState = "loading" | "preview" | "unavailable" | "error";

const REFRESH_MS = 4_000;

/**
 * Development frame preview with honest labelling.
 *
 * The backend serves a buffered JPEG from the in-process pipeline
 * (`/cameras/{id}/preview.jpg`) - there is no media server, so this is never
 * called "live video". States: PREVIEW (frame shown, with capture time) or
 * UNAVAILABLE (no pipeline/buffer for this camera).
 */
export function CameraPreview({
  cameraId,
  className,
  refreshMs = REFRESH_MS,
}: {
  cameraId: string;
  className?: string;
  refreshMs?: number;
}) {
  const [state, setState] = useState<PreviewState>("loading");
  const [src, setSrc] = useState<string | null>(null);
  const [capturedAt, setCapturedAt] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const load = useCallback(
    async (signal: AbortSignal) => {
      try {
        const frame = await fetchPreviewFrame(cameraId, signal);
        if (signal.aborted) return;
        const url = URL.createObjectURL(frame.blob);
        if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = url;
        setSrc(url);
        setCapturedAt(frame.capturedAt);
        setError(null);
        setState("preview");
      } catch (err) {
        if (signal.aborted) return;
        if (err instanceof ApiError) {
          if (err.status === 409 || err.isNotFound) {
            setError(err.message);
            setState("unavailable");
            return;
          }
          setError(err.message);
          setState("error");
          return;
        }
        setError(String(err));
        setState("error");
      }
    },
    [cameraId],
  );

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    const tick = async () => {
      if (!active) return;
      await load(controller.signal);
    };
    void tick();
    const interval = window.setInterval(() => void tick(), refreshMs);
    return () => {
      active = false;
      controller.abort();
      window.clearInterval(interval);
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, [load, refreshMs]);

  if (state === "preview" && src) {
    return (
      <div className={`relative overflow-hidden bg-black ${className ?? ""}`}>
        <img
          src={src}
          alt={`Preview frame for ${cameraId}`}
          className="h-full w-full object-contain"
          data-testid={`preview-${cameraId}`}
        />
        <span className="absolute left-2 top-2 rounded bg-amber-950/90 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-amber-300 ring-1 ring-amber-700">
          Preview
        </span>
        <span className="absolute bottom-1.5 right-2 font-mono text-[10px] text-slate-400">
          {capturedAt ? `frame ${relativeTime(capturedAt)}` : ""}
        </span>
      </div>
    );
  }

  if (state === "unavailable" || state === "error") {
    return (
      <div
        className={`flex flex-col items-center justify-center gap-1 bg-slate-900 text-center ${className ?? ""}`}
        data-testid={`preview-unavailable-${cameraId}`}
      >
        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
          Preview unavailable
        </span>
        <span className="px-3 text-xs text-slate-600">
          {state === "unavailable"
            ? "No live pipeline is feeding this camera."
            : (error ?? "Preview request failed.")}
        </span>
      </div>
    );
  }

  return (
    <div className={`flex items-center justify-center bg-slate-900 ${className ?? ""}`}>
      <span className="text-xs text-slate-500">Loading preview…</span>
    </div>
  );
}

export function PreviewMeta({ capturedAt }: { capturedAt?: string | null }) {
  if (!capturedAt) return null;
  return (
    <span className="text-xs text-slate-500">{formatTimestamp(capturedAt)}</span>
  );
}

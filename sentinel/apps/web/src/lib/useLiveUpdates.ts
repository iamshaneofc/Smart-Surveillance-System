import { useEffect, useRef, useState } from "react";
import { openEventStream } from "../lib/api";

export type LiveMode = "live" | "polling" | "offline";

export interface LiveUpdates {
  mode: LiveMode;
  /** Bump: consumers should refetch their data. */
  tick: number;
}

const POLL_INTERVAL_MS = 10_000;
const RECONNECT_DELAY_MS = 8_000;

/**
 * Real-time updates with an explicit fallback ladder:
 *
 * 1. `live`    – SSE stream connected (server pushes `events.*`, `camera.health`,
 *    `alerts.updated`), tick on every message.
 * 2. `polling` – stream failed or dropped; tick every POLL_INTERVAL_MS and keep
 *    retrying the stream in the background.
 * 3. `offline` – stream fetch itself failed AND polls are erroring; the UI must
 *    show offline affordances instead of stale numbers.
 *
 * Uses fetch-based SSE because native EventSource cannot send X-API-Key.
 */
export function useLiveUpdates(
  options: { enabled?: boolean; onPollError?: () => void } = {},
): LiveUpdates {
  const { enabled = true } = options;
  const [mode, setMode] = useState<LiveMode>(enabled ? "polling" : "offline");
  const [tick, setTick] = useState(0);
  const modeRef = useRef(mode);
  modeRef.current = mode;

  const notify = useRef(() => setTick((n) => n + 1));
  const onPollError = useRef(options.onPollError);
  onPollError.current = options.onPollError;

  useEffect(() => {
    if (!enabled) {
      setMode("offline");
      return;
    }
    let disposed = false;
    let close: (() => void) | null = null;
    let reconnectTimer: number | undefined;
    let hasConnectedOnce = false;

    const scheduleReconnect = () => {
      if (disposed) return;
      reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
    };

    const connect = () => {
      if (disposed) return;
      close = openEventStream({
        onOpen: () => {
          if (disposed) return;
          hasConnectedOnce = true;
          setMode("live");
        },
        onMessage: () => {
          if (disposed) return;
          setMode("live");
          notify.current();
        },
        onClose: () => {
          if (disposed) return;
          setMode(hasConnectedOnce ? "polling" : "offline");
          scheduleReconnect();
        },
        onError: () => {
          if (disposed) return;
          setMode(hasConnectedOnce ? "polling" : "offline");
          scheduleReconnect();
        },
      });
    };

    connect();

    const poll = window.setInterval(() => {
      if (modeRef.current !== "live") notify.current();
    }, POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      close?.();
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      window.clearInterval(poll);
    };
  }, [enabled]);

  return { mode, tick };
}

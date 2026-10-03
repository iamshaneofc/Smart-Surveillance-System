import { createContext, useContext, type ReactNode } from "react";
import { useLiveUpdates, type LiveUpdates } from "./useLiveUpdates";

const LiveContext = createContext<LiveUpdates>({ mode: "offline", tick: 0 });

export function LiveProvider({
  children,
  enabled = true,
}: {
  children: ReactNode;
  enabled?: boolean;
}) {
  const live = useLiveUpdates({ enabled });
  return <LiveContext.Provider value={live}>{children}</LiveContext.Provider>;
}

export function useLive(): LiveUpdates {
  return useContext(LiveContext);
}

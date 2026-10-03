import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { ApiError, fetchMe, setApiKey } from "./api";
import type { WhoAmI } from "./types";

type AuthState = "loading" | "authenticated" | "unauthenticated" | "error";

interface AuthContextValue {
  state: AuthState;
  me: WhoAmI | null;
  error: string | null;
  login: (key: string) => Promise<void>;
  logout: () => void;
  can: (permission: string) => boolean;
  reload: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>("loading");
  const [me, setMe] = useState<WhoAmI | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    setState((prev) => (prev === "authenticated" ? prev : "loading"));
    fetchMe(controller.signal)
      .then((who) => {
        if (cancelled) return;
        setMe(who);
        setError(null);
        setState("authenticated");
      })
      .catch((err: unknown) => {
        if (cancelled || controller.signal.aborted) return;
        if (err instanceof ApiError && err.isUnauthorized) {
          setMe(null);
          setError(null);
          setState("unauthenticated");
          return;
        }
        setMe(null);
        setError(err instanceof Error ? err.message : String(err));
        setState("error");
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [attempt]);

  const login = useCallback(async (key: string) => {
    setApiKey(key);
    const who = await fetchMe();
    setMe(who);
    setError(null);
    setState("authenticated");
  }, []);

  const logout = useCallback(() => {
    setApiKey(null);
    setMe(null);
    setState("unauthenticated");
  }, []);

  const reload = useCallback(() => setAttempt((n) => n + 1), []);

  const can = useCallback(
    (permission: string) => (me ? me.permissions.includes(permission) : false),
    [me],
  );

  const value = useMemo(
    () => ({ state, me, error, login, logout, can, reload }),
    [state, me, error, login, logout, can, reload],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

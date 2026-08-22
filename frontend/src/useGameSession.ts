import { useCallback, useEffect, useRef, useState } from "react";
import {
  acknowledgeCheckpoint,
  ApiError,
  getGameView,
  resignGame,
  submitDecisionPlan,
  submitGameAction,
  submitSupplyPlan,
} from "./api";
import type { GameView } from "./types";

export function useGameSession(gameId: string, token: string) {
  const [view, setView] = useState<GameView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const fetching = useRef(false);
  const active = useRef(true);

  const refresh = useCallback(
    async (signal?: AbortSignal) => {
      if (fetching.current) return;
      fetching.current = true;
      try {
        const next = await getGameView(gameId, token, signal);
        if (active.current) {
          setView(next);
          setError(null);
        }
      } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        if (active.current) {
          setError(cause instanceof Error ? cause.message : "Unable to load this game");
        }
      } finally {
        fetching.current = false;
      }
    },
    [gameId, token],
  );

  useEffect(() => {
    active.current = true;
    const controller = new AbortController();
    void refresh(controller.signal);
    const timer = window.setInterval(() => void refresh(), 1_000);
    const refetch = () => void refresh();
    const onVisibility = () => {
      if (document.visibilityState === "visible") refetch();
    };
    window.addEventListener("focus", refetch);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      active.current = false;
      controller.abort();
      window.clearInterval(timer);
      window.removeEventListener("focus", refetch);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refresh]);

  const submit = useCallback(
    async (operation: (revision: number) => Promise<GameView>) => {
      if (!view || submitting) return null;
      setSubmitting(true);
      setError(null);
      try {
        const next = await operation(view.revision);
        setView(next);
        return next;
      } catch (cause) {
        if (cause instanceof ApiError && cause.status === 409) {
          await refresh();
        } else {
          setError(cause instanceof Error ? cause.message : "Action was not accepted");
        }
        return null;
      } finally {
        setSubmitting(false);
      }
    },
    [refresh, submitting, view],
  );

  const act = useCallback(
    (actionId: number) => submit((revision) => submitGameAction(gameId, token, actionId, revision)),
    [gameId, submit, token],
  );
  const supply = useCallback(
    (planId: string) => submit((revision) => submitSupplyPlan(gameId, token, planId, revision)),
    [gameId, submit, token],
  );
  const decision = useCallback(
    (planId: string) => submit((revision) => submitDecisionPlan(gameId, token, planId, revision)),
    [gameId, submit, token],
  );
  const acknowledge = useCallback(
    (checkpointId: string) => submit((revision) => acknowledgeCheckpoint(gameId, token, checkpointId, revision)),
    [gameId, submit, token],
  );

  const resign = useCallback(async () => {
    if (!view || submitting) return false;
    setSubmitting(true);
    setError(null);
    try {
      await resignGame(gameId, token, view.revision);
      return true;
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 409) await refresh();
      else setError(cause instanceof Error ? cause.message : "Resignation was not accepted");
      return false;
    } finally {
      setSubmitting(false);
    }
  }, [gameId, refresh, submitting, token, view]);

  return { view, error, submitting, act, supply, decision, acknowledge, resign, refresh };
}

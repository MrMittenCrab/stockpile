import type {
  ApiFailure,
  CreateGameRequest,
  CreateGameResponse,
  GameView,
  SetupResponse,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string,
  ) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let failure: ApiFailure | undefined;
    try {
      failure = (await response.json()) as ApiFailure;
    } catch {
      // The server contract is JSON, but retain a readable network failure.
    }
    throw new ApiError(
      failure?.error.message ?? `Request failed (${response.status})`,
      response.status,
      failure?.error.code ?? "request_failed",
    );
  }
  return (await response.json()) as T;
}

export const getSetup = (signal?: AbortSignal) =>
  request<SetupResponse>("/api/v2/setup", { signal });

export const createGame = (body: CreateGameRequest) =>
  request<CreateGameResponse>("/api/v2/games", {
    method: "POST",
    body: JSON.stringify(body),
  });

const authorized = (token: string): HeadersInit => ({
  Authorization: `Bearer ${token}`,
});

export const getGameView = (
  gameId: string,
  token: string,
  signal?: AbortSignal,
) =>
  request<GameView>(`/api/v2/games/${encodeURIComponent(gameId)}/view`, {
    headers: authorized(token),
    signal,
  });

export const submitGameAction = (
  gameId: string,
  token: string,
  actionId: number,
  expectedRevision: number,
) =>
  request<GameView>(`/api/v2/games/${encodeURIComponent(gameId)}/actions`, {
    method: "POST",
    headers: authorized(token),
    body: JSON.stringify({ action_id: actionId, expected_revision: expectedRevision }),
  });

export const submitSupplyPlan = (
  gameId: string,
  token: string,
  planId: string,
  expectedRevision: number,
) =>
  request<GameView>(`/api/v2/games/${encodeURIComponent(gameId)}/supply`, {
    method: "POST",
    headers: authorized(token),
    body: JSON.stringify({ plan_id: planId, expected_revision: expectedRevision }),
  });

export const acknowledgeCheckpoint = (
  gameId: string,
  token: string,
  checkpointId: string,
  expectedRevision: number,
) =>
  request<GameView>(`/api/v2/games/${encodeURIComponent(gameId)}/acknowledgements`, {
    method: "POST",
    headers: authorized(token),
    body: JSON.stringify({ checkpoint_id: checkpointId, expected_revision: expectedRevision }),
  });

export const seatStorageKey = (gameId: string) => `stockpile.seatToken:${gameId}`;

export function claimFragmentToken(gameId: string): string | null {
  const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const candidate = fragment.get("seat");
  if (candidate) {
    sessionStorage.setItem(seatStorageKey(gameId), candidate);
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
    return candidate;
  }
  return sessionStorage.getItem(seatStorageKey(gameId));
}

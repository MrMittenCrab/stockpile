import { useMemo } from "react";
import { claimFragmentToken } from "./api";
import { GamePage } from "./components/GamePage";
import { SetupPage } from "./SetupPage";
import "./global.css";

export function App() {
  const match = window.location.pathname.match(/^\/game\/([^/]+)\/?$/);
  const gameId = match ? decodeURIComponent(match[1]) : null;
  const token = useMemo(() => gameId ? claimFragmentToken(gameId) : null, [gameId]);
  if (!gameId) return <SetupPage />;
  if (!token) {
    return <main className="missing-seat"><small>FIXED SEAT REQUIRED</small><h1>This tab has no seat token.</h1><p>Use one of the seat links created at setup.</p><a href="/">Create a new game</a></main>;
  }
  return <GamePage gameId={gameId} token={token} />;
}

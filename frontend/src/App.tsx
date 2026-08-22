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
    return <main className="missing-seat"><h1>NO SEAT</h1><p>Open a seat link.</p><a href="/">HOME</a></main>;
  }
  return <GamePage gameId={gameId} token={token} />;
}

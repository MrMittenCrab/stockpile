import { useMemo } from "react";
import { claimFragmentToken } from "./api";
import { GamePage } from "./components/GamePage";
import { TextButton } from "./components/Primitives";
import { SetupPage } from "./SetupPage";
import "./global.css";

export function App() {
  const match = window.location.pathname.match(/^\/game\/([^/]+)\/?$/);
  const gameId = match ? decodeURIComponent(match[1]) : null;
  const token = useMemo(() => gameId ? claimFragmentToken(gameId) : null, [gameId]);
  if (!gameId) return <SetupPage />;
  if (!token) {
    return <main className="missing-seat"><span>GAME UNAVAILABLE</span><TextButton onClick={() => window.location.assign("/")}>NEW GAME</TextButton></main>;
  }
  return <GamePage gameId={gameId} token={token} />;
}

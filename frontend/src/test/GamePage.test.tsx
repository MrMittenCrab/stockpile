import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it } from "vitest";
import { GamePage } from "../components/GamePage";
import gameCss from "../components/Game.module.css?raw";
import type { GameView } from "../types";
import { gameView, server } from "./server";

afterEach(cleanup);

describe("fixed-seat Bauhaus game surface", () => {
  it("keeps the flat two-size visual contract", () => {
    expect(gameCss).not.toMatch(/gradient|box-shadow|text-shadow|border-radius|font-style\s*:\s*italic/i);
    expect(new Set(Array.from(gameCss.matchAll(/font-size:\s*var\(--([a-z-]+-size)/g), (match) => match[1]))).toEqual(new Set(["primary-size", "secondary-size"]));
    expect(new Set(Array.from(gameCss.matchAll(/font-weight:\s*([^;}]+)/g), (match) => match[1].trim()))).toEqual(new Set(["400"]));
  });

  it("renders the authoritative prices, pile count, movement, and bid action", async () => {
    let submitted: unknown;
    server.use(http.post("/api/v1/games/unusual/actions", async ({ request }) => {
      submitted = await request.json();
      return HttpResponse.json({ ...gameView, revision: 8, legal_actions: [] });
    }));

    render(<GamePage gameId="unusual" token="seat-secret" />);

    const market = await screen.findByLabelText("Market");
    expect(within(market).getByText("47")).toBeInTheDocument();
    expect(await within(market).findByText("↑2")).toBeInTheDocument();
    expect(await within(market).findByText("↓1")).toBeInTheDocument();
    expect(within(screen.getByLabelText("Stockpiles")).getAllByRole("article")).toHaveLength(7);
    expect(screen.getByRole("button", { name: "Bid 37K" })).toHaveTextContent("37K");
    expect(screen.queryByText("$5K")).not.toBeInTheDocument();
    expect(screen.queryByText(/Fees due/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Private pile knowledge")).toHaveTextContent("S2");
    expect(screen.getByLabelText("Stockpile 7").className).toContain("stockpileSelected");

    await userEvent.click(screen.getByRole("button", { name: "Bid 37K" }));
    await waitFor(() => expect(submitted).toEqual({ action_id: 9123, expected_revision: 7 }));
  });

  it("uses all six blue geometric pattern identities and real card layers", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    await screen.findByLabelText("Stockpiles");

    const patterns = new Set(Array.from(document.querySelectorAll("[data-stock-pattern]")).map((node) => node.getAttribute("data-stock-pattern")));
    expect(patterns).toEqual(new Set(["matrix", "ledger", "molecular", "chevron", "crosshatch", "wave"]));
    const expectedStackCards = gameView.stockpiles.reduce((count, pile) => count + pile.visible_cards.length + pile.hidden_cards.length, 0);
    expect(document.querySelectorAll("[data-stack-card]")).toHaveLength(expectedStackCards);
    expect(document.querySelectorAll('[data-card-scale="stockpile"]')).toHaveLength(expectedStackCards);
    expect(within(screen.getByLabelText("Stockpile 1")).queryByText("COSMIC")).not.toBeInTheDocument();
    expect(within(screen.getByLabelText("Stockpile 1")).getByText("-$4K")).toBeInTheDocument();
  });

  it("keeps hidden cards blank and omits chat, history, player colors, and analysis", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    const hiddenCards = await screen.findAllByLabelText("Hidden card");
    expect(hiddenCards.length).toBeGreaterThan(0);
    for (const card of hiddenCards) expect(card).toHaveTextContent("");
    expect(screen.queryByText("CHAT")).not.toBeInTheDocument();
    expect(screen.queryByText(/history/i)).not.toBeInTheDocument();

    const text = document.body.textContent?.toLowerCase() ?? "";
    for (const forbidden of ["deep cfr", "expected value", "exploitability", "policy percentage", "advantage value"]) expect(text).not.toContain(forbidden);
  });

  it("renders private and public information with one shared card grammar", async () => {
    const view: GameView = {
      ...gameView,
      private: {
        ...gameView.private,
        market_information: [
          { visibility: "private", source: "dealt", card: { visibility: "visible", kind: "company_forecast", company_id: 0, company: "Cosmic Computers", forecast: 3 } },
          { visibility: "public", source: "revealed", card: { visibility: "visible", kind: "company_forecast", company_id: 4, company: "Stanford Steel", forecast: "DIVIDEND" } },
          { visibility: "hidden", source: "unknown", card: { visibility: "hidden" } },
        ],
      },
    };
    server.use(http.get("/api/v1/games/info/view", () => HttpResponse.json(view)));
    render(<GamePage gameId="info" token="seat-secret" />);

    const privateInfo = await screen.findByLabelText("Private information");
    const publicInfo = screen.getByLabelText("Public information");
    expect(within(privateInfo).getByLabelText("COSMIC company card")).toBeInTheDocument();
    expect(within(privateInfo).getByLabelText("Price up 3")).toHaveTextContent("↑3");
    expect(within(publicInfo).getByLabelText("STANFORD company card")).toBeInTheDocument();
    expect(within(publicInfo).getByLabelText("Dividend")).toHaveTextContent("$$");
    expect(within(publicInfo).getAllByLabelText("Hidden card")).toHaveLength(2);
  });

  it("renders authoritative Impact direction and movement without deriving Boom or Bust", async () => {
    const impact = { visibility: "visible" as const, kind: "action" as const, effect: "shift", direction: "up" as const, movement: 9 };
    const view: GameView = {
      ...gameView,
      phase: "action",
      private: { ...gameView.private, available_action_cards: [impact] },
      pending_decision: { ...gameView.pending_decision, kind: "action_card", prompt: "Select impact", selected_stockpile_id: null },
      legal_actions: [{ action_id: 44, control: "action_card", label: "Use shift", target_id: "action:shift", amount: null, placement_visibility: null, sale_preview: null }],
    };
    server.use(http.get("/api/v1/games/impact/view", () => HttpResponse.json(view)));
    render(<GamePage gameId="impact" token="seat-secret" />);

    const action = await screen.findByRole("button", { name: "Use shift" });
    expect(action).toHaveTextContent("↑9");
    expect(action).not.toHaveTextContent(/boom|bust/i);
  });

  it("maps engine state to the short Rebid dock fragment", async () => {
    const view: GameView = {
      ...gameView,
      players: gameView.players.map((player) => player.player_id === gameView.viewer.player_id ? {
        ...player,
        bid_markers: player.bid_markers.map((marker, index) => index === 0 ? { ...marker, status: "rebidding" as const, stockpile_id: null, bid: null } : marker),
      } : player),
      pending_decision: { ...gameView.pending_decision, kind: "bid_pile", prompt: "This deliberately verbose engine prompt is not displayed", selected_stockpile_id: null },
      legal_actions: [],
    };
    server.use(http.get("/api/v1/games/rebid/view", () => HttpResponse.json(view)));
    render(<GamePage gameId="rebid" token="seat-secret" />);

    const dock = await screen.findByLabelText("Action dock");
    expect(dock).toHaveTextContent("Rebid");
    expect(dock).not.toHaveTextContent("deliberately verbose");
  });
});

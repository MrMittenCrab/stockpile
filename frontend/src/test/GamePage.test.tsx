import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it } from "vitest";
import { GamePage } from "../components/GamePage";
import gameCss from "../components/Game.module.css?raw";
import primitiveCss from "../components/Primitives.module.css?raw";
import globalCss from "../global.css?raw";
import type { GameView } from "../types";
import { gameView, server } from "./server";

afterEach(cleanup);

describe("disciplined Stockpile Lite game surface", () => {
  it("uses the canonical flat two-size visual grammar", () => {
    const css = `${globalCss}\n${primitiveCss}\n${gameCss}`;
    expect(css).not.toMatch(/gradient|box-shadow|text-shadow|border-radius\s*:\s*[^0]|font-style\s*:\s*italic|ochre|opacity\s*:/i);
    expect(new Set(Array.from(css.matchAll(/font-size:\s*var\(--([a-z-]+-size)/g), (match) => match[1]))).toEqual(new Set(["primary-size", "secondary-size"]));
    expect(new Set(Array.from(css.matchAll(/font-weight:\s*([^;}]+)/g), (match) => match[1].trim()))).toEqual(new Set(["400"]));
    expect(globalCss).toContain("--blue: #002fa7");
    expect(globalCss).toContain("--grey: #70747a");
    expect(globalCss).toContain("background: #ffffff");
  });

  it("renders server units, financial labels, unusual prices, piles, and bids", async () => {
    let submitted: unknown;
    server.use(http.post("/api/v2/games/unusual/actions", async ({ request }) => {
      submitted = await request.json();
      return HttpResponse.json({ ...gameView, revision: 8, legal_actions: [] });
    }));
    render(<GamePage gameId="unusual" token="seat-secret" />);

    const market = await screen.findByLabelText("Market");
    expect(within(market).getByText("$47 / SHARE")).toBeInTheDocument();
    expect(within(market).getByText("$11 / SHARE")).toBeInTheDocument();
    expect(within(market).queryByText("↑2")).not.toBeInTheDocument();
    expect(within(screen.getByLabelText("Stockpiles")).getAllByRole("article")).toHaveLength(5);
    expect(within(screen.getByLabelText("Portfolio")).getByText("3K")).toBeInTheDocument();

    const players = screen.getByLabelText("Players");
    expect(within(players).getAllByText("CASH")).toHaveLength(2);
    expect(within(players).getByText("POSITION")).toBeInTheDocument();
    expect(within(players).getByText("$141K")).toBeInTheDocument();
    expect(within(players).getByText("−$9K")).toBeInTheDocument();
    expect(within(players).getByText("+$6K")).toBeInTheDocument();
    expect(within(players).queryAllByText("POSITION")).toHaveLength(1);

    const bid = screen.getByRole("button", { name: "Bid 37K" });
    expect(bid).toHaveTextContent("$37K");
    await userEvent.click(bid);
    await waitFor(() => expect(submitted).toEqual({ action_id: 9123, expected_revision: 7 }));
  });

  it("preserves bottom-to-top ordering and reveals remembered identity only while expanded", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    const piles = await screen.findByLabelText("Stockpiles");
    const first = within(piles).getByLabelText("Stockpile 1");
    const layers = first.querySelectorAll("[data-stack-card]");
    expect(Array.from(layers).map((layer) => layer.getAttribute("data-stack-order"))).toEqual(["0", "1", "2"]);
    expect(Array.from(layers).map((layer) => (layer as HTMLElement).style.zIndex)).toEqual(["1", "2", "3"]);

    const remembered = within(piles).getByLabelText("Stockpile 2");
    expect(within(remembered).getByLabelText("Hidden card")).toBeInTheDocument();
    expect(within(remembered).queryByText("FACE DOWN")).not.toBeInTheDocument();
    const inspect = within(remembered).getByRole("button", { name: "Expand stockpile 2" });
    await userEvent.click(inspect);
    expect(inspect).toHaveAttribute("aria-expanded", "true");
    expect(within(remembered).getByLabelText("EPIC stock 1K shares")).toBeInTheDocument();
    expect(within(remembered).getByText("FACE DOWN")).toBeInTheDocument();
    await userEvent.click(within(remembered).getByRole("button", { name: "Collapse stockpile 2" }));
    expect(within(remembered).getByLabelText("Hidden card")).toBeInTheDocument();
  });

  it("keeps pile inspection separate from a legal pile action", async () => {
    let submissions = 0;
    const view: GameView = {
      ...gameView,
      pending_decision: { ...gameView.pending_decision, kind: "bid_pile", selected_stockpile_id: null },
      legal_actions: [{ action_id: 55, control: "stockpile", label: "Select stockpile 1", target_id: "stockpile:0", amount_thousands: null, direction: null, sale_preview: null }],
    };
    server.use(
      http.get("/api/v2/games/separate/view", () => HttpResponse.json(view)),
      http.post("/api/v2/games/separate/actions", async () => {
        submissions += 1;
        return HttpResponse.json({ ...view, revision: 8 });
      }),
    );
    render(<GamePage gameId="separate" token="seat-secret" />);
    await userEvent.click(await screen.findByRole("button", { name: "Expand stockpile 1" }));
    expect(submissions).toBe(0);
    await userEvent.click(screen.getByRole("button", { name: "Select stockpile 1" }));
    await waitFor(() => expect(submissions).toBe(1));
  });

  it("arranges both Supply cards locally and submits one opaque plan on CONFIRM", async () => {
    let submitted: unknown;
    const supplyView: GameView = {
      ...gameView,
      phase: "supply",
      phase_step: "supply_choose_card",
      pending_decision: { kind: "supply", prompt: "Place current pair", selected_stockpile_id: null, selected_action_effect: null, company_id: null },
      legal_actions: [],
      supply_batch: {
        cards: [
          { card_ref: "card-a", card: { visibility: "visible", kind: "stock", company_id: 3, company: "American Automotive", shares_thousands: 1 } },
          { card_ref: "card-b", card: { visibility: "visible", kind: "trading_fee", cash_effect_thousands: -7 } },
        ],
        plans: [
          { plan_id: "plan-opaque", placements: [
            { card_ref: "card-a", stockpile_id: 0, visibility: "face_up" },
            { card_ref: "card-b", stockpile_id: 1, visibility: "face_down" },
          ] },
          { plan_id: "alternate", placements: [
            { card_ref: "card-a", stockpile_id: 1, visibility: "face_down" },
            { card_ref: "card-b", stockpile_id: 0, visibility: "face_up" },
          ] },
          { plan_id: "swapped", placements: [
            { card_ref: "card-a", stockpile_id: 0, visibility: "face_down" },
            { card_ref: "card-b", stockpile_id: 1, visibility: "face_up" },
          ] },
        ],
      },
    };
    server.use(
      http.get("/api/v2/games/supply/view", () => HttpResponse.json(supplyView)),
      http.post("/api/v2/games/supply/supply", async ({ request }) => {
        submitted = await request.json();
        return HttpResponse.json({ ...supplyView, revision: 8, supply_batch: null, pending_decision: { ...supplyView.pending_decision, kind: "waiting" } });
      }),
    );
    render(<GamePage gameId="supply" token="seat-secret" />);

    const first = await screen.findByRole("button", { name: "Supply card card-a" });
    expect(screen.getByRole("button", { name: "Supply card card-b" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "CONFIRM" })).not.toBeInTheDocument();
    await userEvent.click(first);
    await userEvent.click(screen.getByRole("button", { name: "FACE UP" }));
    await userEvent.click(screen.getByRole("button", { name: "Place selected card in stockpile 1" }));
    await userEvent.click(screen.getByRole("button", { name: "Supply card card-b" }));
    await userEvent.click(screen.getByRole("button", { name: "FACE DOWN" }));
    await userEvent.click(screen.getByRole("button", { name: "Place selected card in stockpile 2" }));
    expect(submitted).toBeUndefined();

    const confirm = screen.getByRole("button", { name: "CONFIRM" });
    expect(confirm).toBeEnabled();
    expect(confirm).toHaveAttribute("data-plan-id", "plan-opaque");
    await userEvent.click(first);
    await userEvent.click(screen.getByRole("button", { name: "FACE DOWN" }));
    expect(first).toHaveAttribute("data-assigned-visibility", "face_down");
    expect(screen.getByRole("button", { name: "Supply card card-b" })).toHaveAttribute("data-assigned-visibility", "face_up");
    expect(confirm).toHaveAttribute("data-plan-id", "swapped");
    await userEvent.click(screen.getByRole("button", { name: "FACE UP" }));
    expect(confirm).toHaveAttribute("data-plan-id", "plan-opaque");
    await userEvent.click(confirm);
    await waitFor(() => expect(submitted).toEqual({ plan_id: "plan-opaque", expected_revision: 7 }));
  });

  it("shows checkpoint deltas in PLAYERS and only acknowledges through CONTINUE", async () => {
    let submitted: unknown;
    const checkpointView: GameView = {
      ...gameView,
      checkpoint: { checkpoint_id: "checkpoint-secret", kind: "demand_result", round: 3 },
      pending_decision: { ...gameView.pending_decision, kind: "acknowledge" },
      legal_actions: [{ action_id: 1, control: "generic", label: "Should not render", target_id: null, amount_thousands: null, direction: null, sale_preview: null }],
    };
    server.use(
      http.get("/api/v2/games/result/view", () => HttpResponse.json(checkpointView)),
      http.post("/api/v2/games/result/acknowledgements", async ({ request }) => {
        submitted = await request.json();
        return HttpResponse.json({ ...checkpointView, revision: 8, checkpoint: null });
      }),
    );
    render(<GamePage gameId="result" token="seat-secret" />);
    expect(await screen.findByText("DEMAND RESULT")).toBeInTheDocument();
    expect(screen.queryByText("Should not render")).not.toBeInTheDocument();
    const button = screen.getByRole("button", { name: "CONTINUE" });
    expect(button).toHaveAttribute("data-checkpoint-kind", "demand_result");
    await userEvent.click(button);
    await waitFor(() => expect(submitted).toEqual({ checkpoint_id: "checkpoint-secret", expected_revision: 7 }));
  });

  it("presents the completed Movement batch, including a dividend reveal, until CONTINUE", async () => {
    const roundResult: GameView = {
      ...gameView,
      phase: "ROUND_RESULT",
      checkpoint: { checkpoint_id: "round-result", kind: "round_result", round: 3 },
      pending_decision: { ...gameView.pending_decision, kind: "acknowledge" },
      legal_actions: [],
      recent_events: [
        ...gameView.recent_events,
        { event_id: 3, event_type: "market_reveal", cause: "market_forecast", round: 3, company_id: 4, prior_price_dollars_per_share: 5, price_delta: null, resulting_price_dollars_per_share: 5, forecast: "DIVIDEND", cash_effect_thousands: 2, direction: null },
      ],
    };
    server.use(http.get("/api/v2/games/movement/view", () => HttpResponse.json(roundResult)));
    render(<GamePage gameId="movement" token="seat-secret" />);
    const market = await screen.findByLabelText("Market");
    expect(within(market).getByText("↑2")).toBeInTheDocument();
    expect(within(market).getByText("↓1")).toBeInTheDocument();
    expect(within(market).queryByText("+$")).not.toBeInTheDocument();
    expect(within(screen.getByLabelText("Public information")).getByLabelText("Cash increases by 2K")).toHaveTextContent("+$2K");
  });

  it("uses arrows for price changes and currency signs for cash changes", async () => {
    const informationView: GameView = {
      ...gameView,
      private: {
        ...gameView.private,
        market_information: [
          { visibility: "private", card: { visibility: "visible", kind: "company_forecast", company_id: 0, company: "Cosmic Computers", forecast: 3, cash_effect_thousands: null } },
          { visibility: "public", card: { visibility: "visible", kind: "company_forecast", company_id: 4, company: "Stanford Steel", forecast: "DIVIDEND", cash_effect_thousands: 2 } },
          { visibility: "hidden", card: { visibility: "hidden" } },
        ],
      },
    };
    server.use(http.get("/api/v2/games/info/view", () => HttpResponse.json(informationView)));
    render(<GamePage gameId="info" token="seat-secret" />);
    expect(await screen.findByLabelText("Price up 3")).toHaveTextContent("↑3");
    expect(screen.getByLabelText("Cash increases by 2K")).toHaveTextContent("+$2K");
    expect(screen.getByLabelText("Cash decreases by 4K")).toHaveTextContent("−$4K");
    expect(document.body).not.toHaveTextContent("$$");
  });

  it("keeps the two server-authored dividend choices distinct", async () => {
    const dividendView: GameView = {
      ...gameView,
      phase: "movement",
      phase_step: "dividend_claim",
      pending_decision: { ...gameView.pending_decision, kind: "dividend_claim" },
      legal_actions: [
        { action_id: 300, control: "dividend", label: "Waive dividend", target_id: null, amount_thousands: null, direction: null, sale_preview: null },
        { action_id: 301, control: "dividend", label: "Claim dividend", target_id: null, amount_thousands: null, direction: null, sale_preview: null },
      ],
    };
    server.use(http.get("/api/v2/games/dividend/view", () => HttpResponse.json(dividendView)));
    render(<GamePage gameId="dividend" token="seat-secret" />);
    expect(await screen.findByRole("button", { name: "Waive dividend" })).toHaveTextContent("WAIVE DIVIDEND");
    expect(screen.getByRole("button", { name: "Claim dividend" })).toHaveTextContent("CLAIM DIVIDEND");
  });

  it("contains no multiplayer, analysis, chat, or hidden-card metadata UI", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    const hidden = await screen.findAllByLabelText("Hidden card");
    for (const card of hidden) expect(card).toHaveTextContent("");
    const text = document.body.textContent?.toLowerCase() ?? "";
    for (const forbidden of ["chat", "history", "seat", "deep cfr", "recommendation", "expected value", "exploitability", "policy", "advantage", "s3", "s4"]) {
      expect(text).not.toContain(forbidden);
    }
  });
});

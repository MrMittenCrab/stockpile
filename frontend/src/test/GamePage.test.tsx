import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { seatStorageKey } from "../api";
import { GamePage } from "../components/GamePage";
import gameCss from "../components/Game.module.css?raw";
import primitiveCss from "../components/Primitives.module.css?raw";
import globalCss from "../global.css?raw";
import type { GameView } from "../types";
import { gameView, server } from "./server";

afterEach(cleanup);

function action(id: number, label: string) {
  return { action_id: id, control: "dividend" as const, label, target_id: null, amount_thousands: null, direction: null, sale_preview: null };
}

describe("disciplined Stockpile Trainer game surface", () => {
  it("uses only the canonical flat two-size visual grammar and exact object sizes", () => {
    const css = `${globalCss}\n${primitiveCss}\n${gameCss}`;
    expect(css).not.toMatch(/gradient|box-shadow|text-shadow|border-radius\s*:\s*[^0]|font-style\s*:\s*italic|ochre|opacity\s*:/i);
    expect(new Set(Array.from(css.matchAll(/font-size:\s*var\(--([a-z-]+-size)/g), (match) => match[1]))).toEqual(new Set(["primary-size", "secondary-size"]));
    expect(new Set(Array.from(css.matchAll(/font-weight:\s*([^;}]+)/g), (match) => match[1].trim()))).toEqual(new Set(["400"]));
    expect(globalCss).toContain("--blue: #002fa7");
    expect(globalCss).toContain("--grey: #70747a");
    expect(primitiveCss).toMatch(/\.stockpile\s*\{\s*width:\s*104px;\s*height:\s*139px;/);
    expect(primitiveCss).toMatch(/\.active,[^}]*\.portfolio,[^}]*\.information\s*\{\s*width:\s*54px;\s*height:\s*72px;/s);
    expect(gameCss).toContain(".selected { transform: translateY(-4px); }");
    expect(gameCss).toMatch(/\.hiddenCard\s*\{[^}]*background:\s*var\(--blue\)/s);
    expect(gameCss).toMatch(/\[data-card-scale="portfolio"\] \.cardValue,\s*\[data-card-scale="active"\] \.cardValue,\s*\[data-card-scale="information"\] \.cardValue\s*\{[^}]*right:\s*3px;[^}]*bottom:\s*2px;[^}]*font-size:\s*var\(--secondary-size\)/s);
    expect(gameCss).toMatch(/\.stack\s*\{[^}]*width:\s*104px;[^}]*height:\s*139px;/s);
    expect(gameCss).not.toContain("--stack-count");
    expect(gameCss).toContain("grid-template-columns: minmax(0, 74px) minmax(0, 1fr)");
    expect(gameCss).toMatch(/\.dockControls\s*\{[^}]*gap:\s*8px;/s);
    expect(gameCss).toContain("grid-template-columns: 20px minmax(0, 1fr) 12ch 4ch");
    expect(gameCss).toContain("grid-template-columns: 8ch 6ch 7ch");
    expect(gameCss).not.toMatch(/\.marketCompany \.positive,[^}]*grid-column/s);
  });

  it("renders explicit units and keeps every fact in its canonical section", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    const market = await screen.findByLabelText("Market");
    expect(within(market).getByText("$47 / SHARE")).toBeInTheDocument();
    expect(within(market).getByText("$11 / SHARE")).toBeInTheDocument();
    expect(within(market).getByText("↑2")).toHaveAttribute("data-market-price-delta", "0");
    expect(within(market).getByText("↓1")).toHaveAttribute("data-market-price-delta", "1");
    expect(market.querySelectorAll("[data-market-delta-slot]")).toHaveLength(6);
    expect(market.querySelector('[data-market-delta-slot="2"]')).toBeEmptyDOMElement();
    expect(within(screen.getByLabelText("Stockpiles")).getAllByRole("article")).toHaveLength(5);
    expect(within(screen.getByLabelText("Portfolio")).getByText("3K")).toBeInTheDocument();

    const players = screen.getByLabelText("Players");
    expect(within(players).getAllByText("CASH")).toHaveLength(2);
    expect(within(players).getByText("POSITION")).toBeInTheDocument();
    expect(within(players).getByText("$141K")).toBeInTheDocument();
    expect(within(players).getByText("−$9K")).toBeInTheDocument();
    expect(within(players).getByText("+$6K")).toBeInTheDocument();
    expect(players.querySelectorAll("[data-player-value-slot]")).toHaveLength(3);
    expect(players.querySelectorAll("[data-player-delta-slot]")).toHaveLength(3);
    expect(within(players).queryAllByText("POSITION")).toHaveLength(1);
    expect(screen.getByLabelText("Research")).toBeInTheDocument();
    expect(screen.queryByText("PUBLIC")).not.toBeInTheDocument();
    expect(screen.queryByText("PRIVATE")).not.toBeInTheDocument();
    const controls = screen.getByLabelText("Action dock").querySelectorAll("[data-dock-control]");
    expect(controls).toHaveLength(2);
    expect(screen.getByRole("button", { name: "BACK" })).toBeDisabled();
  });

  it("collapses ordered stacks to white/blue edges and reveals remembered cards only on double-click", async () => {
    const user = userEvent.setup();
    render(<GamePage gameId="unusual" token="seat-secret" />);
    const field = await screen.findByLabelText("Stockpiles");
    const first = within(field).getByLabelText("Stockpile 1");
    const layers = first.querySelectorAll("[data-stack-card]");
    expect(Array.from(layers).map((layer) => layer.getAttribute("data-stack-order"))).toEqual(["0", "1", "2"]);
    expect(Array.from(layers).map((layer) => layer.hasAttribute("data-stack-bottom"))).toEqual([true, false, false]);
    expect(Array.from(layers).map((layer) => layer.hasAttribute("data-stack-top"))).toEqual([false, false, true]);
    expect(Array.from(layers).map((layer) => layer.getAttribute("data-card-edge"))).toEqual(["white", "blue", null]);
    expect(Array.from(layers).map((layer) => (layer as HTMLElement).style.zIndex)).toEqual(["1", "2", "3"]);
    expect(Array.from(layers).map((layer) => (layer as HTMLElement).style.getPropertyValue("--stack-index"))).toEqual(["0", "1", "2"]);

    const remembered = within(field).getByLabelText("Stockpile 2");
    const inspect = remembered.querySelector("[data-stack-inspect]") as HTMLElement;
    expect(inspect).toHaveAttribute("aria-expanded", "false");
    expect(within(remembered).getByLabelText("Hidden card")).toBeInTheDocument();
    await user.dblClick(inspect);
    expect(inspect).toHaveAttribute("aria-expanded", "true");
    expect(within(remembered).getByLabelText("EPIC stock 1K shares")).toBeInTheDocument();
    expect(within(remembered).getByText("FACE DOWN")).toBeInTheDocument();
    await user.dblClick(inspect);
    expect(inspect).toHaveAttribute("aria-expanded", "false");
    inspect.focus();
    await user.keyboard("{Enter}");
    expect(inspect).toHaveAttribute("aria-expanded", "true");
    expect(inspect).toHaveAttribute("aria-keyshortcuts", "Enter Space");
  });

  it("stages a Demand pile and bid, then commits one opaque decision plan", async () => {
    let submitted: unknown;
    server.use(http.post("/api/v2/games/unusual/decisions", async ({ request }) => {
      submitted = await request.json();
      return HttpResponse.json({ ...gameView, revision: 8, decision_batch: null, pending_decision: { ...gameView.pending_decision, kind: "waiting" } });
    }));
    const user = userEvent.setup();
    render(<GamePage gameId="unusual" token="seat-secret" />);
    const targetPile = await screen.findByLabelText("Stockpile 5");
    const inspect = targetPile.querySelector("[data-stack-inspect]") as HTMLElement;
    inspect.focus();
    await user.keyboard("{Shift>}{Enter}{/Shift}");
    expect(inspect).toHaveAttribute("aria-expanded", "true");
    expect(screen.queryByRole("button", { name: "$37K" })).not.toBeInTheDocument();
    await user.keyboard("{Enter}");
    expect(await screen.findByRole("button", { name: "$37K" })).toBeInTheDocument();
    const undo = screen.getByRole("button", { name: "UNDO" });
    expect(undo).toHaveAttribute("data-context-action", "undo");
    await user.click(undo);
    expect(screen.queryByRole("button", { name: "$37K" })).not.toBeInTheDocument();
    expect(submitted).toBeUndefined();
    const target = within(targetPile).getByRole("button", { name: "Select stockpile" });
    await user.click(target);
    const bid = await screen.findByRole("button", { name: "$37K" });
    await user.click(bid);
    const confirm = screen.getByRole("button", { name: "CONFIRM" });
    expect(confirm).toHaveAttribute("data-plan-id", "bid-37");
    expect(submitted).toBeUndefined();
    await user.click(confirm);
    await waitFor(() => expect(submitted).toEqual({ plan_id: "bid-37", expected_revision: 7 }));
  });

  it("stages both Supply cards independently, whites tentative copies, supports precise undo, and confirms once", async () => {
    let submitted: unknown;
    const supplyView: GameView = {
      ...gameView,
      phase: "supply",
      phase_step: "supply",
      pending_decision: { kind: "supply", prompt: "Place current pair", selected_stockpile_id: null, selected_action_effect: null, company_id: null },
      legal_actions: [],
      decision_batch: null,
      stockpiles: gameView.stockpiles.map((pile) => ({ ...pile, cards_bottom_to_top: [], bid: null })),
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
          { plan_id: "swapped-face", placements: [
            { card_ref: "card-a", stockpile_id: 0, visibility: "face_down" },
            { card_ref: "card-b", stockpile_id: 1, visibility: "face_up" },
          ] },
          { plan_id: "swapped-pile", placements: [
            { card_ref: "card-a", stockpile_id: 1, visibility: "face_up" },
            { card_ref: "card-b", stockpile_id: 0, visibility: "face_down" },
          ] },
          { plan_id: "swapped-both", placements: [
            { card_ref: "card-a", stockpile_id: 1, visibility: "face_down" },
            { card_ref: "card-b", stockpile_id: 0, visibility: "face_up" },
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
    const user = userEvent.setup();
    render(<GamePage gameId="supply" token="seat-secret" />);

    const sourceA = await screen.findByRole("button", { name: "Supply card card-a" });
    const firstEmpty = within(screen.getByLabelText("Stockpile 1")).getByLabelText("Empty stockpile");
    expect(firstEmpty).toHaveAttribute("data-card-scale", "stockpile");
    expect(firstEmpty.closest("[data-empty-stockpile]")).toBeInTheDocument();
    await user.click(sourceA);
    await user.click(within(screen.getByLabelText("Stockpile 1")).getByRole("button", { name: "Select stockpile" }));
    await user.click(await screen.findByRole("button", { name: "FACE UP" }));
    expect(sourceA).toHaveAttribute("data-white-out", "true");
    expect(document.querySelector('[data-tentative-card-ref="card-a"]')).toHaveAttribute("data-white-out", "true");
    expect(screen.queryByRole("button", { name: "CONFIRM" })).not.toBeInTheDocument();

    await user.dblClick(sourceA);
    expect(sourceA).not.toHaveAttribute("data-white-out");
    expect(document.querySelector('[data-tentative-card-ref="card-a"]')).not.toBeInTheDocument();

    await user.click(sourceA);
    await user.click(within(screen.getByLabelText("Stockpile 1")).getByRole("button", { name: "Select stockpile" }));
    await user.click(await screen.findByRole("button", { name: "FACE UP" }));
    const sourceB = screen.getByRole("button", { name: "Supply card card-b" });
    await user.click(sourceB);
    await user.click(within(screen.getByLabelText("Stockpile 2")).getByRole("button", { name: "Select stockpile" }));
    await user.click(await screen.findByRole("button", { name: "FACE DOWN" }));

    const confirm = screen.getByRole("button", { name: "CONFIRM" });
    expect(confirm).toHaveAttribute("data-plan-id", "plan-opaque");
    expect(submitted).toBeUndefined();
    await user.click(confirm);
    await waitFor(() => expect(submitted).toEqual({ plan_id: "plan-opaque", expected_revision: 7 }));
  });

  it("stages signed Market Impact direction and company through the atomic decision endpoint", async () => {
    let submitted: unknown;
    const impactView: GameView = {
      ...gameView,
      phase: "action",
      phase_step: "action",
      supply_batch: null,
      legal_actions: [],
      private: {
        ...gameView.private,
        available_action_cards: [
          { visibility: "visible", kind: "action", effect: "Stock Boom", direction: "up", movement: 2 },
          { visibility: "visible", kind: "action", effect: "Stock Bust", direction: "down", movement: -2 },
        ],
      },
      decision_batch: { kind: "market_impact", plans: [
        { plan_id: "impact-up-cosmic", direction: "up", company_id: 0, movement: 2 },
        { plan_id: "impact-down-cosmic", direction: "down", company_id: 0, movement: -2 },
      ] },
    };
    server.use(
      http.get("/api/v2/games/impact/view", () => HttpResponse.json(impactView)),
      http.post("/api/v2/games/impact/decisions", async ({ request }) => {
        submitted = await request.json();
        return HttpResponse.json({ ...impactView, revision: 8, decision_batch: null });
      }),
    );
    const user = userEvent.setup();
    render(<GamePage gameId="impact" token="seat-secret" />);
    const boom = await screen.findByRole("button", { name: "Stock Boom" });
    expect(boom).toHaveAttribute("aria-pressed", "false");
    await user.click(boom);
    const selectedBoom = screen.getByRole("button", { name: "Stock Boom" });
    expect(selectedBoom).toHaveAttribute("aria-pressed", "true");
    await user.click(selectedBoom);
    expect(screen.getByRole("button", { name: "Stock Boom" })).toHaveAttribute("aria-pressed", "false");
    await user.click(screen.getByRole("button", { name: "Stock Boom" }));
    await user.click(screen.getByRole("button", { name: "Select COSMIC" }));
    const confirm = screen.getByRole("button", { name: "CONFIRM" });
    expect(confirm).toHaveAttribute("data-plan-id", "impact-up-cosmic");
    await user.click(confirm);
    await waitFor(() => expect(submitted).toEqual({ plan_id: "impact-up-cosmic", expected_revision: 7 }));
  });

  it("holds checkpoints until CONTINUE and provides a presentation-only BACK before a human decision", async () => {
    let acknowledgements = 0;
    const checkpoint: GameView = {
      ...gameView,
      checkpoint: { checkpoint_id: "demand-result", kind: "demand_result", round: 3 },
      pending_decision: { ...gameView.pending_decision, kind: "acknowledge" },
      decision_batch: null,
      legal_actions: [],
    };
    const next = { ...gameView, revision: 8 };
    server.use(
      http.get("/api/v2/games/checkpoint/view", () => HttpResponse.json(checkpoint)),
      http.post("/api/v2/games/checkpoint/acknowledgements", () => {
        acknowledgements += 1;
        return HttpResponse.json(next);
      }),
    );
    const user = userEvent.setup();
    render(<GamePage gameId="checkpoint" token="seat-secret" />);
    const continueButton = await screen.findByRole("button", { name: "CONTINUE" });
    expect(continueButton).toHaveAttribute("data-checkpoint-kind", "demand_result");
    await user.click(continueButton);
    const back = await screen.findByRole("button", { name: "BACK" });
    expect(acknowledgements).toBe(1);
    await user.click(back);
    expect(screen.getByText("DEMAND RESULT")).toBeInTheDocument();
    const returnLive = screen.getByRole("button", { name: "CONTINUE" });
    expect(returnLive).toBeEnabled();
    await user.click(returnLive);
    expect(await screen.findByRole("button", { name: "BACK" })).toBeInTheDocument();
    expect(acknowledgements).toBe(1);
  });

  it("holds bankruptcy at the round checkpoint, fades only canonical facts, then renders the normalized next round", async () => {
    const bankruptView: GameView = {
      ...gameView,
      active_player_id: null,
      phase: "ROUND_RESULT",
      phase_step: "acknowledge",
      checkpoint: { checkpoint_id: "bankruptcy-result", kind: "round_result", round: 3 },
      pending_decision: { ...gameView.pending_decision, kind: "acknowledge" },
      decision_batch: null,
      legal_actions: [],
      companies: gameView.companies.map((company) => company.company_id === 0
        ? { ...company, price_dollars_per_share: 0, price_delta_dollars_per_share: -3 }
        : company),
      private: {
        ...gameView.private,
        holdings: [
          { company_id: 0, company: "Cosmic Computers", shares_thousands: 3, price_dollars_per_share: 0, market_value_thousands: 0 },
          { company_id: 1, company: "Bottomline Bank", shares_thousands: 2, price_dollars_per_share: 3, market_value_thousands: 6 },
        ],
      },
    };
    const nextRound: GameView = {
      ...gameView,
      revision: 8,
      round: 4,
      companies: bankruptView.companies.map((company) => company.company_id === 0
        ? { ...company, price_dollars_per_share: 5, price_delta_dollars_per_share: null }
        : company),
      private: {
        ...gameView.private,
        holdings: bankruptView.private.holdings.filter((holding) => holding.company_id !== 0),
      },
    };
    server.use(
      http.get("/api/v2/games/bankruptcy/view", () => HttpResponse.json(bankruptView)),
      http.post("/api/v2/games/bankruptcy/acknowledgements", () => HttpResponse.json(nextRound)),
    );
    const user = userEvent.setup();
    render(<GamePage gameId="bankruptcy" token="seat-secret" />);

    const market = await screen.findByLabelText("Market");
    const company = market.querySelector('[data-bankrupt-company="0"]') as HTMLElement;
    expect(company).toBeInTheDocument();
    expect(company.querySelector('[data-stock-pattern="matrix"]')).not.toHaveAttribute("data-white-out");
    expect(company.querySelector('[data-market-company-name="0"]')).toHaveAttribute("data-white-out", "true");
    expect(company.querySelector('[data-market-price-value="0"]')).toHaveTextContent("$0 / SHARE");
    expect(company.querySelector('[data-market-price-value="0"]')).toHaveAttribute("data-white-out", "true");
    expect(company.querySelector('[data-market-delta-slot="0"]')).toHaveTextContent("↓3");
    expect(company.querySelector('[data-market-delta-slot="0"]')).not.toHaveAttribute("data-white-out");

    const portfolio = screen.getByLabelText("Portfolio");
    expect(portfolio.querySelector('[data-portfolio-company-id="0"]')).toHaveAttribute("data-white-out", "true");
    expect(portfolio.querySelector('[data-portfolio-company-id="1"]')).not.toHaveAttribute("data-white-out");

    await user.click(screen.getByRole("button", { name: "CONTINUE" }));
    await waitFor(() => expect(market.querySelector('[data-bankrupt-company="0"]')).not.toBeInTheDocument());
    expect(market.querySelector('[data-market-price-value="0"]')).toHaveTextContent("$5 / SHARE");
    expect(portfolio.querySelector('[data-portfolio-company-id="0"]')).not.toBeInTheDocument();
    expect(portfolio.querySelector('[data-portfolio-company-id="1"]')).toBeInTheDocument();
  });

  it("keeps a settled bid outside the resolved stack white-out", async () => {
    const resolvedView: GameView = {
      ...gameView,
      active_player_id: null,
      checkpoint: { checkpoint_id: "demand-result", kind: "demand_result", round: 3 },
      pending_decision: { ...gameView.pending_decision, kind: "acknowledge" },
      decision_batch: null,
      legal_actions: [],
      stockpiles: gameView.stockpiles.map((pile, index) => ({
        ...pile,
        resolved: true,
        purchaser_id: index % 2,
        bid: pile.bid ?? { player_id: index % 2, marker_index: index % 2, amount_thousands: index + 1 },
      })),
    };
    server.use(http.get("/api/v2/games/resolved/view", () => HttpResponse.json(resolvedView)));
    render(<GamePage gameId="resolved" token="seat-secret" />);

    const pile = await screen.findByLabelText("Stockpile 1");
    const stack = pile.querySelector("[data-stockpile-stack]") as HTMLElement;
    const bid = pile.querySelector("[data-stockpile-bid]") as HTMLElement;
    expect(pile).toHaveAttribute("data-stockpile-resolved", "true");
    expect(pile).not.toHaveAttribute("data-white-out");
    expect(stack).toHaveAttribute("data-white-out", "true");
    expect(stack.contains(bid)).toBe(false);
    expect(bid).toHaveTextContent("YOU $10K");
    expect(bid).not.toHaveAttribute("data-white-out");
  });

  it("stages server-authored dividend choices and never conflates cash with price movement", async () => {
    let submitted: unknown;
    const dividendView: GameView = {
      ...gameView,
      phase: "movement",
      phase_step: "dividend_claim",
      pending_decision: { ...gameView.pending_decision, kind: "dividend_claim" },
      decision_batch: null,
      legal_actions: [action(300, "Waive dividend"), action(301, "Claim dividend")],
      private: {
        ...gameView.private,
        market_information: [
          { visibility: "private", card: { visibility: "visible", kind: "company_forecast", company_id: 0, company: "Cosmic Computers", forecast: 3, cash_effect_thousands: null } },
          { visibility: "private", card: { visibility: "visible", kind: "company_forecast", company_id: 4, company: "Stanford Steel", forecast: "DIVIDEND", cash_effect_thousands: 2 } },
          { visibility: "public", card: { visibility: "visible", kind: "company_forecast", company_id: 1, company: "Bottomline Bank", forecast: -4, cash_effect_thousands: null } },
        ],
      },
    };
    server.use(
      http.get("/api/v2/games/dividend/view", () => HttpResponse.json(dividendView)),
      http.post("/api/v2/games/dividend/actions", async ({ request }) => {
        submitted = await request.json();
        return HttpResponse.json({ ...dividendView, revision: 8, legal_actions: [] });
      }),
    );
    const user = userEvent.setup();
    render(<GamePage gameId="dividend" token="seat-secret" />);
    const research = await screen.findByLabelText("Research");
    expect(within(research).getByLabelText("Price up 3")).toHaveTextContent("↑3");
    expect(within(research).getByLabelText("Cash increases by 2K")).toHaveTextContent("+$2K");
    expect(within(research).queryByLabelText("Price down 4")).not.toBeInTheDocument();
    const claim = screen.getByRole("button", { name: "Claim dividend" });
    expect(screen.getByRole("button", { name: "Waive dividend" })).toHaveTextContent("WAIVE DIVIDEND");
    await user.click(claim);
    expect(submitted).toBeUndefined();
    await user.click(screen.getByRole("button", { name: "CONFIRM" }));
    await waitFor(() => expect(submitted).toEqual({ action_id: 301, expected_revision: 7 }));
  });

  it("shows the current holding beside HOLD and SELL using the sale-preview company fallback", async () => {
    const sellView: GameView = {
      ...gameView,
      phase: "selling",
      phase_step: "selling",
      supply_batch: null,
      decision_batch: null,
      pending_decision: {
        kind: "sell",
        prompt: "SELL",
        selected_stockpile_id: null,
        selected_action_effect: null,
        company_id: null,
      },
      private: {
        ...gameView.private,
        holdings: [
          ...gameView.private.holdings,
          { company_id: 5, company: "Epic Electric", shares_thousands: 2, price_dollars_per_share: 9, market_value_thousands: 18 },
        ],
      },
      legal_actions: [
        {
          action_id: 401,
          control: "sell",
          label: "HOLD",
          target_id: null,
          amount_thousands: null,
          direction: null,
          sale_preview: { company_id: 5, company: "Epic Electric", shares_thousands: 0, price_dollars_per_share: 9, gross_value_thousands: 0, resulting_shares_thousands: 2 },
        },
        {
          action_id: 402,
          control: "sell",
          label: "Sell 1K",
          target_id: null,
          amount_thousands: null,
          direction: null,
          sale_preview: { company_id: 5, company: "Epic Electric", shares_thousands: 1, price_dollars_per_share: 9, gross_value_thousands: 9, resulting_shares_thousands: 1 },
        },
        {
          action_id: 403,
          control: "sell",
          label: "Sell 2K",
          target_id: null,
          amount_thousands: null,
          direction: null,
          sale_preview: { company_id: 5, company: "Epic Electric", shares_thousands: 2, price_dollars_per_share: 9, gross_value_thousands: 18, resulting_shares_thousands: 0 },
        },
      ],
    };
    server.use(http.get("/api/v2/games/sell/view", () => HttpResponse.json(sellView)));
    render(<GamePage gameId="sell" token="seat-secret" />);

    const dock = await screen.findByLabelText("Action dock");
    const sellingCard = within(dock).getByLabelText("EPIC holding 2K shares");
    expect(sellingCard).toHaveAttribute("data-card-scale", "active");
    expect(sellingCard.closest("[data-selling-company-id]")).toHaveAttribute("data-selling-company-id", "5");
    expect(within(dock).getByText("2K", { selector: "[data-card-value]" })).toBeInTheDocument();
    expect(within(dock).getByRole("button", { name: "HOLD" })).toBeInTheDocument();
    expect(within(dock).getByRole("button", { name: "Sell 1K" })).toHaveTextContent("SELL 1K +$9K");
    expect(within(dock).getByRole("button", { name: "Sell 2K" })).toHaveTextContent("SELL 2K +$18K");
    expect(dock.querySelectorAll("[data-selling-company-id]")).toHaveLength(1);
  });

  it("keeps calculations above a shared-scale terminal chart, whites Portfolio, and omits winner copy", async () => {
    const navigate = vi.fn();
    const terminalView: GameView = {
      ...gameView,
      active_player_id: null,
      phase: "terminal",
      phase_step: "terminal",
      checkpoint: null,
      decision_batch: null,
      legal_actions: [],
      pending_decision: { ...gameView.pending_decision, kind: "terminal" },
      terminal_results: {
        winner_ids: [0],
        players: [
          { player_id: 0, player_name: "YOU", cash_before_liquidation_thousands: 13, liquidation_value_thousands: 141, final_cash_thousands: 154, rank: 1, winner: true, liquidation: [
            { company_id: 0, company: "Cosmic Computers", shares_thousands: 3, price_dollars_per_share: 47, value_thousands: 141 },
          ] },
          { player_id: 1, player_name: "COMPUTER", cash_before_liquidation_thousands: -5, liquidation_value_thousands: 90, final_cash_thousands: 85, rank: 2, winner: false, liquidation: [] },
        ],
      },
    };
    server.use(http.get("/api/v2/games/terminal/view", () => HttpResponse.json(terminalView)));
    const user = userEvent.setup();
    render(<GamePage gameId="terminal" token="seat-secret" navigate={navigate} />);
    const field = await screen.findByLabelText("Game end");
    expect(within(field).getByText("3K × $47 / SHARE = $141K")).toBeInTheDocument();
    const chart = screen.getByTestId("terminal-chart");
    expect(chart).toHaveAttribute("data-chart-min", "-5");
    expect(chart.querySelectorAll('[data-chart-segment="position"]')).toHaveLength(2);
    expect(chart.querySelectorAll('[data-chart-segment="cash"]')).toHaveLength(2);
    const calculations = field.querySelector("[class*='rankings']");
    expect(calculations).not.toBeNull();
    expect(calculations!.compareDocumentPosition(chart) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByLabelText("Portfolio").querySelector('[data-white-out="true"]')).toBeInTheDocument();
    expect(document.body).not.toHaveTextContent("WINNER");
    expect(screen.getByRole("button", { name: "RESIGN" })).toBeInTheDocument();
    expect(screen.getByLabelText("Action dock").querySelectorAll("[data-dock-control]")).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "CONTINUE" }));
    expect(navigate).toHaveBeenCalledWith("/");
  });

  it("arms authoritative resignation, clears the seat token only after 204, and returns home", async () => {
    let request: unknown;
    const navigate = vi.fn();
    sessionStorage.setItem(seatStorageKey("resign"), "seat-secret");
    server.use(
      http.get("/api/v2/games/resign/view", () => HttpResponse.json(gameView)),
      http.post("/api/v2/games/resign/resignations", async ({ request: incoming }) => {
        request = await incoming.json();
        return new HttpResponse(null, { status: 204 });
      }),
    );
    const user = userEvent.setup();
    render(<GamePage gameId="resign" token="seat-secret" navigate={navigate} />);
    await user.click(await screen.findByRole("button", { name: "RESIGN" }));
    expect(sessionStorage.getItem(seatStorageKey("resign"))).toBe("seat-secret");
    const confirm = screen.getByRole("button", { name: "CONFIRM" });
    expect(confirm).toHaveAttribute("data-resign-confirm", "true");
    await user.click(confirm);
    await waitFor(() => expect(request).toEqual({ expected_revision: 7 }));
    expect(sessionStorage.getItem(seatStorageKey("resign"))).toBeNull();
    expect(navigate).toHaveBeenCalledWith("/");
  });

  it("contains no multiplayer, public, analysis, chat, or pile-number text", async () => {
    render(<GamePage gameId="unusual" token="seat-secret" />);
    await screen.findByLabelText("Stockpiles");
    const visibleText = document.body.textContent?.toLowerCase() ?? "";
    for (const forbidden of ["chat", "history", "seat", "public", "deep cfr", "recommendation", "expected value", "exploitability", "policy", "advantage", "pile 1", "pile 2", "s3", "s4"]) {
      expect(visibleText).not.toContain(forbidden);
    }
  });
});

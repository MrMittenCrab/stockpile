import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import type { GameView, SetupResponse } from "../types";

export const setupResponse: SetupResponse = {
  schema_version: "2.0",
  mode: "lite",
  round_count: 6,
  options: [
    { key: "trading_fees", label: "Trading Fees", default: false },
    { key: "dividends", label: "Dividends", default: false },
    { key: "sell_order", label: "Sell Order", default: false },
  ],
};

const options = { market_impact: false, trading_fees: false, dividends: false, sell_order: false };
export const gameView: GameView = {
  schema_version: "2.0",
  game_id: "unusual",
  revision: 7,
  configuration: { mode: "lite", player_count: 2, round_count: 6, options },
  round: 3,
  total_rounds: 6,
  phase: "demand",
  phase_step: "demand_bid",
  viewer: { player_id: 0, name: "YOU" },
  active_player_id: 0,
  companies: [
    { company_id: 0, symbol: "A", name: "Cosmic Computers", display_name: "COSMIC", pattern: "matrix", price_dollars_per_share: 47 },
    { company_id: 1, symbol: "B", name: "Bottomline Bank", display_name: "BOTTOMLINE", pattern: "ledger", price_dollars_per_share: 3 },
    { company_id: 2, symbol: "C", name: "Leading Laboratories", display_name: "LEADING", pattern: "molecular", price_dollars_per_share: 11 },
    { company_id: 3, symbol: "D", name: "American Automotive", display_name: "AMERICAN", pattern: "chevron", price_dollars_per_share: 8 },
    { company_id: 4, symbol: "E", name: "Stanford Steel", display_name: "STANFORD", pattern: "crosshatch", price_dollars_per_share: 5 },
    { company_id: 5, symbol: "F", name: "Epic Electric", display_name: "EPIC", pattern: "wave", price_dollars_per_share: 9 },
  ],
  stockpiles: [
    {
      stockpile_id: 0,
      cards_bottom_to_top: [
        { visibility: "visible", kind: "stock", company_id: 0, company: "Cosmic Computers", shares_thousands: 3 },
        { visibility: "hidden" },
        { visibility: "visible", kind: "trading_fee", cash_effect_thousands: -4 },
      ],
      bid: { player_id: 0, marker_index: 0, amount_thousands: 10 },
      locked: false,
      purchaser_id: null,
      resolved: false,
    },
    {
      stockpile_id: 1,
      cards_bottom_to_top: [{ visibility: "remembered", face_down: true, card: { visibility: "visible", kind: "stock", company_id: 5, company: "Epic Electric", shares_thousands: 1 } }],
      bid: null,
      locked: false,
      purchaser_id: null,
      resolved: false,
    },
    ...Array.from({ length: 3 }, (_, offset) => ({
      stockpile_id: offset + 2,
      cards_bottom_to_top: [{ visibility: "hidden" as const }],
      bid: null,
      locked: false,
      purchaser_id: null,
      resolved: false,
    })),
  ],
  players: [
    {
      role: "human",
      player_id: 0,
      name: "YOU",
      cash_thousands: 13,
      cash_delta_thousands: -9,
      position_value_thousands: 141,
      position_delta_thousands: 6,
      active: true,
      status: "Choosing bid",
      bid_markers: [
        { player_id: 0, marker_index: 0, status: "placed", stockpile_id: 0, bid_thousands: 10 },
        { player_id: 0, marker_index: 1, status: "available", stockpile_id: null, bid_thousands: null },
      ],
    },
    {
      role: "computer",
      player_id: 1,
      name: "COMPUTER",
      cash_thousands: 99,
      cash_delta_thousands: 4,
      active: false,
      status: "Waiting",
      bid_markers: [
        { player_id: 1, marker_index: 0, status: "outbid", stockpile_id: null, bid_thousands: null },
        { player_id: 1, marker_index: 1, status: "available", stockpile_id: null, bid_thousands: null },
      ],
    },
  ],
  private: {
    market_information: [{ visibility: "hidden", card: { visibility: "hidden" } }],
    holdings: [{ company_id: 0, company: "Cosmic Computers", shares_thousands: 3, price_dollars_per_share: 47, market_value_thousands: 141 }],
    available_action_cards: [],
  },
  pending_decision: { kind: "bid_pile", prompt: "Choose a legal stockpile", selected_stockpile_id: null, selected_action_effect: null, company_id: null },
  legal_actions: [],
  supply_batch: null,
  decision_batch: { kind: "demand", plans: [
    { plan_id: "bid-37", stockpile_id: 4, amount_thousands: 37, marker_index: 1 },
  ] },
  checkpoint: null,
  recent_events: [
    { event_id: 1, event_type: "market_movement", cause: "market_forecast", round: 3, company_id: 0, prior_price_dollars_per_share: 45, price_delta: 2, resulting_price_dollars_per_share: 47, forecast: 2, cash_effect_thousands: null, direction: "up" },
    { event_id: 2, event_type: "market_movement", cause: "market_forecast", round: 3, company_id: 1, prior_price_dollars_per_share: 4, price_delta: -1, resulting_price_dollars_per_share: 3, forecast: -1, cash_effect_thousands: null, direction: "down" },
  ],
  terminal_results: null,
};

export const server = setupServer(
  http.get("/api/v2/setup", () => HttpResponse.json(setupResponse)),
  http.get("/api/v2/games/:id/view", () => HttpResponse.json(gameView)),
  http.post("/api/v2/games/:id/actions", () => HttpResponse.json({ ...gameView, revision: 8 })),
  http.post("/api/v2/games/:id/supply", () => HttpResponse.json({ ...gameView, revision: 8 })),
  http.post("/api/v2/games/:id/decisions", () => HttpResponse.json({ ...gameView, revision: 8 })),
  http.post("/api/v2/games/:id/acknowledgements", () => HttpResponse.json({ ...gameView, revision: 8, checkpoint: null })),
  http.post("/api/v2/games/:id/resignations", () => new HttpResponse(null, { status: 204 })),
);

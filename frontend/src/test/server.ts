import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import type { GameView, SetupResponse } from "../types";

export const setupResponse: SetupResponse = {
  schema_version: "1.0",
  mode: "lite",
  defaults: { player_count: 2, round_count: 6 },
  player_limits: { minimum: 2, maximum: 5 },
  round_limits: { minimum: 1, maximum: 10 },
  options: [
    { key: "market_impact", label: "Market Impact", description: "Add the Action phase.", default: false },
    { key: "starting_share", label: "Starting Share", description: "Deal one starting share.", default: false },
    { key: "trading_fees", label: "Trading Fees", description: "Include fees.", default: false },
    { key: "dividends", label: "Dividends", description: "Include dividends.", default: false },
    { key: "sell_order", label: "Sell Order", description: "Sequential sales.", default: false },
  ],
};

const options = { market_impact: false, starting_share: false, trading_fees: false, dividends: false, sell_order: false };
export const gameView: GameView = {
  schema_version: "1.0", game_id: "unusual", revision: 7,
  configuration: { mode: "lite", player_count: 2, round_count: 6, options },
  capabilities: { market_impact: false, starting_share: false, trading_fees: false, dividends: false, sequential_selling: false, stock_splits: false, majority_bonus: false, price_ceiling: null },
  round: 3, total_rounds: 6, phase: "demand", phase_step: "demand_bid",
  viewer: { player_id: 0, name: "Ada" }, active_player_id: 0,
  companies: [
    { company_id: 0, symbol: "A", name: "Arc", price: 47, color: "#d33" },
    { company_id: 1, symbol: "B", name: "Bolt", price: 3, color: "#39c" },
  ],
  stockpiles: Array.from({ length: 7 }, (_, index) => ({ stockpile_id: index, visible_cards: index === 0 ? [{ visibility: "visible", kind: "stock", company_id: 0, company: "Arc", quantity: 3 }] : [], hidden_cards: index % 2 ? [{ visibility: "hidden" as const }] : [], marker: null, bid: null, locked: false, purchaser_id: null })),
  players: [
    { player_id: 0, name: "Ada", cash: 13, active: true, status: "Choosing bid", fee_debts: [], bid_markers: [{ player_id: 0, marker_index: 0, status: "placed", stockpile_id: 0, bid: 0 }, { player_id: 0, marker_index: 1, status: "available", stockpile_id: null, bid: null }] },
    { player_id: 1, name: "Lin", cash: 99, active: false, status: "Waiting", fee_debts: [4, 7], bid_markers: [{ player_id: 1, marker_index: 0, status: "outbid", stockpile_id: null, bid: null }, { player_id: 1, marker_index: 1, status: "available", stockpile_id: null, bid: null }] },
  ],
  private: { hand: [], market_information: [{ visibility: "hidden", source: "unknown", card: { visibility: "hidden" } }], holdings: [{ company_id: 0, company: "Arc", regular: 1, split: 0, represented: 1, price: 47 }], known_pile_cards: [{ stockpile_id: 1, card: { visibility: "visible", kind: "stock", company_id: 0, company: "Arc", quantity: 3 } }], available_action_cards: [] },
  pending_decision: { kind: "bid_amount", prompt: "Choose a legal bid", selected_card_index: null, selected_stockpile_id: 6, selected_action_effect: null, company_id: null, private_progress: null, private_total: null },
  legal_actions: [{ action_id: 9123, control: "bid", label: "Bid 37K", target_id: null, amount: 37, placement_visibility: null, sale_preview: null }],
  public_history: [], recent_events: [
    { event_id: 1, event_type: "market_movement", cause: "market_forecast", round: 3, description: "Arc moved", company_id: 0, company: "Arc", prior_price: 45, requested_delta: 2, actual_delta: 2, resulting_price: 47, forecast: 2, effect: null, actor_id: null },
    { event_id: 2, event_type: "market_movement", cause: "market_forecast", round: 3, description: "Bolt moved", company_id: 1, company: "Bolt", prior_price: 4, requested_delta: -1, actual_delta: -1, resulting_price: 3, forecast: -1, effect: null, actor_id: null },
  ], chat: [], terminal_results: null,
};

export const server = setupServer(
  http.get("/api/v1/setup", () => HttpResponse.json(setupResponse)),
  http.get("/api/v1/games/:id/view", () => HttpResponse.json(gameView)),
  http.post("/api/v1/games/:id/actions", () => HttpResponse.json({ ...gameView, revision: 8 })),
  http.post("/api/v1/games/:id/chat", () => HttpResponse.json({ schema_version: "1.0", chat_message: { message_id: 1, player_id: 0, player_name: "Ada", message: "hi", created_at: "now" } }, { status: 201 })),
);

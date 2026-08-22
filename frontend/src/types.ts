export type LiteOptionKey =
  | "market_impact"
  | "starting_share"
  | "trading_fees"
  | "dividends"
  | "sell_order";

export interface SetupResponse {
  schema_version: "1.0";
  mode: "lite";
  defaults: { player_count: number; round_count: number };
  player_limits: { minimum: number; maximum: number };
  round_limits: { minimum: number; maximum: number };
  options: Array<{
    key: LiteOptionKey;
    label: string;
    description: string;
    default: boolean;
  }>;
}

export type LiteOptions = Record<LiteOptionKey, boolean>;

export interface CreateGameRequest {
  player_count: number;
  player_names: string[];
  round_count: number;
  options: LiteOptions;
  seed?: number;
}

export interface CreateGameResponse {
  schema_version: "1.0";
  game_id: string;
  seats: Array<{ player_id: number; player_name: string; url: string }>;
}

export interface HiddenCard { visibility: "hidden" }
export interface StockCard {
  visibility: "visible";
  kind: "stock";
  company_id: number;
  company: string;
  quantity: number;
}
export interface FeeCard {
  visibility: "visible";
  kind: "trading_fee";
  amount: number;
}
export interface ActionCard {
  visibility: "visible";
  kind: "action";
  effect: string;
  direction: "up" | "down";
  movement: number;
}
export interface InformationCard {
  visibility: "visible";
  kind: "company_forecast";
  company_id: number;
  company: string;
  forecast: number | "DIVIDEND";
}
export type VisibleCard = StockCard | FeeCard | ActionCard | InformationCard;
export type Card = HiddenCard | VisibleCard;

export interface Company {
  company_id: number;
  symbol: string;
  name: string;
  display_name: string;
  pattern:
    | "matrix"
    | "ledger"
    | "molecular"
    | "chevron"
    | "crosshatch"
    | "wave";
  price: number;
  color: string;
}

export type BidMarkerStatus =
  | "available"
  | "placed"
  | "outbid"
  | "rebidding"
  | "locked";
export interface BidMarker {
  player_id: number;
  marker_index: number;
  status: BidMarkerStatus;
  stockpile_id: number | null;
  bid: number | null;
}

export interface Stockpile {
  stockpile_id: number;
  visible_cards: VisibleCard[];
  hidden_cards: HiddenCard[];
  marker: BidMarker | null;
  bid: number | null;
  locked: boolean;
  purchaser_id: number | null;
}

export interface PublicPlayer {
  player_id: number;
  name: string;
  cash: number;
  active: boolean;
  status: string;
  fee_debts: number[];
  bid_markers: BidMarker[];
}

export interface SalePreview {
  company_id: number;
  company: string;
  quantity: number;
  unit_price: number;
  gross_value: number;
  resulting_regular: number;
  resulting_split: number;
  resulting_represented: number;
}

export type ActionControl =
  | "card"
  | "stockpile"
  | "bid"
  | "action_card"
  | "company"
  | "sell"
  | "dividend"
  | "continue"
  | "generic";

export interface LegalAction {
  action_id: number;
  control: ActionControl;
  label: string;
  target_id: string | null;
  amount: number | null;
  placement_visibility: "face_up" | "face_down" | null;
  sale_preview: SalePreview | null;
}

export type PendingKind =
  | "supply_card"
  | "supply_face_up_pile"
  | "supply_face_down_pile"
  | "bid_pile"
  | "bid_amount"
  | "action_card"
  | "action_company"
  | "sell"
  | "dividend_claim"
  | "acknowledge"
  | "waiting"
  | "private_selling"
  | "terminal"
  | "generic";

export interface PendingDecision {
  kind: PendingKind;
  prompt: string;
  selected_card_index: number | null;
  selected_stockpile_id: number | null;
  selected_action_effect: string | null;
  company_id: number | null;
  private_progress: number | null;
  private_total: number | null;
}

export interface Holding {
  company_id: number;
  company: string;
  regular: number;
  split: number;
  represented: number;
  price: number;
}

export interface ChatMessage {
  message_id: number;
  player_id: number;
  player_name: string;
  message: string;
  created_at: string;
}

export interface GameView {
  schema_version: "1.0";
  game_id: string;
  revision: number;
  configuration: {
    mode: "lite";
    player_count: number;
    round_count: number;
    options: LiteOptions;
  };
  capabilities: {
    market_impact: boolean;
    starting_share: boolean;
    trading_fees: boolean;
    dividends: boolean;
    sequential_selling: boolean;
    stock_splits: false;
    majority_bonus: false;
    price_ceiling: null;
  };
  round: number;
  total_rounds: number;
  phase: string;
  phase_step: string;
  viewer: { player_id: number; name: string };
  active_player_id: number | null;
  companies: Company[];
  stockpiles: Stockpile[];
  players: PublicPlayer[];
  private: {
    hand: VisibleCard[];
    market_information: Array<{
      visibility: "private" | "public" | "hidden";
      source: "dealt" | "viewed" | "revealed" | "unknown";
      card: InformationCard | HiddenCard;
    }>;
    holdings: Holding[];
    known_pile_cards: Array<{ stockpile_id: number; card: VisibleCard }>;
    available_action_cards: ActionCard[];
  };
  pending_decision: PendingDecision;
  legal_actions: LegalAction[];
  public_history: Array<{
    sequence: number;
    phase: string;
    actor_id: number | null;
    summary: string;
    sale_totals: Record<string, Record<string, number>> | null;
  }>;
  recent_events: Array<{
    event_id: number;
    event_type: string;
    cause: string | null;
    round: number;
    description: string;
    company_id: number | null;
    company: string | null;
    prior_price: number | null;
    requested_delta: number | null;
    actual_delta: number | null;
    resulting_price: number | null;
    forecast: number | "DIVIDEND" | null;
    effect: string | null;
    actor_id: number | null;
  }>;
  chat: ChatMessage[];
  terminal_results: null | {
    players: Array<{
      player_id: number;
      player_name: string;
      cash_before_liquidation: number;
      liquidation_value: number;
      final_cash: number;
      rank: number;
      winner: boolean;
      liquidation: Array<{
        company_id: number;
        company: string;
        represented_shares: number;
        unit_price: number;
        value: number;
      }>;
    }>;
    winner_ids: number[];
  };
}

export interface ApiFailure {
  error: { code: string; message: string };
}

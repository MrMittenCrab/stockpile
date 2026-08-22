export type LiteOptionKey = "market_impact" | "trading_fees" | "dividends" | "sell_order";
export type LiteOptions = Record<LiteOptionKey, boolean>;

export interface SetupResponse {
  schema_version: "2.0";
  mode: "lite";
  round_count: 6;
  options: Array<{ key: LiteOptionKey; label: string; default: boolean }>;
}

export interface CreateGameRequest { options: LiteOptions; seed?: number }
export interface CreateGameResponse { schema_version: "2.0"; game_id: string; game_url: string }

export type StockPatternName = "matrix" | "ledger" | "molecular" | "chevron" | "crosshatch" | "wave";
export interface Company {
  company_id: number;
  symbol: string;
  name: string;
  display_name: string;
  pattern: StockPatternName;
  price_dollars_per_share: number;
}

export interface HiddenCard { visibility: "hidden" }
export interface StockCard {
  visibility: "visible";
  kind: "stock";
  company_id: number;
  company: string;
  shares_thousands: number;
}
export interface FeeCard { visibility: "visible"; kind: "trading_fee"; cash_effect_thousands: number }
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
  cash_effect_thousands?: number | null;
}
export type VisibleCard = StockCard | FeeCard | ActionCard;
export type Card = HiddenCard | VisibleCard | InformationCard;
export interface RememberedPileCard { visibility: "remembered"; face_down: true; card: VisibleCard }
export type PileCard = HiddenCard | VisibleCard | RememberedPileCard;

export type BidMarkerStatus = "available" | "placed" | "outbid" | "rebidding" | "locked";
export interface BidMarker {
  player_id: number;
  marker_index: number;
  status: BidMarkerStatus;
  stockpile_id: number | null;
  bid_thousands: number | null;
}
export interface Stockpile {
  stockpile_id: number;
  cards_bottom_to_top: PileCard[];
  bid: { player_id: number; marker_index: number; amount_thousands: number } | null;
  locked: boolean;
  purchaser_id: number | null;
  resolved: boolean;
}
export interface PublicPlayerBase {
  player_id: number;
  name: string;
  cash_thousands: number;
  cash_delta_thousands: number | null;
  active: boolean;
  status: string;
  bid_markers: BidMarker[];
}
export interface HumanPublicPlayer extends PublicPlayerBase {
  role: "human";
  position_value_thousands: number;
  position_delta_thousands: number | null;
}
export interface ComputerPublicPlayer extends PublicPlayerBase { role: "computer" }
export type PublicPlayer = HumanPublicPlayer | ComputerPublicPlayer;
export interface Holding {
  company_id: number;
  company: string;
  shares_thousands: number;
  price_dollars_per_share: number;
  market_value_thousands: number;
}
export interface MarketInformationSlot {
  visibility: "private" | "public" | "hidden";
  card: InformationCard | HiddenCard;
}

export interface SalePreview {
  company_id: number;
  company: string;
  shares_thousands: number;
  price_dollars_per_share: number;
  gross_value_thousands: number;
  resulting_shares_thousands: number;
}
export type ActionControl = "stockpile" | "bid" | "action_card" | "company" | "sell" | "dividend" | "continue" | "generic";
export interface LegalAction {
  action_id: number;
  control: ActionControl;
  label: string;
  target_id: string | null;
  amount_thousands: number | null;
  direction: "up" | "down" | null;
  sale_preview: SalePreview | null;
}
export type PendingKind = "supply" | "bid_pile" | "bid_amount" | "action_card" | "action_company" | "sell" | "dividend_claim" | "acknowledge" | "waiting" | "private_selling" | "terminal" | "generic";
export interface PendingDecision {
  kind: PendingKind;
  prompt: string;
  selected_stockpile_id: number | null;
  selected_action_effect: string | null;
  company_id: number | null;
}

export interface SupplyPlacement {
  card_ref: string;
  stockpile_id: number;
  visibility: "face_up" | "face_down";
}
export interface SupplyBatch {
  cards: Array<{ card_ref: string; card: VisibleCard }>;
  plans: Array<{ plan_id: string; placements: SupplyPlacement[] }>;
}
export interface DemandDecisionBatch {
  kind: "demand";
  plans: Array<{
    plan_id: string;
    stockpile_id: number;
    amount_thousands: number;
    marker_index: number;
  }>;
}
export interface MarketImpactDecisionBatch {
  kind: "market_impact";
  plans: Array<{
    plan_id: string;
    direction: "up" | "down";
    company_id: number;
    movement: number;
  }>;
}
export type DecisionBatch = DemandDecisionBatch | MarketImpactDecisionBatch;
export interface PresentationCheckpoint {
  checkpoint_id: string;
  kind: "demand_result" | "round_result";
  round: number;
}
export interface MarketEvent {
  event_id: number;
  event_type: string;
  cause: string | null;
  round: number;
  company_id: number | null;
  prior_price_dollars_per_share: number | null;
  price_delta: number | null;
  resulting_price_dollars_per_share: number | null;
  forecast: number | "DIVIDEND" | null;
  cash_effect_thousands: number | null;
  direction: "up" | "down" | null;
}
export interface TerminalResult {
  players: Array<{
    player_id: number;
    player_name: string;
    cash_before_liquidation_thousands: number;
    liquidation_value_thousands: number;
    final_cash_thousands: number;
    rank: number;
    winner: boolean;
    liquidation: Array<{
      company_id: number;
      company: string;
      shares_thousands: number;
      price_dollars_per_share: number;
      value_thousands: number;
    }>;
  }>;
  winner_ids: number[];
}

export interface GameView {
  schema_version: "2.0";
  game_id: string;
  revision: number;
  configuration: { mode: "lite"; player_count: 2; round_count: 6; options: LiteOptions };
  round: number;
  total_rounds: number;
  phase: string;
  phase_step: string;
  viewer: { player_id: 0; name: "YOU" };
  active_player_id: number | null;
  companies: Company[];
  stockpiles: Stockpile[];
  players: PublicPlayer[];
  private: {
    market_information: MarketInformationSlot[];
    holdings: Holding[];
    available_action_cards: ActionCard[];
  };
  pending_decision: PendingDecision;
  legal_actions: LegalAction[];
  supply_batch: SupplyBatch | null;
  decision_batch: DecisionBatch | null;
  checkpoint: PresentationCheckpoint | null;
  recent_events: MarketEvent[];
  terminal_results: TerminalResult | null;
}

export interface ApiFailure { error: { code: string; message: string } }

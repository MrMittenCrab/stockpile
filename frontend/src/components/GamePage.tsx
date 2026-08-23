import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { clearSeatToken } from "../api";
import type {
  Card,
  Company,
  GameView,
  LegalAction,
  PileCard,
  Stockpile,
  SupplyBatch,
  SupplyPlacement,
  VisibleCard,
} from "../types";
import { useGameSession } from "../useGameSession";
import { CardView, CompanyCard, HoldingCard } from "./CardView";
import { CardFrame, SectionLabel, TextButton } from "./Primitives";
import { Selectable } from "./Selectable";
import { StockPattern } from "./StockPattern";
import styles from "./Game.module.css";

type SupplyDraft = Record<string, Partial<Pick<SupplyPlacement, "stockpile_id" | "visibility">>>;
interface SupplySnapshot { selectedCardRef: string | null; draft: SupplyDraft }
interface SupplyUi extends SupplySnapshot { history: SupplySnapshot[] }
interface DecisionSnapshot {
  actionId: number | null;
  planId: string | null;
  stockpileId: number | null;
  direction: "up" | "down" | null;
}
interface DecisionUi extends DecisionSnapshot { history: DecisionSnapshot[] }
interface TentativePlacement {
  cardRef: string;
  card: VisibleCard;
  visibility: SupplyPlacement["visibility"];
}

const EMPTY_SUPPLY: SupplyUi = { selectedCardRef: null, draft: {}, history: [] };
const EMPTY_DECISION: DecisionUi = {
  actionId: null,
  planId: null,
  stockpileId: null,
  direction: null,
  history: [],
};

function money(value: number) { return `$${value}K`; }
function price(value: number) { return `$${value} / SHARE`; }
function companyById(view: GameView, companyId: number | null) {
  return companyId === null ? undefined : view.companies.find((company) => company.company_id === companyId);
}
function decisionSnapshot(value: DecisionUi): DecisionSnapshot {
  return { actionId: value.actionId, planId: value.planId, stockpileId: value.stockpileId, direction: value.direction };
}
function supplySnapshot(value: SupplyUi): SupplySnapshot {
  return { selectedCardRef: value.selectedCardRef, draft: value.draft };
}

function Status({ view }: { view: GameView }) {
  let phase = view.phase.toUpperCase();
  let turn = "WAIT";
  if (view.terminal_results) {
    phase = "GAME END";
    turn = "";
  } else if (view.checkpoint) {
    phase = view.checkpoint.kind === "demand_result" ? "DEMAND RESULT" : "ROUND RESULT";
    turn = "";
  } else if (view.active_player_id === view.viewer.player_id) {
    turn = "YOUR TURN";
  } else if (view.active_player_id !== null) {
    turn = "COMPUTER";
  }
  return (
    <header className={`${styles.module} ${styles.status}`} aria-label="Status">
      <span>ROUND {view.round} / {view.total_rounds}</span>
      <span>{phase}</span>
      <span>{turn}</span>
    </header>
  );
}

function Market({
  view,
  selectedActionId,
  selectedPlanId,
  selectedDirection,
  onSelectAction,
  onSelectImpactCompany,
  disabled,
}: {
  view: GameView;
  selectedActionId: number | null;
  selectedPlanId: string | null;
  selectedDirection: "up" | "down" | null;
  onSelectAction: (actionId: number) => void;
  onSelectImpactCompany: (planId: string) => void;
  disabled: boolean;
}) {
  const actions = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((candidate) => candidate.control === "company")) {
    const target = action.target_id?.match(/^company:(\d+)$/);
    if (target) actions.set(Number(target[1]), action);
  }
  const impactPlans = view.decision_batch?.kind === "market_impact" && selectedDirection
    ? view.decision_batch.plans.filter((plan) => plan.direction === selectedDirection)
    : [];
  const plansByCompany = new Map(impactPlans.map((plan) => [plan.company_id, plan]));

  return (
    <section className={`${styles.module} ${styles.market}`} aria-label="Market">
      <SectionLabel>MARKET</SectionLabel>
      <div className={styles.marketList}>
        {view.companies.map((company) => {
          const action = actions.get(company.company_id);
          const plan = plansByCompany.get(company.company_id);
          const delta = company.price_delta_dollars_per_share;
          const bankrupt = view.checkpoint?.kind === "round_result" && company.price_dollars_per_share === 0;
          const body = (
            <span
              className={styles.marketCompany}
              data-company-id={company.company_id}
              data-bankrupt-company={bankrupt ? company.company_id : undefined}
            >
              <StockPattern pattern={company.pattern} />
              <span
                className={`${styles.marketName} ${bankrupt ? styles.whiteOutObject : ""}`}
                data-market-company-name={company.company_id}
                data-white-out={bankrupt || undefined}
              >{company.display_name}</span>
              <span
                className={`${styles.marketPrice} ${bankrupt ? styles.whiteOutObject : ""}`}
                data-market-price-value={company.company_id}
                data-white-out={bankrupt || undefined}
              >{price(company.price_dollars_per_share)}</span>
              <span
                className={`${styles.marketDelta} ${delta && delta > 0 ? styles.positive : delta && delta < 0 ? styles.negative : ""}`}
                data-market-delta-slot={company.company_id}
                data-market-price-delta={delta ? company.company_id : undefined}
                aria-live="polite"
              >
                {delta ? `${delta > 0 ? "↑" : "↓"}${Math.abs(delta)}` : ""}
              </span>
            </span>
          );
          if (plan) {
            return (
              <Selectable
                key={company.company_id}
                className={styles.objectControl}
                selected={selectedPlanId === plan.plan_id}
                data-decision-plan-id={plan.plan_id}
                aria-label={`Select ${company.display_name}`}
                disabled={disabled}
                onClick={() => onSelectImpactCompany(plan.plan_id)}
              >{body}</Selectable>
            );
          }
          if (action) {
            return (
              <Selectable
                key={company.company_id}
                className={styles.objectControl}
                selected={selectedActionId === action.action_id}
                data-action-id={action.action_id}
                aria-label={action.label}
                disabled={disabled}
                onClick={() => onSelectAction(action.action_id)}
              >{body}</Selectable>
            );
          }
          return <span key={company.company_id}>{body}</span>;
        })}
      </div>
    </section>
  );
}

function InformationPair({ card, companies }: { card: Card; companies: Company[] }) {
  if (card.visibility === "hidden") {
    return (
      <div className={styles.informationPair}>
        <CardView card={card} companies={companies} scale="information" />
        <CardView card={card} companies={companies} scale="information" />
      </div>
    );
  }
  if (card.kind !== "company_forecast") return null;
  const company = companies.find((candidate) => candidate.company_id === card.company_id);
  return (
    <div className={styles.informationPair}>
      {company && <CompanyCard company={company} />}
      <CardView card={card} companies={companies} scale="information" />
    </div>
  );
}

function Research({ view, served }: { view: GameView; served: boolean }) {
  const slots = view.private.market_information.filter((slot) => slot.visibility === "private");
  return (
    <section
      className={`${styles.module} ${styles.research} ${served ? styles.whiteOutRegion : ""}`}
      aria-label="Research"
      data-white-out={served || undefined}
    >
      <SectionLabel>RESEARCH</SectionLabel>
      <div className={styles.informationList}>
        {slots.map((slot, index) => <InformationPair key={index} card={slot.card} companies={view.companies} />)}
      </div>
    </section>
  );
}

function visiblePileCard(card: PileCard, expanded: boolean): { card: Card; faceDownKnown: boolean } {
  if (card.visibility !== "remembered") return { card, faceDownKnown: false };
  if (!expanded) return { card: { visibility: "hidden" }, faceDownKnown: false };
  return { card: card.card, faceDownKnown: true };
}
function isFaceDown(card: PileCard) {
  return card.visibility === "hidden" || card.visibility === "remembered";
}

function Stack({ cards, tentative, companies, expanded, onUndoTentative }: {
  cards: PileCard[];
  tentative: TentativePlacement[];
  companies: Company[];
  expanded: boolean;
  onUndoTentative: (cardRef: string) => void;
}) {
  const entries = [
    ...cards.map((card) => ({ card, tentative: null as TentativePlacement | null })),
    ...tentative.map((placement) => ({
      card: placement.visibility === "face_down" ? ({ visibility: "hidden" } as const) : placement.card,
      tentative: placement,
    })),
  ];
  if (!entries.length) {
    return (
      <span className={styles.stack}>
        <span className={styles.stackLayer} data-empty-stockpile>
          <CardFrame aria-label="Empty stockpile" className={styles.blankCard} scale="stockpile" />
        </span>
      </span>
    );
  }
  const stackExtent = !expanded && entries.length > 1
    ? {
      width: `${104 + (entries.length - 1) * 10}px`,
      height: `${139 + (entries.length - 1) * 3}px`,
    } as CSSProperties
    : undefined;
  return (
    <span
      className={`${styles.stack} ${expanded ? styles.stackExpanded : ""}`}
      style={stackExtent}
    >
      {entries.map(({ card: entry, tentative: tentativeEntry }, index) => {
        const detailed = expanded || index === entries.length - 1;
        const display = visiblePileCard(entry, expanded);
        return (
          <span
            key={tentativeEntry?.cardRef ?? `server:${index}`}
            className={`${styles.stackLayer} ${tentativeEntry ? styles.whiteOutObject : ""}`}
            data-stack-card
            data-stack-order={index}
            data-stack-bottom={index === 0 || undefined}
            data-stack-top={index === entries.length - 1 || undefined}
            data-tentative-card-ref={tentativeEntry?.cardRef}
            data-white-out={tentativeEntry ? true : undefined}
            data-card-edge={!detailed ? (isFaceDown(entry) ? "blue" : "white") : undefined}
            style={{ "--stack-index": index, zIndex: index + 1 } as CSSProperties}
            onClick={tentativeEntry ? (event) => event.stopPropagation() : undefined}
            onDoubleClick={tentativeEntry ? (event) => {
              event.stopPropagation();
              onUndoTentative(tentativeEntry.cardRef);
            } : undefined}
          >
            {detailed
              ? <CardView card={display.card} companies={companies} scale="stockpile" faceDownKnown={display.faceDownKnown} />
              : isFaceDown(entry)
                ? <CardView card={{ visibility: "hidden" }} companies={companies} scale="stockpile" />
                : <CardFrame aria-label="Face-up card edge" className={styles.blankCard} scale="stockpile" />}
          </span>
        );
      })}
    </span>
  );
}

function bidderName(view: GameView, pile: Stockpile) {
  if (!pile.bid) return null;
  return pile.bid.player_id === view.viewer.player_id ? "YOU" : "COMPUTER";
}

function StockpileItem({ pile, view, tentative, selectable, selected, expanded, onSelect, onInspect, onUndoTentative, disabled }: {
  pile: Stockpile;
  view: GameView;
  tentative: TentativePlacement[];
  selectable: boolean;
  selected: boolean;
  expanded: boolean;
  onSelect: () => void;
  onInspect: (stockpileId: number) => void;
  onUndoTentative: (cardRef: string) => void;
  disabled: boolean;
}) {
  const clickTimer = useRef<number | null>(null);
  useEffect(() => () => {
    if (clickTimer.current !== null) window.clearTimeout(clickTimer.current);
  }, []);

  function select() {
    if (!disabled && selectable) onSelect();
  }
  function delayedSelect(event: ReactMouseEvent) {
    if (!selectable || event.detail > 1) {
      if (clickTimer.current !== null) window.clearTimeout(clickTimer.current);
      clickTimer.current = null;
      return;
    }
    clickTimer.current = window.setTimeout(() => {
      select();
      clickTimer.current = null;
    }, 200);
  }
  function inspect(event: ReactMouseEvent) {
    if (clickTimer.current !== null) window.clearTimeout(clickTimer.current);
    clickTimer.current = null;
    event.preventDefault();
    onInspect(pile.stockpile_id);
  }

  return (
    <article
      className={`${styles.stockpile} ${expanded ? styles.stockpileExpanded : ""} ${selected ? styles.stockpileSelected : ""}`}
      aria-label={`Stockpile ${pile.stockpile_id + 1}`}
      data-stockpile-id={pile.stockpile_id}
      data-stockpile-resolved={pile.resolved || undefined}
    >
      {selectable && (
        <button
          type="button"
          className={styles.pileTarget}
          data-stockpile-target={pile.stockpile_id}
          aria-label="Select stockpile"
          aria-pressed={selected}
          disabled={disabled}
          onClick={delayedSelect}
          onDoubleClick={inspect}
        />
      )}
      <div
        className={`${styles.stackInspect} ${pile.resolved ? styles.whiteOutRegion : ""}`}
        data-stockpile-stack={pile.stockpile_id}
        data-white-out={pile.resolved || undefined}
        data-stack-inspect={pile.stockpile_id}
        aria-label={selectable
          ? `Select stockpile; double-click to ${expanded ? "collapse" : "expand"}`
          : `Double-click or press Enter to ${expanded ? "collapse" : "expand"} stockpile`}
        aria-expanded={expanded}
        aria-keyshortcuts={selectable ? "Enter Space Shift+Enter" : "Enter Space"}
        role="button"
        tabIndex={0}
        onClick={delayedSelect}
        onDoubleClick={inspect}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (selectable && !event.shiftKey) select();
            else onInspect(pile.stockpile_id);
          }
        }}
      >
        <Stack cards={pile.cards_bottom_to_top} tentative={tentative} companies={view.companies} expanded={expanded} onUndoTentative={onUndoTentative} />
      </div>
      {pile.bid && <span className={styles.bid} data-stockpile-bid>{bidderName(view, pile)} {money(pile.bid.amount_thousands)}</span>}
    </article>
  );
}

function StockpileField({ view, expanded, selectedStockpileId, selectablePiles, tentativeByPile, onSelectPile, onInspect, onUndoTentative, disabled }: {
  view: GameView;
  expanded: Set<number>;
  selectedStockpileId: number | null;
  selectablePiles: Set<number>;
  tentativeByPile: Map<number, TentativePlacement[]>;
  onSelectPile: (stockpileId: number) => void;
  onInspect: (stockpileId: number) => void;
  onUndoTentative: (cardRef: string) => void;
  disabled: boolean;
}) {
  return (
    <main className={`${styles.module} ${styles.stockpileField}`} aria-label="Stockpiles" data-testid="stockpile-field">
      <div className={styles.stockpileGrid}>
        {view.stockpiles.map((pile) => (
          <StockpileItem
            key={pile.stockpile_id}
            pile={pile}
            view={view}
            tentative={tentativeByPile.get(pile.stockpile_id) ?? []}
            selectable={selectablePiles.has(pile.stockpile_id)}
            selected={selectedStockpileId === pile.stockpile_id || view.pending_decision.selected_stockpile_id === pile.stockpile_id}
            expanded={expanded.has(pile.stockpile_id)}
            onSelect={() => onSelectPile(pile.stockpile_id)}
            onInspect={onInspect}
            onUndoTentative={onUndoTentative}
            disabled={disabled}
          />
        ))}
      </div>
    </main>
  );
}

function Portfolio({ view, served }: { view: GameView; served: boolean }) {
  const bankruptCompanyIds = new Set(
    view.checkpoint?.kind === "round_result"
      ? view.companies.filter((company) => company.price_dollars_per_share === 0).map((company) => company.company_id)
      : [],
  );
  return (
    <section className={`${styles.module} ${styles.portfolio}`} aria-label="Portfolio">
      <SectionLabel>PORTFOLIO</SectionLabel>
      <div className={styles.portfolioCards}>
        {view.private.holdings.filter((holding) => holding.shares_thousands > 0).map((holding) => {
          const company = companyById(view, holding.company_id);
          const whiteOut = served || bankruptCompanyIds.has(holding.company_id);
          return company ? (
            <span
              key={holding.company_id}
              className={whiteOut ? styles.whiteOutObject : ""}
              data-portfolio-company-id={holding.company_id}
              data-white-out={whiteOut || undefined}
            >
              <HoldingCard company={company} sharesThousands={holding.shares_thousands} />
            </span>
          ) : null;
        })}
      </div>
    </section>
  );
}

function Delta({ value }: { value: number | null }) {
  return (
    <span
      className={`${styles.playerDelta} ${value && value > 0 ? styles.positive : value && value < 0 ? styles.negative : ""}`}
      data-player-delta-slot
    >
      {value ? `${value > 0 ? "+" : "−"}$${Math.abs(value)}K` : ""}
    </span>
  );
}

function Players({ view }: { view: GameView }) {
  // During play, COMPUTER holdings stay private. At GAME END the terminal
  // breakdown is public: show each seat's pre-liquidation cash and remaining
  // mark-to-market position (after sell-off, before forced final liquidation).
  const terminalByPlayer = new Map(
    (view.terminal_results?.players ?? []).map((player) => [player.player_id, player]),
  );
  return (
    <section className={`${styles.module} ${styles.players}`} aria-label="Players">
      <SectionLabel>PLAYERS</SectionLabel>
      <div className={styles.playerList}>
        {[...view.players].sort((a, b) => a.role === "human" ? -1 : b.role === "human" ? 1 : 0).map((player) => {
          const terminal = terminalByPlayer.get(player.player_id);
          const cash = terminal?.cash_before_liquidation_thousands ?? player.cash_thousands;
          const cashDelta = terminal ? null : player.cash_delta_thousands;
          const position = terminal
            ? terminal.liquidation_value_thousands
            : player.role === "human"
              ? player.position_value_thousands
              : null;
          const positionDelta = terminal || player.role !== "human"
            ? null
            : player.position_delta_thousands;
          return (
            <div className={styles.player} key={player.player_id} data-player-role={player.role}>
              <span>{player.role === "human" ? "YOU" : "COMPUTER"}</span>
              <div className={styles.metric} data-player-metric="cash"><span>CASH</span><span data-player-value-slot>{money(cash)}</span><Delta value={cashDelta} /></div>
              {position !== null && (
                <div className={styles.metric} data-player-metric="position"><span>POSITION</span><span data-player-value-slot>{money(position)}</span><Delta value={positionDelta} /></div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function planMatchesDraft(plan: SupplyBatch["plans"][number], draft: SupplyDraft, omit?: { cardRef: string; field: "stockpile_id" | "visibility" }) {
  return Object.entries(draft).every(([cardRef, assignment]) => {
    const placement = plan.placements.find((candidate) => candidate.card_ref === cardRef);
    if (!placement) return false;
    if (assignment.stockpile_id !== undefined && !(omit?.cardRef === cardRef && omit.field === "stockpile_id") && placement.stockpile_id !== assignment.stockpile_id) return false;
    if (assignment.visibility !== undefined && !(omit?.cardRef === cardRef && omit.field === "visibility") && placement.visibility !== assignment.visibility) return false;
    return true;
  });
}
function planMatchesPileAssignments(plan: SupplyBatch["plans"][number], draft: SupplyDraft) {
  return Object.entries(draft).every(([cardRef, assignment]) => {
    if (assignment.stockpile_id === undefined) return true;
    return plan.placements.some((placement) => placement.card_ref === cardRef && placement.stockpile_id === assignment.stockpile_id);
  });
}
function exactSupplyPlan(batch: SupplyBatch, draft: SupplyDraft) {
  return batch.plans.find((plan) => plan.placements.every((placement) => {
    const assignment = draft[placement.card_ref];
    return assignment?.stockpile_id === placement.stockpile_id && assignment.visibility === placement.visibility;
  }));
}
function tentativePlacements(batch: SupplyBatch | null, draft: SupplyDraft) {
  const byPile = new Map<number, TentativePlacement[]>();
  if (!batch) return byPile;
  const compatible = batch.plans.find((plan) => planMatchesDraft(plan, draft));
  const ordered = compatible?.placements ?? batch.cards.map(({ card_ref }) => ({
    card_ref,
    stockpile_id: draft[card_ref]?.stockpile_id ?? -1,
    visibility: draft[card_ref]?.visibility ?? "face_up" as const,
  }));
  for (const placement of ordered) {
    const assignment = draft[placement.card_ref];
    const card = batch.cards.find((candidate) => candidate.card_ref === placement.card_ref)?.card;
    if (!card || assignment?.stockpile_id === undefined || !assignment.visibility) continue;
    const current = byPile.get(assignment.stockpile_id) ?? [];
    current.push({ cardRef: placement.card_ref, card, visibility: assignment.visibility });
    byPile.set(assignment.stockpile_id, current);
  }
  return byPile;
}

function prompt(view: GameView, supply: SupplyUi, decision: DecisionUi) {
  if (view.supply_batch) {
    if (!supply.selectedCardRef) return "SELECT CARD";
    const assignment = supply.draft[supply.selectedCardRef];
    if (assignment?.stockpile_id === undefined) return "SELECT PILE";
    if (!assignment.visibility) return "FACE";
    return "SELECT CARD";
  }
  if (view.decision_batch?.kind === "demand") return decision.stockpileId === null ? "SELECT PILE" : decision.planId === null ? "BID" : "";
  if (view.decision_batch?.kind === "market_impact") return decision.direction === null ? "SELECT CARD" : decision.planId === null ? "SELECT COMPANY" : "";
  switch (view.pending_decision.kind) {
    case "sell": return "SELL?";
    case "dividend_claim": return "DIVIDEND";
    case "waiting":
    case "private_selling": return "WAIT";
    case "terminal": return "";
    default: return view.legal_actions.length ? "SELECT" : "WAIT";
  }
}
function actionText(action: LegalAction) {
  if (action.amount_thousands !== null) return money(action.amount_thousands);
  return action.label.toUpperCase();
}

function SupplyContents({ view, supply, onSelectCard, onVisibility, onUndoCard, disabled }: {
  view: GameView;
  supply: SupplyUi;
  onSelectCard: (cardRef: string) => void;
  onVisibility: (visibility: SupplyPlacement["visibility"]) => void;
  onUndoCard: (cardRef: string) => void;
  disabled: boolean;
}) {
  const batch = view.supply_batch!;
  const selected = batch.cards.find((item) => item.card_ref === supply.selectedCardRef);
  const selectedAssignment = selected ? supply.draft[selected.card_ref] : undefined;
  const allowedVisibility = new Set(selected && selectedAssignment?.stockpile_id !== undefined
    ? batch.plans
      .filter((plan) => planMatchesDraft(plan, supply.draft, { cardRef: selected.card_ref, field: "visibility" }))
      .map((plan) => plan.placements.find((placement) => placement.card_ref === selected.card_ref)?.visibility)
      .filter((value): value is SupplyPlacement["visibility"] => value !== undefined)
    : []);
  return (
    <>
      {batch.cards.map(({ card_ref, card }) => {
        const assignment = supply.draft[card_ref];
        const complete = assignment?.stockpile_id !== undefined && assignment.visibility !== undefined;
        return (
          <Selectable
            key={card_ref}
            selected={supply.selectedCardRef === card_ref}
            className={`${styles.cardAction} ${complete ? styles.whiteOutObject : ""}`}
            data-supply-card-ref={card_ref}
            data-assigned-pile={assignment?.stockpile_id}
            data-assigned-visibility={assignment?.visibility}
            data-white-out={complete || undefined}
            aria-label={`Supply card ${card_ref}`}
            disabled={disabled}
            onClick={() => { if (!complete) onSelectCard(card_ref); }}
            onDoubleClick={(event) => {
              if (!complete) return;
              event.preventDefault();
              onUndoCard(card_ref);
            }}
          >
            <CardView card={card} companies={view.companies} scale="active" />
          </Selectable>
        );
      })}
      {selected && selectedAssignment?.stockpile_id !== undefined && (["face_up", "face_down"] as const).map((visibility) => allowedVisibility.has(visibility) && (
        <TextButton key={visibility} selected={selectedAssignment.visibility === visibility} data-supply-visibility={visibility} disabled={disabled} onClick={() => onVisibility(visibility)}>
          {visibility === "face_up" ? "FACE UP" : "FACE DOWN"}
        </TextButton>
      ))}
    </>
  );
}

function DecisionContents({ view, decision, onDirection, onPlan, onAction, disabled }: {
  view: GameView;
  decision: DecisionUi;
  onDirection: (direction: "up" | "down") => void;
  onPlan: (planId: string) => void;
  onAction: (actionId: number) => void;
  disabled: boolean;
}) {
  if (view.decision_batch?.kind === "demand") {
    if (decision.stockpileId === null) return null;
    return <>{view.decision_batch.plans.filter((plan) => plan.stockpile_id === decision.stockpileId).map((plan) => (
      <TextButton key={plan.plan_id} selected={decision.planId === plan.plan_id} data-decision-plan-id={plan.plan_id} disabled={disabled} onClick={() => onPlan(plan.plan_id)}>
        {money(plan.amount_thousands)}
      </TextButton>
    ))}</>;
  }
  if (view.decision_batch?.kind === "market_impact") {
    const directions = [...new Set(view.decision_batch.plans.map((plan) => plan.direction))];
    const displayedDirections = decision.direction === null ? directions : [decision.direction];
    return <>{displayedDirections.map((direction) => {
        const card = view.private.available_action_cards.find((candidate) => candidate.direction === direction);
        return (
          <Selectable key={direction} className={styles.cardAction} selected={decision.direction === direction} data-impact-direction={direction} aria-label={direction === "up" ? "Stock Boom" : "Stock Bust"} disabled={disabled} onClick={() => onDirection(direction)}>
            {card ? <CardView card={card} companies={view.companies} scale="active" /> : direction === "up" ? "↑2" : "↓2"}
          </Selectable>
        );
      })}</>;
  }
  const nonHoldSale = view.legal_actions.find((action) => (
    action.control === "sell"
    && action.sale_preview !== null
    && action.sale_preview.shares_thousands > 0
  ));
  const sellingCompanyId = view.pending_decision.kind === "sell"
    ? view.pending_decision.company_id ?? nonHoldSale?.sale_preview?.company_id ?? null
    : null;
  const sellingCompany = companyById(view, sellingCompanyId);
  const sellingHolding = sellingCompanyId === null
    ? undefined
    : view.private.holdings.find((holding) => holding.company_id === sellingCompanyId);

  return <>
    {sellingCompany && sellingHolding && (
      <span className={styles.cardAction} data-selling-company-id={sellingCompany.company_id}>
        <HoldingCard company={sellingCompany} sharesThousands={sellingHolding.shares_thousands} scale="active" />
      </span>
    )}
    {view.legal_actions.map((action) => {
    if (action.control === "company" || action.control === "stockpile") return null;
    if (action.control === "action_card") {
      const card = view.private.available_action_cards.find((candidate) => candidate.direction === action.direction);
      return (
        <Selectable key={action.action_id} className={styles.cardAction} selected={decision.actionId === action.action_id} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>
          {card ? <CardView card={card} companies={view.companies} scale="active" /> : actionText(action)}
        </Selectable>
      );
    }
    if (action.control === "sell" && action.sale_preview) {
      const sale = action.sale_preview;
      return (
        <TextButton key={action.action_id} selected={decision.actionId === action.action_id} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>
          {sale.shares_thousands === 0 ? "HOLD" : <><span>SELL {sale.shares_thousands}K</span> <span className={styles.positive}>+{money(sale.gross_value_thousands)}</span></>}
        </TextButton>
      );
    }
    return (
      <TextButton key={action.action_id} selected={decision.actionId === action.action_id} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>
        {actionText(action)}
      </TextButton>
    );
    })}
  </>;
}

function ActionDock({ view, supply, decision, contextLabel, contextEnabled, contextPlanId, contextActionId, onContext, onResign, resignArmed, onSupplyCard, onSupplyVisibility, onUndoSupplyCard, onDirection, onPlan, onAction, disabled, contentDisabled }: {
  view: GameView;
  supply: SupplyUi;
  decision: DecisionUi;
  contextLabel: string;
  contextEnabled: boolean;
  contextPlanId?: string;
  contextActionId?: number;
  onContext: () => void;
  onResign: () => void;
  resignArmed: boolean;
  onSupplyCard: (cardRef: string) => void;
  onSupplyVisibility: (visibility: SupplyPlacement["visibility"]) => void;
  onUndoSupplyCard: (cardRef: string) => void;
  onDirection: (direction: "up" | "down") => void;
  onPlan: (planId: string) => void;
  onAction: (actionId: number) => void;
  disabled: boolean;
  contentDisabled: boolean;
}) {
  return (
    <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock">
      <div className={styles.dockHeading}><SectionLabel>ACTION</SectionLabel><span>{view.checkpoint ? "" : prompt(view, supply, decision)}</span></div>
      <div className={styles.dockContents}>
        {view.supply_batch ? (
          <SupplyContents view={view} supply={supply} onSelectCard={onSupplyCard} onVisibility={onSupplyVisibility} onUndoCard={onUndoSupplyCard} disabled={contentDisabled} />
        ) : !view.checkpoint && !view.terminal_results ? (
          <DecisionContents view={view} decision={decision} onDirection={onDirection} onPlan={onPlan} onAction={onAction} disabled={contentDisabled} />
        ) : null}
      </div>
      <div className={styles.dockControls}>
        <span className={styles.controlSlot}>
          <TextButton data-dock-control="context" data-context-action={contextLabel.toLowerCase().replace(" ", "-")} data-resign-confirm={resignArmed || undefined} data-plan-id={contextPlanId} data-action-id={contextActionId} data-checkpoint-id={view.checkpoint?.checkpoint_id} data-checkpoint-kind={view.checkpoint?.kind} disabled={disabled || !contextEnabled} onClick={onContext}>{contextLabel}</TextButton>
        </span>
        <TextButton selected={resignArmed} data-dock-control="resign" data-resign disabled={disabled} onClick={onResign}>RESIGN</TextButton>
      </div>
    </section>
  );
}

function TerminalChart({ view }: { view: GameView }) {
  const results = view.terminal_results!;
  // Bars are cash after sell-off / movement, plus remaining stock still held
  // before terminal liquidation converts that position into final cash.
  const minimum = Math.min(0, ...results.players.map((player) => player.cash_before_liquidation_thousands));
  const maximum = Math.max(0, ...results.players.flatMap((player) => [
    player.liquidation_value_thousands,
    player.final_cash_thousands,
    player.liquidation_value_thousands + Math.max(0, player.cash_before_liquidation_thousands),
  ]));
  const span = Math.max(1, maximum - minimum);
  const percent = (value: number) => `${(value / span) * 100}%`;
  const zero = -minimum;
  return (
    <div className={styles.terminalChart} data-testid="terminal-chart" data-chart-min={minimum} data-chart-max={maximum}>
      {results.players.map((player) => {
        const cash = player.cash_before_liquidation_thousands;
        const position = player.liquidation_value_thousands;
        return (
          <div className={styles.chartRow} key={player.player_id} data-chart-player={player.player_id === view.viewer.player_id ? "human" : "computer"}>
            <span className={styles.chartLabel}>{player.player_id === view.viewer.player_id ? "YOU" : "COMPUTER"}</span>
            <span
              className={styles.chartTrack}
              role="img"
              aria-label={`${player.player_name}: position ${money(position)}, cash ${money(cash)}, final cash ${money(player.final_cash_thousands)}`}
              data-chart-position={position}
              data-chart-cash={cash}
            >
              <span
                className={styles.positionSegment}
                data-chart-segment="position"
                style={{
                  left: percent(zero),
                  width: percent(position),
                  minWidth: position > 0 ? 2 : undefined,
                }}
              />
              <span
                className={styles.cashSegment}
                data-chart-segment="cash"
                style={cash >= 0
                  ? { left: percent(zero + position), width: percent(cash), minWidth: cash > 0 ? 2 : undefined }
                  : { left: percent(zero + cash), width: percent(-cash), minWidth: 2 }}
              />
              <span className={styles.zeroLine} style={{ left: percent(zero) }} />
            </span>
          </div>
        );
      })}
    </div>
  );
}

function TerminalField({ view }: { view: GameView }) {
  const results = view.terminal_results!;
  return (
    <main className={`${styles.module} ${styles.stockpileField} ${styles.terminal}`} aria-label="Game end" data-testid="stockpile-field">
      <SectionLabel>GAME END</SectionLabel>
      <div className={styles.rankings}>
        {[...results.players].sort((a, b) => a.rank - b.rank).map((player) => (
          <div className={styles.ranking} key={player.player_id}>
            <span>#{player.rank}</span>
            <span>{player.player_id === view.viewer.player_id ? "YOU" : "COMPUTER"}</span>
            <span>FINAL CASH {money(player.final_cash_thousands)}</span>
            <div className={styles.liquidation}>
              <span>CASH {money(player.cash_before_liquidation_thousands)}</span>
              <span>LIQUIDATION <span className={styles.positive}>+{money(player.liquidation_value_thousands)}</span></span>
              {player.liquidation.filter((line) => line.shares_thousands > 0).map((line) => {
                const company = companyById(view, line.company_id);
                return <span key={line.company_id}>{company && <StockPattern pattern={company.pattern} />} {line.shares_thousands}K × {price(line.price_dollars_per_share)} = {money(line.value_thousands)}</span>;
              })}
            </div>
          </div>
        ))}
      </div>
      <TerminalChart view={view} />
    </main>
  );
}

function isHumanDecision(view: GameView) {
  return view.active_player_id === view.viewer.player_id
    && !view.checkpoint
    && !view.terminal_results
    && view.pending_decision.kind !== "waiting"
    && view.pending_decision.kind !== "private_selling"
    && Boolean(view.supply_batch || view.decision_batch || view.legal_actions.length);
}

export function GamePage({ gameId, token, navigate = (url: string) => window.location.assign(url) }: { gameId: string; token: string; navigate?: (url: string) => void }) {
  const { view: liveView, error, submitting, act, supply, decision: commitDecision, acknowledge, resign } = useGameSession(gameId, token);
  const [expandedPiles, setExpandedPiles] = useState<Set<number>>(new Set());
  const [supplyUi, setSupplyUi] = useState<SupplyUi>(EMPTY_SUPPLY);
  const [decisionUi, setDecisionUi] = useState<DecisionUi>(EMPTY_DECISION);
  const [backView, setBackView] = useState<GameView | null>(null);
  const [showBackView, setShowBackView] = useState(false);
  const [resignArmed, setResignArmed] = useState(false);
  const supplyKey = liveView?.supply_batch?.cards.map((card) => card.card_ref).join(":") ?? "";
  const decisionKey = liveView ? `${liveView.revision}:${liveView.decision_batch?.kind ?? liveView.pending_decision.kind}` : "";

  useEffect(() => { setSupplyUi(EMPTY_SUPPLY); }, [supplyKey]);
  useEffect(() => { setDecisionUi(EMPTY_DECISION); }, [decisionKey]);

  if (!liveView && !error) return <main className={styles.centerState}>OPENING GAME</main>;
  if (!liveView) return <main className={styles.centerState}><span>GAME UNAVAILABLE</span><TextButton onClick={() => window.location.assign("/")}>NEW GAME</TextButton></main>;

  const view = showBackView && backView ? backView : liveView;
  const supplyDraft = supplyUi.draft;
  const selectedSupplyAssignment = supplyUi.selectedCardRef ? supplyDraft[supplyUi.selectedCardRef] : undefined;
  const supplyTargets = new Set<number>();
  if (view.supply_batch && supplyUi.selectedCardRef && selectedSupplyAssignment?.visibility === undefined) {
    for (const plan of view.supply_batch.plans.filter((candidate) => planMatchesDraft(candidate, supplyDraft, { cardRef: supplyUi.selectedCardRef!, field: "stockpile_id" }))) {
      const placement = plan.placements.find((candidate) => candidate.card_ref === supplyUi.selectedCardRef);
      if (placement) supplyTargets.add(placement.stockpile_id);
    }
  }
  const demandTargets = new Set<number>(view.decision_batch?.kind === "demand" ? view.decision_batch.plans.map((plan) => plan.stockpile_id) : []);
  const actionTargets = new Map<number, number>();
  for (const action of view.legal_actions.filter((candidate) => candidate.control === "stockpile")) {
    const target = action.target_id?.match(/^stockpile:(\d+)$/);
    if (target) actionTargets.set(Number(target[1]), action.action_id);
  }
  const selectablePiles = new Set([...supplyTargets, ...demandTargets, ...actionTargets.keys()]);
  const tentativeByPile = tentativePlacements(view.supply_batch, supplyDraft);
  const exactPlan = view.supply_batch ? exactSupplyPlan(view.supply_batch, supplyDraft) : undefined;
  const supplyPartial = Boolean(supplyUi.selectedCardRef || Object.keys(supplyDraft).length);
  const decisionPartial = decisionUi.actionId !== null || decisionUi.planId !== null || decisionUi.stockpileId !== null || decisionUi.direction !== null;
  const interactionDisabled = submitting || resignArmed || showBackView;

  function beginDecision() {
    setBackView(null);
    setShowBackView(false);
  }
  function updateSupply(updater: (current: SupplySnapshot) => SupplySnapshot) {
    beginDecision();
    setSupplyUi((current) => {
      const before = supplySnapshot(current);
      const next = updater(before);
      return { ...next, history: [...current.history, before] };
    });
  }
  function updateDecision(updater: (current: DecisionSnapshot) => DecisionSnapshot) {
    beginDecision();
    setDecisionUi((current) => {
      const before = decisionSnapshot(current);
      const next = updater(before);
      return { ...next, history: [...current.history, before] };
    });
  }
  function undoSupply() {
    setSupplyUi((current) => {
      const prior = current.history.at(-1) ?? { selectedCardRef: null, draft: {} };
      return { ...prior, history: current.history.slice(0, -1) };
    });
  }
  function undoDecision() {
    setDecisionUi((current) => {
      const prior = current.history.at(-1) ?? decisionSnapshot(EMPTY_DECISION);
      return { ...prior, history: current.history.slice(0, -1) };
    });
  }
  function removeSupplyCard(cardRef: string) {
    updateSupply((current) => {
      const draft = { ...current.draft };
      delete draft[cardRef];
      return { selectedCardRef: null, draft };
    });
  }
  function chooseSupplyPile(stockpileId: number) {
    if (!supplyUi.selectedCardRef || !supplyTargets.has(stockpileId)) return;
    const cardRef = supplyUi.selectedCardRef;
    updateSupply((current) => ({
      selectedCardRef: cardRef,
      draft: { ...current.draft, [cardRef]: { ...current.draft[cardRef], stockpile_id: stockpileId } },
    }));
  }
  function chooseSupplyVisibility(visibility: SupplyPlacement["visibility"]) {
    const cardRef = supplyUi.selectedCardRef;
    if (!view.supply_batch || !cardRef) return;
    const matchingPlan = view.supply_batch.plans.find((plan) =>
      planMatchesDraft(plan, supplyDraft, { cardRef, field: "visibility" })
      && plan.placements.some((placement) => placement.card_ref === cardRef && placement.visibility === visibility)
    );
    if (!matchingPlan) return;
    updateSupply((current) => {
      const draft = { ...current.draft };
      draft[cardRef] = { ...draft[cardRef], visibility };
      return { selectedCardRef: null, draft };
    });
  }
  function selectPile(stockpileId: number) {
    if (supplyTargets.has(stockpileId)) return chooseSupplyPile(stockpileId);
    if (demandTargets.has(stockpileId)) {
      updateDecision((current) => ({ ...current, stockpileId: current.stockpileId === stockpileId ? null : stockpileId, planId: null }));
      return;
    }
    const actionId = actionTargets.get(stockpileId);
    if (actionId !== undefined) updateDecision((current) => ({ ...current, actionId: current.actionId === actionId ? null : actionId }));
  }
  function selectAction(actionId: number) {
    updateDecision((current) => ({ ...current, actionId: current.actionId === actionId ? null : actionId }));
  }
  function selectPlan(planId: string) {
    updateDecision((current) => ({ ...current, planId: current.planId === planId ? null : planId }));
  }
  function selectDirection(direction: "up" | "down") {
    updateDecision((current) => ({ ...current, direction: current.direction === direction ? null : direction, planId: null }));
  }

  let contextLabel: string | null = null;
  if (resignArmed) contextLabel = "CONFIRM";
  else if (showBackView) contextLabel = "CONTINUE";
  else if (view.checkpoint) contextLabel = "CONTINUE";
  else if (exactPlan || decisionUi.actionId !== null || decisionUi.planId !== null) contextLabel = "CONFIRM";
  else if (supplyPartial || decisionPartial) contextLabel = "UNDO";
  else if (backView && !view.terminal_results) contextLabel = "BACK";
  else if (view.terminal_results) contextLabel = "CONTINUE";
  const contextEnabled = contextLabel !== null;
  const displayedContextLabel = contextLabel ?? (isHumanDecision(view) ? "BACK" : "CONTINUE");

  async function handleContext() {
    if (resignArmed) {
      if (await resign()) {
        clearSeatToken(gameId);
        navigate("/");
      }
      return;
    }
    if (showBackView) return void setShowBackView(false);
    if (view.checkpoint) {
      const previous = view;
      const next = await acknowledge(view.checkpoint.checkpoint_id);
      if (next && isHumanDecision(next)) {
        setBackView(previous);
        setShowBackView(false);
      } else if (next) {
        setBackView(null);
        setShowBackView(false);
      }
      return;
    }
    if (exactPlan) {
      if (await supply(exactPlan.plan_id)) setSupplyUi(EMPTY_SUPPLY);
      return;
    }
    if (decisionUi.planId !== null) {
      if (await commitDecision(decisionUi.planId)) setDecisionUi(EMPTY_DECISION);
      return;
    }
    if (decisionUi.actionId !== null) {
      if (await act(decisionUi.actionId)) setDecisionUi(EMPTY_DECISION);
      return;
    }
    if (supplyPartial) return void undoSupply();
    if (decisionPartial) return void undoDecision();
    if (backView) return void setShowBackView(true);
    if (view.terminal_results) navigate("/");
  }

  return (
    <div className={styles.game}>
      {error && <div className={styles.errorBanner} role="alert">{error}</div>}
      <div className={styles.workstation} data-testid="workstation" data-decision-kind={view.pending_decision.kind} data-checkpoint-kind={view.checkpoint?.kind}>
        <Status view={view} />
        <Market view={view} selectedActionId={decisionUi.actionId} selectedPlanId={decisionUi.planId} selectedDirection={decisionUi.direction} onSelectAction={selectAction} onSelectImpactCompany={selectPlan} disabled={interactionDisabled} />
        <Research view={view} served={view.checkpoint?.kind === "round_result" || Boolean(view.terminal_results)} />
        {view.terminal_results ? <TerminalField view={view} /> : (
          <StockpileField
            view={view}
            expanded={expandedPiles}
            selectedStockpileId={selectedSupplyAssignment?.stockpile_id ?? decisionUi.stockpileId}
            selectablePiles={selectablePiles}
            tentativeByPile={tentativeByPile}
            onSelectPile={selectPile}
            onInspect={(id) => setExpandedPiles((current) => {
              const next = new Set(current);
              if (next.has(id)) next.delete(id); else next.add(id);
              return next;
            })}
            onUndoTentative={removeSupplyCard}
            disabled={interactionDisabled}
          />
        )}
        <Portfolio view={view} served={Boolean(view.terminal_results)} />
        <Players view={view} />
        <ActionDock
          view={view}
          supply={supplyUi}
          decision={decisionUi}
          contextLabel={displayedContextLabel}
          contextEnabled={contextEnabled}
          contextPlanId={exactPlan?.plan_id ?? decisionUi.planId ?? undefined}
          contextActionId={decisionUi.actionId ?? undefined}
          onContext={() => void handleContext()}
          onResign={() => setResignArmed((current) => !current)}
          resignArmed={resignArmed}
          onSupplyCard={(cardRef) => updateSupply((current) => ({ selectedCardRef: current.selectedCardRef === cardRef ? null : cardRef, draft: current.draft }))}
          onSupplyVisibility={chooseSupplyVisibility}
          onUndoSupplyCard={removeSupplyCard}
          onDirection={selectDirection}
          onPlan={selectPlan}
          onAction={selectAction}
          disabled={submitting}
          contentDisabled={interactionDisabled}
        />
      </div>
    </div>
  );
}

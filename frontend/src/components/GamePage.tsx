import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type {
  Card,
  Company,
  GameView,
  LegalAction,
  MarketEvent,
  PileCard,
  Stockpile,
  SupplyBatch,
  SupplyPlacement,
} from "../types";
import { useGameSession } from "../useGameSession";
import { CardView, CompanyCard, HoldingCard } from "./CardView";
import { SectionLabel, TextButton } from "./Primitives";
import { Selectable } from "./Selectable";
import { StockPattern } from "./StockPattern";
import styles from "./Game.module.css";

type SupplyDraft = Record<string, Partial<Pick<SupplyPlacement, "stockpile_id" | "visibility">>>;

function money(value: number) { return `$${value}K`; }
function price(value: number) { return `$${value} / SHARE`; }
function targetNumber(target: string | null, prefix: string) {
  if (!target?.startsWith(`${prefix}:`)) return null;
  const value = Number(target.slice(prefix.length + 1));
  return Number.isFinite(value) ? value : null;
}
function companyById(view: GameView, companyId: number | null) {
  return companyId === null ? undefined : view.companies.find((company) => company.company_id === companyId);
}

function Status({ view }: { view: GameView }) {
  let phase = view.phase.toUpperCase();
  let turn = "WAIT";
  if (view.terminal_results) {
    phase = "GAME END";
    turn = "";
  } else if (view.checkpoint) {
    phase = view.checkpoint.kind === "demand_result" ? "DEMAND RESULT" : "ROUND RESULT";
    turn = "CONTINUE";
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

function Market({ view, movements, onAction, disabled }: {
  view: GameView;
  movements: MarketEvent[];
  onAction: (actionId: number) => void;
  disabled: boolean;
}) {
  const actions = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((candidate) => candidate.control === "company")) {
    const companyId = targetNumber(action.target_id, "company");
    if (companyId !== null) actions.set(companyId, action);
  }
  return (
    <section className={`${styles.module} ${styles.market}`} aria-label="Market">
      <SectionLabel>MARKET</SectionLabel>
      <div className={styles.marketList}>
        {view.companies.map((company) => {
          const action = actions.get(company.company_id);
          const movement = movements.find((event) => event.company_id === company.company_id);
          const body = (
            <span className={styles.marketCompany} data-company-id={company.company_id}>
              <StockPattern pattern={company.pattern} />
              <span className={styles.marketName}>{company.display_name}</span>
              <span className={styles.marketPrice}>{price(company.price_dollars_per_share)}</span>
              {movement?.price_delta != null && movement.price_delta !== 0 && (
                <span className={movement.price_delta > 0 ? styles.positive : styles.negative} aria-live="polite">
                  {movement.price_delta > 0 ? "↑" : "↓"}{Math.abs(movement.price_delta)}
                </span>
              )}
            </span>
          );
          return action ? (
            <Selectable
              key={company.company_id}
              className={styles.objectControl}
              data-action-id={action.action_id}
              aria-label={action.label}
              disabled={disabled}
              onClick={() => onAction(action.action_id)}
            >
              {body}
            </Selectable>
          ) : <span key={company.company_id}>{body}</span>;
        })}
      </div>
    </section>
  );
}

function InformationPair({ card, companies }: { card: Card; companies: Company[] }) {
  if (card.visibility === "hidden") {
    return <div className={styles.informationPair}><CardView card={card} companies={companies} scale="information" /><CardView card={card} companies={companies} scale="information" /></div>;
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

function MarketInformation({ view, visibility }: { view: GameView; visibility: "private" | "public" }) {
  const privateSlots = view.private.market_information.filter((slot) => slot.visibility === "private");
  const publicSlots = view.private.market_information.filter((slot) => slot.visibility === "public");
  const hiddenSlots = view.private.market_information.filter((slot) => slot.visibility === "hidden");
  const completedRoundReveals = visibility === "public" && view.checkpoint?.kind === "round_result"
    ? view.recent_events.filter((event) => event.round === view.checkpoint?.round && event.company_id !== null && event.forecast !== null)
      .map((event) => ({
        visibility: "public" as const,
        card: {
          visibility: "visible" as const,
          kind: "company_forecast" as const,
          company_id: event.company_id!,
          company: companyById(view, event.company_id)?.name ?? "",
          forecast: event.forecast!,
          cash_effect_thousands: event.cash_effect_thousands,
        },
      }))
    : [];
  const publicByFact = new Map<string, (typeof publicSlots)[number]>();
  for (const slot of [...publicSlots, ...completedRoundReveals]) {
    if (slot.card.visibility === "hidden") continue;
    publicByFact.set(`${slot.card.company_id}:${slot.card.forecast}`, slot);
  }
  const visiblePublic = [...publicByFact.values()];
  const slots = visibility === "private"
    ? privateSlots
    : [...visiblePublic, ...hiddenSlots.slice(0, Math.max(0, view.companies.length - visiblePublic.length))];
  const label = visibility.toUpperCase();
  return (
    <section className={`${styles.module} ${visibility === "private" ? styles.privateInformation : styles.publicInformation}`} aria-label={`${label[0]}${label.slice(1).toLowerCase()} information`}>
      <SectionLabel>{label}</SectionLabel>
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

function StackEdge({ card, companies }: { card: Card; companies: Company[] }) {
  if (card.visibility === "hidden") return null;
  if (card.kind === "stock") {
    const company = companies.find((candidate) => candidate.company_id === card.company_id);
    return company ? <StockPattern pattern={company.pattern} /> : null;
  }
  if (card.kind === "trading_fee") return <span className={styles.negative}>−$</span>;
  if (card.kind === "action") return <span className={card.direction === "up" ? styles.positive : styles.negative}>{card.direction === "up" ? "↑" : "↓"}</span>;
  if (card.forecast === "DIVIDEND") return <span className={styles.positive}>+$</span>;
  if (card.forecast > 0) return <span className={styles.positive}>↑</span>;
  if (card.forecast < 0) return <span className={styles.negative}>↓</span>;
  return <span>0</span>;
}

function Stack({ cards, companies, expanded }: { cards: PileCard[]; companies: Company[]; expanded: boolean }) {
  return (
    <span
      className={`${styles.stack} ${expanded ? styles.stackExpanded : ""}`}
      style={{ "--stack-count": Math.max(cards.length, 1) } as CSSProperties}
    >
      {cards.map((entry, index) => {
        const display = visiblePileCard(entry, expanded);
        return (
          <span
            key={index}
            className={styles.stackLayer}
            data-stack-card
            data-stack-order={index}
            style={{ "--stack-index": index, zIndex: index + 1 } as CSSProperties}
          >
            <CardView card={display.card} companies={companies} scale="stockpile" faceDownKnown={display.faceDownKnown} />
            {!expanded && index < cards.length - 1 && (display.card.visibility === "hidden"
              ? <span className={styles.hiddenStackEdge} aria-hidden="true" />
              : <span className={styles.stackEdge} aria-hidden="true"><StackEdge card={display.card} companies={companies} /></span>)}
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

function StockpileItem({ pile, view, action, supplyTarget, supplySelected, expanded, onAction, onSupplyPile, onInspect, disabled }: {
  pile: Stockpile;
  view: GameView;
  action?: LegalAction;
  supplyTarget: boolean;
  supplySelected: boolean;
  expanded: boolean;
  onAction: (actionId: number) => void;
  onSupplyPile: (stockpileId: number) => void;
  onInspect: (stockpileId: number) => void;
  disabled: boolean;
}) {
  const selected = supplySelected || view.pending_decision.selected_stockpile_id === pile.stockpile_id;
  const target = supplyTarget || Boolean(action);
  return (
    <article className={styles.stockpile} aria-label={`Stockpile ${pile.stockpile_id + 1}`} data-stockpile-id={pile.stockpile_id}>
      {target && (
        <button
          type="button"
          className={styles.pileTarget}
          data-supply-pile-target={supplyTarget ? pile.stockpile_id : undefined}
          data-action-id={action?.action_id}
          aria-label={supplyTarget ? `Place selected card in stockpile ${pile.stockpile_id + 1}` : action?.label}
          aria-pressed={selected}
          disabled={disabled}
          onClick={() => supplyTarget ? onSupplyPile(pile.stockpile_id) : action && onAction(action.action_id)}
        />
      )}
      <button
        type="button"
        className={styles.stackInspect}
        data-stack-inspect={pile.stockpile_id}
        aria-label={`${expanded ? "Collapse" : "Expand"} stockpile ${pile.stockpile_id + 1}`}
        aria-expanded={expanded}
        onClick={() => onInspect(pile.stockpile_id)}
      >
        <Stack cards={pile.cards_bottom_to_top} companies={view.companies} expanded={expanded} />
      </button>
      {pile.bid && <span className={styles.bid}>{bidderName(view, pile)} {money(pile.bid.amount_thousands)}</span>}
    </article>
  );
}

function StockpileField({ view, expanded, onInspect, onAction, disabled, supplyTargets, selectedSupplyPile, onSupplyPile }: {
  view: GameView;
  expanded: Set<number>;
  onInspect: (stockpileId: number) => void;
  onAction: (actionId: number) => void;
  disabled: boolean;
  supplyTargets: Set<number>;
  selectedSupplyPile: number | null;
  onSupplyPile: (stockpileId: number) => void;
}) {
  const actions = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((candidate) => candidate.control === "stockpile")) {
    const pileId = targetNumber(action.target_id, "stockpile");
    if (pileId !== null) actions.set(pileId, action);
  }
  return (
    <main className={`${styles.module} ${styles.stockpileField}`} aria-label="Stockpiles" data-testid="stockpile-field">
      <div className={styles.stockpileGrid}>
        {view.stockpiles.map((pile) => (
          <StockpileItem
            key={pile.stockpile_id}
            pile={pile}
            view={view}
            action={actions.get(pile.stockpile_id)}
            supplyTarget={supplyTargets.has(pile.stockpile_id)}
            supplySelected={selectedSupplyPile === pile.stockpile_id}
            expanded={expanded.has(pile.stockpile_id)}
            onAction={onAction}
            onSupplyPile={onSupplyPile}
            onInspect={onInspect}
            disabled={disabled}
          />
        ))}
      </div>
    </main>
  );
}

function Portfolio({ view }: { view: GameView }) {
  return (
    <section className={`${styles.module} ${styles.portfolio}`} aria-label="Portfolio">
      <SectionLabel>PORTFOLIO</SectionLabel>
      <div className={styles.portfolioCards}>
        {view.private.holdings.filter((holding) => holding.shares_thousands > 0).map((holding) => {
          const company = companyById(view, holding.company_id);
          return company ? <HoldingCard key={holding.company_id} company={company} sharesThousands={holding.shares_thousands} /> : null;
        })}
      </div>
    </section>
  );
}

function Delta({ value }: { value: number | null }) {
  if (!value) return null;
  return <span className={value > 0 ? styles.positive : styles.negative}>{value > 0 ? "+" : "−"}${Math.abs(value)}K</span>;
}

function Players({ view }: { view: GameView }) {
  return (
    <section className={`${styles.module} ${styles.players}`} aria-label="Players">
      <SectionLabel>PLAYERS</SectionLabel>
      <div className={styles.playerList}>
        {[...view.players].sort((a, b) => a.role === "human" ? -1 : b.role === "human" ? 1 : 0).map((player) => (
          <div className={styles.player} key={player.player_id} data-player-role={player.role}>
            <span>{player.role === "human" ? "YOU" : "COMPUTER"}</span>
            <div className={styles.metric} data-player-metric="cash"><span>CASH</span><span>{money(player.cash_thousands)}</span><Delta value={player.cash_delta_thousands} /></div>
            {player.role === "human" && <div className={styles.metric} data-player-metric="position"><span>POSITION</span><span>{money(player.position_value_thousands)}</span><Delta value={player.position_delta_thousands} /></div>}
          </div>
        ))}
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
  return batch.plans.find((plan) => batch.cards.every(({ card_ref }) => {
    const assignment = draft[card_ref];
    const placement = plan.placements.find((candidate) => candidate.card_ref === card_ref);
    return assignment?.stockpile_id !== undefined && assignment.visibility !== undefined && placement?.stockpile_id === assignment.stockpile_id && placement.visibility === assignment.visibility;
  }));
}

function SupplyDock({ view, draft, selectedCardRef, setSelectedCardRef, setDraft, onConfirm, disabled }: {
  view: GameView;
  draft: SupplyDraft;
  selectedCardRef: string | null;
  setSelectedCardRef: (cardRef: string) => void;
  setDraft: (updater: (current: SupplyDraft) => SupplyDraft) => void;
  onConfirm: (planId: string) => void;
  disabled: boolean;
}) {
  const batch = view.supply_batch!;
  const selected = batch.cards.find((item) => item.card_ref === selectedCardRef);
  const allowedVisibility = new Set(selected ? batch.plans
    .filter((plan) => planMatchesPileAssignments(plan, draft))
    .map((plan) => plan.placements.find((placement) => placement.card_ref === selected.card_ref)?.visibility)
    .filter((value): value is SupplyPlacement["visibility"] => value !== undefined) : []);
  const complete = exactSupplyPlan(batch, draft);

  function chooseVisibility(visibility: SupplyPlacement["visibility"]) {
    if (!selected) return;
    const matchingPlan = batch.plans.find((plan) =>
      planMatchesPileAssignments(plan, draft)
      && plan.placements.some((placement) => placement.card_ref === selected.card_ref && placement.visibility === visibility)
    );
    if (!matchingPlan) return;
    setDraft((current) => {
      const next = { ...current };
      for (const placement of matchingPlan.placements) {
        next[placement.card_ref] = {
          ...next[placement.card_ref],
          visibility: placement.visibility,
        };
      }
      return next;
    });
  }

  return (
    <>
      <div className={styles.dockHeading}><SectionLabel>ACTION</SectionLabel><span>{selected ? "PLACE" : "SELECT CARD"}</span></div>
      <div className={styles.dockContents}>
        {batch.cards.map(({ card_ref, card }) => {
          const assignment = draft[card_ref];
          return (
            <Selectable
              key={card_ref}
              selected={selectedCardRef === card_ref}
              className={styles.cardAction}
              data-supply-card-ref={card_ref}
              data-assigned-pile={assignment?.stockpile_id}
              data-assigned-visibility={assignment?.visibility}
              aria-label={`Supply card ${card_ref}`}
              disabled={disabled}
              onClick={() => setSelectedCardRef(card_ref)}
            >
              <CardView card={card} companies={view.companies} scale="active" />
              {assignment?.stockpile_id !== undefined && assignment.visibility && <span className={styles.assignment}>PILE {assignment.stockpile_id + 1} · {assignment.visibility === "face_up" ? "FACE UP" : "FACE DOWN"}</span>}
            </Selectable>
          );
        })}
        {selected && (["face_up", "face_down"] as const).map((visibility) => allowedVisibility.has(visibility) && (
          <TextButton
            key={visibility}
            selected={draft[selected.card_ref]?.visibility === visibility}
            data-supply-visibility={visibility}
            disabled={disabled}
            onClick={() => chooseVisibility(visibility)}
          >
            {visibility === "face_up" ? "FACE UP" : "FACE DOWN"}
          </TextButton>
        ))}
        {complete && <TextButton data-supply-confirm data-plan-id={complete.plan_id} disabled={disabled} onClick={() => onConfirm(complete.plan_id)}>CONFIRM</TextButton>}
      </div>
    </>
  );
}

function prompt(view: GameView) {
  const human = view.players.find((player) => player.role === "human");
  const rebid = human?.bid_markers.some((marker) => marker.status === "outbid" || marker.status === "rebidding");
  switch (view.pending_decision.kind) {
    case "bid_pile": return rebid ? "REBID" : "SELECT PILE";
    case "bid_amount": return rebid ? "REBID" : "BID";
    case "action_card": return "SELECT CARD";
    case "action_company": return "SELECT COMPANY";
    case "sell": return "SELL?";
    case "dividend_claim": return "DIVIDEND";
    case "acknowledge": return "CONTINUE";
    case "waiting":
    case "private_selling": return "WAIT";
    case "terminal": return "";
    case "supply": return "SELECT CARD";
    case "generic": return view.legal_actions.length ? "CONTINUE" : "WAIT";
  }
}

function actionText(action: LegalAction) {
  if (action.control === "bid" && action.amount_thousands !== null) return money(action.amount_thousands);
  if (action.amount_thousands !== null) return money(action.amount_thousands);
  return action.label.toUpperCase();
}

function ActionDock({ view, draft, selectedCardRef, setSelectedCardRef, setDraft, onAction, onSupply, onAcknowledge, disabled }: {
  view: GameView;
  draft: SupplyDraft;
  selectedCardRef: string | null;
  setSelectedCardRef: (cardRef: string) => void;
  setDraft: (updater: (current: SupplyDraft) => SupplyDraft) => void;
  onAction: (actionId: number) => void;
  onSupply: (planId: string) => void;
  onAcknowledge: (checkpointId: string) => void;
  disabled: boolean;
}) {
  if (view.checkpoint) {
    return (
      <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock">
        <div className={styles.dockHeading}><SectionLabel>ACTION</SectionLabel></div>
        <div className={styles.dockContents}>
          <TextButton data-checkpoint-continue data-checkpoint-kind={view.checkpoint.kind} disabled={disabled} onClick={() => onAcknowledge(view.checkpoint!.checkpoint_id)}>CONTINUE</TextButton>
        </div>
      </section>
    );
  }
  if (view.supply_batch) {
    return <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock"><SupplyDock view={view} draft={draft} selectedCardRef={selectedCardRef} setSelectedCardRef={setSelectedCardRef} setDraft={setDraft} onConfirm={onSupply} disabled={disabled} /></section>;
  }
  if (view.terminal_results) {
    return <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock"><SectionLabel>ACTION</SectionLabel><div className={styles.dockContents}><TextButton onClick={() => window.location.assign("/")}>NEW GAME</TextButton></div></section>;
  }

  const inline = view.legal_actions.filter((action) => action.control !== "stockpile" && action.control !== "company");
  const selectedImpact = view.private.available_action_cards.find((card) => card.effect.toLowerCase() === view.pending_decision.selected_action_effect?.toLowerCase());
  const selectedHolding = view.private.holdings.find((holding) => holding.company_id === view.pending_decision.company_id);
  const holdingCompany = selectedHolding ? companyById(view, selectedHolding.company_id) : undefined;
  return (
    <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock">
      <div className={styles.dockHeading}><SectionLabel>ACTION</SectionLabel><span>{prompt(view)}</span></div>
      <div className={styles.dockContents}>
        {selectedImpact && <CardView card={selectedImpact} companies={view.companies} scale="active" />}
        {view.pending_decision.kind === "sell" && selectedHolding && holdingCompany && <HoldingCard company={holdingCompany} sharesThousands={selectedHolding.shares_thousands} scale="active" />}
        {inline.map((action) => {
          if (action.control === "action_card") {
            const effect = action.target_id?.split(":")[1]?.toLowerCase();
            const card = view.private.available_action_cards.find((candidate) => candidate.effect.toLowerCase() === effect);
            return (
              <Selectable key={action.action_id} className={styles.cardAction} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>
                {card ? <CardView card={card} companies={view.companies} scale="active" /> : actionText(action)}
              </Selectable>
            );
          }
          if (action.control === "sell" && action.sale_preview) {
            const sale = action.sale_preview;
            return (
              <TextButton key={action.action_id} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>
                {sale.shares_thousands === 0 ? "HOLD" : <><span>SELL {sale.shares_thousands}K</span> <span className={styles.positive}>+{money(sale.gross_value_thousands)}</span></>}
              </TextButton>
            );
          }
          if (action.control === "dividend") {
            return <TextButton key={action.action_id} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{action.label.toUpperCase()}</TextButton>;
          }
          return <TextButton key={action.action_id} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{actionText(action)}</TextButton>;
        })}
      </div>
    </section>
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
            {player.winner && <span>WINNER</span>}
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
    </main>
  );
}

function currentMovements(view: GameView) {
  const events = view.recent_events.filter((event) => event.company_id !== null && event.price_delta !== null);
  if (!events.length) return [];
  const last = events.at(-1)!;
  return events.filter((event) => event.round === last.round && event.cause === last.cause);
}

export function GamePage({ gameId, token }: { gameId: string; token: string }) {
  const { view, error, submitting, act, supply, acknowledge } = useGameSession(gameId, token);
  const [expandedPiles, setExpandedPiles] = useState<Set<number>>(new Set());
  const [selectedSupplyCard, setSelectedSupplyCard] = useState<string | null>(null);
  const [supplyDraft, setSupplyDraft] = useState<SupplyDraft>({});
  const supplyKey = view?.supply_batch?.cards.map((card) => card.card_ref).join(":") ?? "";

  useEffect(() => {
    setSelectedSupplyCard(null);
    setSupplyDraft({});
  }, [supplyKey]);

  const movementBatch = useMemo(() => view ? currentMovements(view) : [], [view]);
  const movementKey = movementBatch.map((event) => event.event_id).join(":");
  const movementCheckpointKind = view?.checkpoint?.kind;
  const hasView = view !== null;
  const [movements, setMovements] = useState<MarketEvent[]>([]);
  const movementViewInitialized = useRef(false);
  useEffect(() => {
    if (!hasView) return;
    if (!movementViewInitialized.current) {
      movementViewInitialized.current = true;
      setMovements(movementCheckpointKind === "round_result" ? movementBatch : []);
      return;
    }
    if (!movementKey) {
      setMovements([]);
      return;
    }
    setMovements(movementBatch);
    if (movementCheckpointKind === "round_result") return;
    const timer = window.setTimeout(() => setMovements([]), 2400);
    return () => window.clearTimeout(timer);
  }, [hasView, movementCheckpointKind, movementKey]);

  if (!view && !error) return <main className={styles.centerState}>OPENING GAME</main>;
  if (!view) return <main className={styles.centerState}><span>GAME UNAVAILABLE</span><span>{error}</span><TextButton onClick={() => window.location.assign("/")}>NEW GAME</TextButton></main>;

  const selectedAssignment = selectedSupplyCard ? supplyDraft[selectedSupplyCard] : undefined;
  const supplyTargets = new Set<number>();
  if (view.supply_batch && selectedSupplyCard) {
    for (const plan of view.supply_batch.plans.filter((candidate) => planMatchesDraft(candidate, supplyDraft, { cardRef: selectedSupplyCard, field: "stockpile_id" }))) {
      const placement = plan.placements.find((candidate) => candidate.card_ref === selectedSupplyCard);
      if (placement) supplyTargets.add(placement.stockpile_id);
    }
  }

  function chooseSupplyPile(stockpileId: number) {
    if (!selectedSupplyCard || !supplyTargets.has(stockpileId)) return;
    setSupplyDraft((current) => ({ ...current, [selectedSupplyCard]: { ...current[selectedSupplyCard], stockpile_id: stockpileId } }));
  }

  return (
    <div className={styles.game}>
      {error && <div className={styles.errorBanner} role="alert">{error}</div>}
      <div className={`${styles.workstation} ${view.terminal_results ? styles.workstationTerminal : ""}`} data-testid="workstation" data-decision-kind={view.pending_decision.kind} data-checkpoint-kind={view.checkpoint?.kind}>
        <Status view={view} />
        <Market view={view} movements={movements} onAction={(id) => void act(id)} disabled={submitting} />
        <MarketInformation view={view} visibility="private" />
        <MarketInformation view={view} visibility="public" />
        {view.terminal_results ? <TerminalField view={view} /> : (
          <StockpileField
            view={view}
            expanded={expandedPiles}
            onInspect={(id) => setExpandedPiles((current) => {
              const next = new Set(current);
              if (next.has(id)) next.delete(id); else next.add(id);
              return next;
            })}
            onAction={(id) => void act(id)}
            disabled={submitting}
            supplyTargets={supplyTargets}
            selectedSupplyPile={selectedAssignment?.stockpile_id ?? null}
            onSupplyPile={chooseSupplyPile}
          />
        )}
        {!view.terminal_results && <Portfolio view={view} />}
        {!view.terminal_results && <Players view={view} />}
        <ActionDock
          view={view}
          draft={supplyDraft}
          selectedCardRef={selectedSupplyCard}
          setSelectedCardRef={setSelectedSupplyCard}
          setDraft={setSupplyDraft}
          onAction={(id) => void act(id)}
          onSupply={(planId) => void supply(planId)}
          onAcknowledge={(checkpointId) => void acknowledge(checkpointId)}
          disabled={submitting}
        />
      </div>
    </div>
  );
}

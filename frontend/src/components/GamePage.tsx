import { useEffect, useMemo, useState, type CSSProperties } from "react";
import type { Card, Company, GameView, LegalAction, Stockpile } from "../types";
import { useGameSession } from "../useGameSession";
import { CardView, CompanyCard, HoldingCard } from "./CardView";
import { Selectable } from "./Selectable";
import { StockPattern } from "./StockPattern";
import styles from "./Game.module.css";

type MarketEvent = GameView["recent_events"][number];

function money(value: number) {
  return `$${value}K`;
}

function idNumber(target: string | null, prefix: string) {
  if (!target?.startsWith(`${prefix}:`)) return null;
  const result = Number(target.slice(prefix.length + 1));
  return Number.isFinite(result) ? result : null;
}

function companyById(view: GameView, companyId: number | null) {
  return companyId === null ? undefined : view.companies.find((company) => company.company_id === companyId);
}

function Status({ view }: { view: GameView }) {
  const actor = view.players.find((player) => player.player_id === view.active_player_id);
  let turn = "WAIT";
  if (view.terminal_results) turn = "GAME END";
  else if (view.pending_decision.kind === "private_selling") turn = "PRIVATE SELLING";
  else if (actor?.player_id === view.viewer.player_id) turn = "YOUR TURN";
  else if (actor) turn = `${actor.name.toUpperCase()} TURN`;
  return (
    <header className={`${styles.module} ${styles.status}`} aria-label="Status">
      <span>ROUND {view.round} / {view.total_rounds}</span>
      <span>{view.phase.toUpperCase()}</span>
      <span>{turn}</span>
    </header>
  );
}

function Market({ view, movements, onAction, disabled }: { view: GameView; movements: MarketEvent[]; onAction: (id: number) => void; disabled: boolean }) {
  const targets = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((item) => item.control === "company")) {
    const companyId = idNumber(action.target_id, "company");
    if (companyId !== null) targets.set(companyId, action);
  }
  return (
    <section className={`${styles.module} ${styles.market}`} aria-label="Market">
      <span className={styles.moduleLabel}>MARKET</span>
      <div className={styles.marketList}>
        {view.companies.map((company) => {
          const action = targets.get(company.company_id);
          const movement = movements.find((event) => event.company_id === company.company_id);
          const item = (
            <div className={styles.marketCompany}>
              <StockPattern pattern={company.pattern} />
              <span className={styles.marketName}>{company.display_name}</span>
              <span className={styles.marketPrice}>{company.price}</span>
              {movement?.actual_delta !== null && movement?.actual_delta !== undefined && (
                <span aria-live="polite" className={movement.actual_delta > 0 ? styles.priceUp : styles.priceDown}>
                  {movement.actual_delta > 0 ? "↑" : "↓"}{Math.abs(movement.actual_delta)}
                </span>
              )}
            </div>
          );
          return action ? (
            <Selectable key={company.company_id} className={styles.companyTarget} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{item}</Selectable>
          ) : <div key={company.company_id}>{item}</div>;
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
  if (card.kind !== "company_forecast") return <CardView card={card} companies={companies} scale="information" />;
  const company = companies.find((item) => item.company_id === card.company_id);
  return (
    <div className={styles.informationPair}>
      {company ? <CompanyCard company={company} /> : <span />}
      <CardView card={card} companies={companies} scale="information" />
    </div>
  );
}

function PrivateInformation({ view }: { view: GameView }) {
  const slots = view.private.market_information.filter((slot) => slot.visibility === "private");
  return (
    <section className={`${styles.module} ${styles.privateInformation}`} aria-label="Private information">
      <span className={styles.moduleLabel}>PRIVATE</span>
      <div className={styles.informationList}>{slots.map((slot, index) => <InformationPair key={index} card={slot.card} companies={view.companies} />)}</div>
      {view.private.known_pile_cards.length > 0 && (
        <div className={styles.knownCards} aria-label="Private pile knowledge">
          {view.private.known_pile_cards.map((item, index) => (
            <div key={`${item.stockpile_id}:${index}`}><span>S{item.stockpile_id + 1}</span><CardView card={item.card} companies={view.companies} scale="information" /></div>
          ))}
        </div>
      )}
    </section>
  );
}

function PublicInformation({ view }: { view: GameView }) {
  const slots = view.private.market_information.filter((slot) => slot.visibility !== "private");
  return (
    <section className={`${styles.module} ${styles.publicInformation}`} aria-label="Public information">
      <span className={styles.moduleLabel}>PUBLIC</span>
      <div className={styles.informationList}>{slots.map((slot, index) => <InformationPair key={index} card={slot.card} companies={view.companies} />)}</div>
    </section>
  );
}

function bidderLabel(view: GameView, pile: Stockpile) {
  if (!pile.marker) return null;
  if (pile.marker.player_id === view.viewer.player_id) return "YOU";
  return view.players.find((player) => player.player_id === pile.marker?.player_id)?.name.toUpperCase() ?? `P${pile.marker.player_id + 1}`;
}

function Stack({ cards, companies }: { cards: Card[]; companies: Company[] }) {
  return (
    <div className={styles.stack} style={{ "--stack-count": Math.max(cards.length, 1) } as CSSProperties}>
      {cards.map((card, index) => (
        <div key={index} className={styles.stackLayer} data-stack-card style={{ "--stack-index": index, zIndex: cards.length - index } as CSSProperties}>
          <CardView card={card} companies={companies} scale="stockpile" stackEdge={index > 0} />
        </div>
      ))}
    </div>
  );
}

function StockpileItem({ pile, view, action, onAction, disabled }: { pile: Stockpile; view: GameView; action?: LegalAction; onAction: (id: number) => void; disabled: boolean }) {
  const selected = view.pending_decision.selected_stockpile_id === pile.stockpile_id;
  const cards: Card[] = [...pile.visible_cards, ...pile.hidden_cards];
  const content = (
    <>
      <Stack cards={cards} companies={view.companies} />
      {pile.bid !== null && <span className={styles.bid}>{bidderLabel(view, pile)} {pile.bid}K</span>}
    </>
  );
  return (
    <article className={`${styles.stockpile} ${selected ? styles.stockpileSelected : ""}`} aria-label={`Stockpile ${pile.stockpile_id + 1}`} data-stockpile-id={pile.stockpile_id}>
      {action ? <Selectable className={styles.pileSelector} selected={selected} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{content}</Selectable> : content}
    </article>
  );
}

function StockpileField({ view, onAction, disabled }: { view: GameView; onAction: (id: number) => void; disabled: boolean }) {
  const targets = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((item) => item.control === "stockpile")) {
    const pileId = idNumber(action.target_id, "stockpile");
    if (pileId !== null) targets.set(pileId, action);
  }
  return (
    <main className={`${styles.module} ${styles.stockpileField}`} data-testid="stockpile-field" aria-label="Stockpiles">
      <div className={`${styles.stockpileGrid} ${view.stockpiles.length === 4 ? styles.fourPiles : ""}`}>
        {view.stockpiles.map((pile) => <StockpileItem key={pile.stockpile_id} pile={pile} view={view} action={targets.get(pile.stockpile_id)} onAction={onAction} disabled={disabled} />)}
      </div>
    </main>
  );
}

function Portfolio({ view }: { view: GameView }) {
  return (
    <section className={`${styles.module} ${styles.portfolio}`} aria-label="Portfolio">
      <span className={styles.moduleLabel}>PORTFOLIO</span>
      <div className={styles.portfolioCards}>
        {view.private.holdings.filter((holding) => holding.represented > 0).map((holding) => {
          const company = companyById(view, holding.company_id);
          return company ? <HoldingCard key={holding.company_id} company={company} quantity={holding.represented} /> : null;
        })}
      </div>
    </section>
  );
}

function Players({ view }: { view: GameView }) {
  return (
    <section className={`${styles.module} ${styles.players}`} aria-label="Players">
      <span className={styles.moduleLabel}>PLAYERS</span>
      <div className={styles.playerList}>
        {view.players.map((player) => <div key={player.player_id}><span>{player.player_id === view.viewer.player_id ? "YOU" : player.name.toUpperCase()}</span><span>{money(player.cash)}</span></div>)}
      </div>
    </section>
  );
}

function actionText(action: LegalAction) {
  if (action.control === "bid" && action.amount !== null) return `${action.amount}K`;
  if (action.amount !== null) return money(action.amount);
  return action.label.toUpperCase();
}

function dockPrompt(view: GameView) {
  const viewer = view.players.find((player) => player.player_id === view.viewer.player_id);
  const rebidding = viewer?.bid_markers.some((marker) => marker.status === "outbid" || marker.status === "rebidding") ?? false;
  switch (view.pending_decision.kind) {
    case "supply_card": return "Select card";
    case "supply_face_up_pile":
    case "supply_face_down_pile": return "Select pile";
    case "bid_pile": return rebidding ? "Rebid" : "Select pile";
    case "bid_amount": return rebidding ? "Rebid" : "Bid";
    case "action_card": return "Select card";
    case "action_company": return "Select company";
    case "sell": return "Sell?";
    case "dividend_claim":
    case "acknowledge": return "Continue";
    case "waiting":
    case "private_selling": return "Wait";
    case "terminal": return "";
    case "generic": return view.legal_actions.length ? "Continue" : "Wait";
  }
}

function ActionDock({ view, onAction, disabled }: { view: GameView; onAction: (id: number) => void; disabled: boolean }) {
  if (view.terminal_results) {
    return <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock"><span className={styles.moduleLabel}>ACTION</span><a className={styles.actionControl} href="/">NEW GAME</a></section>;
  }
  const inline = view.legal_actions.filter((action) => action.control !== "stockpile" && action.control !== "company");
  const selectedSupply = view.pending_decision.selected_card_index === null ? null : view.private.hand[view.pending_decision.selected_card_index];
  const selectedImpact = view.private.available_action_cards.find((card) => card.effect.toLowerCase() === view.pending_decision.selected_action_effect?.toLowerCase());
  const holding = view.private.holdings.find((item) => item.company_id === view.pending_decision.company_id);
  const holdingCompany = holding ? companyById(view, holding.company_id) : undefined;
  const prompt = dockPrompt(view);
  return (
    <section className={`${styles.module} ${styles.actionDock}`} aria-label="Action dock">
      <div className={styles.dockHeading}><span className={styles.moduleLabel}>ACTION</span><span>{prompt}</span></div>
      <div className={styles.dockContents}>
        {selectedSupply && <CardView card={selectedSupply} companies={view.companies} scale="active" />}
        {selectedImpact && <CardView card={selectedImpact} companies={view.companies} scale="active" />}
        {view.phase.toLowerCase() === "selling" && holding && holdingCompany && <HoldingCard company={holdingCompany} quantity={holding.represented} scale="active" />}
        {inline.map((action) => {
          if (action.control === "card") {
            const index = idNumber(action.target_id, "hand");
            const card = index === null ? undefined : view.private.hand[index];
            return <Selectable key={action.action_id} className={styles.cardAction} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{card ? <CardView card={card} companies={view.companies} scale="active" /> : actionText(action)}</Selectable>;
          }
          if (action.control === "action_card") {
            const effect = action.target_id?.split(":")[1]?.toLowerCase();
            const card = view.private.available_action_cards.find((item) => item.effect.toLowerCase() === effect);
            return <Selectable key={action.action_id} className={styles.cardAction} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{card ? <CardView card={card} companies={view.companies} scale="active" /> : actionText(action)}</Selectable>;
          }
          if (action.control === "sell" && action.sale_preview) {
            return <Selectable key={action.action_id} className={styles.saleAction} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}><span>{action.sale_preview.quantity === 0 ? "HOLD" : `SELL ${action.sale_preview.quantity}`}</span>{action.sale_preview.quantity > 0 && <span>+{money(action.sale_preview.gross_value)}</span>}</Selectable>;
          }
          return <Selectable key={action.action_id} className={styles.actionControl} data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}>{actionText(action)}</Selectable>;
        })}
      </div>
    </section>
  );
}

function TerminalField({ view }: { view: GameView }) {
  if (!view.terminal_results) return null;
  return (
    <main className={`${styles.module} ${styles.stockpileField} ${styles.terminal}`} data-testid="stockpile-field" aria-label="Game end">
      <span className={styles.moduleLabel}>GAME END</span>
      <div className={styles.rankings}>
        {[...view.terminal_results.players].sort((left, right) => left.rank - right.rank).map((player) => (
          <div key={player.player_id} className={styles.ranking}>
            <span>#{player.rank}</span><span>{player.player_name.toUpperCase()}</span><span>{money(player.final_cash)}</span>{player.winner && <span>WINNER</span>}
            <div className={styles.liquidation}>
              {player.liquidation.filter((line) => line.represented_shares > 0).map((line) => {
                const company = companyById(view, line.company_id);
                return <span key={line.company_id}>{company && <StockPattern pattern={company.pattern} />} {line.represented_shares} × {money(line.unit_price)} = {money(line.value)}</span>;
              })}
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}

function currentMovement(view: GameView) {
  const events = view.recent_events.filter((event) => event.company_id !== null && event.actual_delta !== null);
  if (!events.length) return [];
  const last = events.at(-1)!;
  return events.filter((event) => event.round === last.round && event.cause === last.cause);
}

export function GamePage({ gameId, token }: { gameId: string; token: string }) {
  const { view, error, submitting, act } = useGameSession(gameId, token);
  const movementBatch = useMemo(() => view ? currentMovement(view) : [], [view]);
  const movementKey = movementBatch.map((event) => event.event_id).join(":");
  const [movements, setMovements] = useState<MarketEvent[]>([]);
  useEffect(() => {
    if (!movementKey) return;
    setMovements(movementBatch);
    const timer = window.setTimeout(() => setMovements([]), 2_400);
    return () => window.clearTimeout(timer);
  }, [movementKey]);

  if (!view && !error) return <main className={styles.centerState}>OPENING SEAT</main>;
  if (!view) return <main className={styles.centerState}><span>SEAT UNAVAILABLE</span><span>{error}</span><a href="/">NEW GAME</a></main>;
  return (
    <div className={styles.game}>
      {error && <div className={styles.errorBanner} role="alert">{error}</div>}
      <div className={styles.workstation} data-testid="workstation">
        <Status view={view} />
        <Market view={view} movements={movements} onAction={(id) => void act(id)} disabled={submitting} />
        <PrivateInformation view={view} />
        <PublicInformation view={view} />
        {view.terminal_results ? <TerminalField view={view} /> : <StockpileField view={view} onAction={(id) => void act(id)} disabled={submitting} />}
        <Portfolio view={view} />
        <Players view={view} />
        <ActionDock view={view} onAction={(id) => void act(id)} disabled={submitting} />
      </div>
    </div>
  );
}

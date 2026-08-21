import { useMemo, useState } from "react";
import { seatLink } from "../api";
import type {
  BidMarker,
  Company,
  GameView,
  LegalAction,
  PublicPlayer,
  Stockpile,
} from "../types";
import { useGameSession } from "../useGameSession";
import { CardView } from "./CardView";
import { Selectable } from "./Selectable";
import styles from "./Game.module.css";

function money(value: number) {
  return `$${value}K`;
}

function idNumber(target: string | null, prefix: string) {
  if (!target?.startsWith(`${prefix}:`)) return null;
  const result = Number(target.slice(prefix.length + 1));
  return Number.isFinite(result) ? result : null;
}

function BidToken({ marker, players }: { marker: BidMarker; players: PublicPlayer[] }) {
  const player = players.find((item) => item.player_id === marker.player_id);
  return (
    <span
      className={`${styles.bidToken} ${styles[`player${marker.player_id % 5}`]} ${marker.status === "outbid" || marker.status === "rebidding" ? styles.tokenAlert : ""}`}
      title={`${player?.name ?? "Player"} position ${marker.marker_index + 1}: ${marker.status}`}
      aria-label={`${player?.name ?? "Player"} bid marker ${marker.marker_index + 1}`}
    >
      {marker.marker_index + 1}
    </span>
  );
}

function TopBar({ view, token }: { view: GameView; token: string }) {
  const [menu, setMenu] = useState(false);
  const actor = view.players.find((player) => player.player_id === view.active_player_id);
  const actingLabel =
    view.pending_decision.kind === "private_selling"
      ? "Private selling in progress"
      : actor
        ? `${actor.name} to act`
        : view.terminal_results
          ? "Game complete"
          : view.pending_decision.prompt;
  async function copyLink() {
    await navigator.clipboard.writeText(seatLink(view.game_id, token));
    setMenu(false);
  }
  return (
    <header className={styles.topBar}>
      <a className={styles.brand} href="/">STOCKPILE</a>
      <span className={styles.round}>Round {view.round} / {view.total_rounds}</span>
      <span className={styles.phase}>{view.phase.toUpperCase()}</span>
      <span className={styles.actor}>{actingLabel}</span>
      <div className={styles.menuWrap}>
        <button className={styles.menuButton} type="button" aria-label="Game menu" aria-expanded={menu} onClick={() => setMenu(!menu)}>•••</button>
        {menu && (
          <div className={styles.menu}>
            <button type="button" onClick={() => void copyLink()}>Copy seat link</button>
            <a href="/">New game</a>
          </div>
        )}
      </div>
    </header>
  );
}

function MarketStrip({ view, onAction, disabled }: { view: GameView; onAction: (id: number) => void; disabled: boolean }) {
  const targets = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((item) => item.control === "company")) {
    const company = idNumber(action.target_id, "company");
    if (company !== null) targets.set(company, action);
  }
  return (
    <section className={styles.market} aria-label="Market">
      <div className={styles.sectionLabel}>MARKET</div>
      <div className={styles.marketRow}>
        {view.companies.map((company) => {
          const action = targets.get(company.company_id);
          const content = <CompanyPrice company={company} actionable={Boolean(action)} />;
          return action ? (
            <Selectable key={company.company_id} data-action-id={action.action_id} aria-label={action.label} className={styles.companyTarget} disabled={disabled} onClick={() => onAction(action.action_id)}>{content}</Selectable>
          ) : <div key={company.company_id}>{content}</div>;
        })}
      </div>
    </section>
  );
}

function CompanyPrice({ company, actionable }: { company: Company; actionable: boolean }) {
  return (
    <div className={`${styles.company} ${actionable ? styles.actionable : ""}`} style={{ "--company": company.color } as React.CSSProperties}>
      <span className={styles.companySymbol}>{company.symbol}</span>
      <div><strong>{company.price}</strong><small>{company.name}</small></div>
    </div>
  );
}

function StockpilePanel({ pile, view, action, onAction, disabled }: { pile: Stockpile; view: GameView; action?: LegalAction; onAction: (id: number) => void; disabled: boolean }) {
  const known = view.private.known_pile_cards.filter((item) => item.stockpile_id === pile.stockpile_id);
  const selected = view.pending_decision.selected_stockpile_id === pile.stockpile_id;
  const panel = (
    <article className={`${styles.pile} ${selected ? styles.selectedPile : ""} ${pile.locked ? styles.locked : ""} ${pile.purchaser_id !== null ? styles.purchased : ""}`}>
      <header><span>STOCKPILE {pile.stockpile_id + 1}</span>{pile.locked && <small>LOCKED</small>}</header>
      <div className={styles.cardFan}>
        {pile.visible_cards.map((card, index) => <CardView key={`v${index}`} card={card} compact />)}
        {pile.hidden_cards.map((card, index) => <CardView key={`h${index}`} card={card} compact />)}
        {!pile.visible_cards.length && !pile.hidden_cards.length && <span className={styles.emptyPile}>Awaiting cards</span>}
      </div>
      {known.length > 0 && (
        <div className={styles.privateNote} title="Private knowledge">You know {known.length} hidden card{known.length === 1 ? "" : "s"}</div>
      )}
      <footer>
        <div className={styles.bidStatus}>
          {pile.marker && <BidToken marker={pile.marker} players={view.players} />}
          <strong>{pile.bid === null ? "NO BID" : money(pile.bid)}</strong>
        </div>
        {pile.purchaser_id !== null && <small>Won by {view.players.find((player) => player.player_id === pile.purchaser_id)?.name}</small>}
      </footer>
    </article>
  );
  return action ? (
    <Selectable data-action-id={action.action_id} aria-label={action.label} className={styles.pileButton} selected={selected} disabled={disabled} onClick={() => onAction(action.action_id)}>{panel}</Selectable>
  ) : panel;
}

function StockpileGrid({ view, onAction, disabled }: { view: GameView; onAction: (id: number) => void; disabled: boolean }) {
  const pileActions = new Map<number, LegalAction>();
  for (const action of view.legal_actions.filter((item) => item.control === "stockpile")) {
    const pile = idNumber(action.target_id, "stockpile");
    if (pile !== null) pileActions.set(pile, action);
  }
  return (
    <section className={`${styles.stockpileGrid} ${view.configuration.player_count === 2 && view.stockpiles.length === 4 ? styles.twoPlayerGrid : ""}`} aria-label="Stockpiles">
      {view.stockpiles.map((pile) => <StockpilePanel key={pile.stockpile_id} pile={pile} view={view} action={pileActions.get(pile.stockpile_id)} onAction={onAction} disabled={disabled} />)}
    </section>
  );
}

function ActionButton({ action, onAction, disabled }: { action: LegalAction; onAction: (id: number) => void; disabled: boolean }) {
  return <Selectable data-action-id={action.action_id} aria-label={action.label} disabled={disabled} onClick={() => onAction(action.action_id)}><strong>{action.amount === null ? action.label : money(action.amount)}</strong></Selectable>;
}

function ActionDock({ view, onAction, disabled }: { view: GameView; onAction: (id: number) => void; disabled: boolean }) {
  const inlineActions = view.legal_actions.filter((action) => !["stockpile", "company"].includes(action.control));
  const holding = view.private.holdings.find((item) => item.company_id === view.pending_decision.company_id);
  const choosingSupplyTarget = view.phase.toLowerCase() === "supply" && inlineActions.every((action) => action.control !== "card");
  const choosingActionTarget = view.phase.toLowerCase() === "action" && inlineActions.every((action) => action.control !== "action_card");
  return (
    <section className={styles.actionDock} aria-label="Action dock">
      <div className={styles.dockPrompt}>
        <small>{view.viewer.name}</small>
        <strong>{view.pending_decision.prompt}</strong>
        {holding && <span>{holding.company}: {holding.represented} shares at {money(holding.price)}</span>}
        {view.pending_decision.private_progress !== null && view.pending_decision.private_total !== null && <span>{view.pending_decision.private_progress + 1} / {view.pending_decision.private_total}</span>}
      </div>
      <div className={styles.actionChoices}>
        {choosingSupplyTarget && view.private.hand.map((card, index) => (
          <div key={`supply-context:${index}`} className={`${styles.contextCard} ${view.pending_decision.selected_card_index === index ? styles.contextSelected : ""}`}>
            <CardView card={card} compact />
            <small>{view.pending_decision.selected_card_index === index ? "SELECTED" : "DRAWN"}</small>
          </div>
        ))}
        {choosingActionTarget && view.private.available_action_cards.map((card, index) => (
          <div key={`action-context:${index}`} className={`${styles.contextCard} ${view.pending_decision.selected_action_effect?.toLowerCase() === card.effect.toLowerCase() ? styles.contextSelected : ""}`}>
            <CardView card={card} compact />
            <small>{view.pending_decision.selected_action_effect?.toLowerCase() === card.effect.toLowerCase() ? "SELECTED" : "AVAILABLE"}</small>
          </div>
        ))}
        {view.phase.toLowerCase() === "selling" && (
          <div className={styles.portfolioStrip} aria-label="Your portfolio">
            {view.private.holdings.map((item) => (
              <span key={item.company_id} className={item.company_id === view.pending_decision.company_id ? styles.portfolioCurrent : ""}>
                <b>{item.company.slice(0, 1).toUpperCase()}</b>
                <strong>{item.represented}</strong>
                <small>@ {money(item.price)}</small>
              </span>
            ))}
          </div>
        )}
        {inlineActions.map((action) => {
          if (action.control === "card") {
            const cardIndex = idNumber(action.target_id, "hand");
            const card = cardIndex === null ? undefined : view.private.hand[cardIndex];
            return <Selectable key={action.action_id} data-action-id={action.action_id} aria-label={action.label} className={styles.cardChoice} disabled={disabled} onClick={() => onAction(action.action_id)}>{card ? <CardView card={card} /> : action.label}</Selectable>;
          }
          if (action.control === "action_card") {
            const effect = action.target_id?.split(":")[1];
            const card = view.private.available_action_cards.find((item) => item.effect.toLowerCase() === effect?.toLowerCase());
            return <Selectable key={action.action_id} data-action-id={action.action_id} aria-label={action.label} className={styles.cardChoice} disabled={disabled} onClick={() => onAction(action.action_id)}>{card ? <CardView card={card} /> : action.label}</Selectable>;
          }
          if (action.control === "sell" && action.sale_preview) {
            const preview = action.sale_preview;
            return (
              <Selectable key={action.action_id} data-action-id={action.action_id} aria-label={action.label} className={styles.saleChoice} disabled={disabled} onClick={() => onAction(action.action_id)}>
                <strong>{preview.quantity === 0 ? "Hold" : `Sell ${preview.quantity}`}</strong>
                <span>{preview.quantity === 0 ? `${preview.resulting_represented} retained` : `+${money(preview.gross_value)}`}</span>
              </Selectable>
            );
          }
          return <ActionButton key={action.action_id} action={action} onAction={onAction} disabled={disabled} />;
        })}
        {!inlineActions.length && view.pending_decision.kind === "waiting" && <span className={styles.waiting}>Your seat stays fixed. This view will update automatically.</span>}
        {!inlineActions.length && view.pending_decision.kind === "private_selling" && <span className={styles.waiting}>Other sale choices remain sealed until settlement.</span>}
      </div>
    </section>
  );
}

function ChatPanel({ view, sendChat }: { view: GameView; sendChat: (message: string) => Promise<void> }) {
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!message.trim() || sending) return;
    setSending(true);
    try { await sendChat(message); setMessage(""); } finally { setSending(false); }
  }
  return (
    <section className={`${styles.railPanel} ${styles.chatPanel}`}>
      <h2>CHAT</h2>
      <div className={styles.messages} aria-live="polite">
        {view.chat.map((entry) => <p key={entry.message_id}><strong>{entry.player_name}</strong><span>{entry.message}</span></p>)}
        {!view.chat.length && <span className={styles.muted}>Local game chat</span>}
      </div>
      <form onSubmit={submit}><input aria-label="Chat message" maxLength={500} value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Message" /><button disabled={sending || !message.trim()}>Send</button></form>
    </section>
  );
}

function MarketInfoPanel({ view }: { view: GameView }) {
  return (
    <section className={styles.railPanel}>
      <h2>MARKET INFORMATION</h2>
      <div className={styles.infoGrid}>
        {view.private.market_information.map((slot, index) => (
          <div key={index} className={`${styles.infoSlot} ${styles[slot.visibility]}`}>
            <CardView card={slot.card} compact />
            <small>{slot.visibility}</small>
          </div>
        ))}
      </div>
      {view.private.known_pile_cards.length > 0 && (
        <div className={styles.knownCards} aria-label="Private pile knowledge">
          <small>PRIVATE PILE KNOWLEDGE</small>
          <div>
            {view.private.known_pile_cards.map((item, index) => (
              <span key={`${item.stockpile_id}:${index}`}>
                <b>Pile {item.stockpile_id + 1}</b>
                <CardView card={item.card} compact />
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function PlayersPanel({ view }: { view: GameView }) {
  return (
    <section className={`${styles.railPanel} ${styles.playersPanel}`}>
      <h2>PLAYERS</h2>
      <div className={styles.playerList}>
        {view.players.map((player) => (
          <div key={player.player_id} className={`${styles.playerRow} ${player.active ? styles.activePlayer : ""} ${player.player_id === view.viewer.player_id ? styles.viewer : ""}`}>
            <span className={`${styles.playerDot} ${styles[`player${player.player_id % 5}`]}`} />
            <div>
              <strong>{player.name}</strong>
              <small>{player.status}{player.player_id === view.viewer.player_id ? " · You" : ""}</small>
              {player.fee_debts.length > 0 && <small className={styles.feeDebt}>Fees due: {player.fee_debts.map(money).join(" · ")}</small>}
            </div>
            <span className={styles.playerCash}>{money(player.cash)}</span>
            <div className={styles.markerShelf}>{player.bid_markers.map((marker) => <BidToken key={marker.marker_index} marker={marker} players={view.players} />)}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function latestMovementBatch(view: GameView) {
  const movement = view.recent_events.filter((event) => event.cause === "market_forecast");
  const latestRound = movement.reduce((round, event) => Math.max(round, event.round), -1);
  return movement.filter((event) => event.round === latestRound);
}

function EventToast({ view }: { view: GameView }) {
  const latest = view.recent_events.at(-1);
  if (!latest) return null;
  const movementBatch = latestMovementBatch(view);
  const events = movementBatch.length > 0 ? movementBatch : [latest];
  return (
    <div className={styles.event} aria-label={movementBatch.length > 0 ? "Latest market movement" : "Latest market event"}>
      {movementBatch.length > 0 && <small>ROUND {latest.round} MOVEMENT</small>}
      {events.map((event) => (
        <span key={event.event_id}>
          <b>{event.company ?? "Market"}</b>
          <em>{event.forecast === null ? event.description : String(event.forecast)}</em>
          {event.resulting_price !== null && <strong>{event.prior_price} → {event.resulting_price}</strong>}
        </span>
      ))}
    </div>
  );
}

function GameEnd({ view }: { view: GameView }) {
  if (!view.terminal_results) return null;
  const finalMovement = latestMovementBatch(view);
  return (
    <section className={styles.gameEnd}>
      <header><small>GAME END</small><h1>{view.terminal_results.winner_ids.length > 1 ? "Joint winners" : "Winner"}: {view.terminal_results.players.filter((item) => item.winner).map((item) => item.player_name).join(" & ")}</h1></header>
      {finalMovement.length > 0 && (
        <section className={styles.finalMovement} aria-label="Final market movement">
          <h2>FINAL MARKET MOVEMENT</h2>
          <div>
            {finalMovement.map((event) => (
              <span key={event.event_id}>
                <b>{event.company ?? "Market"}</b>
                <em>{event.forecast === null ? event.description : String(event.forecast)}</em>
                <strong>{event.prior_price} → {event.resulting_price}</strong>
              </span>
            ))}
          </div>
        </section>
      )}
      <div className={styles.rankings}>
        {[...view.terminal_results.players].sort((a, b) => a.rank - b.rank).map((player) => (
          <article key={player.player_id} className={player.winner ? styles.winner : ""}>
            <span className={styles.rank}>#{player.rank}</span><h2>{player.player_name}</h2><strong>{money(player.final_cash)}</strong>
            <small>{money(player.cash_before_liquidation)} cash + {money(player.liquidation_value)} liquidation</small>
            <div>{player.liquidation.filter((line) => line.represented_shares > 0).map((line) => <span key={line.company_id}>{line.company} {line.represented_shares} × {money(line.unit_price)} = {money(line.value)}</span>)}</div>
          </article>
        ))}
      </div>
      <a href="/" className={styles.newGame}>New game</a>
    </section>
  );
}

export function GamePage({ gameId, token }: { gameId: string; token: string }) {
  const { view, error, submitting, act, sendChat } = useGameSession(gameId, token);
  const phaseClass = useMemo(() => view?.phase.toLowerCase().replace(/[^a-z]/g, "") ?? "", [view?.phase]);
  if (!view && !error) return <main className={styles.centerState}><div className={styles.loader} /><span>Opening your seat…</span></main>;
  if (!view) return <main className={styles.centerState}><h1>Seat unavailable</h1><p>{error}</p><a href="/">Create a new game</a></main>;
  return (
    <div className={`${styles.game} ${styles[phaseClass] ?? ""}`}>
      <TopBar view={view} token={token} />
      {view.terminal_results ? <GameEnd view={view} /> : (
        <div className={styles.layout}>
          <main className={styles.board}>
            {error && <div className={styles.errorBanner} role="alert">{error}</div>}
            <MarketStrip view={view} onAction={(id) => void act(id)} disabled={submitting} />
            <StockpileGrid view={view} onAction={(id) => void act(id)} disabled={submitting} />
            <EventToast view={view} />
            <ActionDock view={view} onAction={(id) => void act(id)} disabled={submitting} />
          </main>
          <aside className={styles.rail}>
            <ChatPanel view={view} sendChat={sendChat} />
            <MarketInfoPanel view={view} />
            <PlayersPanel view={view} />
          </aside>
        </div>
      )}
    </div>
  );
}

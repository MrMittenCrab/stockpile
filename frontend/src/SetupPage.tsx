import { useEffect, useMemo, useState } from "react";
import { createGame, getSetup } from "./api";
import type { CreateGameResponse, LiteOptions, SetupResponse } from "./types";
import styles from "./SetupPage.module.css";

const emptyOptions: LiteOptions = {
  market_impact: false,
  starting_share: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

export function SetupPage() {
  const [setup, setSetup] = useState<SetupResponse | null>(null);
  const [playerCount, setPlayerCount] = useState(2);
  const [roundCount, setRoundCount] = useState(6);
  const [names, setNames] = useState(["Player 1", "Player 2"]);
  const [options, setOptions] = useState<LiteOptions>(emptyOptions);
  const [created, setCreated] = useState<CreateGameResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getSetup(controller.signal).then((value) => {
      setSetup(value);
      setPlayerCount(value.defaults.player_count);
      setRoundCount(value.defaults.round_count);
      setOptions({ ...emptyOptions, ...Object.fromEntries(value.options.map((item) => [item.key, item.default])) });
    }).catch((cause: unknown) => {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) setError(cause instanceof Error ? cause.message : "Backend unavailable");
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    setNames((current) => Array.from({ length: playerCount }, (_, index) => current[index] ?? `Player ${index + 1}`));
  }, [playerCount]);

  const valid = useMemo(() => names.every((name) => name.trim()) && new Set(names.map((name) => name.trim().toLowerCase())).size === names.length, [names]);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!valid) return;
    setSubmitting(true);
    setError(null);
    try {
      setCreated(await createGame({ player_count: playerCount, player_names: names, round_count: roundCount, options }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to create game");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.header}><strong>STOCKPILE</strong><span>LITE · LOCAL TABLE</span></header>
      <div className={styles.hero}>
        <section className={styles.intro}>
          <small>FINANCIAL STRATEGY</small>
          <h1>Take your seat<br />at the market.</h1>
          <p>Create a private local table. Each player opens a fixed seat in a separate browser tab.</p>
          <div className={styles.marketGlyphs}><span>A</span><span>B</span><span>C</span><span>D</span><span>E</span><span>F</span></div>
        </section>
        <section className={styles.panel}>
          {!created ? (
            <form onSubmit={submit}>
              <div className={styles.panelHeading}><div><small>NEW GAME</small><h2>Table setup</h2></div><span>LOCAL</span></div>
              {!setup && !error && <p className={styles.loading}>Reading Lite configuration…</p>}
              {error && <p role="alert" className={styles.error}>{error}</p>}
              {setup && <>
                <div className={styles.row}>
                  <label>Players<select aria-label="Player count" value={playerCount} onChange={(event) => setPlayerCount(Number(event.target.value))}>{Array.from({ length: setup.player_limits.maximum - setup.player_limits.minimum + 1 }, (_, index) => setup.player_limits.minimum + index).map((count) => <option key={count}>{count}</option>)}</select></label>
                  <label>Rounds<input aria-label="Round count" type="number" min={setup.round_limits.minimum} max={setup.round_limits.maximum} value={roundCount} onChange={(event) => setRoundCount(Number(event.target.value))} /></label>
                </div>
                <fieldset><legend>Seats</legend>{names.map((name, index) => <label key={index} className={styles.nameField}><span>{index + 1}</span><input aria-label={`Player ${index + 1} name`} maxLength={32} value={name} onChange={(event) => setNames((current) => current.map((item, itemIndex) => itemIndex === index ? event.target.value : item))} /></label>)}</fieldset>
                <fieldset><legend>Lite options</legend><div className={styles.options}>{setup.options.map((option) => <label key={option.key} className={styles.option}><input type="checkbox" checked={options[option.key]} onChange={(event) => setOptions((current) => ({ ...current, [option.key]: event.target.checked }))} /><span><strong>{option.label}</strong><small>{option.description}</small></span></label>)}</div></fieldset>
                <button className={styles.create} type="submit" disabled={submitting || !valid}>{submitting ? "Creating…" : "Create game"}</button>
              </>}
            </form>
          ) : (
            <div className={styles.lobby}>
              <div className={styles.panelHeading}><div><small>GAME READY</small><h2>Open each seat</h2></div><span>{created.seats.length} SEATS</span></div>
              <p>Open every seat in its own tab or window. A seat stays tied to that tab.</p>
              <div className={styles.seats}>{created.seats.map((seat) => <a key={seat.player_id} href={seat.url} target="_blank" rel="noopener noreferrer"><span>{seat.player_id + 1}</span><strong>Open {seat.player_name}</strong><small>New tab ↗</small></a>)}</div>
              <button type="button" className={styles.secondary} onClick={() => setCreated(null)}>Change setup</button>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

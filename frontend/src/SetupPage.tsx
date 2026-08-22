import { useEffect, useState, type SubmitEvent } from "react";
import { createGame, getSetup } from "./api";
import type { CreateGameResponse, LiteOptionKey, LiteOptions, SetupResponse } from "./types";
import styles from "./SetupPage.module.css";

const emptyOptions: LiteOptions = {
  market_impact: false,
  starting_share: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

const playerCounts = [2, 3, 4, 5] as const;

const features: ReadonlyArray<{ key: LiteOptionKey; label: string }> = [
  { key: "dividends", label: "DIVIDEND" },
  { key: "trading_fees", label: "FEES" },
  { key: "market_impact", label: "IMPACT" },
  { key: "sell_order", label: "SELL ORDER" },
];

export function SetupPage() {
  const [setup, setSetup] = useState<SetupResponse | null>(null);
  const [playerCount, setPlayerCount] = useState(2);
  const [options, setOptions] = useState<LiteOptions>(emptyOptions);
  const [created, setCreated] = useState<CreateGameResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getSetup(controller.signal)
      .then((value) => {
        const defaults = Object.fromEntries(
          value.options.map((option) => [option.key, option.default]),
        ) as Partial<LiteOptions>;
        setSetup(value);
        setPlayerCount(value.defaults.player_count);
        setOptions({ ...emptyOptions, ...defaults, starting_share: false });
      })
      .catch((cause: unknown) => {
        if (!(cause instanceof DOMException && cause.name === "AbortError")) {
          setError(cause instanceof Error ? cause.message : "Backend unavailable");
        }
      });
    return () => controller.abort();
  }, []);

  function toggleFeature(key: LiteOptionKey) {
    setOptions((current) => ({ ...current, [key]: !current[key], starting_share: false }));
  }

  async function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!setup) return;
    setSubmitting(true);
    setError(null);
    try {
      setCreated(await createGame({
        player_count: playerCount,
        player_names: Array.from({ length: playerCount }, (_, index) => `Player ${index + 1}`),
        round_count: setup.defaults.round_count,
        options: { ...options, starting_share: false },
      }));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to create game");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.identity}>
        <span>STOCKPILE</span>
        <span>LITE</span>
      </header>

      {!created ? (
        <form className={styles.setup} onSubmit={submit}>
          <section className={styles.group} aria-labelledby="players-heading">
            <h1 id="players-heading">PLAYERS</h1>
            <div className={styles.players}>
              {playerCounts.map((count) => (
                <button
                  key={count}
                  type="button"
                  className={styles.choice}
                  aria-label={`${count} players`}
                  aria-pressed={playerCount === count}
                  onClick={() => setPlayerCount(count)}
                  disabled={!setup || count < setup.player_limits.minimum || count > setup.player_limits.maximum}
                >
                  {count}
                </button>
              ))}
            </div>
          </section>

          <section className={styles.group} aria-labelledby="features-heading">
            <h1 id="features-heading">FEATURES</h1>
            <div className={styles.features}>
              {features.map((feature) => (
                <button
                  key={feature.key}
                  type="button"
                  className={styles.feature}
                  aria-pressed={options[feature.key]}
                  onClick={() => toggleFeature(feature.key)}
                  disabled={!setup}
                >
                  {feature.label}
                </button>
              ))}
            </div>
          </section>

          {error && <p role="alert" className={styles.error}>{error}</p>}
          <button className={styles.start} type="submit" disabled={!setup || submitting}>
            {submitting ? "STARTING" : "START"}
          </button>
        </form>
      ) : (
        <section className={styles.lobby} aria-label="Seats">
          {created.seats.map((seat) => (
            <div className={styles.seat} key={seat.player_id}>
              <span>P{seat.player_id + 1}</span>
              <a href={seat.url} target="_blank" rel="noopener noreferrer">Open Seat</a>
            </div>
          ))}
          <button type="button" className={styles.change} onClick={() => setCreated(null)}>
            Change setup
          </button>
        </section>
      )}
    </main>
  );
}

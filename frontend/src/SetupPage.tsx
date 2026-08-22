import { useEffect, useState, type FormEvent } from "react";
import { createGame, getSetup } from "./api";
import { SectionLabel, TextButton } from "./components/Primitives";
import type { LiteOptionKey, LiteOptions, SetupResponse } from "./types";
import styles from "./SetupPage.module.css";

const emptyOptions: LiteOptions = {
  market_impact: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

const featureOrder: LiteOptionKey[] = ["dividends", "trading_fees", "market_impact", "sell_order"];
const featureLabels: Record<LiteOptionKey, string> = {
  dividends: "DIVIDEND",
  trading_fees: "FEES",
  market_impact: "IMPACT",
  sell_order: "SELL ORDER",
};

export function SetupPage({ navigate = (url: string) => window.location.assign(url) }: { navigate?: (url: string) => void } = {}) {
  const [setup, setSetup] = useState<SetupResponse | null>(null);
  const [options, setOptions] = useState<LiteOptions>(emptyOptions);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getSetup(controller.signal)
      .then((value) => {
        const defaults = Object.fromEntries(value.options.map((option) => [option.key, option.default])) as Partial<LiteOptions>;
        setSetup(value);
        setOptions({ ...emptyOptions, ...defaults });
      })
      .catch((cause: unknown) => {
        if (!(cause instanceof DOMException && cause.name === "AbortError")) {
          setError(cause instanceof Error ? cause.message : "BACKEND UNAVAILABLE");
        }
      });
    return () => controller.abort();
  }, []);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!setup || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const game = await createGame({ options });
      navigate(game.game_url);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "UNABLE TO START");
      setSubmitting(false);
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.identity} aria-label="Stockpile Lite">
        <span>STOCKPILE</span>
        <span>LITE</span>
      </header>

      <form className={styles.setup} onSubmit={submit}>
        <section className={styles.features} aria-labelledby="features-heading">
          <SectionLabel id="features-heading">FEATURES</SectionLabel>
          <div className={styles.featureGrid}>
            {featureOrder.filter((key) => setup?.options.some((option) => option.key === key)).map((key) => (
              <TextButton
                key={key}
                selected={options[key]}
                onClick={() => setOptions((current) => ({ ...current, [key]: !current[key] }))}
              >
                {featureLabels[key]}
              </TextButton>
            ))}
          </div>
        </section>

        {error && <p role="alert" className={styles.error}>{error}</p>}
        <TextButton className={styles.start} type="submit" disabled={!setup || submitting}>
          {submitting ? "STARTING" : "START"}
        </TextButton>
      </form>
    </main>
  );
}

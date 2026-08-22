import { useEffect, useState, type FormEvent } from "react";
import { createGame, getSetup } from "./api";
import { TextButton } from "./components/Primitives";
import type { LiteOptionKey, LiteOptions, SetupResponse } from "./types";
import styles from "./SetupPage.module.css";

type TrainerMode = "lite" | "lite_plus";

const emptyOptions: LiteOptions = {
  market_impact: false,
  trading_fees: false,
  dividends: false,
  sell_order: false,
};

const featureOrder = ["dividends", "trading_fees", "sell_order"] as const satisfies readonly LiteOptionKey[];
const featureLabels: Record<(typeof featureOrder)[number], string> = {
  dividends: "DIVIDEND",
  trading_fees: "FEES",
  sell_order: "SELL ORDER",
};

export function SetupPage({ navigate = (url: string) => window.location.assign(url) }: { navigate?: (url: string) => void } = {}) {
  const [setup, setSetup] = useState<SetupResponse | null>(null);
  const [mode, setMode] = useState<TrainerMode | null>(null);
  const [options, setOptions] = useState<LiteOptions>(emptyOptions);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    getSetup(controller.signal)
      .then((value) => {
        setSetup(value);
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
      const game = await createGame({
        options: mode === "lite"
          ? emptyOptions
          : { ...options, market_impact: false },
      });
      navigate(game.game_url);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "UNABLE TO START");
      setSubmitting(false);
    }
  }

  return (
    <main className={styles.page}>
      <header className={styles.identity} aria-label="Stockpile Trainer">
        <span>STOCKPILE TRAINER</span>
      </header>

      <form className={styles.setup} onSubmit={submit}>
        <section className={styles.modes} aria-label="Trainer mode">
          <TextButton selected={mode === "lite"} onClick={() => setMode("lite")}>LITE</TextButton>
          <TextButton selected={mode === "lite_plus"} onClick={() => setMode("lite_plus")}>LITE+</TextButton>
        </section>

        {mode === "lite_plus" && (
          <section className={styles.features} aria-label="Lite plus features">
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
        )}

        {error && <p role="alert" className={styles.error}>{error}</p>}
        {(mode === "lite" || (mode === "lite_plus" && featureOrder.some((key) => options[key]))) && (
          <TextButton className={styles.play} type="submit" disabled={!setup || submitting}>
            {submitting ? "OPENING" : "PLAY"}
          </TextButton>
        )}
      </form>
    </main>
  );
}

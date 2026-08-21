import type { Card, VisibleCard } from "../types";
import styles from "./Game.module.css";

function faceLabel(card: VisibleCard) {
  switch (card.kind) {
    case "stock":
      return { kicker: card.company, value: `${card.quantity > 0 ? "+" : ""}${card.quantity}`, detail: "SHARES" };
    case "trading_fee":
      return { kicker: "FEE", value: `$${card.amount}K`, detail: "TRADING" };
    case "action":
      return { kicker: "IMPACT", value: card.effect.toUpperCase(), detail: "ACTION" };
    case "company_forecast":
      return {
        kicker: card.company,
        value: typeof card.forecast === "number" ? `${card.forecast > 0 ? "+" : ""}${card.forecast}` : card.forecast,
        detail: "FORECAST",
      };
  }
}

export function CardView({ card, compact = false }: { card: Card; compact?: boolean }) {
  if (card.visibility === "hidden") {
    return <div className={`${styles.card} ${styles.cardBack} ${compact ? styles.cardCompact : ""}`} aria-label="Hidden card"><span>S</span></div>;
  }
  const label = faceLabel(card);
  return (
    <div className={`${styles.card} ${styles.cardFace} ${compact ? styles.cardCompact : ""}`}>
      <small>{label.kicker}</small>
      <strong>{label.value}</strong>
      <span>{label.detail}</span>
    </div>
  );
}

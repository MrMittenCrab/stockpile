import type { Card, Company, InformationCard } from "../types";
import { CardFrame, type CardScale } from "./Primitives";
import styles from "./Game.module.css";
import { StockPattern } from "./StockPattern";

function companyFor(companies: Company[], companyId: number) {
  return companies.find((company) => company.company_id === companyId);
}

function forecastFace(card: InformationCard) {
  if (card.forecast === "DIVIDEND") {
    const amount = card.cash_effect_thousands;
    return {
      className: styles.positive,
      text: amount == null ? "+$" : `+$${amount}K`,
      label: amount == null ? "Dividend" : `Cash increases by ${amount}K`,
    };
  }
  if (card.forecast > 0) {
    return { className: styles.positive, text: `↑${card.forecast}`, label: `Price up ${card.forecast}` };
  }
  if (card.forecast < 0) {
    return { className: styles.negative, text: `↓${Math.abs(card.forecast)}`, label: `Price down ${Math.abs(card.forecast)}` };
  }
  return { className: styles.neutral, text: "0", label: "No price movement" };
}

export function CompanyCard({ company, scale = "information" }: { company: Company; scale?: CardScale }) {
  return (
    <CardFrame aria-label={`${company.display_name} company card`} className={styles.stockCard} scale={scale}>
      <StockPattern pattern={company.pattern} />
    </CardFrame>
  );
}

export function HoldingCard({ company, sharesThousands, scale = "portfolio" }: { company: Company; sharesThousands: number; scale?: CardScale }) {
  return (
    <CardFrame aria-label={`${company.display_name} holding ${sharesThousands}K shares`} className={styles.stockCard} scale={scale}>
      <StockPattern pattern={company.pattern} />
      <span className={styles.cardValue} data-card-value>{sharesThousands}K</span>
    </CardFrame>
  );
}

export function CardView({ card, companies, scale = "active", faceDownKnown = false }: { card: Card; companies: Company[]; scale?: CardScale; faceDownKnown?: boolean }) {
  if (card.visibility === "hidden") {
    return <CardFrame aria-label="Hidden card" className={styles.hiddenCard} scale={scale} />;
  }

  let content;
  if (card.kind === "stock") {
    const company = companyFor(companies, card.company_id);
    content = (
      <CardFrame aria-label={`${company?.display_name ?? card.company} stock ${card.shares_thousands}K shares`} className={styles.stockCard} scale={scale}>
        {company && <StockPattern pattern={company.pattern} />}
        <span className={styles.cardValue} data-card-value>{card.shares_thousands}K</span>
        {faceDownKnown && <span className={styles.faceDownFlag}>FACE DOWN</span>}
      </CardFrame>
    );
  } else if (card.kind === "trading_fee") {
    content = (
      <CardFrame aria-label={`Cash decreases by ${Math.abs(card.cash_effect_thousands)}K`} className={styles.negative} scale={scale}>
        <span className={styles.cardSignal}>−${Math.abs(card.cash_effect_thousands)}K</span>
        {faceDownKnown && <span className={styles.faceDownFlag}>FACE DOWN</span>}
      </CardFrame>
    );
  } else if (card.kind === "action") {
    const up = card.direction === "up";
    content = (
      <CardFrame aria-label={`Price ${up ? "up" : "down"} ${Math.abs(card.movement)}`} className={up ? styles.positive : styles.negative} scale={scale}>
        <span className={styles.cardSignal}>{up ? "↑" : "↓"}{Math.abs(card.movement)}</span>
        {faceDownKnown && <span className={styles.faceDownFlag}>FACE DOWN</span>}
      </CardFrame>
    );
  } else {
    const forecast = forecastFace(card);
    content = (
      <CardFrame aria-label={forecast.label} className={forecast.className} scale={scale}>
        <span className={styles.cardSignal}>{forecast.text}</span>
        {faceDownKnown && <span className={styles.faceDownFlag}>FACE DOWN</span>}
      </CardFrame>
    );
  }
  return content;
}

import type { Card, Company, InformationCard, VisibleCard } from "../types";
import styles from "./Game.module.css";
import { StockPattern } from "./StockPattern";

export type CardScale = "stockpile" | "active" | "portfolio" | "information";

function companyFor(companies: Company[], companyId: number) {
  return companies.find((company) => company.company_id === companyId);
}

function forecastFace(card: InformationCard) {
  if (card.forecast === "DIVIDEND") {
    return { className: styles.cashCard, text: "$$", label: "Dividend" };
  }
  if (card.forecast > 0) {
    return { className: styles.upCard, text: `↑${card.forecast}`, label: `Price up ${card.forecast}` };
  }
  if (card.forecast < 0) {
    return { className: styles.downCard, text: `↓${Math.abs(card.forecast)}`, label: `Price down ${Math.abs(card.forecast)}` };
  }
  return { className: styles.neutralCard, text: "0", label: "No price movement" };
}

export function CompanyCard({ company, scale = "information" }: { company: Company; scale?: CardScale }) {
  return (
    <div aria-label={`${company.display_name} company card`} className={`${styles.card} ${styles.stockCard} ${styles[`card_${scale}`]}`} data-card-scale={scale}>
      <StockPattern pattern={company.pattern} />
    </div>
  );
}

export function HoldingCard({ company, quantity, scale = "portfolio" }: { company: Company; quantity: number; scale?: CardScale }) {
  return (
    <div aria-label={`${company.display_name} holding ${quantity}`} className={`${styles.card} ${styles.stockCard} ${styles[`card_${scale}`]}`} data-card-scale={scale}>
      <StockPattern pattern={company.pattern} />
      <span className={styles.cardValue}>{quantity}</span>
    </div>
  );
}

export function CardView({ card, companies, scale = "active", stackEdge = false }: { card: Card; companies: Company[]; scale?: CardScale; stackEdge?: boolean }) {
  const base = `${styles.card} ${styles[`card_${scale}`]}`;
  if (card.visibility === "hidden") {
    return <div aria-label="Hidden card" className={`${base} ${styles.hiddenCard}`} data-card-scale={scale}>{stackEdge && <span className={styles.hiddenEdge} aria-hidden="true" />}</div>;
  }

  if (card.kind === "stock") {
    const company = companyFor(companies, card.company_id);
    return (
      <div aria-label={`${company?.display_name ?? card.company} stock ${card.quantity}`} className={`${base} ${styles.stockCard}`} data-card-scale={scale}>
        {company && <StockPattern pattern={company.pattern} />}
        <span className={styles.cardValue}>{card.quantity}</span>
        {stackEdge && company && <StockPattern pattern={company.pattern} className={styles.edgePattern} />}
      </div>
    );
  }
  if (card.kind === "trading_fee") {
    return <div aria-label={`Trading fee ${card.amount}K`} className={`${base} ${styles.cashCard}`} data-card-scale={scale}><span className={styles.cardSignal}>{`-$${card.amount}K`}</span>{stackEdge && <span className={styles.edgeSignal} aria-hidden="true">-$</span>}</div>;
  }
  if (card.kind === "action") {
    const up = card.direction === "up";
    return <div aria-label={`${up ? "Price up" : "Price down"} ${card.movement}`} className={`${base} ${up ? styles.upCard : styles.downCard}`} data-card-scale={scale}><span className={styles.cardSignal}>{up ? "↑" : "↓"}{card.movement}</span>{stackEdge && <span className={styles.edgeSignal} aria-hidden="true">{up ? "↑" : "↓"}</span>}</div>;
  }
  const forecast = forecastFace(card);
  return <div aria-label={forecast.label} className={`${base} ${forecast.className}`} data-card-scale={scale}><span className={styles.cardSignal}>{forecast.text}</span>{stackEdge && <span className={styles.edgeSignal} aria-hidden="true">{forecast.text}</span>}</div>;
}

export function visibleCardCompany(card: VisibleCard, companies: Company[]) {
  if (card.kind !== "stock" && card.kind !== "company_forecast") return null;
  return companyFor(companies, card.company_id) ?? null;
}

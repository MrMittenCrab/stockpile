import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  PropsWithChildren,
} from "react";
import styles from "./Primitives.module.css";

export interface TextButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  recommendationStrength?: number;
}

export function TextButton({
  selected,
  recommendationStrength: _recommendationStrength,
  className = "",
  children,
  ...props
}: PropsWithChildren<TextButtonProps>) {
  return (
    <button
      type="button"
      aria-pressed={selected === undefined ? undefined : selected}
      className={`${styles.button} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function SectionLabel({ children, className = "", ...props }: PropsWithChildren<HTMLAttributes<HTMLSpanElement>>) {
  return <span className={`${styles.sectionLabel} ${className}`} {...props}>{children}</span>;
}

export type CardScale = "stockpile" | "active" | "portfolio" | "information";

export function CardFrame({ scale, className = "", children, ...props }: PropsWithChildren<HTMLAttributes<HTMLDivElement> & { scale: CardScale }>) {
  return (
    <div className={`${styles.card} ${styles[scale]} ${className}`} data-card-scale={scale} {...props}>
      {children}
    </div>
  );
}

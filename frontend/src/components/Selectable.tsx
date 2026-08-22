import type { ButtonHTMLAttributes, PropsWithChildren } from "react";
import styles from "./Game.module.css";

export interface SelectableProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  selected?: boolean;
  recommendationStrength?: number;
}

export function Selectable({
  selected,
  recommendationStrength: _recommendationStrength,
  className = "",
  children,
  ...props
}: PropsWithChildren<SelectableProps>) {
  return (
    <button
      type="button"
      aria-pressed={selected === undefined ? undefined : selected}
      className={`${styles.selectable} ${selected ? styles.selected : ""} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

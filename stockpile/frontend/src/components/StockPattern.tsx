import type { Company } from "../types";
import styles from "./Game.module.css";

export function StockPattern({
  pattern,
  className = "",
}: {
  pattern: Company["pattern"];
  className?: string;
}) {
  return (
    <svg
      aria-hidden="true"
      className={`${styles.stockPattern} ${className}`}
      data-stock-pattern={pattern}
      viewBox="0 0 96 64"
      focusable="false"
    >
      {pattern === "matrix" && (
        <g>
          {[8, 32, 56, 80].flatMap((x) =>
            [8, 28, 48].map((y) => <rect key={`${x}:${y}`} x={x} y={y} width="10" height="10" />),
          )}
        </g>
      )}
      {pattern === "ledger" && (
        <g>{[10, 24, 38, 52].map((y) => <rect key={y} x="5" y={y} width="86" height="6" />)}</g>
      )}
      {pattern === "molecular" && (
        <g>
          <path d="M12 16 36 8l24 16 24-8M12 48l24 8 24-16 24 8M36 8v48M60 24v16" fill="none" stroke="currentColor" strokeWidth="3" />
          {[[12, 16], [36, 8], [60, 24], [84, 16], [12, 48], [36, 56], [60, 40], [84, 48]].map(([x, y]) => (
            <circle key={`${x}:${y}`} cx={x} cy={y} r="6" />
          ))}
        </g>
      )}
      {pattern === "chevron" && (
        <g fill="none" stroke="currentColor" strokeWidth="8">
          <path d="m4 8 20 24L4 56" /><path d="m28 8 20 24-20 24" /><path d="m52 8 20 24-20 24" /><path d="m76 8 16 24-16 24" />
        </g>
      )}
      {pattern === "crosshatch" && (
        <g fill="none" stroke="currentColor" strokeWidth="6">
          <path d="M2 4 32 34 2 64M32 0 64 32 32 64M64 0 94 30 64 64M94 0 64 32 94 64M64 0 32 32 64 64M32 0 2 30 32 64" />
        </g>
      )}
      {pattern === "wave" && (
        <g fill="none" stroke="currentColor" strokeWidth="6">
          <path d="M0 12h12l8-8 16 16 16-16 16 16L84 4l12 8" /><path d="M0 34h12l8-8 16 16 16-16 16 16 16-16 12 8" /><path d="M0 56h12l8-8 16 16 16-16 16 16 16-16 12 8" />
        </g>
      )}
    </svg>
  );
}

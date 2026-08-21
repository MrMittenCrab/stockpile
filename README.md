# Stockpile

This repository contains a configurable Stockpile rules engine with a native
OpenSpiel interface, information-set complexity analysis, and an optional Deep
CFR trainer. It supports the `lite`, `classic`, and `deluxe` rules profiles.

## Setup

Create and activate a virtual environment, then install the core dependencies:

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Inspect the CLI or a resolved rules profile:

```console
python -m stockpile --help
python -m stockpile rules --mode lite
python -m stockpile complexity --mode lite
```

Lite defaults to two players and six rounds. Market Impact, starting shares,
fees, dividends, and sell order are optional and off by default. Selling is
therefore sealed: players commit without seeing earlier commitments, and all
sales are revealed and settled together. Enable Lite's Action Cards and Action
phase with `--impact on`. Stock splits, re-splits, majority bonuses, and advanced
price tracks are not Lite rules; Lite prices can rise above 10 normally.

## Browser play

The local browser interface supports complete Stockpile Lite games for two to
five human players. Each player uses a separate fixed-seat browser tab; game
sessions and chat exist only in memory and are cleared when the backend stops.

Install the web and frontend dependencies once:

```console
.venv/bin/python -m pip install -r requirements-web.txt
npm --prefix frontend install
npm --prefix frontend exec -- playwright install chromium
```

Start the backend and frontend in separate terminals:

```console
.venv/bin/python -m stockpile play --mode lite --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
```

Open <http://127.0.0.1:5173>, configure and create a game, then open each
generated seat link in its own tab or window. The browser client submits only
opaque actions offered by Python; it does not implement game rules or expose
other seats' private information.

This interface intentionally has no accounts, matchmaking, bots, hot-seat
mode, WebSockets, persistent storage, production deployment, or Deep CFR
recommendations.

## Deep CFR

Install the optional training dependencies:

```console
python -m pip install -r requirements-training.txt
```

Run the small one-round smoke preset:

```console
python -m stockpile solve --mode lite --rounds 1 --smoke
```

Start the default six-round curriculum:

```console
python -m stockpile solve --mode lite --rounds 6
```

Normal runs are allocated automatically under
`artifacts/deep_cfr/lite/run_XX`; smoke runs use the separate
`artifacts/deep_cfr/smoke/run_XX` namespace. Select a specific unused number
with `--run INT`, or use `--output-dir PATH` for an unmanaged destination.

Analyze the signed sampled-regret history recorded by a run:

```console
python -m stockpile analyze --mode lite --run 3
python -m stockpile analyze --output-dir artifacts/deep_cfr/lite/run_03
```

The analysis saves a plot-ready per-iteration series with an empirical
confidence interval. Older checkpoints and policies without signed traversal
history remain usable, but correctly report that this analysis is unavailable.

The solver targets the canonical two-player compact Lite game with sealed
selling. It uses outcome-sampled Deep CFR and a strict visible-history encoder;
the default curriculum is `1,2,3,4,6`. Its paired policy evaluation is training
telemetry, not an exact exploitability or equilibrium claim. See
[`stockpile/training/README.md`](stockpile/training/README.md) for checkpoints,
resume behavior, memory defaults, and additional training options.

## Tests

```console
python -m unittest discover -s stockpile/tests -v
npm --prefix frontend run typecheck
npm --prefix frontend test
npm --prefix frontend run build
npm --prefix frontend run e2e
```

## Layout

- `stockpile/stockpile_platform.py` — rules, state transitions, scoring, and
  OpenSpiel integration.
- `stockpile/stockpile_interface.py` — UI-neutral configuration and analysis
  facade.
- `stockpile/complexity_cache.py` — semantic fingerprints and remembered
  information-set complexity results.
- `stockpile/training/` — Deep CFR encoding, sampling, models, trainer, policy,
  and evaluation.
- `stockpile/web/` — local FastAPI sessions and privacy-safe browser views.
- `frontend/` — Vite React TypeScript fixed-seat browser client.
- `stockpile/tests/` — engine, interface, CLI, complexity, and training tests.
- `stockpile/docs/` — bundled Stockpile rules and reference documents.

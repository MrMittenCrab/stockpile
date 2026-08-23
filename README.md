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

The local browser interface is a complete two-round Stockpile Lite game between
`YOU` and `COMPUTER`. The computer seat loads the matching Deep CFR
`round_02/policy.pt` from `artifacts/deep_cfr/lite/` (the same artifact family
produced by `python -m stockpile solve`). Use `--policy PATH`, `--run INT`, or
`--policy random` to override. Lite+ options that leave the trained canonical
Lite rules fall back to uniform legal actions for those games. Games exist only
in memory and are cleared when the backend stops.

Install the web and frontend dependencies once:

```console
.venv/bin/python -m pip install -r requirements-web.txt
npm --prefix frontend install
npm --prefix frontend exec -- playwright install chromium
```

Start the complete local trainer with one command:

```console
python -m stockpile play
```

If the virtual environment is not active, use
`.venv/bin/python -m stockpile play` instead.

The command waits for both local services and then prints:

```text
Starting Stockpile Trainer...

http://127.0.0.1:5173
```

It does not open a browser. Open <http://127.0.0.1:5173>, choose `LITE` or
`LITE+`, then press `PLAY`. `LITE+` exposes Dividend, Fees, and Sell Order;
Market Impact remains accepted by the V2 API for compatibility, while Starting
Shares remain engine/legacy-V1 only. Neither is exposed by this browser setup.
The browser moves directly into the human seat. The client submits only opaque
actions and complete server-authored decision plans; it does not implement game
rules or receive the computer's private portfolio.

`--mode lite`, `--host`, and `--port` remain available for compatible local
scripts. The launcher passes a non-default API address to Vite automatically.
Press `Ctrl+C` once to stop both services.

This interface intentionally has no accounts, matchmaking, human multiplayer,
hot-seat mode, WebSockets, persistent storage, or production deployment. The
computer seat uses Deep CFR inference from a local solve artifact; it does not
stream live training. The legacy local multi-seat HTTP API remains available
for compatibility but is not used by the browser product.

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

Summarize the policy's stored paired evaluation against a random legal-action
opponent:

```console
python -m stockpile analyze --mode lite --run 3
python -m stockpile analyze --output-dir artifacts/deep_cfr/lite/run_03
```

Regenerate the learning-curve graph (score vs training traversals with a
pointwise 95% confidence band) from saved checkpoint history:

```console
python -m stockpile analyze --method learning-curve --mode lite --run 3
```

Request sampled-regret convergence explicitly when needed:

```console
python -m stockpile analyze --method regret --mode lite --run 3
python -m stockpile analyze --method regret \
  --output-dir artifacts/deep_cfr/lite/run_03 --confidence 0.90
```

The default evaluation is read from the run's existing training metrics and
does not rerun games. Regret analysis saves a plot-ready per-iteration series
with an empirical confidence interval; its bootstrap progress is written to
standard error so the compact result table on standard output remains clean.
Older checkpoints and policies without the requested telemetry remain usable
and report `N/A`.

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
- `frontend/` — Vite React TypeScript `YOU`-versus-`COMPUTER` browser client.
- `stockpile/tests/` — engine, interface, CLI, complexity, and training tests.
- `stockpile/docs/` — bundled Stockpile rules and reference documents.

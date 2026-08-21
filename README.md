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

Lite defaults to two players and six rounds. Market Impact, Investors, starting
shares, fees, dividends, splits, majority bonuses, advanced stock tracks, and
sell order are off. Selling is therefore sealed: players commit without seeing
earlier commitments, and all sales are revealed and settled together. Each
optional Lite rule can still be enabled explicitly through the CLI.

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

The solver targets the canonical two-player compact Lite game with sealed
selling. It uses outcome-sampled Deep CFR and a strict visible-history encoder;
the default curriculum is `1,2,3,4,6`. Its paired policy evaluation is training
telemetry, not an exact exploitability or equilibrium claim. See
[`stockpile/training/README.md`](stockpile/training/README.md) for checkpoints,
resume behavior, memory defaults, and additional training options.

## Tests

```console
python -m unittest discover -s stockpile/tests -v
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
- `stockpile/tests/` — engine, interface, CLI, complexity, and training tests.
- `stockpile/docs/` — bundled Stockpile rules and reference documents.

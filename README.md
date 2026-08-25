# Stockpile

Configurable Stockpile rules engine with an OpenSpiel interface, information-set
complexity analysis, local browser play against a Deep CFR computer, and an
optional Deep CFR trainer. Rules profiles: `lite`, `classic`, and `deluxe`.

## Setup

```console
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```console
python -m stockpile --help
python -m stockpile rules --mode lite
python -m stockpile complexity --mode lite
```

Lite defaults to two players and six rounds. Market Impact, starting shares,
fees, dividends, and sell order are optional and off by default, so selling is
sealed (commitments are revealed and settled together). Enable Action Cards with
`--impact on`. Stock splits, majority bonuses, and advanced price tracks are not
Lite rules.

## Play

Local one-round Stockpile Lite between `YOU` and `COMPUTER`. The computer loads
the bundled Deep CFR average policy at
`stockpile/artifacts/deep_cfr/lite/run_03/round_01/policy.pt` (the latest
published Lite run). Override with `--policy`, `--run`, or `--policy random`.
Lite+ options that leave the trained Lite rules fall back to uniform legal
actions. Games are in-memory only.

```console
.venv/bin/python -m pip install -r requirements-web.txt
npm --prefix stockpile/frontend install
npm --prefix stockpile/frontend exec -- playwright install chromium
python -m stockpile play
```

Open <http://127.0.0.1:5173>, choose `LITE` or `LITE+`, then `PLAY`. `LITE+`
exposes Dividend, Fees, and Sell Order. The client submits opaque actions and
server-authored plans only; it never sees the computer's private portfolio.

Press `Ctrl+C` to stop. No accounts, multiplayer, WebSockets, or persistent
storage.

## Solve

Outcome-sampled Deep CFR for the canonical two-player compact Lite game (sealed
selling, Market Impact off).

```console
python -m pip install -r requirements-training.txt
python -m stockpile solve --mode lite --rounds 1 --smoke
python -m stockpile solve --mode lite --rounds 6
```

Runs land under `stockpile/artifacts/deep_cfr/lite/run_XX` (smoke under
`stockpile/artifacts/deep_cfr/smoke/`). Use `--run INT` or `--output-dir PATH`.

Default curriculum: `1,2,3,4,6`. Train until a win-rate target vs random:

```console
python -m stockpile solve --mode lite --rounds 1 --until-win-rate 0.70
```

With `--until-win-rate`, evaluations run every `--eval-every` iterations
(default 100), write learning-curve history, and stop after two consecutive hits
or `--max-iterations` (default 10,000).

See [`stockpile/training/README.md`](stockpile/training/README.md) for
checkpoints, resume, memories, and telemetry.

## Analyze

Report stored evaluation, learning-curve history, or sampled regret for one run:

```console
python -m stockpile analyze --mode lite --run 1
python -m stockpile analyze --output-dir stockpile/artifacts/deep_cfr/lite/run_01
python -m stockpile analyze --method learning-curve --mode lite --run 1
python -m stockpile analyze --method regret --mode lite --run 1 --confidence 0.90
```

Default `--method evaluation` reads the last paired evaluation from
`metrics.jsonl` (no new games). Learning-curve mode regenerates the win-rate
plot from saved checkpoints. Regret mode writes
`analysis/sampled_average_regret.json`; it is sampled average regret, not
exploitability or NashConv.

## Tests

```console
python -m unittest discover -s stockpile/tests -v
npm --prefix stockpile/frontend run typecheck
npm --prefix stockpile/frontend test
npm --prefix stockpile/frontend run build
npm --prefix stockpile/frontend run e2e
```

## Layout

- `stockpile/stockpile_platform.py` — rules, transitions, OpenSpiel game
- `stockpile/stockpile_interface.py` — configuration and analysis facade
- `stockpile/complexity_cache.py` — remembered infoset-complexity results
- `stockpile/training/` — Deep CFR trainer, encoder, policy, evaluation
- `stockpile/web/` — local FastAPI sessions and privacy-safe views
- `stockpile/frontend/` — Vite/React `YOU` vs `COMPUTER` client
- `stockpile/artifacts/deep_cfr/lite/run_03/` — bundled play policy (`policy.pt`)
- `stockpile/writeup/` — local write-up notebook and figures (gitignored)
- `stockpile/tests/` — engine, CLI, web, and training tests
- `stockpile/docs/` — bundled Stockpile rules references

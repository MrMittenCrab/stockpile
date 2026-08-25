# Stockpile Lite Deep CFR

The training package implements an outcome-sampled neural CFR solver for the
canonical two-player Stockpile Lite game. Lite uses sealed selling by default:
players commit sales without observing earlier commitments, then all sales are
published and settled as one batch. Training deliberately keeps optional Market
Impact off and retains the compact 18-action policy head.

Install the optional training environment:

```console
python -m pip install -r requirements-training.txt
```

Run the checked-in one-round smoke preset:

```console
python -m stockpile solve --mode lite --rounds 1 --smoke
```

Start the default six-round curriculum:

```console
python -m stockpile solve --mode lite --rounds 6
```

Normal runs are reserved atomically as
`stockpile/artifacts/deep_cfr/lite/run_01`, `run_02`, and so on. Smoke runs are numbered
independently under `stockpile/artifacts/deep_cfr/smoke/`. Use `--run INT` to request a
specific unused number. `--output-dir PATH` remains available for an explicit
unmanaged destination and cannot be combined with `--run`.

The default stages are `1,2,3,4,6`. Round 5 is deliberately omitted, but it
can be requested manually with `--curriculum 1,2,3,4,5,6`. At every horizon
transition only compatible network weights transfer; optimizers, iteration
counters, and both reservoir memories start fresh for the new game.

Resume an interrupted stage from its full checkpoint:

```console
python -m stockpile solve --mode lite --rounds 6 \
  --resume stockpile/artifacts/deep_cfr/lite/run_03/round_04/full.pt
```

Reserved or active new-format runs resume in place unless a different
destination is selected. Completed runs, legacy checkpoints under
`stockpile/artifacts/deep_cfr/default/` or the old smoke layout, and unmarked historical
custom outputs resume into a newly numbered run. The source stays unchanged
and the fork records its checkpoint hash and source-path provenance. Explicit
unmanaged forks record the same information in `resume_provenance.json`.

Each stage writes:

- `full.pt`: networks, optimizers, reservoirs, counters, semantic metadata,
  random-number-generator states, and a self-contained signed-regret snapshot
  for an exact same-stage resume.
- `policy.pt`: the smaller average-policy network and inference metadata.
- `metrics.jsonl`: losses, memory sizes, timing, and paired seat-swapped
  evaluation against a uniform legal-action policy.
- `learning_curve.json` / `learning_curve.csv`: predetermined in-training
  evaluation checkpoints with bootstrap confidence intervals against random.
- `analysis/learning_curve.png`: regenerable win-rate-vs-training graph.
- `sampled_regret/iteration_XXXXXX.npz`: signed per-traversal outcome-sampled
  regret records, stored independently from reservoir memory.

During each stage the trainer pauses at approximately ten evenly spaced
iterations (always including the final iteration), freezes the current
in-memory average policy, and evaluates 500 held-out seat-swapped pairs
against uniform random (1,000 games). The checkpoint score is the strict win
rate (wins / games; ties count as 0) with a pointwise 95% bootstrap confidence
band (10,000 resamples of complete seat-swapped pairs). These evaluation
checkpoints do not write separate policy files and do not mutate training
state.

To keep training until the policy beats the random benchmark, pass
`--until-win-rate` with an iteration budget:

```console
python -m stockpile solve --mode lite --rounds 1 --until-win-rate 0.70
```

Defaults with `--until-win-rate`: evaluate every 100 iterations, 2000 seat-balanced
games per checkpoint, and stop by 10,000 iterations if the target is unmet.
Override with `--eval-every`, `--eval-games`, and `--max-iterations` as needed.

In that mode the solver trains in `--eval-every` iteration increments (default 100),
prints an evaluation line (including `iteration=N`), updates the normal stage
checkpoint, archives a copy under `checkpoints/traversals_*`, and evaluates with a
fresh seat-balanced seed set each time. History is written to
`evaluation_history.csv` (`traversals,games,wins,losses,ties,win_rate,
mean_utility,ci_low,ci_high`) as well as the learning-curve artifacts. Training
stops only after the target win rate is hit on two consecutive evaluations, or
when `--max-iterations` is reached. Without `--until-win-rate`, solve behavior
is unchanged.

Summarize the stored paired evaluation for one run after completion or
interruption:

```console
python -m stockpile analyze --mode lite --run 3
python -m stockpile analyze --output-dir stockpile/artifacts/deep_cfr/lite/run_03
```

This default report reads the last stored evaluation record for each stage
from `metrics.jsonl`; it does not rerun evaluation games. To regenerate the
learning-curve graph from saved checkpoint history:

```console
python -m stockpile analyze --method learning-curve --mode lite --run 3
python -m stockpile analyze --method learning-curve \
  --output-dir stockpile/artifacts/deep_cfr/lite/run_03 \
  --plot stockpile/artifacts/deep_cfr/lite/run_03/analysis/learning_curve.png
```

To analyze sampled regret instead, select the method explicitly:

```console
python -m stockpile analyze --method regret --mode lite --run 3
python -m stockpile analyze --method regret \
  --output-dir stockpile/artifacts/deep_cfr/lite/run_03 --confidence 0.90
```

Regret analysis writes `analysis/sampled_average_regret.json`, containing the
full player 0, player 1, and maximum-player series after every recorded
iteration, while the terminal prints one compact final row per stage. Its
empirical interval resamples complete traversals only within their original
iteration and update-player strata. Bootstrap progress goes to standard error;
the result table stays on standard output. It is sampled average regret, not
exploitability, NashConv, or a formal equilibrium guarantee. Legacy artifacts
without the requested evaluation or signed-regret records report `N/A`; their
reservoir samples, losses, and policy weights are never used as substitutes.

The default batch contains at most 32 samples and each of the three
stage-local reservoirs retains at most 2,000 samples. These are deliberately
conservative strict-history defaults; increase them only after measuring peak
memory and checkpoint size on the target machine.

The encoder uses the current 256-value observation, explicit horizon features,
all of the acting player's remembered observations/actions (including forced
actions), and only public or privately visible event records. Opponent hidden
sale commitments are never encoded.

This is the scalable outcome-sampling variant selected for the large Stockpile
tree. Evaluation reports paired tournament estimates and confidence intervals;
it does not calculate exact NashConv or make an equilibrium-convergence claim.

Fresh runs never reuse numbered directories. `--overwrite` is accepted only
with an explicit unmanaged `--output-dir`; managed and legacy directories are
protected. Full checkpoints use PyTorch's pickle-backed format and must be
loaded only from a trusted source. Float64 advantage networks are required for
the sampled regret range, so automatic device selection uses CUDA when
available and CPU otherwise; MPS training is not supported.

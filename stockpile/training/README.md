# Stockpile Lite Deep CFR

The training package implements an outcome-sampled neural CFR solver for the
canonical two-player Stockpile Lite game. Lite uses sealed selling by default:
players commit sales without observing earlier commitments, then all sales are
published and settled as one batch.

Install the optional training environment:

```console
python -m pip install -r requirements-training.txt
```

Run the checked-in one-round smoke preset:

```console
python -m stockpile solve --mode lite --rounds 1 --smoke \
  --output-dir artifacts/deep_cfr/smoke
```

Start the default six-round curriculum:

```console
python -m stockpile solve --mode lite --rounds 6 \
  --output-dir artifacts/deep_cfr/default
```

The default stages are `1,2,3,4,6`. Round 5 is deliberately omitted, but it
can be requested manually with `--curriculum 1,2,3,4,5,6`. At every horizon
transition only compatible network weights transfer; optimizers, iteration
counters, and both reservoir memories start fresh for the new game.

Resume an interrupted stage from its full checkpoint:

```console
python -m stockpile solve --mode lite --rounds 6 \
  --output-dir artifacts/deep_cfr/default \
  --resume artifacts/deep_cfr/default/round_04/full.pt
```

Each stage writes:

- `full.pt`: networks, optimizers, reservoirs, counters, semantic metadata,
  and random-number-generator states for an exact same-stage resume.
- `policy.pt`: the smaller average-policy network and inference metadata.
- `metrics.jsonl`: losses, memory sizes, timing, and paired seat-swapped
  evaluation against a uniform legal-action policy.

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

Fresh runs refuse to write into a nonempty output directory. Pass
`--overwrite` only when replacing those artifacts is intentional. Full
checkpoints use PyTorch's pickle-backed format and must be loaded only from a
trusted source. Float64 advantage networks are required for the sampled regret
range, so automatic device selection uses CUDA when available and CPU
otherwise; MPS training is not supported.

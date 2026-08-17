# formulate

**Active learning for polymer formulation: find the best copolymer in the fewest experiments.**

![CI](https://github.com/DrGregPitch/formulate/actions/workflows/ci.yml/badge.svg)

An industrial ML chemist does not have 100,000 labelled samples. They have a design space and a budget for forty experiments. The job is not to fit a model to data you already have — it is to *choose which experiments to run*. This repo is that loop: a surrogate model and an acquisition function propose the next formulation to synthesise, an oracle scores it, the model updates, and it repeats under a hard experimental budget.

![Best formulation found vs. number of experiments. Active learning reaches 98% of the optimum in ~16–20 experiments; random screening does not get there in 60.](assets/money_plot.png)

That gap is the entire value proposition. On a design space of 1,440 candidate copolymer formulations, **active learning reaches within 2% of the best possible formulation in about 16 experiments. Random screening does not reach it in 60.** Cutting an experimental campaign from months to weeks is a slide a hiring manager can show their director.

---

## The headline result

| strategy | experiments to reach 98% of the optimum |
|:---|---:|
| random screening | **> 60** (never, within budget) |
| greedy (pure exploitation) | 18 |
| **UCB** (upper confidence bound) | **16** |
| EI (expected improvement) | 20 |

*Median over 20 restarts. Design space: 1,440 copolymer formulations; best achievable Tg 236 °C.*

Every active-learning strategy — whether it exploits greedily or balances exploration with UCB/EI — reaches the target 3–4× faster than random, and random often does not reach it at all within the budget. The specific acquisition function matters less than the fact of *using the surrogate's uncertainty to choose*.

---

## How it works

The loop is four pieces, each swappable:

- **Design space** (`design_space.py`) — a pool of candidate copolymer formulations, each defined by a comonomer pair, a composition, and a sequence (blockiness). The **oracle** is the controlled copolymer-Tg model from [`copolybench`](https://github.com/DrGregPitch/copolybench): cheap to call here, but the loop *pretends each call is an expensive synthesis-and-measure experiment and counts them*.
- **Surrogate** (`surrogate.py`) — a Gaussian process (scikit-learn), or the [`polytools`](https://github.com/DrGregPitch/polytools) gradient-boosting ensemble. Returns a mean *and an uncertainty* at every candidate, which is what acquisition needs.
- **Acquisition** (`acquisition.py`) — `random`, `greedy`, `ucb`, `ei`. Given the surrogate's predictions over the unqueried pool, score each candidate; the loop runs the argmax.
- **Loop** (`loop.py`) — seed with a few random experiments, then propose → measure → update until the budget is spent, tracking the best formulation found.

The surrogate sees interpretable design coordinates — the two homopolymer Tgs, the comonomer polarities, the composition, and the sequence statistics — enough to model the property surface but not the hidden per-pair junction term, so it is *good but imperfect*: exactly the regime where choosing experiments well pays off.

## Cost-aware acquisition

Not all experiments cost the same — exotic comonomers are expensive to source. The loop's cost-aware mode (`cost_aware=True`) divides the acquisition score by a candidate's cost, so it prefers cheaper experiments of comparable promise and reaches the target for **less total spend**, not just fewer runs. The optimization script draws this comparison automatically.

![Best found vs. cumulative experimental cost: cost-aware UCB reaches the target for less spend.](assets/cost_aware.png)

---

## On real data: discovering polymer electrolytes for batteries

The synthetic oracle above proves the loop works with ground truth we control. But the loop is *property-agnostic* — the surrogate, the acquisition functions, and the campaign don't know or care where the numbers come from. So pointing the whole thing at real data is exactly one new function, `build_spe_design_space`, and nothing else changes.

The real design space is **solid polymer electrolyte (SPE) ionic conductivity** — the search for better battery electrolytes — using 6,949 measured formulations curated by [CheMixHub](https://github.com/chemcognition-lab/chemixhub) (MIT-licensed). Each candidate is a real (polymer, lithium salt, composition, temperature) point; the loop treats each measurement as an expensive experiment and counts them.

```bash
python scripts/fetch_spe.py               # download the data (MIT)
python scripts/run_spe.py --outdir results
```

![Active learning vs random screening on real solid-polymer-electrolyte conductivity data.](assets/spe_money_plot.png)

| strategy | experiments to reach the top 10% of conductivity |
|:---|---:|
| random screening | 28 |
| greedy | 11 |
| **UCB** | 14 |
| **EI** | 12 |

*Median over 15 restarts; 6,949 real formulations, best measured conductivity 0.044 S/cm.*

**Active learning reaches a top-decile electrolyte in ~11 experiments; random screening needs 28 — a 2.5× reduction on real, literature-measured data.** The gap is more modest than on the clean synthetic oracle, which is exactly what honest real data looks like: the property surface is noisier, so the surrogate is less certain and random occasionally gets lucky. It still reaches good electrolytes more than twice as fast — the difference between a two-week and a five-week experimental campaign. (The polymer repeat units in this dataset mark their connection points with `[Cu]`/`[Au]`; the loader translates them to the `[*]` the featurizer expects.)

---

## Reproduce it

```bash
git clone https://github.com/DrGregPitch/formulate && cd formulate
uv venv && uv pip install -e ".[dev]"      # pulls polytools and copolybench from GitHub
uv run python scripts/run_optimization.py --outdir results   # regenerates the money plot
uv run pytest tests -v
```

One command reproduces every number and both figures.

---

## Where this sits in the portfolio

This is Project 3 of three, and it composes the other two into the loop an industrial ML chemist actually runs:

- [`polytools`](https://github.com/DrGregPitch/polytools) — the honest evaluation harness and property-prediction models (the surrogate).
- [`copolybench`](https://github.com/DrGregPitch/copolybench) — the copolymer representation study and the controlled Tg oracle (the "lab").
- **`formulate`** — active learning over that space (the campaign).

## Limitations

- **Controlled oracle, not experimental data.** The Tg comes from a known model, so this measures the *loop's* efficiency, not agreement with a real lab. Point the design space at real measured data and the same loop runs unchanged.
- **Pool-based.** The loop selects from a fixed candidate pool rather than optimising a continuous design space with a trust region; the pool is large enough to make the point, and pool-based selection matches how a real candidate library is screened.
- **Single objective.** Real formulation is multi-objective (Tg *and* processability *and* cost). Cost is handled as a constraint on acquisition; a full Pareto-front treatment is the natural extension.

## License

MIT

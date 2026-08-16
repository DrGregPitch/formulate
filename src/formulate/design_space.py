"""The formulation design space, and the oracle that scores it.

An industrial ML chemist does not have 100k labelled samples; they have a design
space and a budget for a few dozen experiments. So the object here is a *pool* of
candidate formulations -- copolymers defined by a comonomer pair, a composition,
and a sequence (blockiness) -- together with an oracle that returns the true
property but is expensive to call, so you want to call it as few times as possible.

The oracle is the controlled copolymer-Tg model from
`copolybench` (the Johnston dyad equation). It is cheap to evaluate here, but the
active-learning loop *pretends* each call is an expensive synthesis-and-measure
experiment and counts them. That is the whole game: reach the best formulation in
the fewest oracle calls.

The surrogate sees interpretable design coordinates -- the two homopolymer Tgs, the
two comonomer polarities, the composition, and the sequence statistics -- enough to
learn the property surface but not the hidden per-pair junction term, so the
surrogate is good but imperfect, exactly the regime where acquisition matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from copolybench import generate_dataset
from polytools.featurize import DescriptorFeaturizer

__all__ = ["DesignSpace", "build_design_space"]

FEATURE_NAMES = ["tg_a", "tg_b", "polarity_a", "polarity_b", "f", "F_AB", "blockiness"]


@dataclass
class DesignSpace:
    """A pool of candidate formulations with a hidden true property and a cost.

    Attributes
    ----------
    X
        ``(n_candidates, n_features)`` surrogate features (design coordinates).
    y_true
        The oracle value for every candidate. The loop may only look at an entry
        after it has "run the experiment" on that candidate.
    cost
        Relative experimental cost per candidate (exotic comonomers cost more),
        used by the cost-aware acquisition extension.
    frame
        The full copolybench record for each candidate, for interpretation.
    """

    X: np.ndarray
    y_true: np.ndarray
    cost: np.ndarray
    feature_names: list[str]
    frame: object  # pandas DataFrame, kept untyped to avoid a hard import here

    def __len__(self) -> int:
        return len(self.y_true)

    @property
    def best_value(self) -> float:
        return float(self.y_true.max())


def build_design_space(
    n_pairs: int = 120,
    points_per_pair: int = 12,
    seed: int = 0,
) -> DesignSpace:
    """Construct the candidate pool from the copolymer oracle (noise-free labels)."""
    df = generate_dataset(
        n_pairs=n_pairs, points_per_pair=points_per_pair, noise_c=0.0, seed=seed
    )

    # comonomer polarity (TPSA per unit), a design-known monomer property
    desc = DescriptorFeaturizer(mode="cap")
    uniq = list(dict.fromkeys(df["psmiles_a"].tolist() + df["psmiles_b"].tolist()))
    pol_lookup = dict(zip(
        uniq,
        desc.transform(uniq)[:, desc.feature_names.index("TPSA_per_unit")],
    ))
    pol_a = df["psmiles_a"].map(pol_lookup).to_numpy(dtype=float)
    pol_b = df["psmiles_b"].map(pol_lookup).to_numpy(dtype=float)

    X = np.column_stack([
        df["tg_a"].to_numpy(float), df["tg_b"].to_numpy(float),
        pol_a, pol_b,
        df["f"].to_numpy(float), df["F_AB"].to_numpy(float),
        df["blockiness"].to_numpy(float),
    ])
    y_true = df["tg"].to_numpy(dtype=float)

    # experimental cost: exotic comonomers (rare in the pool) cost more to source.
    # Frequency of each monomer across the pool -> rarer is pricier.
    counts = (df["psmiles_a"].value_counts() + df["psmiles_b"].value_counts()).fillna(0)
    freq = counts / counts.max()
    rarity_a = 1.0 - df["psmiles_a"].map(freq).to_numpy(dtype=float)
    rarity_b = 1.0 - df["psmiles_b"].map(freq).to_numpy(dtype=float)
    cost = 1.0 + rarity_a + rarity_b  # in [1, 3], cheap common pairs to pricey exotic

    return DesignSpace(X=X, y_true=y_true, cost=cost,
                       feature_names=FEATURE_NAMES, frame=df)

"""A real formulation design space: solid polymer electrolytes.

Everything in :mod:`formulate` above the design space is property-agnostic -- the
surrogate, the acquisition functions, the loop. So pointing the whole apparatus at
real data is exactly one new function: build a :class:`DesignSpace` from a real
dataset. This one uses the **solid polymer electrolyte (SPE)** ionic-conductivity
data curated by CheMixHub (MIT-licensed), which is the battery-electrolyte
formulation problem in the flesh -- mix a polymer with a lithium salt at some
composition and temperature, and measure how well it conducts.

The task the loop then runs: **find the highest-conductivity formulation in the
fewest experiments.** Each of the 6,949 candidates is a real measured
(polymer, salt, composition, temperature) point; the loop treats each as an
expensive experiment and counts them.

Design coordinates the surrogate sees: descriptors of the polymer repeat unit and
the salt, the salt mole fraction, the log polymer molecular weight, and the
temperature. The polymer repeat units use ``[Cu]``/``[Au]`` as connection points
(a convention from the source data); we translate them to the ``[*]`` that
:mod:`polytools` expects.
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from polytools.featurize import DescriptorFeaturizer

from .design_space import DesignSpace

__all__ = ["build_spe_design_space", "SPE_FEATURE_NAMES"]

_SALT_DESC = ["salt_MolWt", "salt_TPSA", "salt_HAcceptors", "salt_F_count"]


def _salt_vector(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import Descriptors as D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        D.MolWt(mol), D.TPSA(mol), float(D.NumHAcceptors(mol)),
        float(sum(a.GetSymbol() == "F" for a in mol.GetAtoms())),
    ], dtype=float)


def _polymer_vector(monomeric_unit: str, featurizer: DescriptorFeaturizer):
    # the source marks connection points with [Cu]/[Au]; polytools wants [*]
    psmiles = monomeric_unit.replace("[Cu]", "[*]").replace("[Au]", "[*]")
    return featurizer.transform([psmiles])[0]


#: Feature block, in order, for the SPE design space.
SPE_FEATURE_NAMES = (
    [f"poly_{n}" for n in DescriptorFeaturizer().feature_names]
    + _SALT_DESC
    + ["salt_mole_fraction", "log10_polymer_mw", "temperature_K"]
)


def build_spe_design_space(
    processed_csv: str,
    compounds_csv: str,
) -> DesignSpace:
    """Build the SPE conductivity design space from the CheMixHub CSVs.

    Keeps the binary (one polymer + one salt) formulations -- the canonical SPE --
    which is the bulk of the data. Target is ``log Conductivity`` (S/cm); higher is
    a better electrolyte, so the loop maximises it.
    """
    from polytools import silence_rdkit
    silence_rdkit()

    df = pd.read_csv(processed_csv)
    comp = pd.read_csv(compounds_csv).set_index("compound_id")
    polf = DescriptorFeaturizer(mode="cap")

    # precompute per-compound lookups once (per-row .loc is far too slow)
    is_polymer = comp["polymer"].astype(int).to_dict()
    is_salt = comp["salt"].astype(int).to_dict()
    monomer = comp["monomeric_unit"].to_dict()
    salt_smiles = comp["smiles"].to_dict()

    pol_cache: dict[int, np.ndarray] = {}
    salt_cache: dict[int, np.ndarray | None] = {}
    rows, targets, salt_ids = [], [], []

    ids_col = df["cmp_ids"].tolist()
    fracs_col = df["cmp_mole_fractions"].tolist()
    mw_col = df["cmp1_mn_or_mw"].tolist()
    temp_col = df["Temperature, K"].tolist()
    val_col = df["value"].tolist()

    for ids_s, fracs_s, mw, temp, val in zip(ids_col, fracs_col, mw_col, temp_col, val_col):
        ids = [int(i) for i in ast.literal_eval(ids_s)]
        fracs = ast.literal_eval(fracs_s)
        if len(ids) != 2:
            continue
        polymers = [i for i in ids if is_polymer.get(i) == 1]
        salts = [i for i in ids if is_salt.get(i) == 1]
        if len(polymers) != 1 or len(salts) != 1:
            continue
        pid, sid = polymers[0], salts[0]
        try:
            if pid not in pol_cache:
                pol_cache[pid] = _polymer_vector(monomer[pid], polf)
            if sid not in salt_cache:
                salt_cache[sid] = _salt_vector(salt_smiles[sid])
        except Exception:
            continue
        if salt_cache[sid] is None:
            continue

        salt_frac = fracs[ids.index(sid)]
        feat = np.concatenate([
            pol_cache[pid], salt_cache[sid],
            [salt_frac, np.log10(mw or 1e4), temp],
        ])
        rows.append(feat)
        targets.append(float(val))
        salt_ids.append(sid)

    X = np.vstack(rows)
    y = np.asarray(targets, dtype=float)

    # cost: rarer salts are pricier to source (same idea as the synthetic space)
    salt_ids = np.asarray(salt_ids)
    uniq, counts = np.unique(salt_ids, return_counts=True)
    freq = dict(zip(uniq.tolist(), counts.tolist()))
    maxc = counts.max()
    cost = np.array([1.0 + (1.0 - freq[s] / maxc) for s in salt_ids])

    frame = pd.DataFrame({"salt_id": salt_ids, "log_conductivity": y})
    return DesignSpace(X=X, y_true=y, cost=cost,
                       feature_names=SPE_FEATURE_NAMES, frame=frame)

"""Synthetic systems with known ground truth (Milestone 9).

This module produces **no result about any real system**, and nothing downstream should
read its output as one. It exists for two reasons, both of which are about validating the
statistics rather than evaluating a model:

1. **The offline demo.** ``evalforge stats`` needs per-example scores, and the repository
   ships no model outputs (running one would need a paid API). Simulated scores let the
   ranking-stability machinery be exercised end to end on the committed fixtures.

2. **Ground truth for coverage.** A confidence interval's only testable property is that it
   contains the true parameter 95% of the time. That sentence is unusable without a true
   parameter, and a real benchmark never has one. Here the true accuracy is an input, so
   ``tests/test_statistics.py`` can measure whether the i.i.d. bootstrap actually covers it
   on clustered data. It does not — 88.4% at nominal 95% — and that measurement is the
   milestone's headline.

The generative model is a one-parameter logistic (Rasch) item-response model, chosen
because it is the simplest thing that produces the two dependencies a benchmark really has:

- **Between systems, on the same example.** A hard item is hard for everyone, so scores are
  *paired*. This is why a paired comparison is more sensitive than two marginal intervals,
  and a simulator with independent systems could not show it.
- **Within a duplicate cluster.** Item difficulty is drawn per :class:`~.bootstrap.Units`
  member, not per example, and ``icc`` controls how strongly the cluster's members share
  their outcome. At ``icc=1`` every member of a cluster gets the identical score — which is
  what a *deterministic* system does on byte-identical examples, not an assumption.

``icc`` mixes two draws that are made up front at fixed shapes, so changing it re-mixes the
same randomness rather than re-drawing it. Sweeping ``icc`` from 0 to 1 therefore traces the
design effect without confounding it with a change of sample.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eval_dataset_forge.statistics.bootstrap import DEFAULT_SEED, Units

# Three systems whose *true* abilities (in logits) are two close and one far. The pair
# exists so the demo can show a benchmark failing to separate systems it should not be able
# to separate at n=56; the outlier exists so the demo can also show one it can.
DEFAULT_ABILITIES: tuple[float, ...] = (1.4, 1.1, 0.1)
DEFAULT_SYSTEM_NAMES: tuple[str, ...] = ("sys_a", "sys_b", "sys_c")

# Item difficulty spread, in the same logit scale as ability.
DIFFICULTY_SD = 1.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_scores(
    units: Units,
    *,
    abilities: Sequence[float] = DEFAULT_ABILITIES,
    names: Sequence[str] | None = None,
    icc: float = 1.0,
    difficulty_sd: float = DIFFICULTY_SD,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[float]]:
    """Per-example binary scores for each system. Deterministic given ``seed``.

    For unit ``u`` and system ``k`` with ability ``a_k``, item difficulty ``theta_u`` is
    drawn once and the success probability is ``sigmoid(a_k - theta_u)``. Each example then
    takes the *unit's* outcome with probability ``icc`` and an independent draw at the same
    probability otherwise. At ``icc = 1`` a cluster's members are identical; at ``icc = 0``
    they are conditionally independent.

    ``icc`` changes the correlation structure, not the marginal: each example is Bernoulli at
    its unit's ``p`` either way, so a system's expected accuracy is the same at every ``icc``
    (measured: 0.766 at ``icc=0`` and 0.760 at ``icc=1`` for ``sys_a``, averaged over 200
    seeds). At a *fixed* seed the realized accuracy still moves, because a different draw is
    consumed. That is what makes an ``icc`` sweep a clean experiment on the design effect: it
    varies the dependence and holds the marginal.
    """
    if not 0.0 <= icc <= 1.0:
        raise ValueError(f"icc must lie in [0, 1], got {icc}")
    labels = tuple(names) if names is not None else DEFAULT_SYSTEM_NAMES[: len(abilities)]
    if len(labels) != len(abilities):
        raise ValueError(f"{len(labels)} names for {len(abilities)} abilities")

    n_units, n_examples = units.n_units, units.n_examples
    owner = np.empty(n_examples, dtype=int)
    for index, member in enumerate(units.members):
        owner[list(member)] = index

    rng = np.random.default_rng(seed)
    difficulty = rng.normal(0.0, difficulty_sd, size=n_units)
    unit_draw = rng.random((len(abilities), n_units))
    example_draw = rng.random((len(abilities), n_examples))
    mixture = rng.random((len(abilities), n_examples))

    probability = _sigmoid(np.asarray(abilities, dtype=float)[:, None] - difficulty[None, :])
    unit_correct = (unit_draw < probability).astype(float)[:, owner]
    example_correct = (example_draw < probability[:, owner]).astype(float)
    scores = np.where(mixture < icc, unit_correct, example_correct)
    return {name: scores[k].tolist() for k, name in enumerate(labels)}


__all__ = [
    "DEFAULT_ABILITIES",
    "DEFAULT_SYSTEM_NAMES",
    "DIFFICULTY_SD",
    "simulate_scores",
]

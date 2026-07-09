#!/usr/bin/env python3
"""Measure what the split seed does to a leaderboard at a fixed dataset.

The question Milestone 10 has to answer before it can decide what a release manifest
records: **is a dataset's content enough to identify it?** If two releases hold the same
examples, are two scores measured on them comparable?

The experiment holds the corpus and the systems fixed and varies only the split seed. Each
system's per-example score vector is generated once, by the Rasch simulator, and never
changes; the seed decides only which examples land in ``test``. Any movement in the reported
number is therefore caused entirely by the split — the one thing a content hash cannot see.

The scores are **synthetic** and describe no real system (see ``docs/STATISTICS.md``). The
block structure they are drawn over is real: it comes from the fixtures through the
deduplication ladder. The claim being measured is about the *estimator*, not about any
model, so the simulator is the right instrument: it is the only way to know the true
corpus-level accuracy that each split's test set is trying to estimate.

    python3 scripts/split_identity_experiment.py                # the table in docs/RELEASES.md
    python3 scripts/split_identity_experiment.py --seeds 1000   # tighter Monte-Carlo error

Runtime is dominated by rebuilding the splits once per seed; 400 seeds × 3 strategies takes
about a minute on one CPU core. Offline, no network, no paid API.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_dataset_forge.datasets import loader  # noqa: E402
from eval_dataset_forge.datasets.versioning import content_id, eval_id  # noqa: E402
from eval_dataset_forge.dedup import find_duplicates  # noqa: E402
from eval_dataset_forge.splits import apply_splits, build_splits  # noqa: E402
from eval_dataset_forge.statistics import build_units, simulate_scores  # noqa: E402

FIXTURES = ("agent_tasks_toy", "bio_toy", "qa_toy", "retrieval_toy", "safety_toy")
STRATEGIES = ("random", "stratified", "adversarial")
SCORE_SEED = 0
TEST = "test"


def load_corpus() -> list:
    fixtures = ROOT / "datasets" / "fixtures"
    return [
        example
        for name in FIXTURES
        for example in loader.load_examples(fixtures / name / "examples.jsonl")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=400, help="split seeds to sweep")
    args = parser.parse_args()

    examples = load_corpus()
    dedup = find_duplicates(examples)
    _per_example, blocks, _keys = build_units(examples, dedup)
    scores = simulate_scores(blocks, seed=SCORE_SEED)
    position = {example.example_id: index for index, example in enumerate(examples)}

    def accuracy(system: str, ids) -> float:
        vector = scores[system]
        chosen = [position[example_id] for example_id in ids]
        return sum(vector[i] for i in chosen) / len(chosen)

    corpus = {system: sum(vector) / len(vector) for system, vector in scores.items()}
    gap = corpus["sys_a"] - corpus["sys_b"]

    print(f"corpus: {len(examples)} examples, {len(blocks.members)} independent blocks")
    print("full-corpus accuracy: " + "  ".join(f"{s}={a:.4f}" for s, a in corpus.items()))
    print(f"gap the benchmark exists to resolve: sys_a - sys_b = {gap:+.4f}\n")

    # The identity claim, checked rather than asserted: two seeds, one dataset.
    first = apply_splits(examples, build_splits(examples, seed=0, dedup_report=dedup).assignment)
    second = apply_splits(examples, build_splits(examples, seed=1, dedup_report=dedup).assignment)
    print(
        f"two seeds, same corpus: content_id equal={content_id(first) == content_id(second)}, "
        f"eval_id equal={eval_id(first) == eval_id(second)}\n"
    )

    # Sub-population accuracies. The strategies below do not all estimate the same quantity,
    # and the mean of a strategy's test number over seeds is only interpretable against the
    # sub-population its test set is drawn from.
    duplicated = {eid for finding in dedup.findings for eid in finding.example_ids}
    singles = [e.example_id for e in examples if e.example_id not in duplicated]
    repeated = [e.example_id for e in examples if e.example_id in duplicated]
    print("estimand                          sys_a    sys_b     gap")
    for label, ids in (
        (f"whole corpus (n={len(examples)})", [e.example_id for e in examples]),
        (f"singletons only (n={len(singles)})", singles),
        (f"duplicate-cluster only (n={len(repeated)})", repeated),
    ):
        a, b = accuracy("sys_a", ids), accuracy("sys_b", ids)
        print(f"  {label:<31} {a:.4f}   {b:.4f}  {a - b:+.4f}")
    print()

    for strategy in STRATEGIES:
        acc_a, acc_b, gaps, leaders = [], [], [], {}
        ids_seen = set()
        for seed in range(args.seeds):
            report = build_splits(examples, strategy=strategy, seed=seed, dedup_report=dedup)
            ids_seen.add(content_id(apply_splits(examples, report.assignment)))
            test_ids = [eid for eid, split in report.assignment.items() if split == TEST]
            a, b = accuracy("sys_a", test_ids), accuracy("sys_b", test_ids)
            acc_a.append(a)
            acc_b.append(b)
            gaps.append(a - b)
            top = max((accuracy(s, test_ids), s) for s in scores)[0]
            winners = [s for s in scores if accuracy(s, test_ids) == top]
            key = "tie" if len(winners) > 1 else winners[0]
            leaders[key] = leaders.get(key, 0) + 1

        n_test = len(
            [
                eid
                for eid, split in build_splits(
                    examples, strategy=strategy, seed=0, dedup_report=dedup
                ).assignment.items()
                if split == TEST
            ]
        )
        agree = sum(1 for g in gaps if g > 0) / len(gaps)
        ties = sum(1 for g in gaps if g == 0) / len(gaps)
        print(f"=== {strategy} (n_test={n_test}, {args.seeds} seeds) ===")
        print(f"  distinct content_id over all seeds : {len(ids_seen)}")
        print(
            f"  sys_a test accuracy : mean={statistics.mean(acc_a):.4f} "
            f"sd={statistics.pstdev(acc_a):.4f} "
            f"range=[{min(acc_a):.3f}, {max(acc_a):.3f}]"
        )
        print(
            f"  sys_b test accuracy : mean={statistics.mean(acc_b):.4f} "
            f"sd={statistics.pstdev(acc_b):.4f}"
        )
        print(
            f"  test gap (a-b)      : mean={statistics.mean(gaps):+.4f} "
            f"sd={statistics.pstdev(gaps):.4f}"
        )
        print(f"  P(test agrees with corpus ordering of the top two) = {agree:.3f}")
        print(f"  P(the two tie on test) = {ties:.3f}")
        print(f"  leader by seed      : {dict(sorted(leaders.items()))}\n")

    print(
        "Reading: content_id is constant down every column — the dataset never changed.\n"
        "The reported number did. A release identified only by its content cannot tell a\n"
        "reader whether two scores may be compared; eval_id can, and that is why the\n"
        "manifest records both and the version bump keys on the second."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

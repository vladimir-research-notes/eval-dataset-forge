#!/usr/bin/env python3
"""Regenerate the committed demo score file for ``qa_toy``.

The scores are **synthetic**. They come from the Rasch simulator in
``eval_dataset_forge.statistics.simulate`` at a fixed seed and describe no real system; they
exist so that ``evalforge stats --scores`` has a committed, inspectable example of the score
file format, and so the reader can see what one looks like without running a model.

Deterministic: running this twice produces a byte-identical file, and
``tests/test_statistics.py`` asserts the committed file matches what this script writes. Any
change to the simulator therefore has to be re-committed on purpose rather than drifting.

    python3 scripts/make_demo_scores.py            # rewrite the fixture
    python3 scripts/make_demo_scores.py --check    # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval_dataset_forge.datasets import loader  # noqa: E402
from eval_dataset_forge.dedup import find_duplicates  # noqa: E402
from eval_dataset_forge.statistics import (  # noqa: E402
    build_units,
    simulate_scores,
    to_table,
    write_scores,
)

DATASET = ROOT / "datasets" / "fixtures" / "qa_toy" / "examples.jsonl"
TARGET = ROOT / "datasets" / "fixtures" / "qa_toy" / "system_scores.jsonl"
SEED = 0


def render() -> str:
    """The exact bytes the fixture should contain."""
    examples = loader.load_examples(DATASET)
    example_ids = [example.example_id for example in examples]
    _, blocks, _ = build_units(examples, find_duplicates(examples))
    table = to_table(simulate_scores(blocks, seed=SEED), example_ids)
    scratch = TARGET.with_suffix(".jsonl.tmp")
    write_scores(table, scratch, example_order=example_ids)
    text = scratch.read_text(encoding="utf-8")
    scratch.unlink()
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="verify the fixture is current; do not write"
    )
    args = parser.parse_args()

    expected = render()
    if args.check:
        actual = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if actual != expected:
            print(f"{TARGET} is stale; run: python3 scripts/make_demo_scores.py")
            return 1
        print(f"{TARGET.relative_to(ROOT)} is up to date")
        return 0

    TARGET.write_text(expected, encoding="utf-8")
    print(f"wrote {len(expected.splitlines())} score records to {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

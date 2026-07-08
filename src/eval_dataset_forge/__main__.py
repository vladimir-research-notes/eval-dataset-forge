"""Enable ``python3 -m eval_dataset_forge`` as an alias for the CLI."""

from eval_dataset_forge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

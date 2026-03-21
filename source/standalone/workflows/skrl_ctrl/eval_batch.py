"""Batch evaluation (B rollouts, bundle, optional memory). Equivalent to ``python cli.py eval-batch ...``."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "eval-batch")
    from cli import main

    main()

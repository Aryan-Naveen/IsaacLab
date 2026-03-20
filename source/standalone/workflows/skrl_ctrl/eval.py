"""Evaluate a checkpoint. Equivalent to ``python cli.py eval ...`` (requires ``--folder``)."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "eval")
    from cli import main

    main()

"""Single-trajectory eval with video. Equivalent to ``python cli.py eval-one ...``."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "eval-one")
    from cli import main

    main()

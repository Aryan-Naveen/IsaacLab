"""Train SAC baseline. Equivalent to ``python cli.py train-sac ...``."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "train-sac")
    from cli import main

    main()

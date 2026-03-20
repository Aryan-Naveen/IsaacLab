"""Train CTRLSAC (default). Equivalent to ``python cli.py train-ctrlsac ...``."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "train-ctrlsac")
    from cli import main

    main()

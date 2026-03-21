"""Few-shot offline finetune. Equivalent to ``python cli.py finetune ...``."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "finetune")
    from cli import main

    main()

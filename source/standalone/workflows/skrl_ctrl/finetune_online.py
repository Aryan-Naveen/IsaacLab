"""Online fine-tune (checkpoint + SequentialTrainer). Equivalent to ``python cli.py finetune-online ...``."""

import sys

if __name__ == "__main__":
    sys.argv.insert(1, "finetune-online")
    from cli import main

    main()

#!/usr/bin/env python3
"""
run.py — thin wrapper around symveig.cli.main().

Lets you reproduce all results with `python run.py [flags]` from the
repository root without installing the package. After `pip install .`
the same entry point is available as the console command `symveig-run`.

See `python run.py --help` for options.
"""
import os
import sys

# Make `import symveig` resolve from the repo root (no install required).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from symveig.cli import main

if __name__ == "__main__":
    main()

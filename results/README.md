# About this directory

These are **example outputs** produced by a reference run of

    python run.py

on a development machine (default settings: 1D models to L = 10, the 3x3
2D lattice). They are included so that a reader can inspect the output
format and the figures without first running anything.

For the archived/published release, regenerate this directory on the
target machine with the full configuration:

    python run.py --full

and (optionally, to keep a laptop cool during the one-time global
baseline)

    python run.py --full --threads 4

The `metadata.json` file records the platform and library versions of
the run that produced the rest of the directory, so the provenance of
any archived results is self-documenting.

See `../docs/reproducibility.md` for the reference table and the expected
cross-machine agreement tolerances.

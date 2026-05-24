# Usage guide

## Installation

From the repository root:

```bash
pip install .
```

or, for development (editable):

```bash
pip install -e ".[dev]"
```

The only hard dependencies are NumPy and SciPy. `matplotlib` is needed
to regenerate figures (`pip install ".[plots]"`).

## Library API

```python
import numpy as np
from symveig import enclose_global, enclose_sectors, summary_stats

# Any Hermitian matrix
A = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=complex)

# Verified enclosures for all eigenvalues
encs = enclose_global(A)
for e in encs:
    print(e)            # Enclosure([lo, hi], mult=..., tier=...)
    print(e.lo, e.hi)   # rigorous interval endpoints
```

Each returned `Enclosure` has:

- `midpoint`, `halfwidth` — the interval is `[midpoint - halfwidth, midpoint + halfwidth]`.
- `lo`, `hi` — convenience properties for the endpoints.
- `multiplicity` — number of true eigenvalues guaranteed inside.
- `tier` — which bound produced it (`'bauer-fike'`, `'merged'`, `'rump-lange'`).

The guarantee: the union of all returned intervals contains every
eigenvalue of `A`, and each interval contains exactly `multiplicity`
of them, under IEEE 754 round-to-nearest arithmetic.

## Sector decomposition

If your operator commutes with a conserved quantity `S` that is diagonal
in the working basis (e.g. total magnetization), pass its diagonal:

```python
from symveig import enclose_sectors
from symveig.models import heisenberg_1d

H, sectors = heisenberg_1d(L=10, periodic=True)   # returns {"Sz": diag}
encs = enclose_sectors(H, sectors["Sz"])
```

This is both tighter and much faster than `enclose_global` because it
never diagonalizes the full matrix — only the per-sector blocks.

## Built-in models

`symveig.models` provides:

- `heisenberg_1d(L, periodic=True, J=1.0)` → conserves `Sz`
- `transverse_field_ising_1d(L, periodic=True, J=1.0, hx=0.5)` → no
  diagonal symmetry (parity is not diagonal in the computational basis)
- `j1j2_heisenberg_1d(L, periodic=True, J1=1.0, J2=0.5)` → conserves `Sz`
- `heisenberg_2d(Lx, Ly, periodic=True, J=1.0)` → conserves `Sz`

Each returns `(H, sectors)` where `sectors` is a dict from observable
name to its diagonal vector.

## Reproducing the paper

```bash
python run.py              # default: L up to 10, ~10 s, all figures
python run.py --full       # adds L=12 and 4x3 2D (heavier; auto-throttles threads)
python run.py --full --threads 4   # cap BLAS threads explicitly
python run.py --tests-only # just the test suite
```

Outputs land in `results/`:

- `metadata.json` — platform, versions, timestamp
- `tests.log` — test output
- `benchmark_<model>.json` — full per-model results
- `summary.csv` — one row per (model, L), ready for the paper tables
- `figures/*.pdf`, `*.png` — all figures

After the benchmark has run once, figures can be regenerated from the
saved JSON without recomputing:

```bash
python -c "from symveig.plot_results import main; main('results')"
```

## Running the tests

```bash
python run.py --tests-only
# or, with pytest:
pytest tests/
```

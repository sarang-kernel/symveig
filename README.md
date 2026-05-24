# symveig

**Verified eigenvalue enclosures for symmetry-decomposed Hermitian matrices.**

`symveig` computes rigorous, machine-checkable intervals that are
*guaranteed* to contain the true eigenvalues of a Hermitian matrix. When
the matrix commutes with an abelian conserved quantity (such as the total
magnetization of a spin model), it exploits the resulting symmetry-sector
block structure to produce enclosures that are both **tighter** and
**faster to compute** than verifying the full matrix.

It is written in pure NumPy/SciPy and requires **no INTLAB and no
MATLAB** — the verified-eigensolver capability that previously lived only
in those commercial tools, now in the standard scientific-Python stack,
and aware of the symmetry structure that quantum lattice models possess.

This is the Phase-2 companion to CERTIFY-ED.

---

## What "verified" means here

For a Hermitian matrix `A`, `enclose_global(A)` returns intervals with
the guarantee:

> Each returned interval `[lo, hi]` contains exactly `multiplicity` true
> eigenvalues of `A`, counting algebraic multiplicity, and the union of
> all intervals contains every eigenvalue — provably, under IEEE 754
> round-to-nearest arithmetic.

The half-widths are rigorous upper bounds on the distance from each
computed eigenvalue to the nearest true eigenvalue, derived with explicit
floating-point error analysis (Higham `gamma_k` accounting). There is no
heuristic slack; see [`docs/theory.md`](docs/theory.md).

## Install

```bash
pip install .
```

Dependencies: NumPy and SciPy (and matplotlib for figures). Python >= 3.9.

## 60-second example

```python
import numpy as np
from symveig import enclose_global, enclose_sectors
from symveig.models import heisenberg_1d

# Verified enclosures for any Hermitian matrix
A = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=complex)
for e in enclose_global(A):
    print(e)            # Enclosure([1.0e+00, 1.0e+00], mult=1, ...)

# For a lattice model with a U(1) symmetry, use the sector path:
H, sectors = heisenberg_1d(L=10, periodic=True)   # conserves total Sz
encs = enclose_sectors(H, sectors["Sz"])          # tighter AND faster
```

Or just run the bundled example:

```bash
python examples/quickstart.py
```

## Reproduce the paper

```bash
python run.py              # default: L up to 10, ~10 s, writes results/
python run.py --full       # adds L=12 and 4x3 2D (auto-throttles threads)
python run.py --tests-only # run only the test suite
```

Everything lands in `results/`: JSON per model, a `summary.csv`, and all
figures as PDF + PNG. See [`docs/reproducibility.md`](docs/reproducibility.md)
for the reference table and expected cross-machine tolerances.

## Why sector decomposition

If `A` commutes with an observable `S` that is diagonal in the working
basis, `A` is block diagonal across the eigenspaces (sectors) of `S`.
Verifying each block independently and taking the union:

- **Tighter:** the worst sector block has dimension `~ n / sqrt(L)`
  rather than `n`, and the half-width scales with the dimension — a
  measured 3–9x improvement for `L = 4–12`.
- **Faster:** the sector path never diagonalizes the full matrix; it
  solves several small blocks instead of one big one — a measured
  25–50x wall-clock speedup at `L = 10–12`.

In practice you only ever run the sector path; the global path exists in
the benchmark solely to provide the comparison baseline.

## Repository layout

```
symveig/
├── LICENSE                     MIT
├── README.md                   this file
├── CITATION.cff                citation metadata (GitHub/Zenodo)
├── CHANGELOG.md                version history
├── AUTHORS.md                  authors and acknowledgements
├── .zenodo.json                Zenodo deposit metadata
├── pyproject.toml              packaging (pip-installable)
├── requirements.txt            dependency pins
├── MANIFEST.in                 source-distribution manifest
├── conftest.py                 pytest path setup
├── run.py                      one-command reproducibility driver
├── docs/
│   ├── cpc_program_summary.md  CPC Program Summary
│   ├── theory.md               full error analysis
│   ├── usage.md                API and CLI guide
│   └── reproducibility.md      how to reproduce, reference table
├── symveig/                    the package
│   ├── __init__.py             public API
│   ├── certify_eig.py          verified enclosure engine
│   ├── models.py               lattice Hamiltonian builders
│   ├── benchmark.py            benchmark engine
│   ├── plot_results.py         figure generation
│   └── cli.py                  the orchestrator (symveig-run)
├── tests/                      unit + reproducibility tests
├── examples/
│   └── quickstart.py           minimal usage example
└── results/                    frozen example outputs (this release)
```

## Tests

```bash
python run.py --tests-only      # or: pytest tests/
```

The suite covers synthetic isolated, clustered, sectored,
ill-conditioned, and near-degenerate matrices; model sanity checks
(Hermiticity, commutation, sector sizes); and a same-machine determinism
check.

## Scope and limitations

- `A` must be Hermitian.
- The symmetry observable must be exactly diagonal in the working basis
  (true for standard abelian U(1) quantities like total `Sz` or particle
  number). Momentum, point-group, and non-abelian (SU(2)) symmetries are
  out of scope for the present sector machinery.
- Dense methods are practical to dimension ~ 4096 (1D spin-1/2 chains to
  `L = 12`) on a workstation; the cost is the `O(n^3)` eigendecomposition.
- An optional Tier-2 cluster refinement (Rump–Lange) is included but
  disabled by default, because without directed-rounding arithmetic it is
  dominated by worst-case error bounds. Tightening it is identified as
  future work.

## License

MIT — see [`LICENSE`](LICENSE).

## Citing

Please cite both the software (via [`CITATION.cff`](CITATION.cff) / the
Zenodo DOI) and the accompanying paper once published. See
[`docs/cpc_program_summary.md`](docs/cpc_program_summary.md).

# Reproducibility

This package is designed so a third party can reproduce every number and
figure in the accompanying paper with a single command.

## One command

```bash
pip install .
python run.py --full
```

`--full` reproduces the complete table (1D models up to L = 12, plus the
4x3 2D lattice). Without `--full`, the run stops at L = 10 and the 3x3
2D lattice, which is enough to establish every trend in ~10 seconds.

All outputs are written to `results/`:

| File                          | Contents                                       |
|-------------------------------|------------------------------------------------|
| `metadata.json`               | platform, Python/NumPy versions, timestamp     |
| `tests.log`                   | output of the unit-test suite                  |
| `benchmark_<model>.json`      | full per-model, per-L results                  |
| `summary.csv`                 | one row per (model, L), paper-table-ready      |
| `figures/fig_*.pdf` / `.png`  | all paper figures                              |

Figures can be regenerated from the saved JSON without recomputing:

```bash
python -c "from symveig.plot_results import main; main('results')"
```

## What is guaranteed to reproduce, and to what precision

The verification is deterministic given the matrix, but the inputs come
from LAPACK's symmetric eigensolver, which is *not* bit-identical across
BLAS implementations or thread counts. Expected agreement across
machines:

- **Ground-state energies:** ~1e-14 (LAPACK accuracy).
- **Enclosure half-widths:** ~10% (floating-point summation order in the
  residual/norm bounds shifts them by a few percent).
- **Global/sector width ratios:** ~0.1.
- **Containment:** exact — every reference eigenvalue lies inside some
  enclosure on every tested platform. This is the property that matters;
  the bounds are larger than the cross-machine input variability.

The unit test `tests/test_reproducibility.py` additionally checks that
two runs *on the same machine* produce bit-identical results (no hidden
nondeterminism such as unseeded randomness).

## Reference results

Produced on Linux x86-64, NumPy 2.4, default thread count. Half-widths
track `h ≈ 6 u d ||A_s||_F` (with `u = 2^-53`) across all model families
spanning four orders of magnitude in `d ||A_s||_F`.

| Model                        | L  | dim  | global max half | sector max half | width ratio |
|------------------------------|----|------|-----------------|-----------------|-------------|
| 1D Heisenberg OBC            | 4  | 16   | 3.15e-14        | 9.33e-15        | 3.38        |
| 1D Heisenberg OBC            | 6  | 64   | 3.16e-13        | 6.26e-14        | 5.05        |
| 1D Heisenberg OBC            | 8  | 256  | 2.96e-12        | 4.56e-13        | 6.49        |
| 1D Heisenberg OBC            | 10 | 1024 | 2.68e-11        | 3.44e-12        | 7.77        |
| 1D Heisenberg OBC            | 12 | 4096 | 2.36e-10        | 2.64e-11        | 8.94        |
| 1D Heisenberg PBC            | 4  | 16   | 3.73e-14        | 1.15e-14        | 3.23        |
| 1D Heisenberg PBC            | 6  | 64   | 3.46e-13        | 6.88e-14        | 5.02        |
| 1D Heisenberg PBC            | 8  | 256  | 3.16e-12        | 4.88e-13        | 6.47        |
| 1D Heisenberg PBC            | 10 | 1024 | 2.82e-11        | 3.64e-12        | 7.75        |
| 1D Heisenberg PBC            | 12 | 4096 | 2.47e-10        | 2.77e-11        | 8.93        |
| 1D J1-J2 Heisenberg (J2=0.5) | 4  | 16   | 4.46e-14        | 1.20e-14        | 3.72        |
| 1D J1-J2 Heisenberg (J2=0.5) | 6  | 64   | 3.87e-13        | 7.42e-14        | 5.22        |
| 1D J1-J2 Heisenberg (J2=0.5) | 8  | 256  | 3.53e-12        | 5.30e-13        | 6.66        |
| 1D J1-J2 Heisenberg (J2=0.5) | 10 | 1024 | 3.15e-11        | 3.99e-12        | 7.91        |
| 1D J1-J2 Heisenberg (J2=0.5) | 12 | 4096 | 2.76e-10        | 3.05e-11        | 9.07        |
| 2D Heisenberg 3x3 PBC        | 9  | 512  | 1.34e-11        | 1.67e-12        | 8.03        |
| 2D Heisenberg 4x3 PBC        | 12 | 4096 | 3.49e-10        | 3.84e-11        | 9.10        |

In every row, every reference eigenvalue is contained in both the global
and the sector enclosures.

## Performance and thermal notes

The dominant cost is the dense complex Hermitian eigendecomposition,
which is `O(n^3)` — about 100 s for the L = 12 *global* baseline
(dim 4096) on a typical laptop, using all cores. The sector path takes a
few seconds for the same case. To keep a laptop cool during the one-time
global baseline, `--full` auto-throttles BLAS to half the available
cores; pass `--threads N` to set the cap explicitly. None of this
affects the computed enclosures, only wall-clock time and CPU load.

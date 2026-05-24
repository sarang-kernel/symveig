# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-05-22

First public release.

### Added
- `enclose_global(H)`: rigorous Bauer–Fike–Wilkinson eigenvalue
  enclosures for Hermitian matrices, with explicit floating-point error
  bounds (Higham gamma_k accounting; no heuristic slack).
- `enclose_sectors(H, S_diag)`: symmetry-sector decomposition that
  yields tighter enclosures and faster computation for matrices with an
  abelian conserved quantity diagonal in the working basis.
- Optional Tier-2 cluster refinement (Rump–Lange Lemma 6.1), off by
  default; documented limitation that it is dominated by Higham-style
  worst-case bounds without directed rounding.
- Model builders: 1D Heisenberg (OBC/PBC), transverse-field Ising,
  J1–J2 Heisenberg, 2D Heisenberg.
- Benchmark engine reporting enclosure widths, global/sector width
  ratios, and global/sector wall-clock speedup.
- One-command reproducibility driver (`run.py` / `symveig-run`) writing
  JSON results, a summary CSV, and publication-quality figures.
- Test suite: synthetic isolated/clustered/sectored matrices,
  ill-conditioned and near-degenerate cases, model sanity checks, and a
  determinism (reproducibility) test.
- Packaging: `pyproject.toml`, `CITATION.cff`, `.zenodo.json`, MIT
  license, CPC program summary.

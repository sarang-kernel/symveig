"""
symveig — Verified eigenvalue enclosures for symmetry-decomposed
          Hermitian matrices.

Public API
----------
enclose_global(H)              Verified enclosures for all eigenvalues of H.
enclose_sectors(H, S_diag)     Sector-decomposed verified enclosures.
Enclosure                      Dataclass for one (cluster) enclosure.
summary_stats(enclosures)      Aggregate statistics.

Model builders (symveig.models) and the benchmark engine
(symveig.benchmark) are also provided for reproducing the accompanying
paper's results.

This package is the Phase-2 companion to CERTIFY-ED. It provides rigorous,
INTLAB-free, MATLAB-free verified eigenvalue enclosures in pure
NumPy/SciPy, with a symmetry-sector decomposition that yields tighter
bounds and faster computation for quantum lattice Hamiltonians with
abelian (U(1)-type) conserved quantities.
"""

from .certify_eig import (
    enclose_global,
    enclose_sectors,
    Enclosure,
    summary_stats,
    U_DOUBLE,
)

__all__ = [
    "enclose_global",
    "enclose_sectors",
    "Enclosure",
    "summary_stats",
    "U_DOUBLE",
]

__version__ = "1.0.0"

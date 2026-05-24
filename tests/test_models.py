"""
tests/test_models.py
====================

Tests for the lattice model builders. Verifies:
  - H is Hermitian
  - [H, S] = 0 for claimed sector observables
  - Sector sizes match expected combinatorics
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from math import comb
from symveig.models import (heisenberg_1d, transverse_field_ising_1d,
                    j1j2_heisenberg_1d, heisenberg_2d)


def _commutator_norm(A, B):
    return float(np.linalg.norm(A @ B - B @ A, 'fro'))


def _diag_to_op(diag):
    return np.diag(diag.astype(np.complex128))


def test_heisenberg_1d():
    for L in [4, 6, 8]:
        for periodic in [True, False]:
            H, sectors = heisenberg_1d(L, periodic=periodic)
            herm = float(np.linalg.norm(H - H.conj().T, 'fro'))
            assert herm < 1e-10, f"L={L} periodic={periodic} H not Hermitian: {herm}"
            S = _diag_to_op(sectors["Sz"])
            comm = _commutator_norm(H, S)
            assert comm < 1e-10, f"L={L} periodic={periodic} [H,Sz] != 0: {comm}"
            # Sector sizes: binomial(L, L/2 + Sz)
            sz_values, counts = np.unique(sectors["Sz"], return_counts=True)
            for sv, cn in zip(sz_values, counts):
                expected = comb(L, int(L/2 + sv))
                assert cn == expected, f"L={L} Sz={sv}: got {cn}, expected {expected}"
    print("  test_heisenberg_1d PASS")


def test_tfi_1d():
    for L in [4, 6, 8]:
        H, sectors = transverse_field_ising_1d(L, periodic=True, hx=0.5)
        herm = float(np.linalg.norm(H - H.conj().T, 'fro'))
        assert herm < 1e-10, f"L={L} TFI not Hermitian: {herm}"
        # TFI has no diagonal symmetry for general hx
        assert sectors == {}, "TFI should expose no diagonal symmetries"
    print("  test_tfi_1d PASS")


def test_j1j2_1d():
    for L in [4, 6, 8]:
        H, sectors = j1j2_heisenberg_1d(L, periodic=True, J2=0.5)
        herm = float(np.linalg.norm(H - H.conj().T, 'fro'))
        assert herm < 1e-10
        S = _diag_to_op(sectors["Sz"])
        assert _commutator_norm(H, S) < 1e-10
    print("  test_j1j2_1d PASS")


def test_heisenberg_2d_small():
    # 2x3 = 6 sites, dim 64, fast
    H, sectors = heisenberg_2d(2, 3, periodic=True)
    herm = float(np.linalg.norm(H - H.conj().T, 'fro'))
    assert herm < 1e-10
    S = _diag_to_op(sectors["Sz"])
    assert _commutator_norm(H, S) < 1e-10
    # 3x3 = 9 sites, dim 512 — also test that it builds
    H, sectors = heisenberg_2d(3, 3, periodic=True)
    assert H.shape == (512, 512)
    print("  test_heisenberg_2d_small PASS")


def main():
    test_heisenberg_1d()
    test_tfi_1d()
    test_j1j2_1d()
    test_heisenberg_2d_small()
    print("All model tests passed.")


if __name__ == "__main__":
    main()

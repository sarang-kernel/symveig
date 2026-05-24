"""
models.py
=========

Hermitian quantum lattice Hamiltonians with their natural commuting
observables, returned as numpy arrays for use with certify_eig.

All builders return (H, sectors) where sectors is a dict mapping a
symmetry-observable name to a (n,) numpy array of its diagonal values
in the computational basis. For symmetries that are not diagonal in
the computational basis (e.g. momentum, parity in 1D), they are not
included here -- the implementation paper focuses on abelian U(1)-type
diagonal symmetries for which the substitution-of-blocking-source story
is cleanest.

The implementations use sparse Kronecker products and convert to dense
at the end; suitable for L up to ~14 in 1D spin-1/2.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp


def _pauli():
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return sx, sy, sz


def _site_op_sparse(op_2x2, site, L):
    """Sparse site operator: 2x2 op at `site`, identity elsewhere.
    Convention: site 0 is the rightmost (lowest bit) in basis index.
    """
    op = sp.csr_matrix(op_2x2)
    I2 = sp.eye(2, format='csr', dtype=np.complex128)
    result = sp.eye(1, format='csr', dtype=np.complex128)
    for k in range(L - 1, -1, -1):
        result = sp.kron(result, op if k == site else I2, format='csr')
    return result


def _total_sz_diag(L):
    """Diagonal of total Sz = sum_i S^z_i in computational basis."""
    dim = 2 ** L
    diag = np.zeros(dim)
    for state in range(dim):
        for s in range(L):
            diag[state] += 0.5 if not ((state >> s) & 1) else -0.5
    return diag


def heisenberg_1d(L, periodic=True, J=1.0):
    """Isotropic Heisenberg chain: H = J sum_<i,j> S_i . S_j.

    Conservation: total Sz (always).

    Returns
    -------
    H : (2^L, 2^L) complex Hermitian dense
    sectors : {"Sz": (2^L,) array}
    """
    sx, sy, sz = _pauli()
    dim = 2 ** L
    H_sp = sp.csr_matrix((dim, dim), dtype=np.complex128)
    bonds = ([(i, (i+1) % L) for i in range(L)] if periodic
             else [(i, i+1) for i in range(L-1)])
    for (i, j) in bonds:
        H_sp = H_sp + J * (_site_op_sparse(sx, i, L) @ _site_op_sparse(sx, j, L))
        H_sp = H_sp + J * (_site_op_sparse(sy, i, L) @ _site_op_sparse(sy, j, L))
        H_sp = H_sp + J * (_site_op_sparse(sz, i, L) @ _site_op_sparse(sz, j, L))
    H = H_sp.toarray()
    H = 0.5 * (H + H.conj().T)
    return H, {"Sz": _total_sz_diag(L)}


def transverse_field_ising_1d(L, periodic=True, J=1.0, hx=0.5):
    """Transverse-field Ising chain: H = -J sum_i sz_i sz_{i+1} - hx sum_i sx_i.

    Conservation: parity P = prod_i sx_i (Z_2 symmetry).
    P is *not* diagonal in computational basis, so we don't include it
    as a sector observable here.

    Returns
    -------
    H : dense Hermitian
    sectors : {} (no diagonal symmetry for general hx)
    """
    sx, _sy, sz = _pauli()
    dim = 2 ** L
    H_sp = sp.csr_matrix((dim, dim), dtype=np.complex128)
    bonds = ([(i, (i+1) % L) for i in range(L)] if periodic
             else [(i, i+1) for i in range(L-1)])
    for (i, j) in bonds:
        H_sp = H_sp - J * (_site_op_sparse(sz, i, L) @ _site_op_sparse(sz, j, L))
    for i in range(L):
        H_sp = H_sp - hx * _site_op_sparse(sx, i, L)
    H = H_sp.toarray()
    H = 0.5 * (H + H.conj().T)
    return H, {}


def j1j2_heisenberg_1d(L, periodic=True, J1=1.0, J2=0.5):
    """J1-J2 Heisenberg chain. Conserves total Sz."""
    sx, sy, sz = _pauli()
    dim = 2 ** L
    H_sp = sp.csr_matrix((dim, dim), dtype=np.complex128)
    bonds_nn = ([(i, (i+1) % L) for i in range(L)] if periodic
                else [(i, i+1) for i in range(L-1)])
    bonds_nnn = ([(i, (i+2) % L) for i in range(L)] if periodic
                 else [(i, i+2) for i in range(L-2)])
    for (i, j) in bonds_nn:
        H_sp = H_sp + J1 * (_site_op_sparse(sx, i, L) @ _site_op_sparse(sx, j, L))
        H_sp = H_sp + J1 * (_site_op_sparse(sy, i, L) @ _site_op_sparse(sy, j, L))
        H_sp = H_sp + J1 * (_site_op_sparse(sz, i, L) @ _site_op_sparse(sz, j, L))
    for (i, j) in bonds_nnn:
        H_sp = H_sp + J2 * (_site_op_sparse(sx, i, L) @ _site_op_sparse(sx, j, L))
        H_sp = H_sp + J2 * (_site_op_sparse(sy, i, L) @ _site_op_sparse(sy, j, L))
        H_sp = H_sp + J2 * (_site_op_sparse(sz, i, L) @ _site_op_sparse(sz, j, L))
    H = H_sp.toarray()
    H = 0.5 * (H + H.conj().T)
    return H, {"Sz": _total_sz_diag(L)}


def heisenberg_2d(Lx, Ly, periodic=True, J=1.0):
    """2D Heisenberg on Lx x Ly lattice. Conserves total Sz.

    Site index: i + Lx * j  where i in [0, Lx), j in [0, Ly).
    """
    sx, sy, sz = _pauli()
    L = Lx * Ly
    dim = 2 ** L
    H_sp = sp.csr_matrix((dim, dim), dtype=np.complex128)
    bonds = []
    for jx in range(Lx):
        for jy in range(Ly):
            s1 = jx + Lx * jy
            # +x bond
            if periodic or jx + 1 < Lx:
                s2 = ((jx + 1) % Lx) + Lx * jy
                bonds.append((s1, s2))
            # +y bond
            if periodic or jy + 1 < Ly:
                s2 = jx + Lx * ((jy + 1) % Ly)
                bonds.append((s1, s2))
    for (i, j) in bonds:
        H_sp = H_sp + J * (_site_op_sparse(sx, i, L) @ _site_op_sparse(sx, j, L))
        H_sp = H_sp + J * (_site_op_sparse(sy, i, L) @ _site_op_sparse(sy, j, L))
        H_sp = H_sp + J * (_site_op_sparse(sz, i, L) @ _site_op_sparse(sz, j, L))
    H = H_sp.toarray()
    H = 0.5 * (H + H.conj().T)
    return H, {"Sz": _total_sz_diag(L)}


# Registry of named models for the benchmark driver.
MODELS = {}

def register_model(name, builder_factory, label, expected_sector_name=None):
    """Register a model under a stable name."""
    MODELS[name] = {
        "builder": builder_factory,
        "label": label,
        "sector_name": expected_sector_name,
    }

# Standard catalogue used by the benchmark.
register_model(
    "heisenberg_1d_pbc",
    lambda L: heisenberg_1d(L, periodic=True),
    "1D Heisenberg PBC",
    "Sz",
)
register_model(
    "heisenberg_1d_obc",
    lambda L: heisenberg_1d(L, periodic=False),
    "1D Heisenberg OBC",
    "Sz",
)
register_model(
    "j1j2_1d_pbc",
    lambda L: j1j2_heisenberg_1d(L, periodic=True, J2=0.5),
    "1D J1-J2 Heisenberg (J2=0.5) PBC",
    "Sz",
)
# 2D models registered separately because they use (Lx, Ly) shape.
TWO_D_MODELS = {
    "heisenberg_2d_3x3_pbc": {
        "builder": lambda: heisenberg_2d(3, 3, periodic=True),
        "label": "2D Heisenberg 3x3 PBC",
        "Lx": 3, "Ly": 3,
        "sector_name": "Sz",
    },
}

TWO_D_MODELS_FULL = {
    **TWO_D_MODELS,
    "heisenberg_2d_4x3_pbc": {
        "builder": lambda: heisenberg_2d(4, 3, periodic=True),
        "label": "2D Heisenberg 4x3 PBC",
        "Lx": 4, "Ly": 3,
        "sector_name": "Sz",
    },
}

"""
benchmark.py
============

Core benchmark engine. Given a Hamiltonian and an optional sector
observable, compute:
  - global verified enclosures (the expensive baseline)
  - sector-decomposed verified enclosures (tighter AND cheaper)
  - per-sector breakdown (single pass, no recomputation)
  - reference eigenvalues
  - timings, the global/sector width ratio, the global/sector speedup

Returns a serialisable dict suitable for JSON dumping.

Performance notes
-----------------
The dominant cost is the dense eigendecomposition np.linalg.eigh, which
is O(n^3). The whole point of symmetry decomposition is that the sector
path NEVER diagonalises the full n x n matrix: it only diagonalises the
per-sector blocks, the largest of which has dimension ~ binomial(L, L/2)
(about n/4 for 1D spin-1/2). The total sector work sum_s n_s^3 is far
less than n^3, so the sector path is both tighter AND substantially
faster. The benchmark reports both.

We do NOT use the SVD-based spectral norm np.linalg.norm(H, ord=2): for
a Hermitian matrix the 2-norm equals max|eigenvalue|, which we already
have from the eigendecomposition. (Calling norm(ord=2) triggers a full
SVD that is more expensive than everything else combined.)

To throttle BLAS threads on a laptop:
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python run.py
"""

from __future__ import annotations
import time
import numpy as np
from .certify_eig import enclose_global, summary_stats


def _hermitian_2norm(eigvals: np.ndarray) -> float:
    """2-norm of a Hermitian matrix = max |eigenvalue|. No SVD."""
    return float(np.max(np.abs(eigvals)))


def benchmark_model(name: str, H: np.ndarray, sectors: dict,
                    sector_to_use: str = None,
                    compute_tier2: bool = False) -> dict:
    """Run global + sector benchmark on one Hamiltonian.

    H is assumed exactly Hermitian (the model builders guarantee this).
    """
    n = H.shape[0]
    norm_H_F = float(np.linalg.norm(H, ord='fro'))  # O(n^2), cheap
    herm_err = float(np.linalg.norm(H - H.conj().T, ord='fro'))

    # ---- Global path: ONE eigendecomposition, reused for everything ----
    t0 = time.time()
    mu_global, X_global = np.linalg.eigh(H)
    t_eigh_global = time.time() - t0
    ref = mu_global  # eigenvalues serve as the reference for containment

    norm_H_2 = _hermitian_2norm(mu_global)  # no SVD

    out = {
        "name": name,
        "dim": n,
        "norm_H_F": norm_H_F,
        "norm_H_2": norm_H_2,
        "hermitian_err": herm_err,
        "ground_state": float(mu_global[0]),
        "highest": float(mu_global[-1]),
        "eigh_global_time": t_eigh_global,
        "global_t1": None,
        "global_t2": None,
        "sector_t1": None,
        "sector_t2": None,
        "per_sector": None,
    }

    def _serialize(enclosures, t):
        s = summary_stats(enclosures)
        s["time"] = t
        return s

    # Global Tier 1, reusing the eigendecomposition (only the residual
    # bound is recomputed; the expensive eigh is not repeated).
    t0 = time.time()
    enc_g_t1 = enclose_global(H, tier2=False,
                              eigh_result=(mu_global, X_global),
                              assume_hermitian=True)
    t_g1 = time.time() - t0
    out["global_t1"] = _serialize(enc_g_t1, t_g1)
    out["global_t1"]["all_contained"] = _check_all_contained(enc_g_t1, ref)
    # Total global cost = eigh + bound (this is the expensive baseline)
    out["global_t1"]["total_time_incl_eigh"] = t_eigh_global + t_g1

    if compute_tier2:
        t0 = time.time()
        enc_g_t2 = enclose_global(H, tier2=True,
                                  eigh_result=(mu_global, X_global),
                                  assume_hermitian=True)
        out["global_t2"] = _serialize(enc_g_t2, time.time() - t0)
        out["global_t2"]["all_contained"] = _check_all_contained(enc_g_t2, ref)

    # ---- Sector path: never touches the full n x n eigh ----
    if sector_to_use is not None and sector_to_use in sectors:
        S_diag = sectors[sector_to_use]
        atol = 1e-12

        all_enclosures = []
        per_sector = []
        sector_int_labels = sorted(set(np.round(S_diag / atol).astype(int)))
        t_sector_start = time.time()
        for svi in sector_int_labels:
            sv = svi * atol
            mask = np.abs(S_diag - sv) <= atol
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            # Slice out the sector block. Already Hermitian (sub-block of
            # a Hermitian matrix in a basis where S is diagonal).
            H_s = np.ascontiguousarray(H[np.ix_(idx, idx)])
            ds = H_s.shape[0]
            mu_s, X_s = np.linalg.eigh(H_s)
            sec_encs = enclose_global(H_s, tier2=False,
                                      eigh_result=(mu_s, X_s),
                                      assume_hermitian=True)
            all_enclosures.extend(sec_encs)
            stats = summary_stats(sec_encs)
            per_sector.append({
                "sector_label": float(sv),
                "dim": ds,
                "norm_H_s_F": float(np.linalg.norm(H_s, ord='fro')),
                "norm_H_s_2": _hermitian_2norm(mu_s),  # no SVD
                "max_halfwidth": stats["max_halfwidth"],
                "median_halfwidth": stats["median_halfwidth"],
            })
        all_enclosures.sort(key=lambda e: e.midpoint)
        t_sector_total = time.time() - t_sector_start
        out["sector_t1"] = _serialize(all_enclosures, t_sector_total)
        out["sector_t1"]["all_contained"] = _check_all_contained(all_enclosures, ref)
        out["per_sector"] = per_sector

        if compute_tier2:
            t0 = time.time()
            all_encs_t2 = []
            for entry in per_sector:
                sv = entry["sector_label"]
                mask = np.abs(S_diag - sv) <= atol
                idx = np.where(mask)[0]
                H_s = np.ascontiguousarray(H[np.ix_(idx, idx)])
                mu_s, X_s = np.linalg.eigh(H_s)
                all_encs_t2.extend(enclose_global(
                    H_s, tier2=True, eigh_result=(mu_s, X_s),
                    assume_hermitian=True))
            all_encs_t2.sort(key=lambda e: e.midpoint)
            out["sector_t2"] = _serialize(all_encs_t2, time.time() - t0)
            out["sector_t2"]["all_contained"] = _check_all_contained(all_encs_t2, ref)

        # Width ratios
        out["ratio_t1_max"] = (out["global_t1"]["max_halfwidth"]
                               / out["sector_t1"]["max_halfwidth"])
        out["ratio_t1_med"] = (out["global_t1"]["median_halfwidth"]
                               / out["sector_t1"]["median_halfwidth"])
        if compute_tier2:
            out["ratio_t2_max"] = (out["global_t2"]["max_halfwidth"]
                                   / out["sector_t2"]["max_halfwidth"])
        # Speedup: full global cost (eigh + bound) vs sector cost
        out["speedup_global_over_sector"] = (
            out["global_t1"]["total_time_incl_eigh"] / max(t_sector_total, 1e-9))

    return out


def _check_all_contained(enclosures, ref):
    """Verify every reference eigenvalue is contained in some enclosure."""
    encs_sorted = sorted(enclosures, key=lambda e: e.midpoint)
    remaining_mult = [e.multiplicity for e in encs_sorted]
    for t in np.sort(ref):
        found = False
        for ei, e in enumerate(encs_sorted):
            if remaining_mult[ei] > 0 and e.lo <= t <= e.hi:
                remaining_mult[ei] -= 1
                found = True
                break
        if not found:
            return False
    return True

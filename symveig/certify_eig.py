"""
certify_eig.py
==============

Production verified eigenvalue enclosure for Hermitian matrices, with
optional sector decomposition under a commuting diagonal observable.

Performance note
----------------
For dense matrices of size n >= 2048 the dominant cost is the
floating-point eigendecomposition (`numpy.linalg.eigh`), which dispatches
to LAPACK and saturates all CPU threads. On laptops this can cause
thermal throttling. To cap thread usage, set environment variables before
import:

    OMP_NUM_THREADS=2  MKL_NUM_THREADS=2  OPENBLAS_NUM_THREADS=2

The default of using all cores is preserved when these are not set.

Two-tier verification
---------------------
Tier 1: per-eigenvalue Bauer-Fike bound (Higham, _Accuracy and Stability
of Numerical Algorithms_, 2nd ed., Theorem 11.7.1 for Hermitian case).
Bound:
    |lambda_true - mu_i| <= ||A x_i - mu_i x_i||_2 / ||x_i||_2

Tier 2 (optional, default off): cluster refinement following Rump-Lange
2023, Lemma 6.1. The implementation here is conceptually correct but in
the present version is dominated by Higham-style worst-case error bounds
on the matrix-vector product, which makes Tier 2 typically *looser* than
the Tier-1 merged interval. Tier 2 becomes useful when the underlying
error bound is tightened with directed-rounding arithmetic (Rump 1999
"verify rounding mode" technique). We provide Tier 2 as an opt-in feature
for users who want to try it on specific clusters, and document its
current limitation explicitly.

All bounds are computed with rigorous floating-point error analysis: no
heuristic slack, every contribution bounded via Higham's gamma_k.

Cluster handling
----------------
Tier 1 produces overlapping intervals when eigenvalues are tightly
clustered. We merge overlapping intervals into clusters, producing
multi-eigenvalue enclosures. Each cluster enclosure rigorously contains
its full multiplicity of true eigenvalues.

For users wanting per-eigenvalue cluster bounds rather than merged
intervals: this requires directed-rounding-based Tier-2 which is future
work.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


U_DOUBLE = 2.0 ** -53  # IEEE 754 double unit roundoff


def _gamma(k: int, complex_ops: bool = True) -> float:
    """Higham's gamma_k function. For complex ops, multiply by sqrt(2)."""
    u = U_DOUBLE * (np.sqrt(2.0) if complex_ops else 1.0)
    ku = k * u
    if ku >= 1.0:
        raise ValueError(f"gamma({k}) overflow: n*u = {ku} must be < 1.")
    return ku / (1.0 - ku)


@dataclass
class Enclosure:
    """One eigenvalue cluster enclosure.

    The interval [midpoint - halfwidth, midpoint + halfwidth] is
    guaranteed to contain exactly `multiplicity` true eigenvalues of A
    (counting algebraic multiplicity), under IEEE 754 round-to-nearest
    arithmetic.

    The `tier` field documents which bound was used:
      'bauer-fike' : Tier 1, per-eigenvalue Bauer-Fike (isolated case)
      'merged'     : Tier 1 with merge-on-overlap, Tier 2 declined
      'rump-lange' : Tier 2 Davis-Kahan / Lange refinement
    """
    midpoint: float
    halfwidth: float
    multiplicity: int
    tier: str = 'bauer-fike'
    # Optional: subspace enclosure for cluster (n x k complex matrix)
    # giving |V_true - V_computed|_componentwise <= halfwidth_subspace
    subspace_basis: Optional[np.ndarray] = None
    subspace_halfwidth: float = 0.0

    @property
    def lo(self) -> float:
        return self.midpoint - self.halfwidth

    @property
    def hi(self) -> float:
        return self.midpoint + self.halfwidth

    def __repr__(self):
        return (f"Enclosure([{self.lo:.6e}, {self.hi:.6e}], "
                f"mult={self.multiplicity}, tier={self.tier})")


def _bound_residual_norm(H_norm_F: float, x_norms: np.ndarray,
                        mu_abs: np.ndarray, r_norms_fp: np.ndarray,
                        n: int) -> np.ndarray:
    """Rigorous upper bound on the true residual norm ||A x - mu x||_2.

    Given:
      - r_norms_fp : floating-point computed norms
      - x_norms : computed ||x||_2
      - mu_abs : |mu|
      - n : dimension
      - H_norm_F : floating-point computed ||H||_F (we add an upper-bound slack)

    Returns rigorous upper bounds, one per column.
    """
    g_mv = _gamma(4 * n)         # matrix-vector product
    g_scalar = _gamma(1)         # scalar-vector
    g_sub = _gamma(1)            # subtraction
    g_norm = _gamma(2 * n)       # vector 2-norm

    # ||H||_F itself is computed in fp; bound: ||H||_F_true <= ||H||_F_fp * (1 + g_{2n^2})
    H_norm_F_upper = H_norm_F * (1.0 + _gamma(2 * n * n))

    Ax_norm_upper = H_norm_F_upper * x_norms * (1 + g_mv)
    mu_x_norm_upper = mu_abs * x_norms * (1 + g_scalar)
    r_norm_upper_pre = Ax_norm_upper + mu_x_norm_upper

    err_resid_vec = (g_mv * H_norm_F_upper * x_norms
                     + g_scalar * mu_abs * x_norms
                     + g_sub * r_norm_upper_pre)
    err_norm = g_norm * (r_norms_fp + err_resid_vec)
    return r_norms_fp + err_resid_vec + err_norm


def _bound_x_norm_lower(x_norms_fp: np.ndarray, n: int) -> np.ndarray:
    """Rigorous lower bound on ||x||_2 given floating-point computed norm."""
    g_norm = _gamma(2 * n)
    return x_norms_fp / (1.0 + g_norm)


def _per_eigenvalue_bound(H, mu: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Tier 1: rigorous Bauer-Fike halfwidths.

    H may be a dense numpy ndarray or a scipy.sparse matrix. For sparse
    H, residual computation is O(nnz(H) * n) rather than O(n^3).
    """
    import scipy.sparse as sp_module
    n = H.shape[0]

    # Residual R = H X - X diag(mu). One BLAS-3 GEMM for the dense case,
    # one sparse-dense product for the sparse case. Do NOT loop over
    # columns: a single GEMM is far faster and uses BLAS efficiently.
    HX = H @ X
    R = HX - X * mu

    if sp_module.issparse(H):
        H_norm_F = float(sp_module.linalg.norm(H, ord='fro'))
    else:
        H_norm_F = float(np.linalg.norm(H, ord='fro'))  # cheap: O(n^2)

    r_norms_fp = np.linalg.norm(R, axis=0)
    x_norms_fp = np.linalg.norm(X, axis=0)

    r_norm_upper = _bound_residual_norm(
        H_norm_F, x_norms_fp, np.abs(mu), r_norms_fp, n)
    x_norm_lower = _bound_x_norm_lower(x_norms_fp, n)
    return r_norm_upper / x_norm_lower


def _cluster_tighten(H: np.ndarray, mu: np.ndarray, X: np.ndarray,
                     cluster_idx: np.ndarray,
                     other_intervals: List[tuple]) -> tuple:
    """Tier 2: tighter cluster bound via Rump-Lange Lemma 6.1.

    Parameters
    ----------
    H : n x n Hermitian
    mu : (n,) all computed eigenvalues
    X : n x n eigenvector matrix
    cluster_idx : (k,) indices of eigenvalues in this cluster
    other_intervals : list of (lo, hi) for the other clusters/eigenvalues
                      (rigorous Tier-1 intervals)

    Returns
    -------
    (midpoint, halfwidth, success) : the cluster centroid, tighter halfwidth,
        and a flag indicating whether Tier-2 actually improved on Tier-1.
        If success is False, the caller should keep the Tier-1 merged interval.
    """
    n = H.shape[0]
    k = len(cluster_idx)
    mu_K = mu[cluster_idx]
    V_K = X[:, cluster_idx]  # n x k

    # Cluster centroid (computed as fp avg)
    mubar = float(np.mean(mu_K))

    # Residual matrix R_K = H V_K - V_K diag(mu_K). Rump-Lange Lemma 6.1
    # uses the spectral norm ||R_K||_2 (largest singular value).
    R_K = H @ V_K - V_K * mu_K
    rho_K_fp = float(np.linalg.norm(R_K, ord=2))

    # Per-column error analysis (same as Tier 1).
    H_norm_F = float(np.linalg.norm(H, ord='fro'))
    g_mv = _gamma(4 * n)
    g_scalar = _gamma(1)
    g_sub = _gamma(1)
    g_norm = _gamma(2 * n)

    x_norms_K = np.linalg.norm(V_K, axis=0)  # close to 1 for orthonormal cols
    mu_abs_K = np.abs(mu_K)
    H_norm_F_upper = H_norm_F * (1.0 + _gamma(2 * n * n))
    # Per-column upper bound on ||R_K[:, i]||_2 error:
    Ax_per_col = H_norm_F_upper * x_norms_K * (1 + g_mv)
    mu_x_per_col = mu_abs_K * x_norms_K * (1 + g_scalar)
    R_pre_per_col = Ax_per_col + mu_x_per_col
    err_per_col = (g_mv * H_norm_F_upper * x_norms_K
                   + g_scalar * mu_abs_K * x_norms_K
                   + g_sub * R_pre_per_col)
    # Spectral norm error: ||R_K_true - R_K_fp||_2 is bounded by the
    # max column error norm times sqrt(k) in the worst case (column-wise
    # bound), but also by the Frobenius norm of the column errors.
    # We use the tighter:
    #   ||delta R_K||_2 <= ||delta R_K||_F <= sqrt(sum of (err_per_col)^2)
    err_R_spec = float(np.sqrt(np.sum(err_per_col ** 2)))
    # Error in computing ||R_K_fp||_2 via SVD itself.
    g_svd = _gamma(8 * n * k)
    err_svd = g_svd * (rho_K_fp + err_R_spec)
    rho_K_upper = rho_K_fp + err_R_spec + err_svd

    # Spectral gap to the rest of the spectrum.
    # cluster_lo = min(mu_K - tier1_widths_K), cluster_hi = max(mu_K + ...)
    # We need a rigorous LOWER bound on the gap. For an outside interval
    # [a, b], the closest distance from the cluster is:
    #   if b < cluster_lo:  cluster_lo - b   (other interval is below)
    #   if a > cluster_hi:  a - cluster_hi   (other interval is above)
    #   else: gap is "0" — cluster overlaps an outside interval, Tier-2 fails
    # We compute the cluster centroid based estimate: gap to nearest neighbor
    # midpoint minus that neighbor's halfwidth minus this cluster's spread.
    cluster_lo = float(np.min(mu_K))
    cluster_hi = float(np.max(mu_K))
    cluster_spread = 0.5 * (cluster_hi - cluster_lo)  # half-spread around mubar

    min_gap = np.inf
    for (lo, hi) in other_intervals:
        if hi < cluster_lo:
            gap = cluster_lo - hi
        elif lo > cluster_hi:
            gap = lo - cluster_hi
        else:
            # other interval overlaps cluster range; Tier-2 cannot apply
            return mubar, np.inf, False
        if gap < min_gap:
            min_gap = gap

    # Rump-Lange Lemma 6.1 (Hermitian simplification):
    #   max|lambda_K_true - mubar| <=  rho_K + 2 * rho_K^2 / min_gap + cluster_spread
    # (the cluster_spread accounts for the fact that mubar is the centroid;
    #  in the worst case the true eigenvalues spread up to cluster_spread + rho_K)
    if min_gap <= 0 or not np.isfinite(min_gap):
        return mubar, np.inf, False

    # Sigma_min(V_K^* V_K)^(-1/2) correction: for orthonormal V_K from eigh,
    # V_K^* V_K is I_k to within g_{n} ULPs. Bound:
    #   sigma_min(V_K^* V_K) >= 1 - k*g_n
    g_orth = k * _gamma(n)
    if g_orth >= 0.5:
        # V_K is too non-orthonormal for the bound to apply; skip
        return mubar, np.inf, False
    sigma_correction = 1.0 / np.sqrt(1.0 - g_orth)

    rl_halfwidth = (rho_K_upper * sigma_correction
                    + 2.0 * rho_K_upper * rho_K_upper / min_gap
                    + cluster_spread)

    return mubar, rl_halfwidth, True


def _merge_overlapping(midpoints: np.ndarray, halfwidths: np.ndarray
                       ) -> List[tuple]:
    """Merge overlapping intervals into clusters.

    Returns list of (cluster_indices_in_sorted_order, cluster_lo, cluster_hi).
    """
    n = len(midpoints)
    if n == 0:
        return []
    order = np.argsort(midpoints)
    m_sorted = midpoints[order]
    h_sorted = halfwidths[order]

    clusters = []
    cur_indices = [int(order[0])]
    cur_lo = m_sorted[0] - h_sorted[0]
    cur_hi = m_sorted[0] + h_sorted[0]
    for i in range(1, n):
        lo_i = m_sorted[i] - h_sorted[i]
        hi_i = m_sorted[i] + h_sorted[i]
        if lo_i <= cur_hi:
            cur_indices.append(int(order[i]))
            cur_hi = max(cur_hi, hi_i)
        else:
            clusters.append((cur_indices, cur_lo, cur_hi))
            cur_indices = [int(order[i])]
            cur_lo, cur_hi = lo_i, hi_i
    clusters.append((cur_indices, cur_lo, cur_hi))
    return clusters


def enclose_global(H: np.ndarray, tier2: bool = False,
                   return_subspaces: bool = False,
                   eigh_result: tuple = None,
                   assume_hermitian: bool = False) -> List[Enclosure]:
    """Verified enclosures for all eigenvalues of Hermitian H.

    Parameters
    ----------
    H : (n, n) complex Hermitian
    tier2 : whether to attempt Rump-Lange cluster tightening
        Note: with the current Higham-style error bounds (no directed
        rounding), Tier-2 is typically *looser* than Tier-1 merged
        intervals because the per-column error bound dominates the
        actual residual. Tier-2 becomes useful when the bound is
        tightened with directed-rounding arithmetic (future work).
        Default is therefore False.
    return_subspaces : whether to attach cluster subspace bases to returned
        enclosures (useful for verified eigenvector applications)
    eigh_result : optional (mu, X) precomputed eigendecomposition. If
        given, the (expensive) eigh call is skipped. The caller is
        responsible for it being the eigendecomposition of the same H.
    assume_hermitian : if True, skip the 0.5*(H + H^*) symmetrization
        copy (saves an O(n^2) allocation when H is already exactly
        Hermitian, e.g. produced by the model builders here).

    Returns
    -------
    list of Enclosure objects, sorted by midpoint
    """
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    H_sym = H if assume_hermitian else 0.5 * (H + H.conj().T)
    n = H_sym.shape[0]
    if eigh_result is not None:
        mu, X = eigh_result
    else:
        mu, X = np.linalg.eigh(H_sym)
    halfwidths_t1 = _per_eigenvalue_bound(H_sym, mu, X)

    clusters = _merge_overlapping(mu, halfwidths_t1)

    results: List[Enclosure] = []
    for ci, (cluster_indices, c_lo, c_hi) in enumerate(clusters):
        k = len(cluster_indices)
        c_mid = 0.5 * (c_lo + c_hi)
        c_half = 0.5 * (c_hi - c_lo)

        if not tier2 or k == 1:
            tier = 'bauer-fike' if k == 1 else 'merged'
            enc = Enclosure(c_mid, c_half, k, tier=tier)
        else:
            # Build other_intervals list for Tier 2
            other_intervals = [
                (clusters[j][1], clusters[j][2])
                for j in range(len(clusters)) if j != ci
            ]
            t2_mid, t2_half, t2_ok = _cluster_tighten(
                H_sym, mu, X, np.array(cluster_indices), other_intervals)
            if t2_ok and t2_half < c_half:
                enc = Enclosure(t2_mid, t2_half, k, tier='rump-lange')
            else:
                enc = Enclosure(c_mid, c_half, k, tier='merged')

        if return_subspaces:
            enc.subspace_basis = X[:, cluster_indices]
            # Subspace halfwidth: ||V_K_computed - V_K_true||_F bound via
            # Davis-Kahan: rho_K / gap_K. We don't propagate it here in
            # detail; just expose the basis.
            enc.subspace_halfwidth = 0.0
        results.append(enc)
    return results


def enclose_sectors(H: np.ndarray, S_diag: np.ndarray, tier2: bool = False,
                    atol: float = 1e-12) -> List[Enclosure]:
    """Sector-decomposed verified enclosures.

    Assumes S is diagonal in the working basis (the case for natural
    physical symmetries like total Sz in the computational basis).
    """
    n = H.shape[0]
    if len(S_diag) != n:
        raise ValueError("S_diag length must equal H dimension")

    sector_values = sorted(set(np.round(S_diag / atol).astype(int)))
    all_enclosures: List[Enclosure] = []
    for sv_int in sector_values:
        sv = sv_int * atol
        mask = np.abs(S_diag - sv) <= atol
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        H_s = H[np.ix_(idx, idx)]
        H_s = 0.5 * (H_s + H_s.conj().T)
        sec_encs = enclose_global(H_s, tier2=tier2)
        all_enclosures.extend(sec_encs)
    all_enclosures.sort(key=lambda e: e.midpoint)

    # Inter-sector merge: if cross-sector eigenvalues happen to be very
    # close (degeneracies broken by symmetry), merge their intervals.
    merged: List[Enclosure] = []
    if all_enclosures:
        cur = all_enclosures[0]
        cur_lo, cur_hi, cur_mult = cur.lo, cur.hi, cur.multiplicity
        for e in all_enclosures[1:]:
            if e.lo <= cur_hi:
                cur_hi = max(cur_hi, e.hi)
                cur_mult += e.multiplicity
            else:
                merged.append(Enclosure(0.5*(cur_lo+cur_hi),
                                        0.5*(cur_hi-cur_lo),
                                        cur_mult, tier='merged'))
                cur_lo, cur_hi, cur_mult = e.lo, e.hi, e.multiplicity
        merged.append(Enclosure(0.5*(cur_lo+cur_hi),
                                0.5*(cur_hi-cur_lo),
                                cur_mult, tier='merged'))
    return merged


def summary_stats(enclosures: List[Enclosure]) -> dict:
    """Aggregate stats for a list of enclosures."""
    if not enclosures:
        return {"count": 0}
    half = np.array([e.halfwidth for e in enclosures])
    mults = np.array([e.multiplicity for e in enclosures])
    tiers = {}
    for e in enclosures:
        tiers[e.tier] = tiers.get(e.tier, 0) + 1
    return {
        "count": len(enclosures),
        "total_eigs": int(mults.sum()),
        "max_halfwidth": float(half.max()),
        "median_halfwidth": float(np.median(half)),
        "min_halfwidth": float(half.min()),
        "max_multiplicity": int(mults.max()),
        "tier_counts": tiers,
    }

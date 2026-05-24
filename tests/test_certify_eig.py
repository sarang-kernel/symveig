"""
tests/test_certify_eig.py
==========================

Unit tests for certify_eig: synthetic isolated, synthetic clustered,
synthetic sectored, ill-conditioned, near-degenerate cluster.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from symveig.certify_eig import (enclose_global, enclose_sectors, Enclosure,
                         summary_stats)


def _check_contains(enclosures, true_eigs):
    encs_sorted = sorted(enclosures, key=lambda e: e.midpoint)
    remaining_mult = [e.multiplicity for e in encs_sorted]
    for t in np.sort(true_eigs):
        found = False
        for ei, e in enumerate(encs_sorted):
            if remaining_mult[ei] > 0 and e.lo <= t <= e.hi:
                remaining_mult[ei] -= 1
                found = True
                break
        if not found:
            return False, t
    return True, None


# ---------- Test 1: synthetic isolated ----------
def test_synthetic_isolated():
    n = 128
    rng = np.random.default_rng(42)
    true_eigs = (np.arange(n) + 0.5).astype(np.float64)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    H = Q @ np.diag(true_eigs.astype(np.complex128)) @ Q.conj().T
    H = 0.5 * (H + H.conj().T)
    encs = enclose_global(H, tier2=True)
    s = summary_stats(encs)
    ok, bad = _check_contains(encs, true_eigs)
    assert ok, f"Test 1: eigenvalue {bad} not contained"
    assert s["total_eigs"] == n, f"Test 1: total mult {s['total_eigs']} != {n}"
    print(f"  test_synthetic_isolated PASS  (count={s['count']}, max_half={s['max_halfwidth']:.3e})")


# ---------- Test 2: synthetic clustered ----------
def test_synthetic_clustered():
    n = 128
    rng = np.random.default_rng(43)
    true_eigs = (np.arange(n) + 0.5).astype(np.float64)
    true_eigs[60:70] = 50.0  # 10-fold degeneracy
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    H = Q @ np.diag(true_eigs.astype(np.complex128)) @ Q.conj().T
    H = 0.5 * (H + H.conj().T)
    encs = enclose_global(H, tier2=True)
    s = summary_stats(encs)
    ok, bad = _check_contains(encs, true_eigs)
    assert ok, f"Test 2: eigenvalue {bad} not contained"
    max_mult = max(e.multiplicity for e in encs)
    assert max_mult >= 10, f"Test 2: expected multiplicity >= 10, got {max_mult}"
    print(f"  test_synthetic_clustered PASS  (max_mult={max_mult}, tiers={s['tier_counts']})")


# ---------- Test 3: synthetic with sector structure ----------
def test_synthetic_sectored():
    sector_sizes = [30, 50, 48]
    n = sum(sector_sizes)
    rng = np.random.default_rng(44)
    sector_eigs = [
        rng.uniform(-2.0, -1.0, sector_sizes[0]),
        rng.uniform(-1.5, +1.5, sector_sizes[1]),
        rng.uniform(+1.0, +2.0, sector_sizes[2]),
    ]
    true_eigs = np.sort(np.concatenate(sector_eigs))
    S_diag = np.concatenate([
        np.full(sector_sizes[0], 0.0),
        np.full(sector_sizes[1], 1.0),
        np.full(sector_sizes[2], 2.0),
    ])
    H = np.zeros((n, n), dtype=np.complex128)
    offset = 0
    for k, sz in enumerate(sector_sizes):
        Mk = rng.standard_normal((sz, sz)) + 1j * rng.standard_normal((sz, sz))
        Qk, _ = np.linalg.qr(Mk)
        Dk = np.diag(sector_eigs[k].astype(np.complex128))
        Hk = Qk @ Dk @ Qk.conj().T
        Hk = 0.5 * (Hk + Hk.conj().T)
        H[offset:offset+sz, offset:offset+sz] = Hk
        offset += sz
    H = 0.5 * (H + H.conj().T)

    enc_g = enclose_global(H, tier2=True)
    enc_s = enclose_sectors(H, S_diag, tier2=True)
    s_g, s_s = summary_stats(enc_g), summary_stats(enc_s)
    ok_g, _ = _check_contains(enc_g, true_eigs)
    ok_s, _ = _check_contains(enc_s, true_eigs)
    assert ok_g and ok_s, "Test 3: containment failed"
    assert s_g['max_halfwidth'] > s_s['max_halfwidth'], "sector should be tighter"
    print(f"  test_synthetic_sectored PASS  (ratio={s_g['max_halfwidth']/s_s['max_halfwidth']:.2f})")


# ---------- Test 4: ill-conditioned (still Hermitian) ----------
def test_ill_conditioned():
    """Hermitian with widely separated eigenvalues (1e-8 to 1e+8 condition).
    Should still produce valid enclosures, possibly looser at extreme eigs.
    """
    n = 64
    rng = np.random.default_rng(45)
    log_eigs = np.linspace(-8, 8, n)
    true_eigs = 10 ** log_eigs
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    H = Q @ np.diag(true_eigs.astype(np.complex128)) @ Q.conj().T
    H = 0.5 * (H + H.conj().T)
    encs = enclose_global(H, tier2=True)
    ok, bad = _check_contains(encs, true_eigs)
    assert ok, f"Test 4: eigenvalue {bad} not contained"
    # All true eigvals contained, that's the success criterion
    print(f"  test_ill_conditioned PASS  (cond ~ 1e16, n={n})")


# ---------- Test 5: near-degenerate cluster (gap = 1e-10) ----------
def test_near_degenerate_cluster():
    """Two eigenvalues separated by 1e-10. Test that:
    (a) both are contained
    (b) they may merge if floating-point width > gap (acceptable)
    """
    n = 64
    rng = np.random.default_rng(46)
    true_eigs = np.arange(n).astype(np.float64)
    true_eigs[10] = 5.0
    true_eigs[11] = 5.0 + 1e-10  # tiny gap
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)
    H = Q @ np.diag(true_eigs.astype(np.complex128)) @ Q.conj().T
    H = 0.5 * (H + H.conj().T)
    encs = enclose_global(H, tier2=True)
    ok, bad = _check_contains(encs, true_eigs)
    assert ok, f"Test 5: eigenvalue {bad} not contained"
    # The two near-degenerate eigenvalues likely merge in the output
    # (gap 1e-10 is smaller than the typical halfwidth ~1e-13 per eigenvalue
    #  in floating point, so merging is expected)
    multiplicities = [e.multiplicity for e in encs]
    print(f"  test_near_degenerate_cluster PASS  (multiplicities at cluster: {sorted(multiplicities, reverse=True)[:3]})")


def main():
    test_synthetic_isolated()
    test_synthetic_clustered()
    test_synthetic_sectored()
    test_ill_conditioned()
    test_near_degenerate_cluster()
    print("All certify_eig tests passed.")


if __name__ == "__main__":
    main()

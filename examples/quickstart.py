#!/usr/bin/env python3
"""
quickstart.py — minimal symveig example.

Run from the repository root:
    python examples/quickstart.py
"""
import os
import sys

# Allow running without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from symveig import enclose_global, enclose_sectors, summary_stats
from symveig.models import heisenberg_1d


def main():
    print("=" * 60)
    print("symveig quickstart")
    print("=" * 60)

    # --- 1. A tiny hand-built Hermitian matrix ---
    print("\n[1] A 2x2 Hermitian matrix with eigenvalues 1 and 3:")
    A = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=complex)
    for e in enclose_global(A):
        print(f"    {e}")

    # --- 2. A 1D Heisenberg chain, global vs sector ---
    L = 8
    print(f"\n[2] 1D Heisenberg chain, L={L}, periodic:")
    H, sectors = heisenberg_1d(L, periodic=True)

    enc_global = enclose_global(H)
    enc_sector = enclose_sectors(H, sectors["Sz"])

    sg = summary_stats(enc_global)
    ss = summary_stats(enc_sector)
    print(f"    global : {sg['count']} clusters, "
          f"max half-width {sg['max_halfwidth']:.3e}")
    print(f"    sector : {ss['count']} clusters, "
          f"max half-width {ss['max_halfwidth']:.3e}")
    print(f"    sector enclosures are "
          f"{sg['max_halfwidth'] / ss['max_halfwidth']:.1f}x tighter")

    # --- 3. Verify containment against a reference ---
    ref = np.linalg.eigvalsh(H)
    inside = all(
        any(e.lo <= lam <= e.hi for e in enc_sector)
        for lam in ref
    )
    print(f"\n[3] All {len(ref)} reference eigenvalues inside the "
          f"sector enclosures: {inside}")


if __name__ == "__main__":
    main()

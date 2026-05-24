# CPC Program Summary

This file contains the _Program Summary_ required by Computer Physics
Communications. It is reproduced (with the DOI filled in) at the front of
the accompanying manuscript. Keeping it in the repository keeps the code
and the paper synchronized.

---

**Program Title:** symveig

**CPC Library link to program files:** (to be assigned on acceptance)

**Developer's repository link:** https://github.com/sarang-kernel/symveig/

**Code Ocean capsule:** (optional; not used)

**Licensing provisions:** MIT

**Programming language:** Python 3 (>= 3.9)

**Supplementary material:** Frozen example outputs (JSON results, summary
CSV, and figures) reproducing the paper's tables and figures are included
in the `results/` directory of the archived release.

**Nature of problem:**
Exact diagonalization of quantum lattice Hamiltonians and other Hermitian
operators returns floating-point eigenvalues whose accuracy is not
certified: rounding error, eigensolver convergence behavior, and
ill-conditioning can all corrupt the result without warning. For
applications that require guaranteed bounds — for example confirming a
spectral gap, certifying a ground-state energy, or validating an
approximate solver — one needs a rigorous interval that is _proven_ to
contain the true eigenvalue. Existing verified eigensolvers of this kind
(notably INTLAB-based tools) require MATLAB and the commercial INTLAB
toolbox, which are not part of the typical computational-physics software
stack and cannot natively exploit the symmetry-sector structure that
quantum lattice models possess.

**Solution method:**
For a Hermitian matrix A, symveig computes an approximate
eigendecomposition with LAPACK (via NumPy) and then bounds the distance
from each computed eigenvalue to the nearest true eigenvalue using the
Hermitian Bauer–Fike–Wilkinson residual estimate
|lambda_true − mu| <= ||A x − mu x||\_2 / ||x||\_2. Every floating-point
operation entering the residual and norm computations is bounded
rigorously with Higham's gamma_k error-accumulation analysis, so the
returned interval is a guaranteed enclosure under IEEE 754
round-to-nearest arithmetic. Overlapping per-eigenvalue intervals are
merged into rigorous multiplicity-counted cluster enclosures. When the
operator commutes with an abelian conserved quantity that is diagonal in
the working basis (for example total S_z for a spin model), symveig
projects A into each symmetry sector and verifies the sectors
independently; because each sector block has dimension at most about
binomial(L, L/2), this yields enclosures that are both tighter (by a
factor ~ n / max_s n_s, observed to be 3–9x across system sizes
L = 4–12) and substantially faster (25–50x at L = 10–12) than verifying
the full matrix.

**Additional comments including restrictions and unusual features:**

- The package is pure NumPy/SciPy; no INTLAB, MATLAB, or compiled
  extensions are required.
- The dominant cost is the dense eigendecomposition, O(n^3); the sector
  path avoids ever diagonalizing the full matrix and is the recommended
  mode of use. Dense methods are practical to dimension ~ 4096
  (1D spin-1/2 chains up to L = 12) on a workstation.
- An optional Tier-2 cluster refinement (Rump–Lange) is provided but is,
  in the present release, dominated by the worst-case Higham error
  bounds and therefore disabled by default; tightening it with
  directed-rounding arithmetic is identified as future work.
- The verification assumes the symmetry observable S is exactly diagonal
  in the working (computational) basis, which holds for the standard
  abelian conserved quantities of lattice models. Extension to
  numerically diagonalized commuting observables requires the additional
  cross-matrix error analysis described in the paper's discussion.

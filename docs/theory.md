# Theory: rigorous eigenvalue enclosures

This document gives the mathematical basis for the bounds computed by
`symveig`. It is the reference for the error analysis implemented in
`symveig/certify_eig.py` and is the basis of the methods section of the
accompanying paper.

## 1. The enclosure problem

Let `A` be a Hermitian matrix of dimension `n`. A numerical eigensolver
returns approximate eigenpairs `(mu_i, x_i)`. We want, for each `i`, a
rigorous interval `[mu_i - h_i, mu_i + h_i]` that is *guaranteed* to
contain a true eigenvalue of `A`, where "guaranteed" means: provably
correct under IEEE 754 double-precision round-to-nearest arithmetic,
accounting for every rounding error in the computation of the bound
itself.

## 2. The Hermitian residual bound (Tier 1)

The foundation is the standard residual estimate for Hermitian (more
generally normal) matrices. If `x` has unit 2-norm and `mu = x* A x`,
and the residual is `r = A x - mu x`, then there exists a true
eigenvalue `lambda` of `A` with

    |lambda - mu| <= ||r||_2 .

This is the Bauer–Fike theorem specialized to the Hermitian case, where
the eigenvector condition number is 1; see Parlett, *The Symmetric
Eigenvalue Problem* (SIAM, 1998), Theorem 11.7.1, or Golub & Van Loan,
*Matrix Computations*. The bound is sharp to first order in the residual.

For a computed eigenvector that is only approximately unit-norm, the
bound becomes

    |lambda - mu| <= ||r||_2 / ||x||_2 .

## 3. Making the bound rigorous in floating point

The quantities `||r||_2` and `||x||_2` are themselves computed in
floating point and are therefore subject to rounding error. To obtain a
*provable* upper bound on `|lambda - mu|`, we replace them with a
rigorous upper bound on `||r||_2` and a rigorous lower bound on
`||x||_2`:

    |lambda - mu| <= ( ||r||_2^fp + delta_r ) / ( ||x||_2^fp - delta_x ) .

The error terms `delta_r` and `delta_x` are bounded using Higham's
`gamma_k` notation. Define the unit roundoff `u = 2^-53` for IEEE 754
double precision. For an inner product or other length-`k` accumulation,

    gamma_k = (k u) / (1 - k u),    valid for k u < 1.

For complex arithmetic we use the standard substitution that replaces
`u` with `sqrt(2) u`, absorbing the factor-of-two cost of complex
multiply–add into the constant (Higham, *Accuracy and Stability of
Numerical Algorithms*, 2nd ed., Lemma 3.5 and Section 3.6). The relevant
contributions, all derived in the docstring of
`certify_eig._bound_residual_norm`, are:

- **Matrix–vector product `A x`.** Each component is a length-`n` inner
  product, so the componentwise error is bounded by
  `gamma_{4n} ||A||_F ||x||_2`, giving
  `|| fl(A x) - A x ||_2 <= gamma_{4n} ||A||_F ||x||_2`.
- **Scalar–vector product `mu x`.** One multiply per component:
  error `<= gamma_1 |mu| ||x||_2`.
- **Subtraction `A x - mu x`.** One add per component:
  error `<= gamma_1 (||A x|| + ||mu x||)`.
- **The 2-norm itself.** A length-`n` sum of squares and a square root:
  relative error `<= gamma_{2n}`.
- **`||A||_F`.** Computed in floating point with relative error
  `<= gamma_{2 n^2}`; we inflate the reported value accordingly so that
  `||A||_F` enters the bound as a rigorous upper bound.

Combining these gives a closed-form, rigorously valid `delta_r`, and the
lower bound `||x||_2 >= ||x||_2^fp / (1 + gamma_{2n})` gives `delta_x`.

The resulting `h_i = ( ||r_i||_2^fp + delta_{r,i} ) / ( ||x_i||_2^fp /
(1 + gamma_{2n}) )` is a guaranteed half-width.

In practice the bound is dominated not by the true residual (which is
typically `~ n u ||A||` from the eigensolver) but by the worst-case
constant `gamma_{4n} ||A||_F` in `delta_r`. This is the price of a
fully rigorous bound without directed-rounding arithmetic; the resulting
half-widths scale as `h ~ C u d ||A||_F` with a model-independent
constant `C` of order a few (empirically `C ≈ 6`; see
`docs/reproducibility.md`).

## 4. Clusters and multiplicities

When eigenvalues are closely spaced (Hermitian lattice models routinely
have exact degeneracies from symmetry), several Tier-1 intervals
overlap. We merge any chain of overlapping intervals into a single
**cluster enclosure** `[c_lo, c_hi]` and report it with a
`multiplicity` equal to the number of merged eigenvalues. The guarantee
upgrades accordingly: the cluster interval contains exactly that many
true eigenvalues, counting algebraic multiplicity. The merge is a
rigorous interval-union operation, so the containment guarantee is
preserved.

## 5. Tier 2 (optional): Rump–Lange cluster refinement

For a cluster `K` with computed invariant subspace `V_K` (the `k`
eigenvectors) and centroid `mubar`, the Davis–Kahan / Lange analysis
(Rump & Lange, *J. Comput. Appl. Math.* 434 (2023) 115332, Lemma 6.1)
bounds the deviation of the true cluster eigenvalues from `mubar` using
the spectral norm of the block residual `R_K = A V_K - V_K diag(mu_K)`
and the spectral gap `delta_K` from `K` to the rest of the spectrum:

    max_{lambda in K} |lambda - mubar|
        <= ||R_K||_2 (1 + O(||R_K||_2 / delta_K)) .

`symveig` implements this (`certify_eig._cluster_tighten`) but, in the
present release, the rigorous floating-point bound on `||R_K||_2`
inherits the same worst-case `gamma_{4n} ||A||_F` term as Tier 1, plus a
factor that grows with cluster size, so the refined bound is typically
*looser* than the merged Tier-1 interval. Tier 2 is therefore disabled
by default. It becomes advantageous once `||R_K||_2` is bounded with
directed-rounding arithmetic (Rump's "verify rounding mode" technique),
which would let the bound track the true residual `~ n u ||A||` rather
than the worst case. This is the natural next step for a follow-up.

## 6. Symmetry-sector decomposition

Suppose `A` commutes with a Hermitian observable `S`, `[A, S] = 0`, and
`S` is diagonal in the working basis with distinct eigenvalues (sectors)
`s`. Then `A` is block diagonal in that basis: it preserves each
eigenspace of `S`. Writing `P_s` for the projector onto sector `s`, the
sector block is `A_s = P_s A P_s`, obtained simply by selecting the rows
and columns whose `S`-value equals `s` (no numerical diagonalization of
`S` is required, since `S` is already diagonal).

Each `A_s` is Hermitian of dimension `d_s`, and the spectrum of `A` is
the disjoint union of the spectra of the `A_s`. Verifying each `A_s`
independently and taking the union of the enclosures therefore
reconstructs a verified enclosure of the full spectrum.

The benefit is twofold and follows directly from the Tier-1 scaling
`h ~ C u d ||A||_F`:

1. **Tighter bounds.** A sector block has `d_s <= max_s d_s`, which for a
   1D spin-1/2 chain is `binomial(L, L/2) ≈ n / sqrt(L)`, roughly `n/4`
   for accessible sizes. Since the half-width scales with the dimension,
   the worst sector half-width is smaller than the global half-width by
   approximately `n / max_s d_s`, modulated by the ratio
   `||A||_F / ||A_s||_F`. Empirically this is a factor 3–9x for
   `L = 4–12`.

2. **Faster computation.** The global path performs one `O(n^3)`
   eigendecomposition; the sector path performs several `O(d_s^3)`
   eigendecompositions with `sum_s d_s = n`, and `sum_s d_s^3 << n^3`.
   The measured wall-clock speedup is 25–50x at `L = 10–12`.

Crucially, the sector path never forms or diagonalizes the full matrix,
so it is the recommended mode of use in practice; the global path exists
in the benchmark only to provide the comparison baseline.

## 7. Scope and assumptions

- `A` must be Hermitian (the symmetrization `(A + A*)/2` is applied
  internally; the residual bound is specific to the Hermitian/normal
  case).
- The symmetry observable `S` must be exactly diagonal in the working
  basis. This holds for the standard abelian U(1)-type conserved
  quantities of lattice models (total magnetization, particle number).
  Non-abelian symmetries (full SU(2)) and symmetries that are not
  diagonal in the computational basis (momentum, point-group) are out of
  scope for the present sector machinery; see the paper's discussion for
  the cross-matrix error analysis that the latter would require.
- All guarantees are under IEEE 754 double-precision round-to-nearest
  arithmetic, as provided by standard x86-64 / ARM hardware and NumPy's
  LAPACK backend.

## References

- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
  ed., SIAM, 2002.
- B. N. Parlett, *The Symmetric Eigenvalue Problem*, SIAM, 1998.
- S. M. Rump and M. Lange, "Fast computation of error bounds for all
  eigenpairs of a Hermitian and all singular pairs of a rectangular
  matrix with emphasis on eigen- and singular value clusters,"
  *J. Comput. Appl. Math.* 434 (2023) 115332.
- S. Miyajima, "Fast enclosure for all eigenvalues and invariant
  subspaces in generalized eigenvalue problems," *SIAM J. Matrix Anal.
  Appl.* 35 (2014) 1205–1225.

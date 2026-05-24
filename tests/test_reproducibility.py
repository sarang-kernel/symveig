"""
tests/test_reproducibility.py
==============================

Run the benchmark twice and verify outputs are identical (modulo timing).
This catches accidental nondeterminism (random seeds not fixed,
thread-dependent reductions, etc).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from symveig.models import heisenberg_1d
from symveig.benchmark import benchmark_model


def _strip_timings(d):
    """Remove all timing-derived keys recursively for comparison."""
    drop_substrings = ("time", "speedup")
    if isinstance(d, dict):
        return {k: _strip_timings(v) for k, v in d.items()
                if not any(s in k for s in drop_substrings)}
    if isinstance(d, list):
        return [_strip_timings(x) for x in d]
    return d


def test_reproducibility_heis_L6():
    H, sec = heisenberg_1d(6, periodic=True)
    r1 = benchmark_model("heis_L6", H, sec, sector_to_use="Sz")
    r2 = benchmark_model("heis_L6", H, sec, sector_to_use="Sz")
    s1, s2 = _strip_timings(r1), _strip_timings(r2)
    assert s1 == s2, "Benchmark is not deterministic"
    print("  test_reproducibility_heis_L6 PASS")


def test_reproducibility_heis_L8():
    H, sec = heisenberg_1d(8, periodic=True)
    r1 = benchmark_model("heis_L8", H, sec, sector_to_use="Sz")
    r2 = benchmark_model("heis_L8", H, sec, sector_to_use="Sz")
    s1, s2 = _strip_timings(r1), _strip_timings(r2)
    assert s1 == s2, "Benchmark is not deterministic"
    print("  test_reproducibility_heis_L8 PASS")


def main():
    test_reproducibility_heis_L6()
    test_reproducibility_heis_L8()
    print("All reproducibility tests passed.")


if __name__ == "__main__":
    main()

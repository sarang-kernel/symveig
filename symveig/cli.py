"""
run.py
======

One-command driver. Reproduces every number in the paper.

Usage:
    python run.py                  # runs everything
    python run.py --tests-only     # just the test suite
    python run.py --quick          # smaller L's for fast smoke test
    python run.py --skip-2d        # skip the 2D model (it's the slowest)

Output structure:
    results/
        metadata.json                    # platform, version, timestamp, etc.
        tests.log                        # test suite output
        benchmark_<model>.json           # one per model, list of per-L dicts
        summary.csv                      # all rows in one table
        figures/                         # all PDF/PNG figures
        run.log                          # everything that went to stdout

Exits 0 if all tests pass and all benchmarks complete; non-zero otherwise.
"""

from __future__ import annotations
import argparse
import os
import sys

# --- Thread throttling MUST happen before numpy/scipy import anywhere ---
# Parse just the --threads flag early and set BLAS env vars accordingly.
# This prevents the dense eigendecomposition from saturating every core
# (which is what "cooks the laptop"). Default: leave threads unset (use
# the library default). Pass --threads N to cap.
def _maybe_set_threads():
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            n = sys.argv[i + 1]
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS"):
                os.environ[var] = str(n)
            return n
    return None

_THREADS = _maybe_set_threads()

import json
import platform
import time
import traceback
from datetime import datetime, timezone


def _maybe_throttle_threads(full_mode: bool):
    """Auto-cap BLAS threads on --full to avoid saturating every core
    (which heats laptops and, on some machines, actually runs slower due
    to oversubscription). Skipped if the user already set thread vars
    (including via --threads). Must run BEFORE importing numpy.
    """
    if not full_mode:
        return
    thread_vars = ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                   "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    if any(os.environ.get(v) for v in thread_vars):
        return  # user already set something (e.g. via --threads)
    cpu_count = os.cpu_count() or 4
    cap = max(2, cpu_count // 2)
    for v in thread_vars:
        os.environ[v] = str(cap)
    print(f"  (auto-throttled BLAS to {cap} of {cpu_count} threads on "
          f"--full; override with --threads N)")


# Pre-parse --full so thread caps go in before numpy import.
_pre_args = set(sys.argv[1:])
_maybe_throttle_threads(full_mode=("--full" in _pre_args))

# Repo root = parent of the directory containing this file (the package dir).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PKG_DIR)
# Make `import symveig` work whether or not the package is pip-installed.
sys.path.insert(0, _REPO_ROOT)

import numpy as np


def _setup_results_dir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "figures"), exist_ok=True)
    return path


def _write_metadata(results_dir):
    np_config = ""
    try:
        np_config = str(np.show_config(mode='dicts'))
    except Exception:
        pass
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "machine": platform.machine(),
        "node": platform.node(),
    }
    with open(os.path.join(results_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def _run_tests(log_path):
    """Run all unit tests, write to log file, return True if all passed."""
    import subprocess
    test_dir = os.path.join(_REPO_ROOT, "tests")
    if not os.path.isdir(test_dir):
        print("  (tests/ directory not found; skipping — install from source"
              " repository to run the test suite)")
        with open(log_path, 'w') as logf:
            logf.write("tests/ directory not found; skipped.\n")
        return True
    tests = [
        os.path.join(test_dir, "test_certify_eig.py"),
        os.path.join(test_dir, "test_models.py"),
        os.path.join(test_dir, "test_reproducibility.py"),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    with open(log_path, 'w') as logf:
        for t in tests:
            name = os.path.basename(t)
            logf.write(f"\n===== {name} =====\n")
            print(f"  running {name}")
            r = subprocess.run([sys.executable, t],
                               capture_output=True, text=True,
                               cwd=_REPO_ROOT, env=env)
            logf.write(r.stdout)
            logf.write(r.stderr)
            if r.returncode != 0:
                logf.write(f"FAILED with code {r.returncode}\n")
                print(f"    FAIL: {r.stderr[-200:]}")
                return False
            else:
                print(f"    PASS")
    return True


def _run_benchmarks(results_dir, quick=False, full=False,
                    skip_2d=False, verbose=True):
    """Run all benchmark configurations."""
    from symveig.models import MODELS, TWO_D_MODELS
    from symveig.benchmark import benchmark_model

    if quick:
        L_values_1d = [4, 6, 8]
    elif full:
        L_values_1d = [4, 6, 8, 10, 12]
    else:
        L_values_1d = [4, 6, 8, 10]

    all_summary = []  # list of (model, L, ...) rows

    for model_name, info in sorted(MODELS.items()):
        print(f"\n=== Model: {model_name} ({info['label']}) ===")
        per_L = []
        for L in L_values_1d:
            if 2**L > 4096 and quick:
                continue
            print(f"  L={L} (dim={2**L})", end=" ", flush=True)
            try:
                H, sectors = info['builder'](L)
            except Exception as e:
                print(f"build failed: {e}")
                continue
            sector_name = info.get('sector_name')
            t_start = time.time()
            result = benchmark_model(model_name, H, sectors,
                                     sector_to_use=sector_name,
                                     compute_tier2=False)
            result['L'] = L
            result['label'] = info['label']
            elapsed = time.time() - t_start
            speedup = result.get('speedup_global_over_sector')
            eigh_t = result.get('eigh_global_time', 0.0)
            sec_t = result['sector_t1']['time'] if result.get('sector_t1') else None
            sec_w = (result['sector_t1']['max_halfwidth']
                     if result.get('sector_t1') else None)
            msg = (f"done in {elapsed:.1f}s "
                   f"(global eigh {eigh_t:.1f}s")
            if sec_t is not None:
                msg += f", sector {sec_t:.1f}s"
            msg += f"); global={result['global_t1']['max_halfwidth']:.2e}"
            if sec_w is not None:
                msg += f", sector={sec_w:.2e}"
            if speedup:
                msg += f"; sector {speedup:.1f}x faster"
            print(msg)
            per_L.append(result)
            all_summary.append(_summarize_row(result))
        if per_L:
            with open(os.path.join(results_dir, f"benchmark_{model_name}.json"), 'w') as f:
                json.dump(per_L, f, indent=2, default=_json_default)

    if not skip_2d:
        from symveig.models import TWO_D_MODELS_FULL
        two_d_set = TWO_D_MODELS_FULL if full else TWO_D_MODELS
        for model_name, info in two_d_set.items():
            print(f"\n=== 2D Model: {model_name} ({info['label']}) ===")
            try:
                H, sectors = info['builder']()
            except Exception as e:
                print(f"  build failed: {e}")
                continue
            sector_name = info.get('sector_name')
            t_start = time.time()
            result = benchmark_model(model_name, H, sectors,
                                     sector_to_use=sector_name,
                                     compute_tier2=False)
            result['L'] = info.get('Lx', 0) * info.get('Ly', 0)
            result['Lx'] = info.get('Lx')
            result['Ly'] = info.get('Ly')
            result['label'] = info['label']
            elapsed = time.time() - t_start
            print(f"  done in {elapsed:.1f}s")
            with open(os.path.join(results_dir, f"benchmark_{model_name}.json"), 'w') as f:
                json.dump([result], f, indent=2, default=_json_default)
            all_summary.append(_summarize_row(result))

    # Write summary CSV
    _write_summary_csv(all_summary, os.path.join(results_dir, "summary.csv"))


def _json_default(obj):
    """Handle non-serialisable types in JSON dump."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


def _summarize_row(result):
    """Extract paper-table-relevant row from a benchmark result."""
    g1 = result.get('global_t1', {})
    g2 = result.get('global_t2', {})
    s1 = result.get('sector_t1', {})
    s2 = result.get('sector_t2', {})
    return {
        "model": result.get("name"),
        "label": result.get("label"),
        "L": result.get("L"),
        "dim": result.get("dim"),
        "norm_H_F": result.get("norm_H_F"),
        "ground_state": result.get("ground_state"),
        "global_t1_max_half": g1.get("max_halfwidth"),
        "global_t1_med_half": g1.get("median_halfwidth"),
        "global_t2_max_half": g2.get("max_halfwidth") if g2 else None,
        "sector_t1_max_half": s1.get("max_halfwidth") if s1 else None,
        "sector_t1_med_half": s1.get("median_halfwidth") if s1 else None,
        "sector_t2_max_half": s2.get("max_halfwidth") if s2 else None,
        "ratio_t1_max": result.get("ratio_t1_max"),
        "ratio_t1_med": result.get("ratio_t1_med"),
        "ratio_t2_max": result.get("ratio_t2_max"),
        "eigh_global_time": result.get("eigh_global_time"),
        "sector_time": s1.get("time") if s1 else None,
        "speedup_global_over_sector": result.get("speedup_global_over_sector"),
    }


def _write_summary_csv(rows, path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w') as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                ("" if r.get(k) is None else f"{r[k]}") for k in keys
            ) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Stop at L=8 for faster smoke test.")
    parser.add_argument("--full", action="store_true",
                        help="Include L=12 (much slower, ~3-5 min/model).")
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--threads", default=None,
                        help="Cap BLAS threads (e.g. --threads 4) to keep "
                             "laptops cool. Default: library default.")
    args = parser.parse_args()

    results_dir = _setup_results_dir(args.results_dir)
    print(f"Writing results to: {results_dir}")
    meta = _write_metadata(results_dir)
    print(f"Platform: {meta['platform']}, NumPy {meta['numpy_version']}")

    t_start = time.time()

    # Tests
    if not args.skip_tests:
        print("\n=== Running tests ===")
        tests_ok = _run_tests(os.path.join(results_dir, "tests.log"))
        if not tests_ok:
            print("Tests FAILED. See results/tests.log")
            sys.exit(1)
        print("All tests PASSED.")

    if args.tests_only:
        sys.exit(0)

    # Benchmarks
    print("\n=== Running benchmarks ===")
    try:
        _run_benchmarks(results_dir, quick=args.quick, full=args.full,
                        skip_2d=args.skip_2d)
    except Exception as e:
        print(f"Benchmark FAILED: {e}")
        traceback.print_exc()
        sys.exit(2)

    # Plots
    if not args.skip_plots:
        print("\n=== Generating plots ===")
        from symveig.plot_results import main as plot_main
        plot_main(results_dir)

    elapsed = time.time() - t_start
    print(f"\nAll done in {elapsed:.1f}s. Outputs in {results_dir}/")


if __name__ == "__main__":
    main()

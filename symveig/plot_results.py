"""
plot_results.py
===============

Read JSON benchmark results from `results/` and produce all paper figures.
Saves to `results/figures/` as both PDF (vector) and PNG (raster).

Reads:
  results/benchmark_<model>.json   (one per model)
  results/summary.csv              (aggregate)

Writes:
  results/figures/fig_ratio_vs_L.{pdf,png}
  results/figures/fig_sector_widths_heisenberg_L12.{pdf,png}
  results/figures/fig_width_vs_dim_norm.{pdf,png}
  results/figures/fig_tier1_vs_tier2.{pdf,png}
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Visual settings: paper-friendly defaults
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})


def _load_benchmark_files(results_dir):
    """Load all benchmark_*.json files. Returns dict[model_name -> list[result_dict]]."""
    out = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("benchmark_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                data = json.load(f)
            # data is a list of per-L results
            model_name = fname.replace("benchmark_", "").replace(".json", "")
            out[model_name] = data
    return out


def _save(fig, path_no_ext):
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png")
    plt.close(fig)


def plot_ratio_vs_L(benchmarks, outdir):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (model, runs) in enumerate(sorted(benchmarks.items())):
        runs_with_sector = [r for r in runs if r.get('sector_t1') is not None]
        if not runs_with_sector:
            continue
        Ls = [r['L'] for r in runs_with_sector]
        ratios = [r['ratio_t1_max'] for r in runs_with_sector]
        ax.plot(Ls, ratios,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=runs_with_sector[0].get('label', model))
    ax.set_xlabel('System size $L$')
    ax.set_ylabel('Width ratio: global / sector')
    ax.set_title('Sector decomposition tightness gain vs system size')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    _save(fig, os.path.join(outdir, "fig_ratio_vs_L"))


def plot_sector_widths_largest_L(benchmarks, outdir):
    """For the largest L of Heisenberg 1D PBC, plot per-sector widths."""
    if 'heisenberg_1d_pbc' not in benchmarks:
        return
    runs = benchmarks['heisenberg_1d_pbc']
    # pick the largest L with per_sector data
    candidate = None
    for r in runs:
        if r.get('per_sector') is not None and (candidate is None or r['L'] > candidate['L']):
            candidate = r
    if candidate is None:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    sectors = candidate['per_sector']
    labels = [s['sector_label'] for s in sectors]
    widths = [s['max_halfwidth'] for s in sectors]
    dims = [s['dim'] for s in sectors]
    # Sort by sector label
    order = np.argsort(labels)
    labels = [labels[i] for i in order]
    widths = [widths[i] for i in order]
    dims = [dims[i] for i in order]
    bars = ax.bar(range(len(labels)), widths, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{int(l)}" if l == int(l) else f"{l:.1f}" for l in labels])
    ax.set_xlabel('Sector ($S_z$)')
    ax.set_ylabel('Max halfwidth')
    ax.set_yscale('log')
    ax.set_title(f"1D Heisenberg L={candidate['L']}: per-sector max halfwidth")
    # Annotate with dim
    for i, (w, d) in enumerate(zip(widths, dims)):
        ax.text(i, w * 1.4, f"d={d}", ha='center', fontsize=7, rotation=0)
    ax.grid(True, axis='y', alpha=0.3)
    _save(fig, os.path.join(outdir, "fig_sector_widths_largestL"))


def plot_width_vs_dim_norm(benchmarks, outdir):
    """For each per-sector record across all models, plot halfwidth vs
    dim*norm_H_s_F. If our error analysis is right, the points should
    lie on a line."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    all_x = []
    all_y = []
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (model, runs) in enumerate(sorted(benchmarks.items())):
        x_vals, y_vals = [], []
        for r in runs:
            ps = r.get('per_sector')
            if not ps:
                continue
            for s in ps:
                x = s['dim'] * s['norm_H_s_F']
                y = s['max_halfwidth']
                x_vals.append(x)
                y_vals.append(y)
                all_x.append(x)
                all_y.append(y)
        if x_vals:
            ax.scatter(x_vals, y_vals,
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       alpha=0.6,
                       label=runs[0].get('label', model) if runs else model)
    # Plot the predicted line y = C*u*x where C is fitted from the data
    if all_x:
        all_x_arr = np.array(all_x)
        all_y_arr = np.array(all_y)
        u = 2.0**-53
        # Fit y = C * u * x via least squares in log space
        # log(y) = log(C*u) + log(x)
        # Better: just pick C as median(y/(u*x)) to anchor the line
        C = float(np.median(all_y_arr / (u * all_x_arr)))
        xs = np.logspace(np.log10(min(all_x)), np.log10(max(all_x)), 50)
        ax.plot(xs, C * u * xs, 'k--', alpha=0.5,
                label=fr'$y = {C:.0f}\,u\,d\,\|H_s\|_F$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$d_s \cdot \|H_s\|_F$')
    ax.set_ylabel('Max halfwidth')
    ax.set_title(r'Halfwidth scaling: $w \propto d \cdot \|H\|$')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=7)
    _save(fig, os.path.join(outdir, "fig_width_vs_dim_norm"))


def plot_tier1_vs_tier2(benchmarks, outdir):
    """Compare global Tier-1 vs Tier-2 widths."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    plotted = False
    for i, (model, runs) in enumerate(sorted(benchmarks.items())):
        runs_with_t2 = [r for r in runs if r.get('global_t2') is not None
                                          and r.get('global_t1') is not None]
        if not runs_with_t2:
            continue
        Ls = [r['L'] for r in runs_with_t2]
        ratios = [r['global_t2']['max_halfwidth'] / r['global_t1']['max_halfwidth']
                  for r in runs_with_t2]
        ax.plot(Ls, ratios,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=runs_with_t2[0].get('label', model))
        plotted = True
    if not plotted:
        return
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Tier 1 = Tier 2')
    ax.set_xlabel('System size $L$')
    ax.set_ylabel('Width ratio: Tier 2 / Tier 1')
    ax.set_title('Effect of Rump-Lange Tier-2 cluster refinement')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    _save(fig, os.path.join(outdir, "fig_tier1_vs_tier2"))


def plot_speedup_vs_L(benchmarks, outdir):
    """Plot global/sector wall-clock speedup vs L."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plotted = False
    for i, (model, runs) in enumerate(sorted(benchmarks.items())):
        pts = [(r['L'], r.get('speedup_global_over_sector'))
               for r in runs
               if r.get('speedup_global_over_sector') is not None
               and r.get('dim', 0) >= 256]  # ignore tiny-matrix overhead noise
        pts = [(L, s) for (L, s) in pts if s is not None]
        if not pts:
            continue
        Ls = [p[0] for p in pts]
        sp = [p[1] for p in pts]
        ax.plot(Ls, sp,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=runs[0].get('label', model))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('System size $L$')
    ax.set_ylabel('Wall-clock speedup: global / sector')
    ax.set_title('Sector decomposition is faster than global verification')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    _save(fig, os.path.join(outdir, "fig_speedup_vs_L"))


def main(results_dir="results"):
    benchmarks = _load_benchmark_files(results_dir)
    figdir = os.path.join(results_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    plot_ratio_vs_L(benchmarks, figdir)
    plot_sector_widths_largest_L(benchmarks, figdir)
    plot_width_vs_dim_norm(benchmarks, figdir)
    plot_tier1_vs_tier2(benchmarks, figdir)
    plot_speedup_vs_L(benchmarks, figdir)
    print(f"Figures written to {figdir}/")


if __name__ == "__main__":
    main()

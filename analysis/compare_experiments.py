"""
compare_experiments.py

Runs the full DQN-vs-A* comparison across MULTIPLE goal configurations
and (optionally) multiple trials per goal, then produces both a summary
table and comparison charts.

WORKFLOW:
1. For each goal you want to test, run deploy.py, then deploy_astar.py,
   with the SAME GOAL_DISTANCE_M/GOAL_ANGLE_DEG in both files.
2. Save each terminal log into a folder (default: ./logs/) using this
   naming convention:

       {method}_{distance}m_{angle}deg_trial{n}.log

   method is "dqn" or "astar". Examples:
       logs/dqn_1.2m_0deg_trial1.log
       logs/astar_1.2m_0deg_trial1.log
       logs/dqn_0.8m_25deg_trial1.log
       logs/dqn_0.8m_25deg_trial2.log     <- multiple trials per goal
       logs/astar_0.8m_25deg_trial1.log
       logs/astar_0.8m_25deg_trial2.log

   You don't need the same number of trials for every goal/method, and
   "_trial1" can be omitted if you only ever run one trial per goal
   (it defaults to trial 1).

3. Run this script:
       python3 compare_experiments.py
   or, to use a different folder:
       python3 compare_experiments.py --logs-dir my_logs/

This produces:
    - A printed summary table (one row per goal x method, averaged
      across trials, with success rate)
    - comparison_charts.png: bar charts of success rate, steps to
      goal, path efficiency, and turn ratio, DQN vs A* side by side
      across all tested goals
    - all_runs.csv: every individual run's metrics, one row per run,
      for your own further analysis or inclusion as a raw data table

Requires matplotlib (pip3 install matplotlib --break-system-packages
if you don't already have it from training).
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # no display needed, just saves files
import matplotlib.pyplot as plt
import numpy as np

# Reuse the exact same parsing logic as analyze_run.py, so both tools
# always agree on what a "step" and a "success" mean.
from analyze_run import parse_log, compute_metrics


FILENAME_RE = re.compile(
    r"(dqn|astar)_([\d.]+)m_([\d.]+)deg(?:_trial(\d+))?\.log$",
    re.IGNORECASE,
)


def discover_runs(logs_dir):
    """Scans logs_dir for files matching the naming convention and
    returns a list of (filepath, method, distance, angle, trial)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        fname = os.path.basename(path)
        m = FILENAME_RE.search(fname)
        if not m:
            print(f"  (skipping '{fname}' -- doesn't match naming convention)")
            continue
        method = m.group(1).lower()
        distance = float(m.group(2))
        angle = float(m.group(3))
        trial = int(m.group(4)) if m.group(4) else 1
        runs.append((path, method, distance, angle, trial))
    return runs


def load_all_runs(logs_dir):
    """Parses every discovered log and returns a list of result dicts,
    one per run, with method/goal info attached to each."""
    results = []
    for path, method, distance, angle, trial in discover_runs(logs_dir):
        with open(path) as f:
            steps, outcome = parse_log(f.read())
        metrics = compute_metrics(steps, outcome)
        if metrics is None:
            print(f"  (warning: no parseable steps in '{path}', skipping)")
            continue
        metrics["method"] = method
        metrics["goal_distance"] = distance
        metrics["goal_angle"] = angle
        metrics["trial"] = trial
        metrics["file"] = path
        results.append(metrics)
    return results


def aggregate(results):
    """Groups runs by (method, goal) and averages metrics across
    trials, including a success rate (fraction of trials that succeeded)."""
    groups = defaultdict(list)
    for r in results:
        key = (r["method"], r["goal_distance"], r["goal_angle"])
        groups[key].append(r)

    aggregated = []
    for (method, distance, angle), runs in sorted(groups.items()):
        n = len(runs)
        success_rate = sum(1 for r in runs if r["outcome"] == "success") / n
        aggregated.append({
            "method": method,
            "goal_distance": distance,
            "goal_angle": angle,
            "n_trials": n,
            "success_rate": success_rate,
            "avg_steps": np.mean([r["total_steps"] for r in runs]),
            "avg_path_length": np.mean([r["path_length_m"] for r in runs]),
            "avg_path_efficiency": np.mean([r["path_efficiency"] for r in runs]),
            "avg_turn_ratio": np.mean([r["turn_ratio"] for r in runs]),
            "avg_reversals": np.mean([r["direction_reversals"] for r in runs]),
            "avg_final_dist": np.mean([r["final_dist_to_goal_m"] for r in runs]),
        })
    return aggregated


def print_summary(aggregated):
    print(f"\n{'Goal':<16}{'Method':<8}{'N':>4}{'Success':>10}{'Avg Steps':>12}{'Path Eff.':>12}{'Turn Ratio':>12}{'Reversals':>11}")
    print("-" * 87)
    for row in aggregated:
        goal_label = f"{row['goal_distance']}m/{row['goal_angle']}°"
        print(
            f"{goal_label:<16}{row['method']:<8}{row['n_trials']:>4}"
            f"{row['success_rate']:>9.0%} {row['avg_steps']:>11.1f}"
            f"{row['avg_path_efficiency']:>11.1%} {row['avg_turn_ratio']:>11.1%}"
            f"{row['avg_reversals']:>11.1f}"
        )


def write_csv(results, path="all_runs.csv"):
    if not results:
        return
    fieldnames = [
        "file", "method", "goal_distance", "goal_angle", "trial", "outcome",
        "total_steps", "path_length_m", "path_efficiency", "turn_ratio",
        "direction_reversals", "evasive_count", "final_dist_to_goal_m",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nWrote per-run data to {path}")


def make_charts(aggregated, out_path="comparison_charts.png"):
    if not aggregated:
        print("No data to chart.")
        return

    goals = sorted(set((row["goal_distance"], row["goal_angle"]) for row in aggregated))
    goal_labels = [f"{d}m\n{a}\u00b0" for d, a in goals]

    def get(method, metric):
        vals = []
        for d, a in goals:
            match = [row for row in aggregated if row["method"] == method
                     and row["goal_distance"] == d and row["goal_angle"] == a]
            vals.append(match[0][metric] if match else 0)
        return vals

    x = np.arange(len(goals))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    panels = [
        ("success_rate", "Success Rate", "{:.0%}"),
        ("avg_steps", "Avg. Steps to Goal", "{:.1f}"),
        ("avg_path_efficiency", "Avg. Path Efficiency", "{:.0%}"),
        ("avg_turn_ratio", "Avg. Turn Ratio", "{:.0%}"),
    ]

    for ax, (metric, title, fmt) in zip(axes.flat, panels):
        dqn_vals = get("dqn", metric)
        astar_vals = get("astar", metric)

        ax.bar(x - width / 2, dqn_vals, width, label="DQN", color="#4C72B0")
        ax.bar(x + width / 2, astar_vals, width, label="A*", color="#DD8452")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(goal_labels)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        for i, v in enumerate(dqn_vals):
            ax.text(i - width / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        for i, v in enumerate(astar_vals):
            ax.text(i + width / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8)

    fig.suptitle("DQN vs. A*: Real-Hardware Navigation Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved comparison charts to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="logs", help="Folder containing your .log files (default: logs/)")
    args = parser.parse_args()

    if not os.path.isdir(args.logs_dir):
        print(f"Folder '{args.logs_dir}' doesn't exist. Create it and put your logs there, e.g.:")
        print(f"  {args.logs_dir}/dqn_1.2m_0deg_trial1.log")
        print(f"  {args.logs_dir}/astar_1.2m_0deg_trial1.log")
        sys.exit(1)

    print(f"Scanning '{args.logs_dir}' for logs...")
    results = load_all_runs(args.logs_dir)

    if not results:
        print("\nNo valid logs found. Check your filenames match:")
        print("  {method}_{distance}m_{angle}deg_trial{n}.log")
        print("  e.g. dqn_1.2m_0deg_trial1.log")
        sys.exit(1)

    print(f"Loaded {len(results)} runs.")

    aggregated = aggregate(results)
    print_summary(aggregated)
    write_csv(results)
    make_charts(aggregated)


if __name__ == "__main__":
    main()

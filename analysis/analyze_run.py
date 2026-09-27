"""
analyze_run.py

Parses pasted terminal output from deploy.py (DQN) or deploy_astar.py
(A*) and computes the metrics that actually matter for a DQN-vs-A*
comparison: success/collision/timeout, path length and efficiency,
turn ratio, direction reversals ("choppiness"), and evasive-override
count.

Usage:
    Analyze one run:
        python3 analyze_run.py run1.log

    Compare two runs side by side (e.g. a DQN run vs an A* run):
        python3 analyze_run.py dqn_run.log astar_run.log

To use it: paste a terminal log into a text file (e.g. dqn_run.log),
then run this script on it. Works on logs copied straight from
deploy.py or deploy_astar.py, including ones with EVASIVE override
lines and sensor warnings mixed in.
"""

import math
import re
import sys


STEP_LINE_RE = re.compile(
    r"Step\s+(\d+)\s*\|\s*"
    r"(?:action:\s*(\w[\w\s]*?)|EVASIVE:\s*(\w[\w\s]*?)(?:\s*\(obstacle at ([\d.]+)m\))?)\s*\|\s*"
    r"pos:\s*\(\s*(-?[\d.]+),\s*(-?[\d.]+)\)\s*\|\s*"
    r"dist to goal:\s*([\d.]+)m"
)


def parse_log(text):
    """Extracts a list of step dicts from raw terminal log text."""
    steps = []
    for line in text.splitlines():
        m = STEP_LINE_RE.search(line)
        if not m:
            continue
        step_num = int(m.group(1))
        action = (m.group(2) or m.group(3) or "").strip()
        is_evasive = m.group(3) is not None
        x, y = float(m.group(5)), float(m.group(6))
        dist_to_goal = float(m.group(7))
        steps.append({
            "step": step_num,
            "action": action,
            "is_evasive": is_evasive,
            "pos": (x, y),
            "dist_to_goal": dist_to_goal,
        })

    outcome = "unknown"
    if re.search(r"GOAL REACHED", text):
        outcome = "success"
    elif re.search(r"SAFETY STOP", text):
        outcome = "safety_stop"
    elif re.search(r"Max steps reached", text):
        outcome = "timeout"
    elif re.search(r"Stopped by user", text):
        outcome = "interrupted"

    return steps, outcome


def compute_metrics(steps, outcome):
    if not steps:
        return None

    total_steps = len(steps)

    # path length: sum of straight-line distance between consecutive positions
    path_length = 0.0
    for i in range(1, len(steps)):
        x1, y1 = steps[i - 1]["pos"]
        x2, y2 = steps[i]["pos"]
        path_length += math.hypot(x2 - x1, y2 - y1)

    start_x, start_y = steps[0]["pos"]
    end_x, end_y = steps[-1]["pos"]
    straight_line_dist = math.hypot(end_x - start_x, end_y - start_y)
    path_efficiency = (straight_line_dist / path_length) if path_length > 0 else float("nan")

    initial_dist_to_goal = steps[0]["dist_to_goal"]
    final_dist_to_goal = steps[-1]["dist_to_goal"]

    action_counts = {}
    for s in steps:
        action_counts[s["action"]] = action_counts.get(s["action"], 0) + 1

    turn_actions = action_counts.get("turn left", 0) + action_counts.get("turn right", 0)
    turn_ratio = turn_actions / total_steps if total_steps else 0.0

    # direction reversals: turn left immediately followed by turn right, or vice versa
    reversals = 0
    for i in range(1, len(steps)):
        prev_a, cur_a = steps[i - 1]["action"], steps[i]["action"]
        if {prev_a, cur_a} == {"turn left", "turn right"}:
            reversals += 1

    evasive_count = sum(1 for s in steps if s["is_evasive"])

    return {
        "outcome": outcome,
        "total_steps": total_steps,
        "path_length_m": path_length,
        "straight_line_dist_m": straight_line_dist,
        "path_efficiency": path_efficiency,
        "initial_dist_to_goal_m": initial_dist_to_goal,
        "final_dist_to_goal_m": final_dist_to_goal,
        "action_counts": action_counts,
        "turn_ratio": turn_ratio,
        "direction_reversals": reversals,
        "evasive_count": evasive_count,
    }


def print_single(metrics, label="Run"):
    print(f"\n=== {label} ===")
    print(f"  Outcome:                {metrics['outcome']}")
    print(f"  Total steps:            {metrics['total_steps']}")
    print(f"  Path length:            {metrics['path_length_m']:.2f} m")
    print(f"  Straight-line dist:     {metrics['straight_line_dist_m']:.2f} m")
    print(f"  Path efficiency:        {metrics['path_efficiency']:.2%}  (straight-line / actual path)")
    print(f"  Initial dist to goal:   {metrics['initial_dist_to_goal_m']:.2f} m")
    print(f"  Final dist to goal:     {metrics['final_dist_to_goal_m']:.2f} m")
    print(f"  Turn ratio:             {metrics['turn_ratio']:.1%}  (fraction of steps that were turns)")
    print(f"  Direction reversals:    {metrics['direction_reversals']}  (turn-left immediately followed by turn-right, or vice versa)")
    print(f"  Evasive overrides:      {metrics['evasive_count']}")
    print(f"  Action breakdown:       {metrics['action_counts']}")


def print_comparison(metrics_a, metrics_b, label_a="Run A", label_b="Run B"):
    print(f"\n{'Metric':<28}{label_a:>18}{label_b:>18}")
    print("-" * 64)
    rows = [
        ("Outcome", metrics_a["outcome"], metrics_b["outcome"]),
        ("Total steps", metrics_a["total_steps"], metrics_b["total_steps"]),
        ("Path length (m)", f"{metrics_a['path_length_m']:.2f}", f"{metrics_b['path_length_m']:.2f}"),
        ("Path efficiency", f"{metrics_a['path_efficiency']:.1%}", f"{metrics_b['path_efficiency']:.1%}"),
        ("Turn ratio", f"{metrics_a['turn_ratio']:.1%}", f"{metrics_b['turn_ratio']:.1%}"),
        ("Direction reversals", metrics_a["direction_reversals"], metrics_b["direction_reversals"]),
        ("Evasive overrides", metrics_a["evasive_count"], metrics_b["evasive_count"]),
        ("Final dist to goal (m)", f"{metrics_a['final_dist_to_goal_m']:.2f}", f"{metrics_b['final_dist_to_goal_m']:.2f}"),
    ]
    for name, a, b in rows:
        print(f"{name:<28}{str(a):>18}{str(b):>18}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_run.py <log_file> [<second_log_file>]")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        steps_a, outcome_a = parse_log(f.read())
    metrics_a = compute_metrics(steps_a, outcome_a)

    if metrics_a is None:
        print(f"No parseable step lines found in {sys.argv[1]}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        with open(sys.argv[2]) as f:
            steps_b, outcome_b = parse_log(f.read())
        metrics_b = compute_metrics(steps_b, outcome_b)
        if metrics_b is None:
            print(f"No parseable step lines found in {sys.argv[2]}")
            sys.exit(1)
        print_comparison(metrics_a, metrics_b, label_a=sys.argv[1], label_b=sys.argv[2])
    else:
        print_single(metrics_a, label=sys.argv[1])


if __name__ == "__main__":
    main()

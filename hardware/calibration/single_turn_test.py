"""
single_turn_test.py

Measures how many REAL degrees a single turn action produces in EACH
direction (left and right separately -- they use opposite motor/direction
combinations, so there's no guarantee they rotate by the same amount),
using the EXACT same TURN_STEP_DURATION_S and motor scales that deploy.py
uses per turn decision (NOT the same as deploy.py's forward STEP_DURATION_S
-- turns get their own, shorter duration). This checks whether a single
turn action's real effect matches what the simulation assumes (~17
degrees, from train_corridor.py's TURN_SPEED_RAD=0.30 radians) -- a large
mismatch here would explain persistent oscillating turn behavior, since
the policy learned assuming small ~17-degree nudges per turn decision.

It also prints a suggested ANGLE_TOLERANCE_RAD for deploy_astar.py:
AStarAgent's angle tolerance needs to be comfortably wider than a single
real burst's rotation, or every turn overshoots the tolerance window and
the robot oscillates correcting back and forth forever.

Run on the Pi:
    python3 single_turn_test.py
"""

import math
import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

from odometry import Odometry

Device.pin_factory = LGPIOFactory()

# EXACT match to deploy.py's real per-turn-decision values (its
# TURN_STEP_DURATION_S, not its forward STEP_DURATION_S -- keep in sync)
STEP_DURATION_S = 0.157
MOTOR_POWER = 0.5
TURN_LEFT_SCALE = 1.1104
TURN_RIGHT_SCALE = 0.9234

EXPECTED_SIM_DEGREES = math.degrees(0.30)  # train_corridor.py's TURN_SPEED_RAD
NUM_TRIALS = 5

motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)


def run_trials(direction):
    """direction: 'left' or 'right'. Returns list of measured degrees
    (signed, per odometry heading -- left turns increase heading,
    right turns decrease it)."""
    left_power = min(MOTOR_POWER * TURN_LEFT_SCALE, 1.0)
    right_power = min(MOTOR_POWER * TURN_RIGHT_SCALE, 1.0)

    odom = Odometry()
    results = []
    for trial in range(NUM_TRIALS):
        odom.reset()
        pre_left_count = odom.left_encoder.count
        pre_right_count = odom.right_encoder.count

        if direction == "left":
            motor_a.backward(left_power)
            motor_b.forward(right_power)
        else:
            motor_a.forward(left_power)
            motor_b.backward(right_power)

        time.sleep(STEP_DURATION_S)
        motor_a.stop()
        motor_b.stop()

        time.sleep(0.3)  # let things settle before reading
        x, y, heading = odom.update()
        heading_deg = math.degrees(heading)
        results.append(heading_deg)
        # pulses THIS BURST (not cumulative) alongside heading -- if a trial
        # reads ~0 degrees AND ~0 pulses this burst, the wheels genuinely
        # didn't turn (stall/slip/wiring, not a math problem). If pulses are
        # non-trivial but heading is still ~0, the problem is in the math.
        delta_left = odom.left_encoder.count - pre_left_count
        delta_right = odom.right_encoder.count - pre_right_count
        print(
            f"  Trial {trial+1}/{NUM_TRIALS}: single {direction} burst produced {heading_deg:+.1f} degrees "
            f"| L pulses this burst: {delta_left:5d} (rejected: {odom.left_encoder.rejected_count}) "
            f"| R pulses this burst: {delta_right:5d} (rejected: {odom.right_encoder.rejected_count})"
        )
        time.sleep(1)

    odom.close()
    return results


def report(direction, results):
    avg = sum(results) / len(results)
    avg_mag = abs(avg)
    print(f"\nAverage single-{direction}-burst rotation: {avg:+.1f} degrees")
    print(f"Simulation assumes: {EXPECTED_SIM_DEGREES:.1f} degrees")
    print(f"Ratio (real/sim): {avg_mag/EXPECTED_SIM_DEGREES:.2f}x")

    if abs(avg_mag / EXPECTED_SIM_DEGREES - 1.0) > 0.3:
        print("-> LARGE mismatch vs. the simulated turn magnitude. This likely explains a")
        print("   lot of oscillating turn behavior -- the policy/planner expects small")
        print("   nudges, reality gives much bigger swings.")
        print(f"   Fix: scale STEP_DURATION_S for turns by ~{EXPECTED_SIM_DEGREES/avg_mag:.2f}x")
        print(f"   New turn step duration suggestion: {STEP_DURATION_S * EXPECTED_SIM_DEGREES/avg_mag:.3f}s")
    return avg_mag


def main():
    print(f"Simulation assumes ~{EXPECTED_SIM_DEGREES:.1f} degrees per single turn action.")
    print(f"Testing what ONE real {STEP_DURATION_S}s turn burst actually produces, each direction...\n")
    time.sleep(9)

    print("--- Turn LEFT ---")
    left_results = run_trials("left")
    left_avg_mag = report("left", left_results)

    print("\n--- Turn RIGHT ---")
    right_results = run_trials("right")
    right_avg_mag = report("right", right_results)

    diff_pct = abs(left_avg_mag - right_avg_mag) / max(left_avg_mag, right_avg_mag) * 100
    print(f"\nLeft vs. right burst magnitude differ by {diff_pct:.0f}%.")
    if diff_pct > 20:
        print("-> Meaningful left/right asymmetry: TURN_LEFT_SCALE/TURN_RIGHT_SCALE were")
        print("   calibrated from a turn-LEFT-only drift test (turn_drift_test.py), then")
        print("   reused as-is for turning right. That assumes each motor's forward vs.")
        print("   backward speed is identical at the same commanded power, which this")
        print("   result says isn't true here. Re-run turn_drift_test.py (it now tests")
        print("   both directions) and consider separate scale factors per direction.")

    # AStarAgent's angle_tolerance must be comfortably wider than the LARGER
    # of the two real per-burst rotations, or turns will chronically overshoot
    # the tolerance window and oscillate. 1.5x margin above the larger measured
    # burst, matching whichever unit (radians) deploy_astar.py expects.
    worst_case_deg = max(left_avg_mag, right_avg_mag)
    suggested_tolerance_rad = math.radians(worst_case_deg) * 1.5
    print(f"\nSuggested ANGLE_TOLERANCE_RAD for deploy_astar.py: {suggested_tolerance_rad:.3f}"
          f" (~{math.degrees(suggested_tolerance_rad):.1f} degrees)")
    print("Set that constant in deploy_astar.py before trusting it on real hardware.")

    motor_a.stop()
    motor_b.stop()


if __name__ == "__main__":
    main()

"""
turn_drift_test.py

Isolated diagnostic: commands a fixed number of pure turn actions (no
DQN policy involved at all) and reports how far the odometry thinks the
robot drifted linearly. A perfect in-place turn should show ~0 net
(x, y) displacement -- any meaningful drift here points to a real
wheel-speed mismatch between the two motors (very common with cheap DC
gearmotors, even "identical" ones), not a software bug.

Tests BOTH turn directions, separately. TURN_LEFT_SCALE/TURN_RIGHT_SCALE
were originally derived from a turn-LEFT-only run of this test (motor_a
backward + motor_b forward), then reused as-is for turning right (motor_a
forward + motor_b backward) elsewhere in the project. That silently
assumes each motor's forward-direction speed matches its backward-
direction speed at the same commanded power -- not guaranteed for a DC
gearmotor/H-bridge. Running both directions here shows whether that
assumption actually holds for your hardware.

Run on the Pi:
    python3 turn_drift_test.py
"""

import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

from odometry import Odometry

Device.pin_factory = LGPIOFactory()

NUM_TURNS = 20
STEP_DURATION_S = 0.3
MOTOR_POWER = 0.5

# same calibration as deploy.py -- update both together if you
# recompute new scale factors from a fresh drift test
#
# NOTE: the first attempt used the full computed correction
# (1.2208 / 0.8468) and made the imbalance WORSE, not better --
# DC motor speed doesn't scale linearly with commanded power, so a
# single-point measurement doesn't reliably predict behavior at a
# different power level. Backed off to a conservative HALFWAY
# correction here; iterate gradually (re-run this test, average
# toward whichever direction still has more drift) rather than
# recomputing the full "ideal" ratio each time.
LEFT_POWER_SCALE = 1.1104
RIGHT_POWER_SCALE = 0.9234

motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right


def run_direction(odom, direction):
    """direction: 'left' or 'right'. Returns final drift distance (m)."""
    odom.reset()
    drift = 0.0

    for i in range(NUM_TURNS):
        if direction == "left":
            motor_a.backward(min(MOTOR_POWER * LEFT_POWER_SCALE, 1.0))
            motor_b.forward(min(MOTOR_POWER * RIGHT_POWER_SCALE, 1.0))
        else:
            motor_a.forward(min(MOTOR_POWER * LEFT_POWER_SCALE, 1.0))
            motor_b.backward(min(MOTOR_POWER * RIGHT_POWER_SCALE, 1.0))
        time.sleep(STEP_DURATION_S)
        motor_a.stop()
        motor_b.stop()

        x, y, heading = odom.update()
        drift = (x**2 + y**2) ** 0.5
        print(
            f"Turn {i+1:2d}/{NUM_TURNS} | pos: ({x:6.3f}, {y:6.3f}) | "
            f"drift from start: {drift:.3f}m | "
            f"L pulses: {odom.left_encoder.count:6d} (rejected: {odom.left_encoder.rejected_count}) | "
            f"R pulses: {odom.right_encoder.count:6d} (rejected: {odom.right_encoder.rejected_count})"
        )

    print(f"\nFinal drift after {NUM_TURNS} pure turn-{direction} actions: {drift:.3f}m")
    print(f"Average drift per turn: {drift/NUM_TURNS*1000:.1f}mm")
    if drift > 0.15:
        print(f"-> Significant drift detected turning {direction}. Points to a real wheel-speed")
        print("   mismatch between motor_a and motor_b at the same commanded power, in this direction.")
    else:
        print(f"-> Drift turning {direction} is small. Turning this direction looks OK.")
    return drift


def main():
    odom = Odometry()

    print(f"Commanding {NUM_TURNS} pure turn actions per direction, no policy involved.")
    print("Watching odometry-reported (x, y) drift -- should stay near (0, 0)")
    print("for a perfect in-place turn.\n")
    print("Starting in 9 seconds -- get in position near the robot now...")
    time.sleep(9)

    try:
        print("--- Turn LEFT ---")
        left_drift = run_direction(odom, "left")

        print("\n--- Turn RIGHT ---")
        right_drift = run_direction(odom, "right")

        diff_pct = abs(left_drift - right_drift) / max(left_drift, right_drift, 1e-6) * 100
        print(f"\nLeft-turn drift {left_drift:.3f}m vs. right-turn drift {right_drift:.3f}m ({diff_pct:.0f}% different).")
        if diff_pct > 30:
            print("-> The two directions drift meaningfully differently. TURN_LEFT_SCALE/")
            print("   TURN_RIGHT_SCALE (measured turning left only) may not be accurate for")
            print("   turning right -- consider deriving a separate scale pair per direction.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()


if __name__ == "__main__":
    main()

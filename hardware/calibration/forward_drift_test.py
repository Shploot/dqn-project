"""
forward_drift_test.py

Isolated diagnostic: commands a fixed number of pure "forward" actions
(no policy involved, no turning) and reports how far the robot curves
off a straight line via odometry. A perfectly matched pair of wheels
should drive dead straight; any lateral drift here confirms a
wheel-speed mismatch specifically under FORWARD load -- which can
differ from the mismatch measured while turning (turn_drift_test.py),
since DC motors don't always behave identically under different load
conditions.

Run on the Pi:
    python3 forward_drift_test.py
"""

import math
import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

from odometry import Odometry

Device.pin_factory = LGPIOFactory()

NUM_STEPS = 15
STEP_DURATION_S = 0.3
MOTOR_POWER = 0.5

# forward-specific calibration, derived from this test's own clean
# (pre-collision) data -- keep in sync with deploy.py's FORWARD_LEFT_SCALE/
# FORWARD_RIGHT_SCALE if you recompute further
LEFT_POWER_SCALE = 0.9572
RIGHT_POWER_SCALE = 1.0468

motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right


def main():
    odom = Odometry()
    odom.reset()

    print(f"Commanding {NUM_STEPS} pure forward actions, no policy involved.")
    print("A perfectly straight drive should show ~0 lateral deviation and")
    print("~0 degrees of heading change.\n")
    print("Starting in 9 seconds -- get in position near the robot now...")
    time.sleep(9)

    try:
        for i in range(NUM_STEPS):
            motor_a.forward(min(MOTOR_POWER * LEFT_POWER_SCALE, 1.0))
            motor_b.forward(min(MOTOR_POWER * RIGHT_POWER_SCALE, 1.0))
            time.sleep(STEP_DURATION_S)
            motor_a.stop()
            motor_b.stop()

            x, y, heading = odom.update()
            distance_traveled = (x**2 + y**2) ** 0.5
            heading_deg = math.degrees(heading)
            # lateral deviation: perpendicular distance from the straight
            # line the robot SHOULD have driven along (the x-axis, since
            # it started facing heading=0)
            lateral_deviation = y

            print(
                f"Step {i+1:2d}/{NUM_STEPS} | pos: ({x:6.3f}, {y:6.3f}) | "
                f"distance traveled: {distance_traveled:.3f}m | "
                f"heading: {heading_deg:6.1f} deg | "
                f"lateral deviation: {lateral_deviation:+.3f}m | "
                f"L pulses: {odom.left_encoder.count:6d} (rejected: {odom.left_encoder.rejected_count}) | "
                f"R pulses: {odom.right_encoder.count:6d} (rejected: {odom.right_encoder.rejected_count})"
            )

        print(f"\nFinal lateral deviation after {NUM_STEPS} forward steps: {lateral_deviation:+.3f}m")
        print(f"Final heading drift: {heading_deg:.1f} degrees (should be ~0 for straight driving)")
        if abs(lateral_deviation) > 0.05 or abs(heading_deg) > 5:
            print("\n-> Significant curving detected during FORWARD motion specifically.")
            print("   This confirms the wheel mismatch affects forward driving too,")
            print("   and likely needs its own calibration separate from the turn one.")
            print(f"   Raw pulse counts -- left: {odom.left_encoder.count}, right: {odom.right_encoder.count}")
        else:
            print("\n-> Forward driving looks reasonably straight with current calibration.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()


if __name__ == "__main__":
    main()

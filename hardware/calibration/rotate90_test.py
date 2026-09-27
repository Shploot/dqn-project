"""
rotate90_test.py

Commands the robot to turn until the ODOMETRY reports a 90-degree
(pi/2 radian) heading change, then stops so you can physically check
with a protractor / right-angle reference whether the real rotation
actually matches. This tells you whether the wheel calibration
(LEFT_POWER_SCALE / RIGHT_POWER_SCALE) produces accurate turn angles,
not just low position drift -- the two are related but not identical.

Run on the Pi:
    python3 rotate90_test.py
Press Ctrl+C to stop immediately (also stops motors).
"""

import math
import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

from odometry import Odometry

Device.pin_factory = LGPIOFactory()

TARGET_DEGREES = 90.0
TARGET_RAD = math.radians(TARGET_DEGREES)

STEP_DURATION_S = 0.1  # short bursts for finer stopping precision than the usual 0.3s
MOTOR_POWER = 0.5

# same wheel calibration as deploy.py -- keep in sync
LEFT_POWER_SCALE = 1.1104
RIGHT_POWER_SCALE = 0.9234

motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right


def main():
    odom = Odometry()
    odom.reset()

    print(f"Turning left until odometry reports {TARGET_DEGREES} degrees.")
    print("Place a protractor or right-angle reference under the robot's")
    print("starting orientation so you can check the real result.\n")
    time.sleep(9)

    left_power = min(MOTOR_POWER * LEFT_POWER_SCALE, 1.0)
    right_power = min(MOTOR_POWER * RIGHT_POWER_SCALE, 1.0)

    try:
        step = 0
        while True:
            x, y, heading = odom.update()
            heading_deg = math.degrees(heading)

            if abs(heading) >= TARGET_RAD:
                break

            motor_a.backward(left_power)
            motor_b.forward(right_power)
            time.sleep(STEP_DURATION_S)
            motor_a.stop()
            motor_b.stop()

            step += 1
            if step % 5 == 0:
                print(f"  ...at {heading_deg:.1f} degrees so far")

        x, y, heading = odom.update()
        heading_deg = math.degrees(heading)
        print(f"\nStopped. Odometry reports: {heading_deg:.1f} degrees (target: {TARGET_DEGREES})")
        print("Now physically check: does the robot's real orientation look like")
        print("a genuine 90-degree turn from where it started?")
        print("  - If yes: odometry heading tracking is accurate, trust it.")
        print("  - If no (real turn is noticeably more/less than 90): the wheel")
        print("    calibration needs further tuning, or there's real slip on")
        print("    this surface specifically.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()


if __name__ == "__main__":
    main()

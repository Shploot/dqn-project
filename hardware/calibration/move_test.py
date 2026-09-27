"""
move_test.py

Drives the robot forward 5 inches, then backward 5 inches, using real
encoder feedback (via odometry.py) to stop at the exact target
distance -- this is closed-loop control, not a timed guess, so it
should be accurate regardless of battery charge level or friction.

Run on the Pi:
    python3 move_test.py
Press Ctrl+C to stop immediately (also stops motors).
"""

import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

from odometry import Odometry

Device.pin_factory = LGPIOFactory()

INCHES_TO_MOVE = 5.0
DISTANCE_M = INCHES_TO_MOVE * 0.0254

MOTOR_POWER = 0.4  # moderate speed for a controlled, easy-to-observe test
STOP_TOLERANCE_M = 0.005  # stop within 5mm of target
RAMP_DOWN_START_M = 0.06  # start slowing down within 6cm of target (widened from 3cm)
MIN_POWER = 0.15  # don't go so slow the motors stall out entirely
BRAKE_PULSE_S = 0.04  # brief reverse pulse at the end to actively kill momentum

motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right


def drive_distance(odom: Odometry, target_distance_m, forward=True):
    """Drives straight until the encoder-measured distance traveled
    (straight-line displacement from the start of this call) reaches
    target_distance_m, then stops."""
    odom.reset()
    direction = "forward" if forward else "backward"
    print(f"Driving {direction} {target_distance_m*39.37:.1f} inches...")

    while True:
        x, y, _ = odom.update()
        traveled = (x**2 + y**2) ** 0.5
        remaining = target_distance_m - traveled

        if remaining <= STOP_TOLERANCE_M:
            break

        # ramp power down linearly as we approach the target, instead
        # of running full power until an abrupt stop -- reduces the
        # momentum-based overshoot/slip that happens when a heavier
        # robot coasts forward after the motors cut off
        if remaining < RAMP_DOWN_START_M:
            power = MIN_POWER + (MOTOR_POWER - MIN_POWER) * (remaining / RAMP_DOWN_START_M)
        else:
            power = MOTOR_POWER

        if forward:
            motor_a.forward(power)
            motor_b.forward(power)
        else:
            motor_a.backward(power)
            motor_b.backward(power)

        time.sleep(0.02)  # tight loop for responsive stopping

    motor_a.stop()
    motor_b.stop()

    # active brake: a brief pulse in the opposite direction kills
    # remaining momentum far more effectively than just cutting power
    # and coasting -- important for a heavier robot
    if forward:
        motor_a.backward(MIN_POWER)
        motor_b.backward(MIN_POWER)
    else:
        motor_a.forward(MIN_POWER)
        motor_b.forward(MIN_POWER)
    time.sleep(BRAKE_PULSE_S)
    motor_a.stop()
    motor_b.stop()

    final_x, final_y, _ = odom.update()
    final_traveled = (final_x**2 + final_y**2) ** 0.5
    print(f"  Stopped at {final_traveled*39.37:.2f} inches (target: {target_distance_m*39.37:.1f} inches)")
    print(f"  Encoder pulses -- left: {odom.left_encoder.count}, right: {odom.right_encoder.count}")


def main():
    odom = Odometry()

    try:
        print(f"Test: forward {INCHES_TO_MOVE} inches, then backward {INCHES_TO_MOVE} inches.")
        print("Make sure the robot has clearance in front and behind.")
        time.sleep(9)

        drive_distance(odom, DISTANCE_M, forward=True)
        time.sleep(1)
        drive_distance(odom, DISTANCE_M, forward=False)

        print("\nDone. Measure the robot's actual final position with a tape")
        print("measure -- it should be very close to where it started, since")
        print("it drove the same distance forward and back.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()


if __name__ == "__main__":
    main()

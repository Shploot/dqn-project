"""
forward_then_turn.py

Drives forward a specific distance (default: 4/5 of the corridor
length), then turns left a specific angle (default: 90 degrees) --
both using real encoder feedback via odometry.py, not timed guesses.

Uses the current forward/turn calibration scales from deploy.py, so
it should track about as straight/accurately as your real DQN runs do.

Run on the Pi:
    python3 forward_then_turn.py
Press Ctrl+C to stop immediately (also stops motors).
"""

import math
import time

from gpiozero import Device, Motor
from gpiozero.pins.lgpio import LGPIOFactory

from odometry import Odometry

Device.pin_factory = LGPIOFactory()

# ------------------------------------------------------------------ #
CORRIDOR_LENGTH_M = 1.83
FORWARD_DISTANCE_M = CORRIDOR_LENGTH_M * 4 / 5  # 1.464m
TURN_DEGREES = 90.0

STEP_DURATION_S = 0.1  # short bursts for responsive stopping
MOTOR_POWER = 0.5

# current calibration -- keep in sync with deploy.py
FORWARD_LEFT_SCALE = 0.9572
FORWARD_RIGHT_SCALE = 1.0468
TURN_LEFT_SCALE = 1.1104
TURN_RIGHT_SCALE = 0.9234

RAMP_DOWN_START_M = 0.06
MIN_POWER = 0.15
BRAKE_PULSE_S = 0.04

motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right


def drive_forward(odom: Odometry, target_distance_m):
    odom.reset()
    print(f"Driving forward {target_distance_m:.3f}m ({target_distance_m*39.37:.1f} in)...")

    while True:
        x, y, _ = odom.update()
        traveled = math.hypot(x, y)
        remaining = target_distance_m - traveled

        if remaining <= 0.005:
            break

        if remaining < RAMP_DOWN_START_M:
            power_frac = MIN_POWER + (MOTOR_POWER - MIN_POWER) * (remaining / RAMP_DOWN_START_M)
        else:
            power_frac = MOTOR_POWER

        left_power = min(power_frac * FORWARD_LEFT_SCALE, 1.0)
        right_power = min(power_frac * FORWARD_RIGHT_SCALE, 1.0)
        motor_a.forward(left_power)
        motor_b.forward(right_power)
        time.sleep(0.02)

    motor_a.stop()
    motor_b.stop()

    # active brake: brief reverse pulse to kill momentum
    motor_a.backward(MIN_POWER)
    motor_b.backward(MIN_POWER)
    time.sleep(BRAKE_PULSE_S)
    motor_a.stop()
    motor_b.stop()

    x, y, heading = odom.update()
    final_dist = math.hypot(x, y)
    print(f"  Stopped at {final_dist:.3f}m ({final_dist*39.37:.1f} in) -- target was {target_distance_m:.3f}m")


def turn_left(odom: Odometry, target_degrees):
    target_rad = math.radians(target_degrees)
    print(f"\nTurning left {target_degrees:.0f} degrees...")

    start_x, start_y, start_heading = odom.update()

    left_power = min(MOTOR_POWER * TURN_LEFT_SCALE, 1.0)
    right_power = min(MOTOR_POWER * TURN_RIGHT_SCALE, 1.0)

    while True:
        x, y, heading = odom.update()
        turned = heading - start_heading
        # normalize in case of wraparound
        turned = (turned + math.pi) % (2 * math.pi) - math.pi

        if abs(turned) >= target_rad:
            break

        motor_a.backward(left_power)
        motor_b.forward(right_power)
        time.sleep(STEP_DURATION_S)
        motor_a.stop()
        motor_b.stop()

    x, y, heading = odom.update()
    turned_deg = math.degrees((heading - start_heading + math.pi) % (2 * math.pi) - math.pi)
    print(f"  Stopped after turning {turned_deg:.1f} degrees (target: {target_degrees:.0f})")


def main():
    odom = Odometry()

    print(f"Corridor length assumed: {CORRIDOR_LENGTH_M}m")
    print(f"Plan: drive forward {FORWARD_DISTANCE_M:.3f}m (4/5 of corridor), then turn left {TURN_DEGREES:.0f} degrees.")
    print("Starting in 9 seconds -- place robot at its start position now...")
    time.sleep(9)

    try:
        drive_forward(odom, FORWARD_DISTANCE_M)
        time.sleep(0.5)
        turn_left(odom, TURN_DEGREES)
        print("\nDone.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()


if __name__ == "__main__":
    main()

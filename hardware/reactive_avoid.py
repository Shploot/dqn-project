"""
reactive_avoid.py

Pure reactive obstacle avoidance -- no odometry, no goal position, no
planning. Every step: read the 3 sensors, turn toward whichever one
reports the MOST open space, or go forward if front is clearly the
most open. This is a classic Braitenberg-style reactive controller --
robust specifically because it never depends on position tracking,
so it sidesteps the odometry/wheel-calibration drift issues entirely.

Decision rule each step:
    - If anything is closer than SAFETY_STOP_M: hard stop (safety backstop).
    - Otherwise, turn toward whichever of front/left/right reads the
      LARGEST distance (most open path) -- never toward the smallest.
      If front itself is the largest (or within FRONT_PREFERENCE_M of
      the largest), just go forward instead of unnecessarily turning.

Run on the Pi:
    python3 reactive_avoid.py
Press Ctrl+C to stop immediately (also stops motors).
"""

import time

from gpiozero import Device, Motor, DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory

Device.pin_factory = LGPIOFactory()

STEP_DURATION_S = 0.3
MOTOR_POWER = 0.5
MAX_SENSOR_RANGE_M = 2.0

# same wheel calibration as deploy.py -- keep in sync.
# Forward and turn use SEPARATE scales because forward load (both wheels
# same direction) produces a different real speed mismatch than turn load
# (wheels opposite directions) -- see deploy.py's calibration comment.
TURN_LEFT_SCALE = 1.1104
TURN_RIGHT_SCALE = 0.9234
FORWARD_LEFT_SCALE = 0.9572
FORWARD_RIGHT_SCALE = 1.0468

# hard safety backstop -- halt regardless of steering logic if
# anything is this close (matches deploy.py's real-robot-sized value)
SAFETY_STOP_M = 0.15

# if front's reading is within this much of the overall largest
# reading, prefer going straight over turning -- avoids needless
# zig-zagging when the path ahead is already basically clear
FRONT_PREFERENCE_M = 0.10


def get_filtered_distance(sensor, num_samples=5):
    readings = sorted(sensor.distance for _ in range(num_samples))
    return readings[len(readings) // 2]


def choose_action(front_m, left_m, right_m):
    """Pure decision logic, no hardware -- kept separate from
    execute_action so it can be unit-tested with synthetic readings."""
    min_reading = min(front_m, left_m, right_m)
    if min_reading < SAFETY_STOP_M:
        return 3  # stop

    largest = max(front_m, left_m, right_m)

    # prefer going straight if front is close to the most-open reading,
    # even if not literally the single largest -- avoids unnecessary
    # zig-zag turning when the path ahead is already clear enough
    if front_m >= largest - FRONT_PREFERENCE_M:
        return 0  # forward

    if left_m >= right_m:
        return 1  # turn left, toward the more open side
    else:
        return 2  # turn right


def execute_action(motor_a, motor_b, action):
    if action == 0:  # forward -- both wheels same direction, use forward-specific calibration
        left_power = min(MOTOR_POWER * FORWARD_LEFT_SCALE, 1.0)
        right_power = min(MOTOR_POWER * FORWARD_RIGHT_SCALE, 1.0)
        motor_a.forward(left_power)
        motor_b.forward(right_power)
    elif action == 1:  # turn left -- wheels opposite directions, use turn-specific calibration
        left_power = min(MOTOR_POWER * TURN_LEFT_SCALE, 1.0)
        right_power = min(MOTOR_POWER * TURN_RIGHT_SCALE, 1.0)
        motor_a.backward(left_power)
        motor_b.forward(right_power)
    elif action == 2:  # turn right
        left_power = min(MOTOR_POWER * TURN_LEFT_SCALE, 1.0)
        right_power = min(MOTOR_POWER * TURN_RIGHT_SCALE, 1.0)
        motor_a.forward(left_power)
        motor_b.backward(right_power)
    elif action == 3:
        motor_a.stop()
        motor_b.stop()

    time.sleep(STEP_DURATION_S)
    motor_a.stop()
    motor_b.stop()


def main():
    motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
    motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right

    left_sensor = DistanceSensor(echo=4, trigger=17, max_distance=MAX_SENSOR_RANGE_M)
    middle_sensor = DistanceSensor(echo=18, trigger=27, max_distance=MAX_SENSOR_RANGE_M)
    right_sensor = DistanceSensor(echo=22, trigger=23, max_distance=MAX_SENSOR_RANGE_M)

    action_names = {0: "forward", 1: "turn left", 2: "turn right", 3: "stop"}

    print("Pure reactive obstacle avoidance -- no odometry, no goal, just live sensors.")
    print("Turns toward whichever direction has the most open space.")
    print("Press Ctrl+C to stop.\n")
    time.sleep(9)

    step = 0
    try:
        while True:
            front_m = get_filtered_distance(middle_sensor)
            left_m = get_filtered_distance(left_sensor)
            right_m = get_filtered_distance(right_sensor)

            action = choose_action(front_m, left_m, right_m)

            print(
                f"Step {step:4d} | sensors F/L/R: {front_m:.2f}/{left_m:.2f}/{right_m:.2f}m | "
                f"action: {action_names[action]}"
            )

            execute_action(motor_a, motor_b, action)
            step += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        print("Motors stopped.")


if __name__ == "__main__":
    main()

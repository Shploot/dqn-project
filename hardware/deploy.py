"""
deploy.py

Runs the trained DQN policy on the real robot: reads the 3 ultrasonic
sensors, tracks the robot's actual position via encoder-based odometry
(odometry.py -- real wheel rotation counts, not a guessed speed), feeds
that state into the trained model, and drives the motors according to
the chosen action.

No calibration needed for position tracking -- odometry.py uses your
measured wheel diameter, encoder pulses-per-rev, and wheel track width
to compute real distance/heading from actual encoder counts.

Run on the Pi:
    python3 deploy.py
Press Ctrl+C to stop immediately (also stops motors).
"""

import math
import time
from collections import deque

import numpy as np
import torch
from gpiozero import Device, Motor, DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory

from dqn_agent import DQNAgent
from odometry import Odometry

Device.pin_factory = LGPIOFactory()

# ------------------------------------------------------------------ #
STEP_DURATION_S = 0.3  # forward/stop actions: how long each runs before choosing the next one
MOTOR_POWER = 0.5      # 0.0-1.0

# TURN_STEP_DURATION_S: an earlier attempt to shorten this to ~0.101s (to
# match the simulation's assumed ~17-degree turn magnitude) made real
# performance WORSE and was reverted to match STEP_DURATION_S. At the time,
# odometry.py's heading was being computed with an inverted sign (turning
# left registered as negative heading, opposite of what nav_env.py assumes)
# -- so every corrective turn was reasoning about direction backwards, and
# shortening the burst just made those backwards corrections happen more
# often per episode. That sign bug is now fixed (see odometry.py).
#
# With the sign fixed, single_turn_test.py measured real single-burst turns
# of ~31.4 degrees left / ~34.3 degrees right at STEP_DURATION_S=0.3s --
# still ~2x the simulation's 17.2-degree assumption. Shortening the turn
# burst to bring that back in line with what the policy was trained on is
# worth retrying now that the earlier likely cause (the sign bug) is gone.
# Re-run single_turn_test.py and adjust this if you change MOTOR_POWER or
# the turn scale constants below.
TURN_STEP_DURATION_S = 0.157

# ------------------------------------------------------------------ #
# Wheel speed calibration -- confirmed via turn_drift_test.py AND
# forward_drift_test.py that the two motors spin at meaningfully
# different rates for the same commanded power, and that the mismatch
# under FORWARD load (both wheels same direction) is genuinely
# different from the mismatch under TURN load (wheels opposite
# directions) -- DC motors don't always behave identically under
# different load conditions, so each motion type gets its own
# calibration rather than one number covering both. Re-run the
# relevant drift test and recompute if drift is still meaningful:
#   avg = (left_pulses + right_pulses) / 2
#   full_correction_left  = avg / left_pulses
#   full_correction_right = avg / right_pulses
#   (average that with the CURRENT scale below, don't jump straight
#    to the full computed value -- motor response isn't perfectly
#    linear with commanded power)
# ------------------------------------------------------------------ #
TURN_LEFT_SCALE = 1.1104
TURN_RIGHT_SCALE = 0.9234
FORWARD_LEFT_SCALE = 0.9572
FORWARD_RIGHT_SCALE = 1.0468

# ------------------------------------------------------------------ #
# MUST MATCH the arena dimensions used in train_corridor.py exactly --
# the observation normalization below depends on these being the same
# values the loaded model was actually trained with. If you retrain
# with different dimensions, update these to match.
# ------------------------------------------------------------------ #
ARENA_WIDTH_M = 1.83   # long dimension -- robot's forward (heading=0) axis
ARENA_LENGTH_M = 0.91  # short dimension -- sideways axis

# ------------------------------------------------------------------ #
# Goal specification -- since the robot has no absolute positioning
# beyond its own odometry, you tell it where the goal is relative to
# its starting pose: how far ahead (meters) and at what angle
# (degrees, 0 = straight ahead, positive = to the left) from wherever
# you place the robot at start. Keep this well within ARENA_WIDTH_M
# (the model was never trained on goals farther than that).
# ------------------------------------------------------------------ #
GOAL_DISTANCE_M = 1.2  # was 1.4 -- that landed right at/past the arena's usable travel span (1.41m) once the robot's real 0.21m collision radius is accounted for; 1.2m leaves comfortable margin on both ends
GOAL_ANGLE_DEG = 0

GOAL_REACHED_THRESHOLD_M = 0.2
MAX_STEPS = 150  # matches max_steps used during corridor training
MAX_SENSOR_RANGE_M = 2.0  # must match what the corridor model was trained with

MODEL_PATH = "nav_dqn_1obs.pt"

# ------------------------------------------------------------------ #
# Hardware setup -- matches your confirmed wiring
# ------------------------------------------------------------------ #
motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left, wiring was reversed
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right

left_sensor = DistanceSensor(echo=4, trigger=17, max_distance=MAX_SENSOR_RANGE_M)
middle_sensor = DistanceSensor(echo=18, trigger=27, max_distance=MAX_SENSOR_RANGE_M)
right_sensor = DistanceSensor(echo=22, trigger=23, max_distance=MAX_SENSOR_RANGE_M)


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def get_filtered_distance(sensor, num_samples=5):
    """Reads a sensor multiple times in quick succession and returns the
    median value, rejecting single-sample outlier spikes -- these are
    common while the motors are running (electrical noise from motor
    PWM switching can corrupt an individual ultrasonic echo reading,
    e.g. a real 0.6m becoming a spurious 4.5m for one sample)."""
    readings = []
    for _ in range(num_samples):
        readings.append(sensor.distance)
    readings.sort()
    return readings[len(readings) // 2]


class GoalTracker:
    """Holds the fixed goal position (relative to start) and computes
    distance/angle to it using the *real* pose reported by Odometry."""

    def __init__(self, goal_distance_m, goal_angle_deg):
        goal_angle_rad = math.radians(goal_angle_deg)
        self.goal_x = goal_distance_m * math.cos(goal_angle_rad)
        self.goal_y = goal_distance_m * math.sin(goal_angle_rad)

    def dist_to_goal(self, x, y):
        return math.hypot(self.goal_x - x, self.goal_y - y)

    def reached_goal(self, x, y):
        return self.dist_to_goal(x, y) < GOAL_REACHED_THRESHOLD_M


def get_observation(x, y, heading, goal: GoalTracker):
    """Builds the same 8-element observation vector the model was
    trained on, using real odometry position + live sensor readings.

    This normalization must exactly match nav_env.py's _get_obs() --
    a mismatched formula here means the trained network receives
    inputs on a different scale than it learned on, which produces
    unpredictable, erratic behavior even though the model itself is
    fine (this was a real bug in an earlier version of this file)."""
    dx = goal.goal_x - x
    dy = goal.goal_y - y
    dist = math.hypot(dx, dy)

    dx_norm = np.clip(dx / ARENA_WIDTH_M, -1, 1)
    dy_norm = np.clip(dy / ARENA_LENGTH_M, -1, 1)
    max_possible_dist = math.hypot(ARENA_WIDTH_M, ARENA_LENGTH_M)
    dist_norm = np.clip(dist / max_possible_dist, 0, 1)

    angle_to_goal = math.atan2(dy, dx) - heading
    angle_to_goal = wrap_angle(angle_to_goal) / math.pi

    # gpiozero's DistanceSensor.distance returns the reading in real
    # METERS, not a normalized [0,1] fraction -- must divide by
    # MAX_SENSOR_RANGE_M here to match nav_env.py's _simulate_sensor(),
    # which explicitly normalizes to [0,1]. Missing this was a real
    # bug in an earlier version: the model received raw meter values
    # where it expected normalized fractions, a scale mismatch large
    # enough to produce genuinely erratic navigation.
    sensor_front = np.clip(get_filtered_distance(middle_sensor) / MAX_SENSOR_RANGE_M, 0, 1)
    sensor_left = np.clip(get_filtered_distance(left_sensor) / MAX_SENSOR_RANGE_M, 0, 1)
    sensor_right = np.clip(get_filtered_distance(right_sensor) / MAX_SENSOR_RANGE_M, 0, 1)

    heading_norm = heading / math.pi

    obs = np.array(
        [dx_norm, dy_norm, dist_norm, angle_to_goal, sensor_front, sensor_left, sensor_right, heading_norm],
        dtype=np.float32,
    )
    return obs


# ------------------------------------------------------------------ #
# Stall/slip detection -- wheel encoders measure WHEEL ROTATION, not
# chassis position. They assume the wheels roll without slipping, so if
# the robot gets mechanically stuck (e.g. wedged against an edge, or in
# contact with an obstacle) the wheels can keep spinning and odometry
# will confidently report forward progress that never actually
# happened -- encoders alone can't tell the difference. This
# cross-checks odometry's claimed progress against the front ultrasonic
# sensor, which should show a corresponding decrease if the robot is
# really moving toward whatever is ahead of it.
#
# IMPORTANT: a stall warning on one step means the position estimate
# was corrupted AT THAT MOMENT -- but odometry is cumulative (each
# step's position builds on the last), so a single bad step can leave
# the position estimate unreliable for several steps afterward too,
# not just the one step that triggered the warning. A real trial
# showed exactly this: the warning fired on step 19, then step 20
# immediately triggered GOAL REACHED using a position built right on
# top of that same flagged, unverified data. STALL_DISTRUST_STEPS
# fixes this by keeping goal-reached suppressed for a few steps after
# a stall fires, not just the triggering step itself.
# ------------------------------------------------------------------ #
STALL_CHECK_WINDOW = 5        # steps of history to compare against
STALL_MIN_ODOM_PROGRESS_M = 0.08  # only check once odometry claims at least this much recent movement
STALL_SENSOR_CONFIRM_RATIO = 0.3  # front sensor should close by at least this fraction of the claimed progress
STALL_DISTRUST_STEPS = 3      # after a stall fires, keep distrusting position for goal-reached purposes for this many additional steps


def check_stall(history, x, y, front_m):
    """history: deque of (x, y, front_m) from the last STALL_CHECK_WINDOW
    steps. Returns True if odometry claims meaningful forward progress that
    the front sensor doesn't corroborate."""
    if len(history) < STALL_CHECK_WINDOW:
        return False
    old_x, old_y, old_front = history[0]
    odom_progress = math.hypot(x - old_x, y - old_y)
    if odom_progress < STALL_MIN_ODOM_PROGRESS_M:
        return False
    front_decrease = old_front - front_m
    return front_decrease < odom_progress * STALL_SENSOR_CONFIRM_RATIO


def choose_safest_turn(front_m, left_m, right_m):
    """Reactive evasion logic: steer AWAY from whichever single sensor
    reads the closest obstacle, rather than picking whichever direction
    has the highest OVERALL reading. A real trial exposed the flaw in
    the old approach: front=1.49m, left=1.37m, right=0.05m -- something
    was critically close on the right, but the old logic picked
    "forward" anyway since front had the highest raw value. That let
    the chassis' own width clip the obstacle while driving past it,
    since the robot's body isn't a single point -- something very
    close on ONE side is dangerous regardless of how open the other
    readings are."""
    closest = min(front_m, left_m, right_m)
    if closest == right_m:
        return 1  # turn left -- away from the right-side threat
    elif closest == left_m:
        return 2  # turn right -- away from the left-side threat
    else:
        # front is the closest/most urgent -- turn toward whichever side is more open
        return 1 if left_m >= right_m else 2


def execute_action(action):
    if action == 0:  # forward -- both wheels same direction, use forward-specific calibration
        left_power = min(MOTOR_POWER * FORWARD_LEFT_SCALE, 1.0)
        right_power = min(MOTOR_POWER * FORWARD_RIGHT_SCALE, 1.0)
        motor_a.forward(left_power)
        motor_b.forward(right_power)
        step_duration = STEP_DURATION_S
    elif action == 1:  # turn left -- wheels opposite directions, use turn-specific calibration + shorter duration
        left_power = min(MOTOR_POWER * TURN_LEFT_SCALE, 1.0)
        right_power = min(MOTOR_POWER * TURN_RIGHT_SCALE, 1.0)
        motor_a.backward(left_power)
        motor_b.forward(right_power)
        step_duration = TURN_STEP_DURATION_S
    elif action == 2:  # turn right
        left_power = min(MOTOR_POWER * TURN_LEFT_SCALE, 1.0)
        right_power = min(MOTOR_POWER * TURN_RIGHT_SCALE, 1.0)
        motor_a.forward(left_power)
        motor_b.backward(right_power)
        step_duration = TURN_STEP_DURATION_S
    elif action == 3:  # stop
        motor_a.stop()
        motor_b.stop()
        step_duration = STEP_DURATION_S

    time.sleep(step_duration)
    motor_a.stop()
    motor_b.stop()


def main():
    device = torch.device("cpu")  # Pi has no CUDA; inference is light enough for CPU
    agent = DQNAgent(state_dim=8, action_dim=4, device=device)
    agent.load(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")

    odom = Odometry()
    goal = GoalTracker(GOAL_DISTANCE_M, GOAL_ANGLE_DEG)
    print(f"Goal set: {GOAL_DISTANCE_M}m at {GOAL_ANGLE_DEG} degrees from start.")
    print("Starting in 9 seconds -- place robot at its start position now...")
    time.sleep(9)
    odom.reset()

    stall_history = deque(maxlen=STALL_CHECK_WINDOW)
    steps_since_stall = None  # None = position currently trusted; otherwise counts up from 0 since the last stall
    decision_times = []  # per-step policy inference time in ms, for computational-cost comparison against A*

    try:
        for step in range(MAX_STEPS):
            x, y, heading = odom.update()
            obs = get_observation(x, y, heading, goal)

            front_m = obs[4] * MAX_SENSOR_RANGE_M
            left_m = obs[5] * MAX_SENSOR_RANGE_M
            right_m = obs[6] * MAX_SENSOR_RANGE_M
            min_sensor_m = min(front_m, left_m, right_m)

            action_names = {0: "forward", 1: "turn left", 2: "turn right", 3: "stop"}
            pulse_info = f"L/R pulses: {odom.left_encoder.count:6d}/{odom.right_encoder.count:6d}"

            # check safety BEFORE committing to an action, not after --
            # if something's too close, override the DQN's decision
            # with a reactive evasive turn instead of dead-stopping and
            # ending the run. Once clear, the DQN resumes normal control.
            if min_sensor_m < 0.15:  # matches real robot collision radius (~0.21m half-diagonal) with margin
                action = choose_safest_turn(front_m, left_m, right_m)
                print(
                    f"Step {step:3d} | EVASIVE: {action_names[action]:10s} (obstacle at {min_sensor_m:.2f}m) | "
                    f"pos: ({x:6.3f}, {y:6.3f}) | "
                    f"sensors F/L/R: {front_m:.2f}/{left_m:.2f}/{right_m:.2f}m | {pulse_info}"
                )
            else:
                # Time just the policy's decision call, not sensor reads or
                # motor execution -- this is the actual per-step computational
                # cost being compared against A*'s planning/waypoint-following.
                decision_start = time.perf_counter()
                action = agent.select_action(obs, greedy=True)
                decision_time_ms = (time.perf_counter() - decision_start) * 1000
                decision_times.append(decision_time_ms)
                print(
                    f"Step {step:3d} | action: {action_names[action]:10s} | "
                    f"pos: ({x:6.3f}, {y:6.3f}) | "
                    f"dist to goal: {goal.dist_to_goal(x, y):5.2f}m | "
                    f"sensors F/L/R: {front_m:.2f}/{left_m:.2f}/{right_m:.2f}m | "
                    f"decision: {decision_time_ms:.2f}ms | {pulse_info}"
                )

            execute_action(action)
            x, y, heading = odom.update()  # refresh pose after moving

            stall_history.append((x, y, front_m))
            stalled = check_stall(stall_history, x, y, front_m)

            if stalled:
                steps_since_stall = 0
                print(
                    "         WARNING: odometry shows forward progress the front sensor doesn't "
                    "confirm -- likely wheel slip or the robot is mechanically stuck. Not trusting "
                    "this position for goal-reached this step."
                )
            elif steps_since_stall is not None:
                steps_since_stall += 1
                if steps_since_stall >= STALL_DISTRUST_STEPS:
                    steps_since_stall = None  # distrust window expired -- back to trusting position normally

            position_trusted = steps_since_stall is None

            if goal.reached_goal(x, y):
                if position_trusted:
                    print("\nGOAL REACHED.")
                    break
                else:
                    print(
                        f"         (within goal threshold, but position still distrusted from a "
                        f"recent stall -- {STALL_DISTRUST_STEPS - steps_since_stall} more step(s) "
                        f"before goal-reached will be trusted again)"
                    )
        else:
            print("\nMax steps reached without finding goal.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()
        if decision_times:
            avg_decision_ms = sum(decision_times) / len(decision_times)
            print(f"Avg decision time: {avg_decision_ms:.2f}ms over {len(decision_times)} policy decisions")
        print("Motors stopped.")


if __name__ == "__main__":
    main()
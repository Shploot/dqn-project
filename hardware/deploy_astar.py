"""
deploy_astar.py

Runs the CLASSICAL A* baseline on the real robot -- same sensors,
motors, and encoder-based odometry as deploy.py, but replacing the
DQN with AStarPlanner/AStarAgent (from astar_planner.py) for
navigation decisions.

Purpose: isolate whether erratic deploy.py behavior comes from the
DQN policy itself, or from the underlying hardware pipeline (odometry
drift, motor mismatch) that both methods share. If this also behaves
erratically, the problem is the hardware pipeline, not the model.

Since the real corridor has no known obstacles ahead of time, A* here
computes ONE straight-ish path at the very start (no live obstacle
mapping/replanning -- that's a genuine, honest limitation of a classical
planner without a live map) and then follows those waypoints using
LIVE odometry each step. The ultrasonic safety-stop backstop is kept
active regardless, for physical safety.

Run on the Pi:
    python3 deploy_astar.py
Press Ctrl+C to stop immediately (also stops motors).
"""

import math
import time
from collections import deque

import numpy as np
from gpiozero import Device, Motor, DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory

from astar_planner import AStarPlanner, AStarAgent
from odometry import Odometry

Device.pin_factory = LGPIOFactory()

# ------------------------------------------------------------------ #
STEP_DURATION_S = 0.3
MOTOR_POWER = 0.5

# TURN_STEP_DURATION_S: shortened from STEP_DURATION_S to match measured
# real turn magnitude now that odometry.py's heading sign bug is fixed --
# see deploy.py for the full history/reasoning. Keep in sync with deploy.py.
TURN_STEP_DURATION_S = 0.157

# same wheel calibration as deploy.py -- keep these in sync
TURN_LEFT_SCALE = 1.1104
TURN_RIGHT_SCALE = 0.9234
FORWARD_LEFT_SCALE = 0.9572
FORWARD_RIGHT_SCALE = 1.0468

# must match deploy.py / train_corridor.py for a fair, consistent comparison
ARENA_WIDTH_M = 1.83
ARENA_LENGTH_M = 0.91
ROBOT_RADIUS_M = 0.21
GOAL_RADIUS_M = 0.25
MOVE_SPEED_M = 0.10   # used by AStarAgent's lookahead distance calc, not actual motor speed
TURN_SPEED_RAD = 0.30 # kept only for AStarAgent's lookahead ratio; NOT used for angle tolerance (see below)

# AStarAgent's default angle_tolerance is derived from TURN_SPEED_RAD above,
# which is the SIMULATED per-step turn angle -- not how far a single real
# TURN_STEP_DURATION_S burst actually rotates the robot. If angle_tolerance
# is left at the sim-derived value (~10 degrees) while a real burst swings
# much wider, every turn overshoots the tolerance window and the robot
# oscillates correcting back and forth -- the same failure mode already
# seen with the DQN in deploy.py.
#
# single_turn_test.py measured ~25.0 degrees/burst turning left, ~25.1
# turning right, directly at the current TURN_STEP_DURATION_S=0.157s (see
# deploy.py) -- confirmed stable and symmetric across repeated runs, using
# the rewritten Encoder (full 4x decoding, correct heading sign -- see
# odometry.py) which reads both wheels reliably; earlier measurements
# taken with the older single-channel Encoder are not trustworthy and were
# superseded. Re-run single_turn_test.py and update these two numbers if
# you change STEP_DURATION_S, MOTOR_POWER, or the turn scale constants
# above.
MEASURED_LEFT_BURST_DEG = 25.0
MEASURED_RIGHT_BURST_DEG = 25.1
ANGLE_TOLERANCE_RAD = math.radians(max(MEASURED_LEFT_BURST_DEG, MEASURED_RIGHT_BURST_DEG) * 1.5)

GOAL_DISTANCE_M = 1.2  # was 1.4 -- that landed right at/past the arena's usable travel span (1.41m) once the robot's real 0.21m collision radius is accounted for; 1.2m leaves comfortable margin on both ends
GOAL_ANGLE_DEG = 0.0

MAX_STEPS = 150
MAX_SENSOR_RANGE_M = 2.0

# ------------------------------------------------------------------ #
motor_a = Motor(forward=6, backward=5, pwm=True, enable=12)  # left
motor_b = Motor(forward=20, backward=21, pwm=True, enable=13)  # right

left_sensor = DistanceSensor(echo=4, trigger=17, max_distance=MAX_SENSOR_RANGE_M)
middle_sensor = DistanceSensor(echo=18, trigger=27, max_distance=MAX_SENSOR_RANGE_M)
right_sensor = DistanceSensor(echo=22, trigger=23, max_distance=MAX_SENSOR_RANGE_M)


def get_filtered_distance(sensor, num_samples=5):
    readings = sorted(sensor.distance for _ in range(num_samples))
    return readings[len(readings) // 2]


# ------------------------------------------------------------------ #
# Stall/slip detection -- see deploy.py for the full rationale. Wheel
# encoders measure wheel rotation, not chassis position, so a robot
# mechanically stuck (e.g. wedged against an edge, or in contact with
# an obstacle) can have its wheels spinning while odometry confidently
# reports forward progress that never happened -- which here would
# make AStarAgent think it reached waypoints (and eventually the goal)
# it never actually reached. Cross-checks odometry's claimed progress
# against the front sensor, which should show a corresponding decrease
# if the robot is really moving forward.
#
# IMPORTANT: a stall warning on one step means the position estimate
# was corrupted AT THAT MOMENT -- but odometry is cumulative, so a
# single bad step can leave the estimate unreliable for several steps
# afterward too, not just the one that triggered it. STALL_DISTRUST_STEPS
# keeps goal-reached suppressed for a few steps after a stall fires,
# not just the triggering step -- ported from deploy.py, where a real
# trial showed a stall fire on one step and GOAL REACHED fire on the
# very next, using that same flagged, unverified position.
# ------------------------------------------------------------------ #
STALL_CHECK_WINDOW = 5
STALL_MIN_ODOM_PROGRESS_M = 0.08
STALL_SENSOR_CONFIRM_RATIO = 0.3
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


class FakeEnv:
    """Minimal stand-in exposing exactly what AStarPlanner/AStarAgent
    need, so the real robot's live odometry can drive the same
    planner/follower code already validated in simulation."""

    def __init__(self):
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_heading = 0.0
        self.goal_pos = np.array([0.0, 0.0])
        self.arena_width = ARENA_WIDTH_M
        self.arena_length = ARENA_LENGTH_M
        self.obstacles = np.zeros((0, 2))  # none known a priori -- real corridor is unmapped
        self.robot_radius = ROBOT_RADIUS_M
        self.obstacle_radius = 0.10
        self.goal_radius = GOAL_RADIUS_M
        self.move_speed = MOVE_SPEED_M
        self.turn_speed = TURN_SPEED_RAD

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi


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
    elif action == 3:
        motor_a.stop()
        motor_b.stop()
        step_duration = STEP_DURATION_S

    time.sleep(step_duration)
    motor_a.stop()
    motor_b.stop()


def choose_safest_turn(front_m, left_m, right_m):
    """Reactive evasion logic: steer AWAY from whichever single sensor
    reads the closest obstacle, rather than picking whichever direction
    has the highest OVERALL reading. Ported from deploy.py, where a real
    trial exposed the flaw in the old approach: front=1.49m, left=1.37m,
    right=0.05m -- something was critically close on the right, but the
    old logic picked "forward" anyway since front had the highest raw
    value. That let the chassis' own width clip the obstacle while
    driving past it, since the robot's body isn't a single point --
    something very close on ONE side is dangerous regardless of how
    open the other readings are."""
    closest = min(front_m, left_m, right_m)
    if closest == right_m:
        return 1  # turn left -- away from the right-side threat
    elif closest == left_m:
        return 2  # turn right -- away from the left-side threat
    else:
        # front is the closest/most urgent -- turn toward whichever side is more open
        return 1 if left_m >= right_m else 2


def main():
    odom = Odometry()
    odom.reset()

    fake_env = FakeEnv()

    # Coordinate frame fix: the simulation (and AStarPlanner's wall
    # checks) assume (0,0) is the arena's bottom-left CORNER. Real
    # odometry instead treats (0,0) as wherever the robot physically
    # starts -- typically somewhere in the middle of the corridor, not
    # jammed against a wall. Offset odometry readings by this amount
    # so they land in a valid, sensible spot within the arena's
    # coordinate frame: near one end (lengthwise), centered (widthwise).
    start_offset = np.array([ROBOT_RADIUS_M + 0.05, ARENA_LENGTH_M / 2.0])

    goal_angle_rad = math.radians(GOAL_ANGLE_DEG)
    fake_env.robot_pos = start_offset.copy()
    fake_env.goal_pos = start_offset + np.array([
        GOAL_DISTANCE_M * math.cos(goal_angle_rad),
        GOAL_DISTANCE_M * math.sin(goal_angle_rad),
    ])

    agent = AStarAgent(fake_env, grid_resolution=0.1, angle_tolerance=ANGLE_TOLERANCE_RAD)
    found_path = agent.reset_plan()
    print(f"A* planned path found: {found_path} ({len(agent.waypoints)} waypoints)")
    if not found_path:
        print("No valid path found -- goal may be unreachable given arena/robot size. Exiting.")
        return

    print(f"Goal set: {GOAL_DISTANCE_M}m at {GOAL_ANGLE_DEG} degrees from start.")
    print("Starting in 9 seconds -- place robot at its start position now...")
    time.sleep(9)

    action_names = {0: "forward", 1: "turn left", 2: "turn right", 3: "stop"}
    stall_history = deque(maxlen=STALL_CHECK_WINDOW)
    steps_since_stall = None  # None = position currently trusted; otherwise counts up from 0 since the last stall
    decision_times = []  # per-step waypoint-following decision time in ms

    try:
        for step in range(MAX_STEPS):
            x, y, heading = odom.update()
            fake_env.robot_pos = start_offset + np.array([x, y])
            fake_env.robot_heading = heading

            front_m = get_filtered_distance(middle_sensor)
            left_m = get_filtered_distance(left_sensor)
            right_m = get_filtered_distance(right_sensor)
            min_sensor_m = min(front_m, left_m, right_m)

            dist_to_goal = np.linalg.norm(fake_env.goal_pos - fake_env.robot_pos)

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

            # raw cumulative pulse counts alongside higher-precision pos --
            # makes a stalled/slipping robot's true wheel activity visible,
            # instead of only the (possibly misleading) x/y already derived
            # from those same pulses.
            pulse_info = f"L/R pulses: {odom.left_encoder.count:6d}/{odom.right_encoder.count:6d}"

            # check safety BEFORE committing to A*'s planned action --
            # if something's too close, override with reactive evasion
            # instead of dead-stopping and ending the run. Once clear,
            # A* resumes following its planned waypoints.
            if min_sensor_m < 0.15:
                action = choose_safest_turn(front_m, left_m, right_m)
                print(
                    f"Step {step:3d} | EVASIVE: {action_names[action]:10s} (obstacle at {min_sensor_m:.2f}m) | "
                    f"pos: ({x:6.3f}, {y:6.3f}) | "
                    f"sensors F/L/R: {front_m:.2f}/{left_m:.2f}/{right_m:.2f}m | {pulse_info}"
                )
            else:
                # Time just the waypoint-following decision call, not sensor
                # reads or motor execution -- comparable to DQN's decision timing.
                decision_start = time.perf_counter()
                action = agent.select_action()
                decision_time_ms = (time.perf_counter() - decision_start) * 1000
                decision_times.append(decision_time_ms)
                print(
                    f"Step {step:3d} | action: {action_names[action]:10s} | "
                    f"pos: ({x:6.3f}, {y:6.3f}) | "
                    f"dist to goal: {dist_to_goal:5.2f}m | "
                    f"sensors F/L/R: {front_m:.2f}/{left_m:.2f}/{right_m:.2f}m | "
                    f"waypoint {agent.waypoint_idx}/{len(agent.waypoints)} | "
                    f"decision: {decision_time_ms:.3f}ms | {pulse_info}"
                )

            if action == 3 and agent.waypoint_idx >= len(agent.waypoints):
                if position_trusted:
                    print("\nGOAL REACHED (per A* waypoint completion).")
                    break
                else:
                    print(
                        f"         (waypoints complete, but position still distrusted from a "
                        f"recent stall -- {STALL_DISTRUST_STEPS - steps_since_stall} more step(s) "
                        f"before goal-reached will be trusted again)"
                    )

            execute_action(action)
        else:
            print("\nMax steps reached without finishing path.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        motor_a.stop()
        motor_b.stop()
        odom.close()
        if decision_times:
            avg_decision_ms = sum(decision_times) / len(decision_times)
            print(f"Avg decision time: {avg_decision_ms:.3f}ms over {len(decision_times)} planning decisions")
        print("Motors stopped.")


if __name__ == "__main__":
    main()
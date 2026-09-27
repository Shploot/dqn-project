"""
nav_env.py

A custom Gymnasium environment simulating a differential-drive robot
(matching the real robot: two-wheeled, ultrasonic distance sensors)
navigating to a goal point while avoiding obstacles.

Supports either a square arena (arena_size) or a rectangular arena
(arena_width x arena_length) -- e.g. a narrow corridor test track.
If arena_width/arena_length are not given, falls back to a square
arena of arena_size x arena_size (original behavior, unchanged).

State (observation), 8 floats:
    [0] dx_to_goal      - x distance to goal, normalized to [-1, 1]
    [1] dy_to_goal      - y distance to goal, normalized to [-1, 1]
    [2] dist_to_goal    - straight-line distance to goal, normalized [0, 1]
    [3] angle_to_goal   - angle to goal relative to heading, normalized [-1, 1] (-1=behind, 0=ahead)
    [4] sensor_front    - simulated ultrasonic reading, normalized [0, 1] (1 = max range / clear)
    [5] sensor_left     - simulated ultrasonic reading, front-left diagonal
    [6] sensor_right    - simulated ultrasonic reading, front-right diagonal
    [7] heading         - robot's heading angle, normalized [-1, 1]

Actions, discrete (4):
    0 = move forward
    1 = turn left (in place)
    2 = turn right (in place)
    3 = stop / do nothing

Episode ends when:
    - goal reached (success)
    - collision with obstacle (failure)
    - max steps reached (timeout)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NavEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        arena_size=10.0,        # meters, used as both width and length if arena_width/arena_length not given
        arena_width=None,       # meters, x-axis extent -- overrides arena_size if given
        arena_length=None,      # meters, y-axis extent -- overrides arena_size if given
        num_obstacles=5,
        obstacle_radius=0.4,
        robot_radius=0.15,
        goal_radius=0.3,
        max_sensor_range=3.0,   # HC-SR04-ish effective range for obstacle avoidance
        move_speed=0.3,         # meters per step when moving forward
        turn_speed=0.35,        # radians per step when turning
        max_steps=300,
        sensor_noise_std=0.02,  # domain randomization: sensor noise (helps sim-to-real)
        render_mode=None,
    ):
        super().__init__()

        # rectangular arena support: width = x-axis extent, length = y-axis extent.
        # falls back to a square arena_size x arena_size if not specified, so
        # existing code (astar_planner.py, compare.py) is unaffected.
        self.arena_width = arena_width if arena_width is not None else arena_size
        self.arena_length = arena_length if arena_length is not None else arena_size
        self.arena_size = arena_size  # kept for backward compatibility with any code reading it directly

        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.robot_radius = robot_radius
        self.goal_radius = goal_radius
        self.max_sensor_range = max_sensor_range
        self.move_speed = move_speed
        self.turn_speed = turn_speed
        self.max_steps = max_steps
        self.sensor_noise_std = sensor_noise_std
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # sensor angles relative to heading (front, front-left, front-right)
        self._sensor_angles = [0.0, np.deg2rad(35), -np.deg2rad(35)]

        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------ #
    # Core Gym API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # margin scales down for small/narrow arenas so there's still
        # room to place the robot/goal without the margin eating the
        # whole space (a 1.0m margin doesn't work in a 0.9m-wide corridor).
        # ALWAYS at least robot_radius plus a small buffer, though --
        # otherwise a large robot in a small arena could be placed
        # already overlapping a wall at the start of an episode.
        margin_x = max(min(1.0, self.arena_width * 0.15), self.robot_radius * 1.2)
        margin_y = max(min(1.0, self.arena_length * 0.15), self.robot_radius * 1.2)

        self.robot_pos = np.array([
            self._rng.uniform(margin_x, self.arena_width - margin_x),
            self._rng.uniform(margin_y, self.arena_length - margin_y),
        ])
        self.robot_heading = self._rng.uniform(-np.pi, np.pi)

        # place goal, ensuring it's not on top of the robot.
        # IMPORTANT: capped at a fixed number of attempts -- an
        # uncapped "while True" here could spin for a very long time
        # (or effectively forever) in a small/narrow arena if an
        # unlucky sequence of random draws keeps landing too close to
        # the robot. If no valid spot is found within the cap, we
        # relax to just taking the farthest of the attempted
        # candidates instead of looping indefinitely.
        min_goal_sep = min(2.0, max(self.arena_width, self.arena_length) * 0.4)
        best_candidate = None
        best_dist = -1.0
        for _ in range(200):
            candidate = np.array([
                self._rng.uniform(margin_x, self.arena_width - margin_x),
                self._rng.uniform(margin_y, self.arena_length - margin_y),
            ])
            dist = np.linalg.norm(candidate - self.robot_pos)
            if dist > best_dist:
                best_dist = dist
                best_candidate = candidate
            if dist > min_goal_sep:
                break
        self.goal_pos = best_candidate

        # place obstacles, avoiding robot start and goal
        self.obstacles = []
        attempts = 0
        min_obs_sep = min(1.2, max(self.arena_width, self.arena_length) * 0.25)
        min_goal_obs_sep = min(1.0, max(self.arena_width, self.arena_length) * 0.2)
        while len(self.obstacles) < self.num_obstacles and attempts < 200:
            attempts += 1
            candidate = np.array([
                self._rng.uniform(margin_x, self.arena_width - margin_x),
                self._rng.uniform(margin_y, self.arena_length - margin_y),
            ])
            if np.linalg.norm(candidate - self.robot_pos) < min_obs_sep:
                continue
            if np.linalg.norm(candidate - self.goal_pos) < min_goal_obs_sep:
                continue
            self.obstacles.append(candidate)
        self.obstacles = np.array(self.obstacles) if self.obstacles else np.zeros((0, 2))

        self.steps = 0
        self._prev_dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1

        if action == 0:  # forward
            new_pos = self.robot_pos + self.move_speed * np.array(
                [np.cos(self.robot_heading), np.sin(self.robot_heading)]
            )
            self.robot_pos = new_pos
        elif action == 1:  # turn left
            self.robot_heading += self.turn_speed
        elif action == 2:  # turn right
            self.robot_heading -= self.turn_speed
        elif action == 3:  # stop
            pass

        self.robot_heading = self._wrap_angle(self.robot_heading)
        self.robot_pos = np.clip(
            self.robot_pos, 0.0, [self.arena_width, self.arena_length]
        )

        dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        collided = self._check_collision()
        reached_goal = dist_to_goal < self.goal_radius

        reward = self._compute_reward(dist_to_goal, collided, reached_goal, action)
        self._prev_dist_to_goal = dist_to_goal

        terminated = bool(collided or reached_goal)
        truncated = bool(self.steps >= self.max_steps)

        obs = self._get_obs()
        info = {
            "collided": collided,
            "reached_goal": reached_goal,
            "dist_to_goal": dist_to_goal,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_reward(self, dist_to_goal, collided, reached_goal, action):
        if reached_goal:
            return 100.0
        if collided:
            return -100.0

        # shaped reward: progress toward goal
        progress = self._prev_dist_to_goal - dist_to_goal
        reward = progress * 10.0

        # small penalty per step to encourage efficient paths
        reward -= 0.05

        # small penalty for turning in place too much (encourages committing to forward motion)
        if action in (1, 2):
            reward -= 0.02

        return reward

    def _check_collision(self):
        # walls (checked per-axis since width/length can differ)
        if (
            self.robot_pos[0] <= self.robot_radius
            or self.robot_pos[0] >= self.arena_width - self.robot_radius
            or self.robot_pos[1] <= self.robot_radius
            or self.robot_pos[1] >= self.arena_length - self.robot_radius
        ):
            return True

        if len(self.obstacles) == 0:
            return False
        dists = np.linalg.norm(self.obstacles - self.robot_pos, axis=1)
        if np.any(dists < (self.obstacle_radius + self.robot_radius)):
            return True
        return False

    def _simulate_sensor(self, angle_offset):
        """Cast a ray from the robot at (heading + angle_offset), return
        normalized distance to nearest obstacle/wall, clipped to max range."""
        angle = self.robot_heading + angle_offset
        direction = np.array([np.cos(angle), np.sin(angle)])

        min_dist = self.max_sensor_range
        arena_extent = [self.arena_width, self.arena_length]

        # distance to walls along this ray (per axis, using that axis's extent)
        for axis in range(2):
            d = direction[axis]
            if abs(d) > 1e-6:
                if d > 0:
                    wall_dist = (arena_extent[axis] - self.robot_pos[axis]) / d
                else:
                    wall_dist = (0.0 - self.robot_pos[axis]) / d
                if wall_dist > 0:
                    min_dist = min(min_dist, wall_dist)

        # distance to each obstacle along this ray (simple circle-ray check)
        for obs_pos in self.obstacles:
            to_obs = obs_pos - self.robot_pos
            proj = np.dot(to_obs, direction)
            if proj < 0:
                continue
            closest_point = self.robot_pos + proj * direction
            perp_dist = np.linalg.norm(obs_pos - closest_point)
            if perp_dist <= self.obstacle_radius:
                chord = np.sqrt(max(self.obstacle_radius**2 - perp_dist**2, 0))
                hit_dist = proj - chord
                if hit_dist > 0:
                    min_dist = min(min_dist, hit_dist)

        min_dist = np.clip(min_dist, 0, self.max_sensor_range)
        # domain randomization: add sensor noise so the policy doesn't overfit
        # to a perfectly clean simulated sensor (helps sim-to-real transfer)
        noisy = min_dist + self._rng.normal(0, self.sensor_noise_std)
        noisy = np.clip(noisy, 0, self.max_sensor_range)
        return noisy / self.max_sensor_range  # normalize to [0, 1]

    def _get_obs(self):
        delta = self.goal_pos - self.robot_pos
        dx = np.clip(delta[0] / self.arena_width, -1, 1)
        dy = np.clip(delta[1] / self.arena_length, -1, 1)
        dist = np.linalg.norm(delta)
        max_possible_dist = np.hypot(self.arena_width, self.arena_length)
        dist_norm = np.clip(dist / max_possible_dist, 0, 1)

        angle_to_goal = np.arctan2(delta[1], delta[0]) - self.robot_heading
        angle_to_goal = self._wrap_angle(angle_to_goal) / np.pi  # [-1, 1]

        sensor_front = self._simulate_sensor(self._sensor_angles[0])
        sensor_left = self._simulate_sensor(self._sensor_angles[1])
        sensor_right = self._simulate_sensor(self._sensor_angles[2])

        heading_norm = self.robot_heading / np.pi

        obs = np.array(
            [dx, dy, dist_norm, angle_to_goal, sensor_front, sensor_left, sensor_right, heading_norm],
            dtype=np.float32,
        )
        return obs

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

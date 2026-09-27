"""
astar_planner.py

Classical path-planning baseline using A* search, built to run in the
exact same NavEnv arena as the DQN agent so the two approaches can be
compared head-to-head on identical metrics (success rate, time-to-goal,
path efficiency).

Approach:
    1. Discretize the continuous arena into a grid.
    2. Mark grid cells as blocked if they fall inside an obstacle
       (inflated by the robot's radius, so the planned path keeps a
       safe margin -- same idea as "configuration space" in planning).
    3. Run A* from the robot's start cell to the goal cell.
    4. Convert the resulting grid path into a sequence of the same
       discrete actions the DQN uses (forward / turn left / turn right),
       so both methods are evaluated by literally replaying actions
       through the same env.step() and reward function.
"""

import heapq
import math

import numpy as np

from nav_env import NavEnv


class AStarPlanner:
    def __init__(self, env: NavEnv, grid_resolution=0.25):
        """
        env: a NavEnv instance (used for arena size / obstacle layout)
        grid_resolution: size of each grid cell in meters. Smaller =
            more precise paths but slower planning.
        """
        self.env = env
        self.res = grid_resolution
        # grid dimensions per-axis, since arena_width and arena_length
        # can differ (e.g. a narrow corridor) -- a single grid_size
        # value silently breaks non-square arenas: it undersizes one
        # axis and oversizes the other, so bounds/blocked checks stop
        # matching the arena the robot is actually in.
        self.grid_w = int(np.ceil(env.arena_width / grid_resolution))
        self.grid_h = int(np.ceil(env.arena_length / grid_resolution))

    # ------------------------------------------------------------------ #
    # Grid <-> world conversions
    # ------------------------------------------------------------------ #

    def world_to_grid(self, pos):
        gx = int(pos[0] / self.res)
        gy = int(pos[1] / self.res)
        gx = np.clip(gx, 0, self.grid_w - 1)
        gy = np.clip(gy, 0, self.grid_h - 1)
        return (gx, gy)

    def grid_to_world(self, cell):
        # center of the cell
        x = (cell[0] + 0.5) * self.res
        y = (cell[1] + 0.5) * self.res
        return np.array([x, y])

    def _is_blocked(self, cell):
        world_pos = self.grid_to_world(cell)
        # inflate obstacles by robot radius so planned path keeps clearance
        safety_margin = self.env.robot_radius + 0.05
        for obs_pos in self.env.obstacles:
            if np.linalg.norm(world_pos - obs_pos) < (self.env.obstacle_radius + safety_margin):
                return True
        # wall margin, checked per-axis since width/length can differ
        if (
            world_pos[0] <= self.env.robot_radius
            or world_pos[0] >= self.env.arena_width - self.env.robot_radius
            or world_pos[1] <= self.env.robot_radius
            or world_pos[1] >= self.env.arena_length - self.env.robot_radius
        ):
            return True
        return False

    # ------------------------------------------------------------------ #
    # A* search
    # ------------------------------------------------------------------ #

    def plan(self, start_pos, goal_pos):
        """Returns a list of world-space waypoints from start to goal,
        or None if no path exists."""
        start = self.world_to_grid(start_pos)
        goal = self.world_to_grid(goal_pos)

        if self._is_blocked(start) or self._is_blocked(goal):
            return None

        # 8-connected grid (allows diagonal movement)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = [(heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            _, cost, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            if current in visited:
                continue
            visited.add(current)

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)

                if not (0 <= neighbor[0] < self.grid_w and 0 <= neighbor[1] < self.grid_h):
                    continue
                if self._is_blocked(neighbor):
                    continue

                step_cost = math.hypot(dx, dy)  # 1.0 for straight, sqrt(2) for diagonal
                tentative_g = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None  # no path found

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return [self.grid_to_world(cell) for cell in path]


class AStarAgent:
    """Wraps AStarPlanner to produce the same discrete actions
    (0=forward, 1=turn left, 2=turn right, 3=stop) that the DQN uses,
    so it can be dropped into the same evaluation loop."""

    def __init__(self, env: NavEnv, grid_resolution=0.25, angle_tolerance=None):
        self.env = env
        self.planner = AStarPlanner(env, grid_resolution)
        # angle_tolerance must be wider than half a turn-step, or the
        # robot oscillates forever trying to face a waypoint exactly
        # (overshoots past the target angle every turn, never settling).
        self.angle_tolerance = angle_tolerance if angle_tolerance is not None else env.turn_speed * 0.6
        # how far ahead of the robot to look when picking a waypoint to
        # steer toward -- lets it skip past waypoints it has already
        # effectively passed instead of overshooting and backtracking
        self.lookahead_dist = env.move_speed * 1.5
        self.waypoints = []
        self.waypoint_idx = 0

    def reset_plan(self):
        """Call after env.reset() to compute a fresh path for the new
        start/goal/obstacle layout."""
        path = self.planner.plan(self.env.robot_pos, self.env.goal_pos)
        self.waypoints = path if path is not None else []
        self.waypoint_idx = 1  # skip index 0 (that's the start position itself)
        return path is not None

    def _advance_waypoint(self):
        """Skip forward through any waypoints that are already within
        lookahead distance, so small overshoots don't cause backtracking."""
        while self.waypoint_idx < len(self.waypoints) - 1:
            dist = np.linalg.norm(self.waypoints[self.waypoint_idx] - self.env.robot_pos)
            if dist < self.lookahead_dist:
                self.waypoint_idx += 1
            else:
                break

    def select_action(self, obs=None):
        """Returns the next discrete action to follow the planned path.
        obs is accepted (unused) so this has the same call signature as
        DQNAgent.select_action, making the eval loop identical for both."""
        if self.waypoint_idx >= len(self.waypoints):
            return 3  # stop, path complete or no path

        self._advance_waypoint()

        # final waypoint uses a tighter arrival radius (it's the goal)
        target = self.waypoints[self.waypoint_idx]
        dist = np.linalg.norm(target - self.env.robot_pos)
        is_final = self.waypoint_idx == len(self.waypoints) - 1
        arrival_radius = self.env.goal_radius * 0.8 if is_final else self.lookahead_dist

        if dist < arrival_radius:
            self.waypoint_idx += 1
            if self.waypoint_idx >= len(self.waypoints):
                return 3
            target = self.waypoints[self.waypoint_idx]

        to_target = target - self.env.robot_pos
        target_angle = np.arctan2(to_target[1], to_target[0])
        angle_diff = self.env._wrap_angle(target_angle - self.env.robot_heading)

        if abs(angle_diff) < self.angle_tolerance:
            return 0  # forward
        elif angle_diff > 0:
            return 1  # turn left
        else:
            return 2  # turn right
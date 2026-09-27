"""
train_ppo.py

Trains a PPO agent on the SAME corridor environment used for the DQN
comparison (train_corridor.py), using stable-baselines3's standard PPO
implementation. Added per mentor request to expand the RL comparison
beyond DQN.

Uses the EXACT same discrete action space (forward/turn left/turn
right/stop) as DQN and A*, so results are directly comparable -- no
action-space reformulation needed, unlike SAC (which requires
continuous actions and was skipped for this reason given time
constraints).

This is a SIMULATION-ONLY comparison (no real hardware deployment) --
matches the scope decision made given deadline constraints.

Install stable-baselines3 first if you don't have it:
    pip install stable-baselines3 --break-system-packages

Usage:
    python train_ppo.py
"""

import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from nav_env import NavEnv


# ------------------------------------------------------------------ #
# Same corridor dimensions and robot/obstacle parameters as
# train_corridor.py, for a fair, directly comparable result.
# ------------------------------------------------------------------ #
ARENA_WIDTH_M = 1.83
ARENA_LENGTH_M = 0.91
ROBOT_RADIUS_M = 0.21
OBSTACLE_RADIUS_M = 0.10

# SET THIS to match whichever condition you're comparing against:
#   0 obstacles: NUM_OBSTACLES = 0, SAVE_PATH = "ppo_corridor.zip"
#   1 obstacle:  NUM_OBSTACLES = 1, SAVE_PATH = "ppo_1obs.zip"
#   2 obstacles: NUM_OBSTACLES = 2, SAVE_PATH = "ppo_2obs.zip"
NUM_OBSTACLES = 2
SAVE_PATH = "ppo_2obs.zip"

GOAL_RADIUS_M = 0.25
MAX_SENSOR_RANGE_M = 2.0
MOVE_SPEED_M = 0.10
TURN_SPEED_RAD = 0.30

TOTAL_TIMESTEPS = 150_000  # roughly comparable training budget to DQN's 1500 episodes x ~100 steps/ep


class PrintProgressCallback(BaseCallback):
    """Mirrors train_corridor.py's periodic success-rate printout AND its
    save-best-checkpoint logic: tracks a rolling 100-episode success rate
    and saves the model whenever a new best is reached, rather than only
    saving whatever the final episode's weights happen to be. Without
    this, a training run that ends on a rough patch (which PPO's
    1-obstacle run showed real signs of) would save a worse model than
    what was actually achieved earlier in training."""

    def __init__(self, save_path, print_every=20, best_window=100, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.print_every = print_every
        self.best_window = best_window
        self.episode_outcomes = []
        self.start_time = time.time()
        self.best_success_rate = -1.0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        new_outcome_this_step = False
        for info, done in zip(infos, dones):
            if not done:
                continue
            if info.get("reached_goal"):
                self.episode_outcomes.append("goal")
            elif info.get("collided"):
                self.episode_outcomes.append("collision")
            else:
                self.episode_outcomes.append("timeout")
            new_outcome_this_step = True

        if new_outcome_this_step and len(self.episode_outcomes) >= self.best_window:
            recent_window = self.episode_outcomes[-self.best_window:]
            current_success_rate = recent_window.count("goal") / self.best_window
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
                self.model.save(self.save_path)

        if new_outcome_this_step and len(self.episode_outcomes) % self.print_every == 0:
            recent = self.episode_outcomes[-self.print_every:]
            success_rate = recent.count("goal") / len(recent)
            collision_rate = recent.count("collision") / len(recent)
            elapsed = time.time() - self.start_time
            print(
                f"Episode {len(self.episode_outcomes):5d} | "
                f"success: {success_rate:5.1%} | "
                f"collision: {collision_rate:5.1%} | "
                f"best (window={self.best_window}): {self.best_success_rate:5.1%} | "
                f"elapsed: {elapsed:6.1f}s"
            )
        return True


def main():
    print(f"Corridor arena: {ARENA_WIDTH_M}m x {ARENA_LENGTH_M}m")
    print(f"Obstacles: {NUM_OBSTACLES} | Save path: {SAVE_PATH}")

    env = NavEnv(
        arena_width=ARENA_WIDTH_M,
        arena_length=ARENA_LENGTH_M,
        num_obstacles=NUM_OBSTACLES,
        obstacle_radius=OBSTACLE_RADIUS_M,
        robot_radius=ROBOT_RADIUS_M,
        goal_radius=GOAL_RADIUS_M,
        max_sensor_range=MAX_SENSOR_RANGE_M,
        move_speed=MOVE_SPEED_M,
        turn_speed=TURN_SPEED_RAD,
        max_steps=150,
    )

    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    callback = PrintProgressCallback(save_path=SAVE_PATH, print_every=20, best_window=100)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    print(f"\nTraining complete. Best checkpoint (success rate {callback.best_success_rate:.1%} over a {callback.best_window}-episode window) saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()

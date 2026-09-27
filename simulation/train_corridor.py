"""
train_corridor.py

Trains a DQN agent in a NavEnv sized to match the real plywood test
track (~0.91m wide x 1.83m long), instead of the original 10m x 10m
open arena. This closes the sim-to-real "domain gap" that caused the
robot to oscillate endlessly when deployed in the small corridor --
it had only ever learned to navigate open space.

Run this on your PC (GPU-accelerated), same as the original train.py.
Produces nav_dqn_corridor.pt -- a SEPARATE file from nav_dqn.pt, so
your original open-arena model isn't overwritten (useful to keep both
for the sim-to-real domain gap comparison in your Polygence writeup).

Usage:
    python train_corridor.py
"""

import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from nav_env import NavEnv
from dqn_agent import DQNAgent


# ------------------------------------------------------------------ #
# Corridor dimensions -- matches the real plywood test track.
#
# IMPORTANT: in the simulation's motion model, "forward" (heading=0)
# always moves along the X axis, which is controlled by arena_width.
# So arena_width must be the corridor's LONG dimension (the direction
# the robot actually walks down), and arena_length is the SHORT/side
# dimension -- NOT their literal real-world "width"/"length" labels.
# Update these if you re-measure or use a different track.
# ------------------------------------------------------------------ #
ARENA_WIDTH_M = 1.83   # long dimension (~6 ft) -- this is the robot's forward axis
ARENA_LENGTH_M = 0.91  # short dimension (~3 ft) -- this is the sideways axis

# Robot/sensor physical parameters -- should roughly match your real
# robot's footprint and the HC-SR04's effective range in a small room.
#
# ROBOT_RADIUS_M: your real chassis is 360mm x 191mm (measured from
# CAD), NOT a small ~9cm robot. Since it rotates in place, the point
# that matters for collision clearance is the farthest corner from
# its center -- the half-diagonal of the rectangle, ~0.204m. Using
# the true footprint (rather than an underestimate) is what makes the
# model learn to keep real clearance before turning, instead of
# swinging its rear end into obstacles it thought were already clear.
ROBOT_RADIUS_M = 0.21     # half-diagonal of the real 360mm x 191mm chassis, rounded up slightly
OBSTACLE_RADIUS_M = 0.10  # size of any small real obstacles you might add to the track
NUM_OBSTACLES = 0        # the real track is empty -- matches actual test conditions; with the corrected robot size, the corridor's usable sideways space is only ~0.49m, tight enough that an obstacle could make some placements nearly unsolvable
GOAL_RADIUS_M = 0.25      # increased alongside robot size -- a smaller robot goal radius isn't meaningful for a robot this large
MAX_SENSOR_RANGE_M = 2.0  # sensor readings beyond this aren't meaningful in a space this small

# Movement per action -- smaller than the original open-arena training
# since the whole corridor is only ~1.8m long; large steps would let
# the robot cross the entire space in 2-3 actions, giving little room
# to actually learn fine-grained control.
MOVE_SPEED_M = 0.10
TURN_SPEED_RAD = 0.30

NUM_EPISODES = 1500
PRINT_EVERY = 20
SAVE_PATH = "nav_dqn_corridor.pt"


def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    print(f"Corridor arena: {ARENA_WIDTH_M}m x {ARENA_LENGTH_M}m")

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
        max_steps=150,  # shorter episodes make sense in a much smaller space
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, device)

    episode_rewards = []
    episode_outcomes = []  # "goal", "collision", or "timeout"
    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, float(terminated))
            agent.update()

            state = next_state
            episode_reward += reward

        if info.get("reached_goal"):
            outcome = "goal"
        elif info.get("collided"):
            outcome = "collision"
        else:
            outcome = "timeout"

        episode_rewards.append(episode_reward)
        episode_outcomes.append(outcome)

        if episode % PRINT_EVERY == 0:
            recent_rewards = episode_rewards[-PRINT_EVERY:]
            recent_outcomes = episode_outcomes[-PRINT_EVERY:]
            success_rate = recent_outcomes.count("goal") / len(recent_outcomes)
            collision_rate = recent_outcomes.count("collision") / len(recent_outcomes)
            elapsed = time.time() - start_time
            print(
                f"Ep {episode:5d} | "
                f"avg reward: {np.mean(recent_rewards):7.2f} | "
                f"success: {success_rate:5.1%} | "
                f"collision: {collision_rate:5.1%} | "
                f"epsilon: {agent.epsilon():.3f} | "
                f"elapsed: {elapsed:6.1f}s"
            )

    agent.save(SAVE_PATH)
    print(f"\nTraining complete. Model saved to {SAVE_PATH}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(episode_rewards, alpha=0.3, label="raw")
    ax1.plot(
        range(len(moving_average(episode_rewards))),
        moving_average(episode_rewards),
        label="moving avg (50 ep)",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Corridor training reward")
    ax1.legend()

    window = 50
    success_over_time = [
        episode_outcomes[max(0, i - window):i + 1].count("goal") / len(episode_outcomes[max(0, i - window):i + 1])
        for i in range(len(episode_outcomes))
    ]
    ax2.plot(success_over_time)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success rate (rolling 50 ep)")
    ax2.set_title("Corridor goal-reaching success rate")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("training_progress_corridor.png", dpi=150)
    print("Saved training plot to training_progress_corridor.png")


if __name__ == "__main__":
    main()

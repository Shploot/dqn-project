"""
train.py

Trains a DQN agent in the NavEnv simulation and saves the resulting
model weights to nav_dqn.pt. Run this on your PC (GPU-accelerated).

Usage:
    python train.py
"""

import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from nav_env import NavEnv
from dqn_agent import DQNAgent


NUM_EPISODES = 1500
PRINT_EVERY = 20
SAVE_PATH = "nav_dqn.pt"


def moving_average(data, window=50):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    env = NavEnv()
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

    # plot training progress
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(episode_rewards, alpha=0.3, label="raw")
    ax1.plot(
        range(len(moving_average(episode_rewards))),
        moving_average(episode_rewards),
        label="moving avg (50 ep)",
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Training reward")
    ax1.legend()

    window = 50
    success_over_time = [
        episode_outcomes[max(0, i - window):i + 1].count("goal") / len(episode_outcomes[max(0, i - window):i + 1])
        for i in range(len(episode_outcomes))
    ]
    ax2.plot(success_over_time)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success rate (rolling 50 ep)")
    ax2.set_title("Goal-reaching success rate")
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=150)
    print("Saved training plot to training_progress.png")


if __name__ == "__main__":
    main()

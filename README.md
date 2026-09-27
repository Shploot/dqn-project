# DQN vs. A* vs. PPO: Autonomous Navigation Comparison

Independent research project (Polygence) comparing a Deep Q-Network (DQN)
and Proximal Policy Optimization (PPO) against classical A* path planning
for autonomous robot navigation with obstacle avoidance -- evaluated in
simulation and, for DQN and A*, on real hardware.

## Overview

A custom simulation environment models a differential-drive robot with
three ultrasonic sensors navigating a narrow corridor (0.91m x 1.83m)
toward a goal, with 0, 1, or 2 obstacles present. DQN, PPO, and A* were
all trained/implemented against this same environment. DQN and A* were
then deployed on a real Raspberry Pi 5 robot built from low-cost,
hobbyist components, and tested across 18 real trials.

## Key findings

- **A performance crossover**: with no obstacles, DQN reached the goal
  in far fewer steps than A* (7.7 vs. 14.0 average). The moment even one
  obstacle was introduced, this reversed sharply -- DQN's step count and
  turning behavior jumped to a high plateau that barely changed between
  one and two obstacles, while A* degraded more gradually and
  proportionally with obstacle count.
- **Success rate vs. clean avoidance**: both methods reached the goal in
  100% of real trials across every condition, but every obstacle-present
  trial for both methods involved physical contact with an obstacle at
  some point -- success rate alone significantly overstates how cleanly
  either method actually avoided obstacles.
- **Six real sim-to-real hardware defects** were found, diagnosed from
  logged data, and fixed during deployment, including an inverted
  encoder sign convention, load-dependent wheel-speed asymmetry, and a
  false-positive goal-detection bug during mechanical stalls.
- In simulation, **PPO slightly outperformed DQN** at both
  obstacle-present densities, though both showed the same sharp
  difficulty jump from zero to one obstacle -- suggesting this is a
  property of the task itself, not one algorithm's instability.

## Repository structure

- **`simulation/`** -- training code that runs on a PC: the custom
  Gymnasium-style environment (`nav_env.py`), DQN implementation
  (`dqn_agent.py`, `train_corridor.py`), PPO training
  (`train_ppo.py`, via stable-baselines3), and the A* planner
  (`astar_planner.py`).
- **`hardware/`** -- code that runs on the Raspberry Pi 5: real-robot
  deployment for DQN (`deploy.py`) and A* (`deploy_astar.py`),
  encoder-based odometry (`odometry.py`), a reactive-only baseline
  (`reactive_avoid.py`), and a `calibration/` subfolder of hardware
  diagnostic/calibration scripts.
- **`models/`** -- trained model weights: DQN (`.pt`) and PPO (`.zip`)
  models for each obstacle-density condition.
- **`logs/`** -- raw terminal logs from all 18 real hardware trials
  (DQN and A*, 0/1/2 obstacles, 3 trials each).
- **`analysis/`** -- tools to parse and aggregate trial logs
  (`analyze_run.py`, `compare_experiments.py`) and their outputs
  (comparison charts, a combined results CSV).

## Hardware

Raspberry Pi 5, 3x HC-SR04 ultrasonic sensors, TB6612FNG motor driver,
2x DFRobot FIT0450 DC gearmotors with quadrature encoders, 3D-printed
chassis (360mm x 191mm).

## Status

Actively in progress. A research paper based on this work is being
developed with my Polygence mentor.

---
Mentored by Morteza, Polygence.

"""
Train a PPO agent for the QuantumStabilizationEnv.

The environment (src/rl_env.py) presents continuous weak measurement of a qubit
as a reinforcement-learning control problem. This module builds the environment
and trains a PPO policy with stable-baselines3.

Run:
    python -m src.train_agent
"""

import os
from typing import Optional, Tuple

from src.rl_env import QuantumStabilizationEnv
from src.hamiltonian.lindblad import DecoherenceParams
from src.hamiltonian.stochastic import MeasurementParams


def build_env(
    dt: float = 0.05,
    max_steps: int = 200,
    T1: float = 20.0,
    T2: float = 15.0,
    strength: float = 5.0,
) -> QuantumStabilizationEnv:
    """Construct the qubit stabilization environment with the given physics."""
    decoherence = DecoherenceParams(T1=T1, T2=T2)
    measurement = MeasurementParams(strength=strength)
    return QuantumStabilizationEnv(
        dt=dt,
        max_steps=max_steps,
        decoherence=decoherence,
        measurement=measurement,
    )


def train(
    total_timesteps: int = 5000,
    env: Optional[QuantumStabilizationEnv] = None,
    seed: Optional[int] = None,
    verbose: int = 0,
    save_path: Optional[str] = None,
    n_steps: int = 512,
    batch_size: int = 64,
):
    """
    Train a PPO policy on the stabilization environment.

    stable-baselines3 is imported lazily so this module remains importable (and
    ``build_env`` testable) without the heavy RL dependency installed.

    Returns
    -------
    (model, env) : the trained PPO model and the environment it was trained on.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    if env is None:
        env = build_env()
    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model.save(save_path)

    return model, env


def evaluate(model, env, n_eval_episodes: int = 10) -> Tuple[float, float]:
    """Evaluate a trained policy; returns (mean_reward, std_reward)."""
    from stable_baselines3.common.evaluation import evaluate_policy

    return evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)


def main():
    """Train, save, and evaluate a PPO stabilizer (demonstration run)."""
    env = build_env()
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "ppo_quantum_stabilizer"
    )
    model, env = train(total_timesteps=5000, env=env, verbose=1, save_path=save_path)
    print(f"Model saved to {save_path}")

    mean_reward, std_reward = evaluate(model, env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
